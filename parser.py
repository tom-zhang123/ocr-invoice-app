"""
表格解析器：把 RapidOCR 的输出（文字+坐标）还原成结构化行列。
针对 KERRY 入库单版式：PALLET / BOX / SKU# / Name / Qty / EXP / MFG / Batch
"""
import re
import json
from datetime import date

# 列的 x 区间（像素，基于样图实测；不同扫描件可按比例微调）
COLUMNS = [
    ("pallet",  90, 175),
    ("box",     175, 235),
    ("sku",     235, 390),
    ("name",    390, 725),
    ("qty",     725, 785),
    ("exp",     785, 905),
    ("mfg",     905, 1025),
    ("batch",   1025, 1185),
]

HEADER_Y = 155          # 表头/单据头在这条线以下开始算数据行
TABLE_BOTTOM = 1100     # 表格底部；下方内容属于备注/签名区域
ROW_TOLERANCE = 16      # 同一行 y 中心容差（像素）


def _col_of(x_center):
    for name, x1, x2 in COLUMNS:
        if x1 <= x_center < x2:
            return name
    return None


def parse_items(ocr_result):
    """ocr_result: RapidOCR 返回的 list[[box, text, score]]"""
    items = []
    for box, text, score in ocr_result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "text": text.strip(),
            "score": float(score),
            "xc": (min(xs) + max(xs)) / 2,
            "yc": (min(ys) + max(ys)) / 2,
            "x1": min(xs),
            "x2": max(xs),
            "y1": min(ys),
            "y2": max(ys),
        })
    return items


def extract_header(items):
    """从单据头提取 PO 号、入库日期等"""
    header = {
        "po_no": "",
        "inbound_date": "",
        "page_number": None,
        "total_pages": None,
    }
    for it in items:
        t = it["text"]
        m = re.search(r"PO\s*(\d{5,})", t, re.I)
        if m:
            header["po_no"] = "PO" + m.group(1)
        m = re.search(r"INBOUND\s*DATE\s*([\d/]+)", t, re.I)
        if m:
            header["inbound_date"] = m.group(1)
        m = re.search(r"PAGE\s*(\d+)\s*OF\s*(\d+)", t, re.I)
        if m:
            header["page_number"] = int(m.group(1))
            header["total_pages"] = int(m.group(2))
    return header


def extract_total_qty(items, columns=COLUMNS):
    """从合计行提取印刷数量；无法同时确认 Total 标记和数量时返回 None。"""
    total_markers = [
        item for item in items
        if re.search(r"TOTA[L1I]|合计", item["text"], re.I)
    ]
    if not total_markers:
        return None

    qty_x1, qty_x2 = next(
        (x1, x2) for name, x1, x2 in columns if name == "qty"
    )
    candidates = []
    for marker in total_markers:
        for item in items:
            if abs(item["yc"] - marker["yc"]) > ROW_TOLERANCE * 2:
                continue
            if not qty_x1 <= item["xc"] < qty_x2:
                continue
            digits = clean_digits(item["text"])
            if digits:
                candidates.append((abs(item["yc"] - marker["yc"]), int(digits)))
    if not candidates:
        return None
    return min(candidates, key=lambda value: value[0])[1]


def parse_table(items):
    """把数据项聚类成行，再按列归位。返回 list[dict]"""
    data = [it for it in items if HEADER_Y < it["yc"] <= TABLE_BOTTOM]
    data.sort(key=lambda it: it["yc"])

    # 按 y 聚类成行
    rows = []
    for it in data:
        placed = False
        for row in rows:
            if abs(row["yc"] - it["yc"]) <= ROW_TOLERANCE:
                row["items"].append(it)
                # 更新行中心为均值
                row["yc"] = sum(x["yc"] for x in row["items"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append({"yc": it["yc"], "items": [it]})

    result = []
    for row in rows:
        cells = {name: [] for name, _, _ in COLUMNS}
        for it in row["items"]:
            col = _col_of(it["xc"])
            if col:
                cells[col].append(it)
        record = {}
        conf = {}
        for name, _, _ in COLUMNS:
            fragments = sorted(cells[name], key=lambda x: x["x1"])
            text = clean_text(" ".join(f["text"] for f in fragments))
            record[name] = text
            # 空格没有识别证据，不能按 100% 置信度处理。
            conf[name] = round(min((f["score"] for f in fragments)), 3) if fragments else None

        record["raw_sku"] = record["sku"]
        record = postprocess(record)
        record = reconcile_handwritten_quantity(record)
        if is_data_row_candidate(record):
            record["confidence"] = conf
            key_fields = ["sku", "qty", "exp", "mfg", "batch"]
            scores = [conf[k] for k in key_fields if conf[k] is not None]
            record["row_confidence"] = round(sum(scores) / len(scores), 3) if scores else 0.0
            record["validation_errors"] = validate_row(record)
            result.append(record)
    return result


def clean_text(text):
    """清理 OCR 文本：去表格线斜杠、多余空格、末尾标点"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[/／]+$", "", text).strip()      # 去末尾表格线斜杠
    text = re.sub(r"^[/／]+", "", text).strip()
    return text


def clean_sku(text):
    """只移除 SKU 两端的表格线和空白，保留内部疑似错字供人工复核。"""
    if not text:
        return ""
    text = re.sub(r"\s+", "", clean_text(text))
    text = text.strip("/／\\|:：;；,.，。")
    edge_trimmed = re.sub(r"^\D+|\D+$", "", text)
    if edge_trimmed != text and is_valid_gtin(edge_trimmed):
        return edge_trimmed
    if not text.isdigit() or is_valid_gtin(text):
        return text

    expected_length = len(text) - 1
    if expected_length in (8, 12, 13, 14):
        candidates = [text[:expected_length], text[-expected_length:]]
        valid_candidates = list(dict.fromkeys(
            candidate for candidate in candidates if is_valid_gtin(candidate)
        ))
        if len(valid_candidates) == 1:
            return valid_candidates[0]
    return text


def is_valid_gtin(value):
    """校验 GTIN-8/12/13/14 的末位校验码。"""
    if not re.fullmatch(r"\d{8}|\d{12}|\d{13}|\d{14}", value or ""):
        return False
    body = value[:-1]
    total = sum(
        int(digit) * (3 if index % 2 == 0 else 1)
        for index, digit in enumerate(reversed(body))
    )
    expected_check_digit = (10 - total % 10) % 10
    return expected_check_digit == int(value[-1])


# 仅用于纯数字语境（Qty）的保守字符纠错：只把明显的 OCR 形似认错改成数字
_DIGIT_MAP = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "S": "5", "s": "5",
})


def clean_digits(text):
    """只保留数字（Qty/箱数用），仅在纯数字语境做形似纠错"""
    s = text.translate(_DIGIT_MAP)
    return re.sub(r"\D", "", s)


def clean_date(text):
    """
    日期轻量清洗：统一分隔符、去空格，保留 OCR 原文，不强行重组避免改错。
    """
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"[.／:：]", "-", s)     # . : / 统一成 -
    s = re.sub(r"\s+", "", s)
    return s


def clean_batch(text):
    """批次号：只去杂符号，原样保留字母数字，不做字符映射（字母有业务含义）"""
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    return s


def parse_batch_candidates(value):
    """Parse a JSON array or delimiter-separated batch candidate string."""
    if not value:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = re.split(r"[\s,，;；]+", value)
    if not isinstance(parsed, list):
        raise ValueError("批次候选必须是数组或分隔字符串")
    unique = {}
    for item in parsed:
        candidate = str(item).strip()
        normalized = clean_batch(candidate).upper()
        if not normalized:
            continue
        if len(normalized) > 64:
            raise ValueError("单个批次号不能超过 64 个字符")
        unique.setdefault(normalized, candidate)
    if len(unique) > 50:
        raise ValueError("单次最多提交 50 个批次号")
    return list(unique.values())


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def match_batch_candidate(raw_value, candidates):
    normalized = clean_batch(str(raw_value or "")).upper()
    if not normalized or not candidates:
        return None
    scored = sorted(
        (
            _edit_distance(normalized, clean_batch(candidate).upper()),
            candidate,
        )
        for candidate in candidates
    )
    threshold = max(1, int(len(normalized) * 0.25))
    unique_best = len(scored) == 1 or scored[0][0] < scored[1][0]
    return scored[0][1] if unique_best and scored[0][0] <= threshold else None


def reconcile_batch_candidate(row, candidates):
    if not candidates:
        return row
    raw_value = str(row.get("batch", ""))
    row["raw_batch"] = raw_value
    matched = match_batch_candidate(raw_value, candidates)
    if matched is not None:
        row["batch"] = matched
        row.setdefault("corrections", {})["batch"] = "allowed_batch_match"
        row.get("review_reasons", {}).pop("batch", None)
    else:
        row["batch"] = ""
        row.setdefault("review_reasons", {})["batch"] = "manual_required"
    return row


def postprocess(row):
    """对单行结果做轻量清洗，不强行纠错，避免把对的改错"""
    row["sku"] = clean_sku(row.get("sku", ""))
    row["qty"] = clean_digits(row.get("qty", ""))
    row["box"] = clean_digits(row.get("box", ""))
    row["exp"] = clean_date(row.get("exp", ""))
    row["mfg"] = clean_date(row.get("mfg", ""))
    row["batch"] = clean_batch(row.get("batch", ""))
    return row


def is_valid_date(text):
    """Require YYYY-MM-DD and validate its calendar date."""
    if not text:
        return True
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return False
    year, month, day = match.groups()
    if not 1900 <= int(year) <= 2200 or not 1 <= int(month) <= 12:
        return False
    try:
        date(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def validate_row(row):
    """返回字段级结构错误；置信度阈值仍由调用方决定。"""
    errors = {}
    sku = row.get("sku", "")
    qty = row.get("qty", "")

    if not sku:
        errors["sku"] = "missing"
    elif not re.fullmatch(r"\d{8,}", sku):
        errors["sku"] = "invalid_format"

    if not qty:
        errors["qty"] = "missing"
    elif not re.fullmatch(r"[1-9]\d*", qty):
        errors["qty"] = "invalid_format"

    for field in ("exp", "mfg"):
        if row.get(field) and not is_valid_date(row[field]):
            errors[field] = "invalid_format"

    return errors


def reconcile_handwritten_quantity(row):
    """仅在箱号佐证时移除被识别成数字 1 的手写末尾勾线。"""
    qty = str(row.get("qty", ""))
    box = str(row.get("box", ""))
    if len(qty) > 1 and qty.endswith("1") and qty[:-1] == box:
        row["raw_qty"] = qty
        row["qty"] = qty[:-1]
        row.setdefault("corrections", {})["qty"] = "trailing_check_mark"
    return row


def reconcile_quantities_with_total(rows, document_total):
    """Resolve OCR disagreements only when one candidate combination matches the total."""
    if document_total is None:
        return rows

    current_total = sum(int(row.get("qty") or 0) for row in rows)
    target_delta = int(document_total) - current_total
    if target_delta == 0:
        return rows

    candidate_rows = []
    for index, row in enumerate(rows):
        if (row.get("review_reasons") or {}).get("qty") != "model_disagreement":
            continue
        current_value = int(row.get("qty") or 0)
        raw_candidates = (row.get("field_candidates") or {}).get("qty") or []
        if isinstance(raw_candidates, dict):
            raw_candidates = [raw_candidates]
        values = {current_value}
        for candidate in raw_candidates:
            value = clean_digits(str(candidate.get("text", "")))
            if re.fullmatch(r"[1-9]\d*", value):
                values.add(int(value))
        if len(values) > 1:
            candidate_rows.append((index, current_value, sorted(values)))

    paths = {0: [()]}
    for index, current_value, values in candidate_rows:
        next_paths = {}
        for accumulated_delta, alternatives in paths.items():
            for value in values:
                delta = accumulated_delta + value - current_value
                changes = () if value == current_value else ((index, value),)
                bucket = next_paths.setdefault(delta, [])
                for alternative in alternatives:
                    candidate_path = alternative + changes
                    if candidate_path not in bucket and len(bucket) < 2:
                        bucket.append(candidate_path)
        paths = next_paths
        if len(paths) > 50000:
            paths = dict(sorted(
                paths.items(),
                key=lambda item: abs(item[0] - target_delta),
            )[:50000])

    solutions = [path for path in paths.get(target_delta, []) if path]
    if len(solutions) == 1:
        for index, corrected_value in solutions[0]:
            row = rows[index]
            row["raw_qty"] = row["qty"]
            row["qty"] = str(corrected_value)
            row.setdefault("corrections", {})["qty"] = "invoice_total_candidates"
            row.get("review_reasons", {}).pop("qty", None)
        return rows

    single_digit_candidates = []
    for row in rows:
        if (row.get("review_reasons") or {}).get("qty") != "model_disagreement":
            continue
        raw_qty = str(row.get("qty", ""))
        if not re.fullmatch(r"[1-9]\d*", raw_qty):
            continue
        corrected_qty = str(int(raw_qty) + target_delta)
        if (
            not re.fullmatch(r"[1-9]\d*", corrected_qty)
            or len(corrected_qty) != len(raw_qty)
        ):
            continue
        if sum(left != right for left, right in zip(raw_qty, corrected_qty)) == 1:
            single_digit_candidates.append((row, raw_qty, corrected_qty))

    if len(single_digit_candidates) == 1:
        row, raw_qty, corrected_qty = single_digit_candidates[0]
        row["raw_qty"] = raw_qty
        row["qty"] = corrected_qty
        row.setdefault("corrections", {})["qty"] = "invoice_total_single_digit"
        row.get("review_reasons", {}).pop("qty", None)
        return rows

    excess = current_total - int(document_total)
    if excess <= 0:
        return rows

    candidates = []
    for row in rows:
        qty = str(row.get("qty", ""))
        if len(qty) <= 1 or not qty.endswith("1"):
            continue
        corrected = qty[:-1]
        if not corrected or int(qty) - int(corrected) != excess:
            continue
        candidates.append((row, qty, corrected))

    if len(candidates) == 1:
        row, raw_qty, corrected_qty = candidates[0]
        row["raw_qty"] = raw_qty
        row["qty"] = corrected_qty
        row.setdefault("corrections", {})["qty"] = "invoice_total_check"
    return rows


def reconcile_date_years(row, header=None, current_year=None):
    """Repair a unique one-character year error and flag implausible dates."""
    header = header or {}
    current_year = int(current_year or date.today().year)
    inbound_years = [
        int(value)
        for value in re.findall(r"(?:19|20|21)\d{2}", str(header.get("inbound_date", "")))
    ]
    anchor_year = inbound_years[0] if inbound_years else current_year
    allowed_years = {
        "mfg": range(anchor_year - 3, anchor_year + 2),
        "exp": range(anchor_year, anchor_year + 7),
    }

    for field, valid_years in allowed_years.items():
        value = str(row.get(field, ""))
        match = re.fullmatch(r"(\d{4})(-\d{2}-\d{2})", value)
        if not match:
            continue
        raw_year, suffix = match.groups()
        year = int(raw_year)
        valid_years = list(valid_years)
        if year in valid_years:
            continue
        distances = {
            candidate: sum(left != right for left, right in zip(raw_year, str(candidate)))
            for candidate in valid_years
        }
        minimum_distance = min(distances.values())
        candidates = [
            candidate
            for candidate, distance in distances.items()
            if distance == minimum_distance
        ]
        if minimum_distance <= 2 and len(candidates) == 1:
            row[f"raw_{field}"] = value
            row[field] = f"{candidates[0]}{suffix}"
            row.setdefault("corrections", {})[field] = "context_year"
        else:
            row.setdefault("review_reasons", {})[field] = "implausible_year"

    exp = str(row.get("exp", ""))
    mfg = str(row.get("mfg", ""))
    if is_valid_date(exp) and is_valid_date(mfg) and exp and mfg:
        exp_key = tuple(int(value) for value in exp.split("-"))
        mfg_key = tuple(int(value) for value in mfg.split("-"))
        if exp_key < mfg_key:
            row.setdefault("review_reasons", {})["exp"] = "date_order"
            row.setdefault("review_reasons", {})["mfg"] = "date_order"

    for field in ("exp", "mfg"):
        value = str(row.get(field, ""))
        implausible = (row.get("review_reasons") or {}).get(field) == "implausible_year"
        if value and (not is_valid_date(value) or implausible):
            row.setdefault(f"raw_{field}", value)
            row[field] = ""
            row.setdefault("corrections", {})[field] = "invalid_date_cleared"
            row.setdefault("review_reasons", {})[field] = "manual_required"
    return row


def is_data_row_candidate(row):
    """保留疑似商品行，即使 SKU 识别异常；排除表头、合计和备注噪声。"""
    values = [str(row.get(name, "")).strip() for name, _, _ in COLUMNS]
    if not any(values):
        return False

    if is_summary_row(row):
        return False

    raw_sku = str(row.get("raw_sku", row.get("sku", ""))).strip()
    if re.fullmatch(r"\d{8,}", row.get("sku", "")):
        return True
    if raw_sku and (row.get("name") or row.get("qty")):
        return True

    support_fields = ("exp", "mfg", "batch", "pallet", "box")
    support_count = sum(bool(row.get(field)) for field in support_fields)
    return bool(row.get("name") and row.get("qty") and support_count >= 1)


def is_summary_row(row):
    values = [str(row.get(name, "")).strip() for name, _, _ in COLUMNS]
    combined = " ".join(values)
    raw_sku = str(row.get("raw_sku", row.get("sku", ""))).strip()
    return bool(
        re.search(r"\bTOTAL\b|合计", combined, re.I)
        or re.match(r"^PO\s*\d+", raw_sku, re.I)
        or re.fullmatch(r"SKU\s*#?", raw_sku, re.I)
    )


def extract_notes(items, table_bottom=1100):
    """提取表格下方的手写备注（y > table_bottom 的文字，按行拼接）"""
    notes_items = [it for it in items if it["yc"] > table_bottom]
    if not notes_items:
        return ""
    notes_items.sort(key=lambda it: (it["yc"], it["x1"]))
    # 按 y 聚成多行
    lines = []
    for it in notes_items:
        placed = False
        for line in lines:
            if abs(line["yc"] - it["yc"]) <= 18:
                line["texts"].append((it["x1"], it["text"]))
                placed = True
                break
        if not placed:
            lines.append({"yc": it["yc"], "texts": [(it["x1"], it["text"])]})
    result_lines = []
    for line in sorted(lines, key=lambda l: l["yc"]):
        txt = " ".join(t for _, t in sorted(line["texts"]))
        txt = txt.strip()
        # 过滤合计行
        if txt and not re.search(r"PO\d+\s*Total", txt, re.I):
            result_lines.append(txt)
    return " | ".join(result_lines)


def parse(ocr_result):
    items = parse_items(ocr_result)
    header = extract_header(items)
    rows = parse_table(items)
    notes = extract_notes(items)
    return {"header": header, "rows": rows, "row_count": len(rows), "notes": notes}


if __name__ == "__main__":
    import cv2

    from ocr_models import recognize_page

    result = recognize_page(cv2.imread("sample.jpg"))
    out = parse(result)
    print("HEADER:", out["header"])
    print("ROWS:", out["row_count"])
    for i, r in enumerate(out["rows"], 1):
        print(f'{i:2d} | {r["pallet"]:>5} | {r["box"]:>5} | {r["sku"]:>14} | qty={r["qty"]:>5} | exp={r["exp"]:>12} | mfg={r["mfg"]:>12} | batch={r["batch"]:>14}')
