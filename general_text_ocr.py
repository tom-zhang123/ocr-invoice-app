"""Free text recognition sharing the package model without business-field parsing."""
import time
import math
import re

from package_ocr import _decode_image, _resize_if_needed, _contrast_variant, _elapsed_ms
from package_ocr_models import recognize_package_image, recognize_package_line
from text_dates import parse_text_dates


def recognize_text(image_bytes, date_order="auto", recognizer=recognize_package_image,
                   text_mode="paragraph", line_recognizer=recognize_package_line):
    if date_order not in {"auto", "dmy", "mdy"}:
        raise ValueError("date_order must be auto, dmy or mdy")
    if text_mode not in {"identifier", "line", "paragraph"}:
        raise ValueError("text_mode must be identifier, line or paragraph")
    started = time.perf_counter()
    image = _resize_if_needed(_decode_image(image_bytes))
    timings = {"decode": _elapsed_ms(started), "line_ocr": 0, "primary_ocr": 0, "enhanced_ocr": 0}
    lines = []
    if text_mode == "identifier":
        stage_started = time.perf_counter()
        try:
            candidate = line_recognizer(image)
            if confident_single_line(candidate):
                lines = candidate
        except Exception:
            # The full pipeline remains the fallback for unsupported or failed line inference.
            lines = []
        timings["line_ocr"] = _elapsed_ms(stage_started)
    if not lines:
        stage_started = time.perf_counter()
        lines = recognizer(image)
        timings["primary_ocr"] = _elapsed_ms(stage_started)
    # Retry only an empty reading; do not merge two readings and duplicate a paragraph.
    if not any(str(line.get("text", "")).strip() for line in lines):
        stage_started = time.perf_counter()
        lines = recognizer(_contrast_variant(image))
        timings["enhanced_ocr"] = _elapsed_ms(stage_started)
    ordered = reading_lines(lines)
    raw_text = "\n".join(line["text"] for line in ordered)
    return {
        "raw_text": raw_text,
        "lines": ordered,
        **parse_text_dates(raw_text, date_order),
        "timing_ms": {**timings, "total": _elapsed_ms(started)},
        "model": "PP-OCRv6-small",
    }


def confident_single_line(lines):
    if len(lines) != 1 or not str(lines[0].get("text", "")).strip():
        return False
    try:
        text = str(lines[0]["text"]).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]{1,127}", text) or not any(c.isdigit() for c in text):
            return False
        score = float(lines[0].get("confidence", 0))
        return math.isfinite(score) and score >= 0.95
    except (TypeError, ValueError):
        return False


def reading_lines(lines):
    rows = []
    for item in sorted(lines, key=lambda item: min(p[1] for p in item["box"])):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        box = item["box"]
        top, bottom = min(p[1] for p in box), max(p[1] for p in box)
        center, height = (top + bottom) / 2, max(1, bottom - top)
        row = next((row for row in rows if abs(row["center"] - center) <= min(row["height"], height) * 0.45), None)
        if row is None:
            row = {"center": center, "height": height, "items": []}
            rows.append(row)
        row["items"].append(item)
    result = []
    for row in sorted(rows, key=lambda row: row["center"]):
        items = sorted(row["items"], key=lambda item: min(p[0] for p in item["box"]))
        result.append({
            "text": " ".join(str(item["text"]).strip() for item in items),
            "confidence": min(float(item["confidence"]) for item in items),
        })
    return result
