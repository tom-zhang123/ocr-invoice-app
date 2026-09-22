"""KERRY 入库单：整页定位后按单元格使用手写/泰文模型识别。"""
import re
import time

import cv2
import numpy as np

from ocr_models import recognize_lines, recognize_page, recognize_region
from parser import (
    COLUMNS,
    HEADER_Y,
    ROW_TOLERANCE,
    TABLE_BOTTOM,
    clean_batch,
    clean_digits,
    clean_sku,
    clean_text,
    extract_header,
    extract_notes,
    extract_total_qty,
    is_data_row_candidate,
    is_summary_row,
    is_valid_gtin,
    is_valid_date,
    match_batch_candidate,
    parse_items,
    reconcile_batch_candidate,
    reconcile_date_years,
    reconcile_handwritten_quantity,
    reconcile_quantities_with_total,
    validate_row,
)


COLUMN_BOUNDS = {name: (x1, x2) for name, x1, x2 in COLUMNS}
WITHOUT_BATCH_BOUNDS = {
    "pallet": (68, 147),
    "box": (153, 235),
    "sku": (241, 440),
    "name": (446, 839),
    "qty": (845, 934),
    "exp": (940, 1082),
    "mfg": (1088, 1231),
}
FIELD_ORDER = tuple(name for name, _, _ in COLUMNS)
HANDWRITTEN_FIELDS = ("pallet", "box", "qty", "exp", "mfg", "batch")
REFERENCE_WIDTH = 1280
REFERENCE_TABLE_LEFT = 74
REFERENCE_TABLE_TOP = 124
REFERENCE_TABLE_WIDTH = 1153
COLUMN_SCALE = 2.0
COLUMN_GAP = 24
FALLBACK_MIN_CONFIDENCE = {
    "sku": 0.90,
    "pallet": 0.75,
    "box": 0.80,
    "qty": 0.85,
    "exp": 0.80,
    "mfg": 0.80,
    "batch": 0.80,
}


def detect_table_geometry(image):
    """Return table corners plus header/bottom lines for photographed invoices."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 18), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 25))),
    )
    grid = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    boxes = [
        box for box in boxes
        if box[2] >= width * 0.5 and box[3] >= height * 0.35
    ]
    if not boxes:
        return None
    table_x, table_y, table_width, table_height = max(
        boxes,
        key=lambda box: box[2] * box[3],
    )

    projection = (horizontal[
        table_y:table_y + table_height,
        table_x:table_x + table_width,
    ] > 0).sum(axis=1)
    smoothed = np.convolve(projection, np.ones(9, dtype=np.int32), mode="same")
    line_indexes = np.where(smoothed > max(100, table_width * 0.3))[0]
    groups = []
    for index in line_indexes:
        if not groups or index > groups[-1][1] + 1:
            groups.append([int(index), int(index)])
        else:
            groups[-1][1] = int(index)
    line_centers = [table_y + (start + end) / 2 for start, end in groups]
    if len(line_centers) < 3:
        return None

    top_y = line_centers[0]
    header_y = line_centers[1]
    bottom_y = line_centers[-1]
    if bottom_y - top_y < table_height * 0.7:
        return None

    edges = cv2.Canny(gray, 50, 150)
    detected_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(200, int(table_height * 0.45)),
        maxLineGap=40,
    )
    vertical_lines = []
    if detected_lines is not None:
        for detected in detected_lines:
            x1, y1, x2, y2 = map(float, detected.flatten())
            if abs(y2 - y1) < table_height * 0.45:
                continue
            if abs(x2 - x1) > abs(y2 - y1) * 0.15:
                continue
            vertical_lines.append((x1, y1, x2, y2))

    def edge_x(target_x, y_value):
        nearby = []
        for x1, y1, x2, y2 in vertical_lines:
            mid_x = (x1 + x2) / 2
            if abs(mid_x - target_x) <= max(45, table_width * 0.06):
                nearby.extend(((y1, x1), (y2, x2)))
        if len(nearby) < 4:
            return float(target_x)
        ys = np.array([point[0] for point in nearby], dtype=np.float32)
        xs = np.array([point[1] for point in nearby], dtype=np.float32)
        slope, intercept = np.polyfit(ys, xs, 1)
        return float(slope * y_value + intercept)

    left_top = edge_x(table_x, top_y)
    left_bottom = edge_x(table_x, bottom_y)
    right_target = table_x + table_width
    right_top = edge_x(right_target, top_y)
    right_bottom = edge_x(right_target, bottom_y)
    if min(right_top - left_top, right_bottom - left_bottom) < width * 0.45:
        return None

    return {
        "bounds": (table_x, table_y, table_width, table_height),
        "corners": np.float32([
            [left_top, top_y],
            [right_top, top_y],
            [right_bottom, bottom_y],
            [left_bottom, bottom_y],
        ]),
        "top_y": top_y,
        "header_y": header_y,
        "bottom_y": bottom_y,
        "line_centers": line_centers,
    }


def normalize_document_image(image):
    """Align a detected table to the canonical KERRY template coordinate system."""
    height, width = image.shape[:2]
    geometry = detect_table_geometry(image)
    if geometry is not None:
        corners = geometry["corners"]
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        scale = REFERENCE_TABLE_WIDTH / ((top_width + bottom_width) / 2)
        normalized_table_height = max(
            1,
            round((geometry["bottom_y"] - geometry["top_y"]) * scale),
        )
        table_bottom = REFERENCE_TABLE_TOP + normalized_table_height
        output_height = table_bottom + 24
        destination = np.float32([
            [REFERENCE_TABLE_LEFT, REFERENCE_TABLE_TOP],
            [REFERENCE_TABLE_LEFT + REFERENCE_TABLE_WIDTH, REFERENCE_TABLE_TOP],
            [REFERENCE_TABLE_LEFT + REFERENCE_TABLE_WIDTH, table_bottom],
            [REFERENCE_TABLE_LEFT, table_bottom],
        ])
        matrix = cv2.getPerspectiveTransform(corners, destination)
        normalized = cv2.warpPerspective(
            image,
            matrix,
            (REFERENCE_WIDTH, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        header_ratio = (
            (geometry["header_y"] - geometry["top_y"])
            / (geometry["bottom_y"] - geometry["top_y"])
        )
        header_y = REFERENCE_TABLE_TOP + round(header_ratio * normalized_table_height)
        row_lines = [
            REFERENCE_TABLE_TOP + round(
                (line_y - geometry["top_y"])
                / (geometry["bottom_y"] - geometry["top_y"])
                * normalized_table_height
            )
            for line_y in geometry["line_centers"][1:]
        ]
        return normalized, {
            "scale": scale,
            "header_y": header_y,
            "table_bottom": table_bottom,
            "table_detected": True,
            "source_table": list(geometry["bounds"]),
            "row_lines": row_lines,
        }

    scale = REFERENCE_WIDTH / width
    normalized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    normalized = cv2.resize(
        image,
        (REFERENCE_WIDTH, normalized_height),
        interpolation=interpolation,
    )
    return normalized, {
        "scale": scale,
        "header_y": HEADER_Y,
        "table_bottom": TABLE_BOTTOM,
        "table_detected": False,
        "source_table": None,
        "row_lines": None,
    }


def detect_column_bounds(image, header_y):
    """Detect seven/eight-column variants from vertical rules in the header row."""
    top = max(0, REFERENCE_TABLE_TOP - 3)
    body_probe = min(320, max(120, (image.shape[0] - header_y) // 3))
    bottom = min(image.shape[0], header_y + body_probe)
    roi = image[top:bottom]
    if roi.size == 0:
        return COLUMN_BOUNDS.copy(), "fixed"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, roi.shape[0] // 2))),
    )
    projection = (vertical > 0).sum(axis=0)
    smoothed = np.convolve(projection, np.ones(5, dtype=np.int32), mode="same")
    indexes = np.where(smoothed > roi.shape[0] * 0.45)[0]
    groups = []
    for index in indexes:
        if not groups or index > groups[-1][1] + 1:
            groups.append([int(index), int(index)])
        else:
            groups[-1][1] = int(index)
    candidates = [int(round((start + end) / 2)) for start, end in groups]
    candidates = [
        value for value in candidates
        if REFERENCE_TABLE_LEFT - 25 <= value <= REFERENCE_TABLE_LEFT + REFERENCE_TABLE_WIDTH + 25
    ]
    if len(candidates) < 8:
        return COLUMN_BOUNDS.copy(), "fixed"

    expected_right = REFERENCE_TABLE_LEFT + REFERENCE_TABLE_WIDTH
    left = min(candidates, key=lambda value: abs(value - REFERENCE_TABLE_LEFT))
    right = min(candidates, key=lambda value: abs(value - expected_right))
    if right <= left:
        return COLUMN_BOUNDS.copy(), "fixed"

    internal = sorted(
        value for value in candidates
        if left + 60 <= value <= right - 60
    )
    lines = [left]
    for value in internal:
        if value - lines[-1] >= 45:
            lines.append(value)
    lines.append(right)

    if len(lines) == 8:
        fields = ("pallet", "box", "sku", "name", "qty", "exp", "mfg")
        layout = "without_batch"
    elif len(lines) == 9:
        fields = FIELD_ORDER
        layout = "with_batch"
    else:
        return COLUMN_BOUNDS.copy(), "fixed"

    bounds = {
        field: (lines[index] + 3, lines[index + 1] - 3)
        for index, field in enumerate(fields)
    }
    return bounds, layout


def fit_regular_row_centers(observed_y, expected_count, fallback_centers):
    """Fit a stable row sequence from partially recognized values in one column."""
    if expected_count <= 0:
        return []
    fallback = list(fallback_centers[:expected_count])
    if len(fallback) != expected_count or len(observed_y) < 3:
        return fallback

    base_spacing = float(np.median(np.diff(fallback)))
    if base_spacing < 10:
        return fallback
    observed = sorted(float(value) for value in observed_y)
    if len(observed) == expected_count:
        observed_gaps = np.diff(observed)
        observed_spacing = float(np.median(observed_gaps))
        if observed_spacing >= 10 and all(
            observed_spacing * 0.55 <= gap <= observed_spacing * 1.45
            for gap in observed_gaps
        ):
            return observed
    best = None
    spacing_values = np.linspace(base_spacing * 0.85, base_spacing * 1.15, 31)
    first_values = np.arange(
        fallback[0] - base_spacing * 0.45,
        fallback[0] + base_spacing * 0.45 + 0.5,
        0.5,
    )
    for spacing in spacing_values:
        for first in first_values:
            indexes = [int(round((value - first) / spacing)) for value in observed]
            valid = [
                (value, index)
                for value, index in zip(observed, indexes)
                if 0 <= index < expected_count
            ]
            unique_count = len({index for _, index in valid})
            if unique_count < min(3, expected_count):
                continue
            residual = sum(
                abs(value - (first + index * spacing))
                for value, index in valid
            ) / len(valid)
            duplicate_count = len(valid) - unique_count
            missing_count = len(observed) - len(valid)
            score = (
                residual
                + duplicate_count * base_spacing * 0.35
                + missing_count * base_spacing * 0.25
                + abs(first - fallback[0]) * 0.03
                - unique_count * 0.02
            )
            if best is None or score < best[0]:
                best = (score, first, spacing, unique_count)

    if best is None or best[3] < max(3, int(min(expected_count, len(observed)) * 0.55)):
        return fallback
    _, first, spacing, _ = best
    return [first + index * spacing for index in range(expected_count)]


def row_centers_to_lines(centers, minimum_y, maximum_y):
    if not centers:
        return []
    if len(centers) == 1:
        return [minimum_y, maximum_y]
    first_gap = centers[1] - centers[0]
    last_gap = centers[-1] - centers[-2]
    lines = [max(minimum_y, centers[0] - first_gap / 2)]
    lines.extend((left + right) / 2 for left, right in zip(centers, centers[1:]))
    lines.append(min(maximum_y, centers[-1] + last_gap / 2))
    return lines


def refine_column_layout(items, header_y, fallback_bounds, fallback_layout):
    """Prefer printed header labels over raw line counts when selecting a schema."""
    header_items = [
        (re.sub(r"[^A-Z]", "", item["text"].upper()), float(item["xc"]))
        for item in items
        if REFERENCE_TABLE_TOP - 15 <= item["yc"] <= header_y + 12
    ]
    header_texts = [text for text, _ in header_items]
    has_batch = any("BATCH" in text or text.startswith("BATC") for text in header_texts)
    has_mfg = any("MFG" in text or text.startswith("MF") for text in header_texts)
    has_exp = any(text == "EXP" or text.startswith("EXP") for text in header_texts)
    has_qty = any(
        (len(text) == 3 and text.endswith("TY")) or text.endswith("QTY")
        for text in header_texts
    )

    if has_batch:
        return COLUMN_BOUNDS.copy(), "with_batch"

    def header_field(text):
        if text.startswith("PALL"):
            return "pallet"
        if text == "BOX":
            return "box"
        if "SKU" in text:
            return "sku"
        if text == "NAME":
            return "name"
        if (len(text) == 3 and text.endswith("TY")) or text.endswith("QTY"):
            return "qty"
        if text.startswith("EXP"):
            return "exp"
        if text.startswith("MF"):
            return "mfg"
        return None

    positioned_headers = [
        (field, x_center)
        for text, x_center in header_items
        if (field := header_field(text)) is not None
    ]
    discriminating_fields = {"name", "qty", "exp", "mfg"}
    if any(field in discriminating_fields for field, _ in positioned_headers):
        def layout_score(bounds):
            return sum(
                abs(x_center - sum(bounds[field]) / 2)
                for field, x_center in positioned_headers
            ) / len(positioned_headers)

        without_batch_score = layout_score(WITHOUT_BATCH_BOUNDS)
        with_batch_score = layout_score(COLUMN_BOUNDS)
        if without_batch_score + 20 < with_batch_score:
            return WITHOUT_BATCH_BOUNDS.copy(), "without_batch"

    if has_mfg and (has_exp or has_qty):
        return WITHOUT_BATCH_BOUNDS.copy(), "without_batch"

    if fallback_layout == "fixed" and not positioned_headers:
        def content_score(bounds):
            sku_x1, sku_x2 = bounds["sku"]
            qty_x1, qty_x2 = bounds["qty"]
            sku_count = 0
            qty_count = 0
            date_count = 0
            for item in items:
                compact = re.sub(r"\s+", "", item["text"])
                digits = re.sub(r"\D", "", compact)
                if sku_x1 <= item["xc"] < sku_x2 and 8 <= len(digits) <= 14:
                    sku_count += 1
                if qty_x1 <= item["xc"] < qty_x2 and 1 <= len(digits) <= 4:
                    qty_count += 1
                for field in ("exp", "mfg"):
                    x1, x2 = bounds[field]
                    if x1 <= item["xc"] < x2 and len(digits) in (6, 8):
                        date_count += 1
                        break
            return sku_count * 2 + qty_count * 3 + date_count

        with_batch_score = content_score(COLUMN_BOUNDS)
        without_batch_score = content_score(WITHOUT_BATCH_BOUNDS)
        if without_batch_score >= with_batch_score + 6:
            return WITHOUT_BATCH_BOUNDS.copy(), "without_batch"
        if with_batch_score >= without_batch_score + 6:
            return COLUMN_BOUNDS.copy(), "with_batch"
    return fallback_bounds, fallback_layout


def prepare_column(image, field, header_y, table_bottom, column_bounds):
    """裁出整列并按字段规则渲染成白底黑字。"""
    _, width = image.shape[:2]
    x1, x2 = column_bounds[field]
    left = max(0, int(x1) + 3)
    right = min(width, int(x2) - 3)
    crop = image[header_y:table_bottom, left:right]
    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        None,
        fx=COLUMN_SCALE,
        fy=COLUMN_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    return render_region_black(crop, field)


def prepare_cell(image, field, y1, y2, column_bounds):
    """Crop one physical table cell for a context-independent second opinion."""
    height, width = image.shape[:2]
    x1, x2 = column_bounds[field]
    left = max(0, int(x1) + 3)
    right = min(width, int(x2) - 3)
    top = max(0, int(y1) + 2)
    bottom = min(height, int(y2) - 2)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    scale = max(1.0, 64 / crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(
        crop,
        8,
        8,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return render_region_black(crop, field)


def prepare_page_for_ocr(
    image,
    header_y,
    table_bottom,
    column_bounds,
    table_layout,
    row_lines,
    ocr_bottom,
    ocr_right,
):
    """Mask product-name cell contents while preserving table rules and coordinates."""
    page = image[:ocr_bottom, :ocr_right].copy()
    if not row_lines or len(row_lines) < 2 or "name" not in column_bounds:
        return page, False

    name_x1, name_x2 = column_bounds["name"]
    if table_layout == "fixed":
        name_x1 = max(name_x1, WITHOUT_BATCH_BOUNDS["name"][0])
        name_x2 = min(name_x2, COLUMN_BOUNDS["name"][1])
    left = max(0, int(name_x1) + 2)
    right = min(page.shape[1], int(name_x2) - 2)
    if right <= left:
        return page, False

    usable_lines = [
        int(round(value))
        for value in row_lines
        if header_y - 2 <= value <= table_bottom + 2
    ]
    if len(usable_lines) < 2:
        return page, False
    row_intervals = list(zip(usable_lines, usable_lines[1:]))
    # Keep the last interval intact because it may contain the printed total label.
    for top, bottom in row_intervals[:-1]:
        cell_top = max(0, top + 2)
        cell_bottom = min(page.shape[0], bottom - 2)
        if cell_bottom > cell_top:
            page[cell_top:cell_bottom, left:right] = 255
    return page, True


def prepare_handwritten_sheet(
    image,
    header_y,
    table_bottom,
    column_bounds,
    fields=None,
):
    """横向拼接指定手写列，通过一次检测/识别完成按需补识别。"""
    columns = []
    segments = []
    cursor = 0
    selected_fields = tuple(fields or HANDWRITTEN_FIELDS)
    for field in selected_fields:
        if field not in HANDWRITTEN_FIELDS:
            continue
        if field == "batch" and field not in column_bounds:
            continue
        column = prepare_column(image, field, header_y, table_bottom, column_bounds)
        if column is None:
            continue
        if columns:
            gap = np.full((column.shape[0], COLUMN_GAP, 3), 255, dtype=np.uint8)
            columns.append(gap)
            cursor += COLUMN_GAP
        start = cursor
        end = start + column.shape[1]
        segments.append((field, start, end))
        columns.append(column)
        cursor = end

    if not columns:
        return None, []
    return np.concatenate(columns, axis=1), segments


def render_region_black(region, field):
    """把区域内容渲染为白底黑字；手写列优先只保留蓝色墨迹。"""
    if region is None or region.size == 0:
        return None

    if field in HANDWRITTEN_FIELDS:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        blue_hsv = cv2.inRange(
            hsv,
            np.array([80, 25, 15], dtype=np.uint8),
            np.array([145, 255, 255], dtype=np.uint8),
        )

        blue, green, red = cv2.split(region)
        blue_i = blue.astype(np.int16)
        green_i = green.astype(np.int16)
        red_i = red.astype(np.int16)
        blue_dominant = (
            (blue_i - red_i > 8)
            & (blue_i - green_i > 2)
            & (red_i < 210)
        ).astype(np.uint8) * 255
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        dark_ink = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        horizontal_kernel = np.ones(
            (1, max(20, dark_ink.shape[1] // 2)),
            dtype=np.uint8,
        )
        horizontal_lines = cv2.morphologyEx(
            dark_ink,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )
        dark_ink = cv2.subtract(dark_ink, horizontal_lines)

        red_low = cv2.inRange(
            hsv,
            np.array([0, 35, 30], dtype=np.uint8),
            np.array([18, 255, 255], dtype=np.uint8),
        )
        red_high = cv2.inRange(
            hsv,
            np.array([165, 35, 30], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        red_ink = cv2.bitwise_or(red_low, red_high)
        dark_ink[red_ink > 0] = 0

        if field == "qty":
            diagonal_size = max(17, min(31, dark_ink.shape[1] // 5))
            diagonal = np.eye(diagonal_size, dtype=np.uint8)
            diagonal_reverse = np.fliplr(diagonal)
            long_diagonals = cv2.bitwise_or(
                cv2.morphologyEx(dark_ink, cv2.MORPH_OPEN, diagonal),
                cv2.morphologyEx(dark_ink, cv2.MORPH_OPEN, diagonal_reverse),
            )
            dark_ink = cv2.subtract(dark_ink, long_diagonals)

        ink_mask = cv2.bitwise_or(dark_ink, cv2.bitwise_or(blue_hsv, blue_dominant))
        ink_mask = cv2.morphologyEx(
            ink_mask,
            cv2.MORPH_CLOSE,
            np.ones((2, 2), dtype=np.uint8),
        )
        if cv2.countNonZero(ink_mask) < max(12, int(ink_mask.size * 0.0005)):
            return None

        rendered = np.full(region.shape[:2], 255, dtype=np.uint8)
        rendered[ink_mask > 0] = 0
    else:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, rendered = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ink = 255 - rendered
        horizontal_kernel = np.ones((1, max(20, rendered.shape[1] // 2)), dtype=np.uint8)
        horizontal_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horizontal_kernel)
        rendered = 255 - cv2.subtract(ink, horizontal_lines)

    return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)


def parse_date(text):
    """把 OCR 日期规整成 YYYY-MM-DD，保留原始值，不强制清空"""
    if not text:
        return ""
    digits = re.sub(r"[^\d]", "", text)
    if len(digits) >= 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    elif len(digits) >= 6:
        return f"{digits[0:4]}-{digits[4:6]}"
    return text.strip()


def normalize_field(field, text):
    value = clean_text(text)
    if field == "sku":
        return clean_sku(value)
    if field in ("qty", "box"):
        return clean_digits(value)
    if field in ("exp", "mfg"):
        return parse_date(value)
    if field == "batch":
        return clean_batch(value)
    return value


def field_value_is_valid(field, value):
    if not value:
        return False
    if field == "sku":
        if not re.fullmatch(r"\d{8,14}", value):
            return False
        if len(value) in (8, 12, 13, 14):
            return is_valid_gtin(value)
        return True
    if field in ("qty", "box"):
        return re.fullmatch(r"[1-9]\d*", value) is not None
    if field in ("exp", "mfg"):
        return is_valid_date(value)
    if field == "pallet":
        return re.fullmatch(r"\d+/\d+", value) is not None
    if field == "batch":
        return re.fullmatch(r"[A-Za-z0-9]{4,}", value) is not None
    return len(value) >= 2


def should_use_cell_result(field, current_text, current_score, new_text, new_score):
    if not new_text or new_score < 0.30:
        return False
    if not current_text:
        return True
    if field == "name":
        return new_score >= 0.35

    current_valid = field_value_is_valid(field, current_text)
    new_valid = field_value_is_valid(field, new_text)
    if field == "sku" and current_valid and new_valid and current_text != new_text:
        return False
    if (
        field == "qty"
        and current_valid
        and new_valid
        and current_text.endswith("1")
        and new_text == current_text[:-1]
    ):
        return True
    if new_valid != current_valid:
        return new_valid
    if new_valid:
        return current_score is None or new_score >= current_score - 0.05
    return current_score is None or new_score > current_score


def select_quantity_candidate(current_text, current_score, candidates, box_value=""):
    """Combine page, column, and cell readings; disagreement always requires review."""
    readings = []
    if field_value_is_valid("qty", current_text):
        readings.append({
            "source": "page",
            "text": current_text,
            "score": float(current_score or 0),
        })
    for candidate in candidates:
        text = normalize_field("qty", candidate.get("text", ""))
        if field_value_is_valid("qty", text):
            readings.append({
                "source": candidate.get("source", "column"),
                "text": text,
                "score": float(candidate.get("score") or 0),
            })
    if not readings:
        return current_text, current_score, False, []

    grouped = {}
    for reading in readings:
        grouped.setdefault(reading["text"], []).append(reading)
    if len(grouped) > 1:
        values = list(grouped)
        common_suffix = ""
        for offset in range(1, min(len(value) for value in values) + 1):
            suffix = values[0][-offset:]
            if all(value.endswith(suffix) for value in values):
                common_suffix = suffix
            else:
                break
        scores = [reading["score"] for reading in readings]
        if (
            len(common_suffix) >= 2
            and all(len(value) == len(common_suffix) + 1 for value in values)
            and max(scores) < 0.85
        ):
            return common_suffix, max(scores), True, readings
    if field_value_is_valid("box", box_value) and box_value in grouped:
        selected_text = box_value
        selected_readings = grouped[box_value]
    else:
        selected_text, selected_readings = max(
            grouped.items(),
            key=lambda item: (len(item[1]), max(value["score"] for value in item[1])),
        )
    selected_score = max(value["score"] for value in selected_readings)
    return selected_text, selected_score, len(grouped) > 1, readings


def select_sku_candidate(current_text, current_score, candidates):
    """Use majority agreement across page, cell, and dedicated-column readings."""
    readings = []
    if field_value_is_valid("sku", current_text):
        readings.append({
            "source": "page",
            "text": current_text,
            "score": float(current_score or 0),
        })
    for candidate in candidates:
        text = normalize_field("sku", candidate.get("text", ""))
        if field_value_is_valid("sku", text):
            readings.append({
                "source": candidate.get("source", "column"),
                "text": text,
                "score": float(candidate.get("score") or 0),
            })
    if not readings:
        return current_text, current_score, False, []

    grouped = {}
    for reading in readings:
        grouped.setdefault(reading["text"], []).append(reading)
    selected_text, selected_readings = max(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            any(value["source"] == "page" for value in item[1]),
            max(value["score"] for value in item[1]),
        ),
    )
    selected_score = max(value["score"] for value in selected_readings)
    counts = sorted((len(values) for values in grouped.values()), reverse=True)
    ambiguous = len(counts) > 1 and counts[0] == counts[1]
    if (
        field_value_is_valid("sku", current_text)
        and selected_text != current_text
        and len(selected_text) < len(current_text)
    ):
        return current_text, current_score, True, readings
    return selected_text, selected_score, ambiguous, readings


def select_structured_candidate(field, current_text, current_score, candidates):
    readings = []
    if current_text:
        readings.append({
            "source": "page",
            "text": current_text,
            "score": float(current_score or 0),
        })
    for candidate in candidates:
        text = normalize_field(field, candidate.get("text", ""))
        if text:
            readings.append({
                "source": candidate.get("source", "column"),
                "text": text,
                "score": float(candidate.get("score") or 0),
            })
    if not readings:
        return current_text, current_score, False, []

    grouped = {}
    for reading in readings:
        grouped.setdefault(reading["text"], []).append(reading)
    valid_groups = {
        text: values
        for text, values in grouped.items()
        if field_value_is_valid(field, text)
    }
    if not valid_groups:
        return current_text, current_score, False, readings
    pool = valid_groups
    selected_text, selected_readings = max(
        pool.items(),
        key=lambda item: (
            len(item[1]),
            any(value["source"] == "page" for value in item[1]),
            max(value["score"] for value in item[1]),
        ),
    )
    selected_score = max(value["score"] for value in selected_readings)
    return selected_text, selected_score, len(valid_groups) > 1, readings


def select_aligned_fragments(fragments, target_y, target_x):
    """Keep only OCR fragments that share one baseline near the target row."""
    positioned = [fragment for fragment in fragments if fragment.get("yc") is not None]
    if len(positioned) <= 1:
        return fragments
    heights = [
        max(1.0, float(fragment.get("y2", fragment["yc"])) - float(fragment.get("y1", fragment["yc"])))
        for fragment in positioned
    ]
    baseline_tolerance = max(3.0, float(np.median(heights)) * 0.2)
    groups = []
    for fragment in sorted(positioned, key=lambda value: value["yc"]):
        if not groups:
            groups.append([fragment])
            continue
        group_y = sum(value["yc"] for value in groups[-1]) / len(groups[-1])
        if abs(fragment["yc"] - group_y) <= baseline_tolerance:
            groups[-1].append(fragment)
        else:
            groups.append([fragment])

    def fragment_center_x(fragment):
        default_x = fragment.get("xc", target_x)
        return (
            fragment.get("x1", default_x) + fragment.get("x2", default_x)
        ) / 2

    selected = min(
        groups,
        key=lambda group: (
            abs(sum(value["yc"] for value in group) / len(group) - target_y),
            abs(
                sum(fragment_center_x(value) for value in group)
                / len(group)
                - target_x
            ),
            -max(float(value.get("score") or 0) for value in group),
        ),
    )
    return sorted(selected, key=lambda value: value.get("x1", value.get("xc", 0)))


def table_row_field_reading(
    table_row,
    field,
    column_bounds,
    field_row_centers=None,
    row_index=None,
):
    fragments = sorted(
        table_row["rec"][field],
        key=lambda value: value.get("x1", value.get("xc", 0)),
    )
    if fragments and field != "name" and field in column_bounds:
        target_y = table_row["yc"]
        if field_row_centers and row_index is not None:
            centers = field_row_centers.get(field)
            if centers and row_index < len(centers):
                target_y = centers[row_index]
        target_x = sum(column_bounds[field]) / 2
        fragments = select_aligned_fragments(fragments, target_y, target_x)
    text = normalize_field(
        field,
        " ".join(value["text"] for value in fragments),
    )
    score = (
        min(float(value["score"]) for value in fragments)
        if fragments else table_row["conf"].get(field)
    )
    return text, score


def select_fallback_fields(
    table_rows,
    fields,
    ink_cells,
    column_bounds,
    field_row_centers,
    batch_candidates=None,
):
    """Select only columns whose visible content still lacks a reliable reading."""
    selected = []
    batch_candidates = list(batch_candidates or [])
    for field in fields:
        if field not in column_bounds:
            continue
        threshold = FALLBACK_MIN_CONFIDENCE[field]
        for index, table_row in enumerate(table_rows):
            text, score = table_row_field_reading(
                table_row,
                field,
                column_bounds,
                field_row_centers,
                index,
            )
            has_content = (
                field in ("sku", "qty")
                or bool(text)
                or (index, field) in ink_cells
            )
            if not has_content:
                continue
            if (
                not field_value_is_valid(field, text)
                or score is None
                or score < threshold
                or field in table_row.get("review_reasons", {})
                or (
                    field == "batch"
                    and batch_candidates
                    and match_batch_candidate(text, batch_candidates) is None
                )
            ):
                selected.append(field)
                break
    return selected


def recognize_column_candidates(
    image,
    field,
    header_y,
    table_bottom,
    column_bounds,
    language="general",
):
    column = prepare_column(
        image,
        field,
        header_y,
        table_bottom,
        column_bounds,
    )
    if column is None:
        return []
    candidates = []
    for box, text, score in recognize_region(column, language=language):
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        candidates.append({
            "text": text,
            "score": float(score),
            "x1": min(xs),
            "x2": max(xs),
            "yc": ((min(ys) + max(ys)) / 2 / COLUMN_SCALE) + header_y,
            "y1": min(ys) / COLUMN_SCALE + header_y,
            "y2": max(ys) / COLUMN_SCALE + header_y,
            "source": "column",
        })
    return candidates


def append_column_matches(
    matches,
    candidates,
    field,
    table_rows,
    field_row_centers,
):
    if not table_rows:
        return
    row_centers = field_row_centers.get(field)
    for candidate in candidates:
        original_y = candidate["yc"]
        closest_index = min(
            range(len(table_rows)),
            key=lambda index: abs(
                (row_centers[index] if row_centers else table_rows[index]["yc"])
                - original_y
            ),
        )
        closest_row = table_rows[closest_index]
        row_tolerance = (
            max(
                ROW_TOLERANCE * 1.75,
                (closest_row["y2"] - closest_row["y1"]) * 0.75,
            )
            if closest_row["y1"] is not None else ROW_TOLERANCE * 1.75
        )
        closest_y = row_centers[closest_index] if row_centers else closest_row["yc"]
        if abs(closest_y - original_y) <= row_tolerance:
            matches.setdefault((closest_index, field), []).append(candidate)


def apply_recognition_candidates(
    table_rows,
    matches,
    field_row_centers,
    column_bounds,
):
    for (index, field), candidates in matches.items():
        candidates.sort(key=lambda value: value["x1"])
        table_row = table_rows[index]
        fragments = sorted(
            table_row["rec"][field],
            key=lambda value: value.get("x1", value.get("xc", 0)),
        )
        row_centers = field_row_centers.get(field)
        target_y = row_centers[index] if row_centers else table_row["yc"]
        target_x = sum(column_bounds[field]) / 2
        if field != "name":
            fragments = select_aligned_fragments(fragments, target_y, target_x)
            table_row["rec"][field] = fragments
        current_text = normalize_field(
            field,
            " ".join(value["text"] for value in fragments),
        )
        current_score = (
            min(float(value["score"]) for value in fragments)
            if fragments else table_row["conf"][field]
        )
        if field == "sku":
            prior_candidates = [
                candidate
                for candidate in table_row.get("field_candidates", {}).get(field, [])
                if candidate.get("source") != "page"
            ]
            selected_text, selected_score, ambiguous, readings = select_sku_candidate(
                current_text,
                current_score,
                prior_candidates + candidates,
            )
            if selected_text:
                x1, x2 = column_bounds[field]
                table_row["rec"][field] = [{
                    "text": selected_text,
                    "score": selected_score,
                    "xc": (x1 + x2) / 2,
                }]
                table_row["conf"][field] = selected_score
            if readings:
                table_row.setdefault("field_candidates", {})[field] = readings
            if ambiguous:
                table_row.setdefault("review_reasons", {})[field] = "model_disagreement"
            else:
                table_row.get("review_reasons", {}).pop(field, None)
            continue
        if field == "qty":
            column_candidates = [
                candidate for candidate in candidates
                if candidate.get("source") == "column"
            ]
            column_candidates = select_aligned_fragments(
                column_candidates,
                target_y,
                target_x,
            )
            resolved_candidates = [
                candidate for candidate in candidates
                if candidate.get("source") != "column"
            ]
            if column_candidates:
                resolved_candidates.append({
                    "source": "column",
                    "text": " ".join(
                        value["text"] for value in column_candidates
                    ),
                    "score": min(value["score"] for value in column_candidates),
                })
            selected_text, selected_score, ambiguous, readings = select_quantity_candidate(
                current_text,
                current_score,
                resolved_candidates,
                normalize_field(
                    "box",
                    " ".join(
                        value["text"]
                        for value in sorted(
                            table_row["rec"]["box"],
                            key=lambda value: value.get("x1", value.get("xc", 0)),
                        )
                    ),
                ),
            )
            if selected_text:
                x1, x2 = column_bounds[field]
                table_row["rec"][field] = [{
                    "text": selected_text,
                    "score": selected_score,
                    "xc": (x1 + x2) / 2,
                }]
                table_row["conf"][field] = selected_score
            if readings:
                table_row.setdefault("field_candidates", {})[field] = readings
            if ambiguous:
                table_row.setdefault("review_reasons", {})[field] = "model_disagreement"
            continue

        if field in ("exp", "mfg"):
            source_readings = []
            for source in ("column", "cell"):
                source_fragments = [
                    candidate
                    for candidate in candidates
                    if candidate.get("source") == source
                ]
                if not source_fragments:
                    continue
                if source == "column":
                    source_fragments = select_aligned_fragments(
                        source_fragments,
                        target_y,
                        target_x,
                    )
                source_readings.append({
                    "source": source,
                    "text": " ".join(
                        value["text"]
                        for value in sorted(
                            source_fragments,
                            key=lambda value: value["x1"],
                        )
                    ),
                    "score": min(value["score"] for value in source_fragments),
                })
            selected_text, selected_score, ambiguous, readings = select_structured_candidate(
                field,
                current_text,
                current_score,
                source_readings,
            )
            if selected_text:
                x1, x2 = column_bounds[field]
                table_row["rec"][field] = [{
                    "text": selected_text,
                    "score": selected_score,
                    "xc": (x1 + x2) / 2,
                }]
                table_row["conf"][field] = selected_score
            if readings:
                table_row.setdefault("field_candidates", {})[field] = readings
            if ambiguous:
                table_row.setdefault("review_reasons", {})[field] = "model_disagreement"
            continue

        candidates = select_aligned_fragments(candidates, target_y, target_x)
        new_text = normalize_field(
            field,
            " ".join(value["text"] for value in candidates),
        )
        new_score = min(value["score"] for value in candidates)
        if should_use_cell_result(
            field,
            current_text,
            current_score,
            new_text,
            new_score,
        ):
            x1, x2 = column_bounds[field]
            table_row["rec"][field] = [{
                "text": new_text,
                "score": new_score,
                "xc": (x1 + x2) / 2,
            }]
            table_row["conf"][field] = new_score


def recognize_enhanced(image_path_or_bytes, batch_candidates=None):
    """主识别流程：整页定位，随后按字段选择手写或泰文模型。"""
    started_at = time.perf_counter()
    timings = {}
    if isinstance(image_path_or_bytes, str):
        img = cv2.imread(image_path_or_bytes)
    else:
        img = cv2.imdecode(np.frombuffer(image_path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片无法解码，请上传有效的 JPG、PNG 或 WebP 图片")
    batch_candidates = list(batch_candidates or [])

    source_height, source_width = img.shape[:2]
    stage_started = time.perf_counter()
    img, normalization = normalize_document_image(img)
    timings["normalize"] = round((time.perf_counter() - stage_started) * 1000)
    height, width = img.shape[:2]
    header_y = int(normalization["header_y"])
    table_bottom = int(normalization["table_bottom"])
    row_lines = normalization.get("row_lines")
    column_bounds, table_layout = detect_column_bounds(img, header_y)
    if height <= header_y or table_bottom <= header_y or width < REFERENCE_WIDTH:
        raise ValueError(
            f"图片归一化后尺寸不足，当前为 {width}x{height}，无法覆盖固定表格区域"
        )

    # 只识别表头与表格；下方手写备注由页面人工填写，避免处理大块无关区域。
    ocr_bottom = min(height, table_bottom + 20)
    is_already_normalized = 0.95 <= float(normalization["scale"]) <= 1.05
    ocr_right = (
        min(width, REFERENCE_TABLE_LEFT + REFERENCE_TABLE_WIDTH + 20)
        if is_already_normalized else width
    )
    page_for_ocr, name_column_skipped = prepare_page_for_ocr(
        img,
        header_y,
        table_bottom,
        column_bounds,
        table_layout,
        row_lines,
        ocr_bottom,
        ocr_right,
    )
    stage_started = time.perf_counter()
    full_result = recognize_page(page_for_ocr)
    timings["page_ocr"] = round((time.perf_counter() - stage_started) * 1000)
    items = parse_items(full_result)
    column_bounds, table_layout = refine_column_layout(
        items,
        header_y,
        column_bounds,
        table_layout,
    )
    header_aliases = {
        "pallet": lambda text: text.startswith("PALL"),
        "box": lambda text: text == "BOX",
        "sku": lambda text: "SKU" in text,
        "name": lambda text: text == "NAME",
        "qty": lambda text: (len(text) == 3 and text.endswith("TY")) or text.endswith("QTY"),
        "exp": lambda text: text.startswith("EXP"),
        "mfg": lambda text: text.startswith("MF"),
        "batch": lambda text: text.startswith("BATC"),
    }
    header_fields = {
        field
        for item in items
        if REFERENCE_TABLE_TOP - 15 <= item["yc"] <= header_y + 12
        for field, matcher in header_aliases.items()
        if matcher(re.sub(r"[^A-Z]", "", item["text"].upper()))
    }
    has_table_header = bool(header_fields & {"sku", "name", "qty"})
    if not has_table_header and row_lines and row_lines[0] > REFERENCE_TABLE_TOP + 5:
        row_lines = [REFERENCE_TABLE_TOP] + list(row_lines)
        header_y = REFERENCE_TABLE_TOP

    active_columns = [
        (field, *column_bounds[field])
        for field in FIELD_ORDER
        if field in column_bounds
    ]
    header = extract_header(items)
    document_total_qty = extract_total_qty(items, columns=active_columns)
    notes = extract_notes(items, table_bottom=table_bottom)
    sku_column_results = []
    sku_fallback_used = not row_lines or len(row_lines) < 2
    timings["sku_fallback"] = 0
    if sku_fallback_used:
        stage_started = time.perf_counter()
        sku_column_results = recognize_column_candidates(
            img,
            "sku",
            header_y,
            table_bottom,
            column_bounds,
        )
        timings["sku_fallback"] = round(
            (time.perf_counter() - stage_started) * 1000
        )

    if row_lines and len(row_lines) >= 2:
        row_spacing = float(np.median(np.diff(row_lines)))
        has_summary = any(re.search(r"TOTA[L1I]|合计", item["text"], re.I) for item in items)
        slot_count = len(row_lines) - 1
        data_row_count = slot_count - 1 if has_summary else slot_count
        data_row_count = max(1, data_row_count)
        data_lines = list(row_lines[:data_row_count + 1])
        fallback_centers = [
            (top + bottom) / 2
            for top, bottom in zip(data_lines, data_lines[1:])
        ]
        field_row_lines = {field: data_lines for field in column_bounds}
        field_row_centers = {
            field: list(fallback_centers)
            for field in column_bounds
        }
        field_observations = {field: [] for field in column_bounds}
        for item in items:
            field = next(
                (name for name, x1, x2 in active_columns if x1 <= item["xc"] < x2),
                None,
            )
            if field is None or field == "name":
                continue
            compact = re.sub(r"\s+", "", item["text"])
            digits = re.sub(r"\D", "", compact)
            if field == "sku" and not 8 <= len(digits) <= 14:
                continue
            if field in ("qty", "box") and not 1 <= len(digits) <= 4:
                continue
            if field in ("exp", "mfg") and len(digits) not in (6, 8):
                continue
            if field == "pallet" and not re.search(r"\d", compact):
                continue
            if field == "batch" and len(re.sub(r"[^A-Za-z0-9]", "", compact)) < 4:
                continue
            if field == "qty" and document_total_qty is not None and int(digits) == int(document_total_qty):
                continue
            field_observations[field].append(item["yc"])
        dedicated_sku_observations = [
            candidate["yc"]
            for candidate in sku_column_results
            if 8 <= len(re.sub(r"\D", "", candidate["text"])) <= 14
        ]
        if len(dedicated_sku_observations) >= 3:
            field_observations["sku"] = dedicated_sku_observations
        for field, observations in field_observations.items():
            centers = fit_regular_row_centers(
                observations,
                data_row_count,
                fallback_centers,
            )
            if len(centers) != data_row_count:
                continue
            field_row_centers[field] = centers
            field_row_lines[field] = row_centers_to_lines(
                centers,
                data_lines[0],
                data_lines[-1],
            )
        data_top = min(lines[0] for lines in field_row_lines.values())
        data = [it for it in items if data_top < it["yc"] <= table_bottom]
        data.sort(key=lambda it: it["yc"])
        rows = [
            {
                "yc": (top + bottom) / 2,
                "y1": top,
                "y2": bottom,
                "items": [],
            }
            for top, bottom in zip(data_lines, data_lines[1:])
        ]
        for item in data:
            field = next(
                (name for name, x1, x2 in active_columns if x1 <= item["xc"] < x2),
                None,
            )
            if field is None:
                continue
            centers = field_row_centers[field]
            index = min(range(len(centers)), key=lambda value: abs(centers[value] - item["yc"]))
            if abs(centers[index] - item["yc"]) <= row_spacing * 0.75:
                rows[index]["items"].append(item)
    else:
        field_row_lines = {}
        field_row_centers = {}
        data = [it for it in items if header_y < it["yc"] <= table_bottom]
        data.sort(key=lambda it: it["yc"])
        rows = []
        for item in data:
            placed = False
            for row in rows:
                if abs(row["yc"] - item["yc"]) <= ROW_TOLERANCE:
                    row["items"].append(item)
                    row["yc"] = sum(value["yc"] for value in row["items"]) / len(row["items"])
                    placed = True
                    break
            if not placed:
                rows.append({"yc": item["yc"], "items": [item]})

    table_rows = []
    for row in rows:
        rec = {name: [] for name in FIELD_ORDER}
        conf = {name: None for name in FIELD_ORDER}
        for it in row["items"]:
            for name, x1, x2 in active_columns:
                if x1 <= it["xc"] < x2:
                    rec[name].append(it)
                    score = float(it["score"])
                    conf[name] = score if conf[name] is None else min(conf[name], score)
        table_rows.append({
            "yc": row["yc"],
            "y1": row.get("y1"),
            "y2": row.get("y2"),
            "structural": row.get("y1") is not None,
            "rec": rec,
            "conf": conf,
        })

    if not table_rows:
        timings["total"] = round((time.perf_counter() - started_at) * 1000)
        return {
            "header": header,
            "rows": [],
            "row_count": 0,
            "document_total_qty": document_total_qty,
            "batch_candidates": batch_candidates,
            "pipeline": {
                "name_column_skipped": name_column_skipped,
                "sku_fallback": sku_fallback_used,
                "handwriting_fallback_fields": [],
            },
            "timing_ms": timings,
            "notes": notes,
            "models": {
                "general": "PP-OCRv6-small",
                "handwriting": "PP-OCRv6-small",
            },
        }

    matches = {}
    ink_cells = set()
    structured_cell_jobs = []
    for index, table_row in enumerate(table_rows):
        for field in ("sku", "qty", "exp", "mfg"):
            if field not in column_bounds:
                continue
            if field == "sku":
                current_text, current_score = table_row_field_reading(
                    table_row,
                    field,
                    column_bounds,
                    field_row_centers,
                    index,
                )
                if (
                    field_value_is_valid(field, current_text)
                    and current_score is not None
                    and current_score >= FALLBACK_MIN_CONFIDENCE[field]
                ):
                    continue
            lines = field_row_lines.get(field, row_lines or [])
            if index + 1 >= len(lines):
                continue
            cell = prepare_cell(
                img,
                field,
                lines[index],
                lines[index + 1],
                column_bounds,
            )
            if cell is not None:
                ink_cells.add((index, field))
                structured_cell_jobs.append((index, field, cell))
    stage_started = time.perf_counter()
    structured_cell_results = recognize_lines(
        [job[2] for job in structured_cell_jobs],
        language="handwriting",
    )
    timings["structured_cells"] = round(
        (time.perf_counter() - stage_started) * 1000
    )
    for (index, field, _), (text, score) in zip(
        structured_cell_jobs,
        structured_cell_results,
    ):
        matches.setdefault((index, field), []).append({
            "text": text,
            "score": float(score),
            "x1": 0,
            "source": "cell",
        })

    apply_recognition_candidates(
        table_rows,
        matches,
        field_row_centers,
        column_bounds,
    )
    matches = {}

    # Cell preparation is cheap compared with model inference and prevents
    # intentionally blank optional fields from activating the slow path.
    for index, table_row in enumerate(table_rows):
        for field in HANDWRITTEN_FIELDS:
            if field not in column_bounds or (index, field) in ink_cells:
                continue
            lines = field_row_lines.get(field, row_lines or [])
            if index + 1 >= len(lines):
                continue
            if prepare_cell(
                img,
                field,
                lines[index],
                lines[index + 1],
                column_bounds,
            ) is not None:
                ink_cells.add((index, field))

    if not sku_fallback_used:
        sku_fallback_used = bool(select_fallback_fields(
            table_rows,
            ("sku",),
            ink_cells,
            column_bounds,
            field_row_centers,
            batch_candidates,
        ))
        if sku_fallback_used:
            stage_started = time.perf_counter()
            sku_column_results = recognize_column_candidates(
                img,
                "sku",
                header_y,
                table_bottom,
                column_bounds,
            )
            timings["sku_fallback"] = round(
                (time.perf_counter() - stage_started) * 1000
            )
    if sku_column_results:
        append_column_matches(
            matches,
            sku_column_results,
            "sku",
            table_rows,
            field_row_centers,
        )

    if not row_lines or len(row_lines) < 2:
        handwriting_fallback_fields = [
            field for field in HANDWRITTEN_FIELDS
            if field in column_bounds
        ]
    else:
        handwriting_fallback_fields = select_fallback_fields(
            table_rows,
            HANDWRITTEN_FIELDS,
            ink_cells,
            column_bounds,
            field_row_centers,
            batch_candidates,
        )

    timings["handwriting_fallback"] = 0
    if handwriting_fallback_fields:
        handwriting_sheet, segments = prepare_handwritten_sheet(
            img,
            header_y,
            table_bottom,
            column_bounds,
            handwriting_fallback_fields,
        )
        if handwriting_sheet is not None:
            stage_started = time.perf_counter()
            handwriting_result = recognize_region(
                handwriting_sheet,
                language="handwriting",
            )
            timings["handwriting_fallback"] = round(
                (time.perf_counter() - stage_started) * 1000
            )
            candidates_by_field = {}
            for box, text, score in handwriting_result:
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                x_center = (min(xs) + max(xs)) / 2
                field = next(
                    (
                        name for name, start, end in segments
                        if start <= x_center < end
                    ),
                    None,
                )
                if field is None:
                    continue
                original_y = (
                    (min(ys) + max(ys)) / 2 / COLUMN_SCALE
                ) + header_y
                candidates_by_field.setdefault(field, []).append({
                    "text": text,
                    "score": float(score),
                    "x1": min(xs),
                    "x2": max(xs),
                    "yc": original_y,
                    "y1": min(ys) / COLUMN_SCALE + header_y,
                    "y2": max(ys) / COLUMN_SCALE + header_y,
                    "source": "column",
                })
            for field, candidates in candidates_by_field.items():
                append_column_matches(
                    matches,
                    candidates,
                    field,
                    table_rows,
                    field_row_centers,
                )

    apply_recognition_candidates(
        table_rows,
        matches,
        field_row_centers,
        column_bounds,
    )

    result_rows = []
    for tr in table_rows:
        rec = tr["rec"]
        conf = tr["conf"]
        row = {}
        row_conf_parts = []
        for name in FIELD_ORDER:
            fragments = sorted(rec[name], key=lambda x: x.get("x1", x.get("xc", 0)))
            raw_text = " ".join(f["text"] for f in fragments).strip()
            row[name] = normalize_field(name, raw_text)
            score = conf.get(name)
            if name in ("sku", "qty", "exp", "mfg", "batch") and score is not None:
                row_conf_parts.append(score)

        row["raw_sku"] = " ".join(
            value["text"] for value in rec["sku"]
        ).strip()
        row = reconcile_handwritten_quantity(row)
        row = reconcile_date_years(row, header)
        if "batch" in column_bounds:
            row = reconcile_batch_candidate(row, batch_candidates)
        is_candidate = is_data_row_candidate(row)
        if is_candidate or (tr["structural"] and not is_summary_row(row)):
            row["name"] = ""
            row["name_source"] = "system_lookup_required"
            conf["name"] = None
            if tr.get("field_candidates"):
                row["field_candidates"] = tr["field_candidates"]
            if tr.get("review_reasons"):
                row.setdefault("review_reasons", {}).update(tr["review_reasons"])
            if not is_candidate:
                row["structural_row_missing"] = True
            row["confidence"] = {
                key: round(value, 3) if value is not None else None
                for key, value in conf.items()
            }
            row["row_confidence"] = (
                round(sum(row_conf_parts) / len(row_conf_parts), 3)
                if row_conf_parts else 0.0
            )
            row["validation_errors"] = validate_row(row)
            for field in row.get("review_reasons", {}):
                row["validation_errors"][field] = "ambiguous"
            result_rows.append(row)

    reconciliation_total = (
        document_total_qty
        if not header.get("total_pages") or header.get("total_pages") == 1
        else None
    )
    result_rows = reconcile_quantities_with_total(result_rows, reconciliation_total)
    for row in result_rows:
        row["validation_errors"] = validate_row(row)
        for field in row.get("review_reasons", {}):
            row["validation_errors"][field] = "ambiguous"

    timings["total"] = round((time.perf_counter() - started_at) * 1000)
    return {
        "header": header,
        "rows": result_rows,
        "row_count": len(result_rows),
        "document_total_qty": document_total_qty,
        "batch_candidates": batch_candidates,
        "pipeline": {
            "name_column_skipped": name_column_skipped,
            "sku_fallback": sku_fallback_used,
            "handwriting_fallback_fields": handwriting_fallback_fields,
        },
        "timing_ms": timings,
        "image": {
            "source_width": source_width,
            "source_height": source_height,
            "normalized_width": width,
            "normalized_height": height,
            "scale": round(float(normalization["scale"]), 4),
            "table_detected": normalization["table_detected"],
            "source_table": normalization["source_table"],
            "table_layout": table_layout,
            "columns": {
                field: [int(x1), int(x2)]
                for field, (x1, x2) in column_bounds.items()
            },
            "header_y": header_y,
            "table_bottom": table_bottom,
            "row_slots": len(row_lines) - 1 if row_lines else None,
        },
        "notes": notes,
        "models": {
            "general": "PP-OCRv6-small",
            "handwriting": "PP-OCRv6-small",
        },
    }


if __name__ == "__main__":
    d = recognize_enhanced("sample.jpg")
    print("单据头:", d["header"])
    print("行数:", d["row_count"])
    print("备注:", d["notes"])
    for i, r in enumerate(d["rows"], 1):
        print(f'{i:2d}|sku={r["sku"]}|qty={r["qty"]:>5}|exp={r["exp"]:>12}|mfg={r["mfg"]:>12}|batch={r["batch"]:>14}|conf={r["row_confidence"]}')
