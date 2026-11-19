"""Free text recognition sharing the package model without business-field parsing."""
import time

from package_ocr import _decode_image, _resize_if_needed, _contrast_variant, _elapsed_ms
from package_ocr_models import recognize_package_image
from text_dates import parse_text_dates


def recognize_text(image_bytes, date_order="auto", recognizer=recognize_package_image):
    if date_order not in {"auto", "dmy", "mdy"}:
        raise ValueError("date_order must be auto, dmy or mdy")
    started = time.perf_counter()
    image = _resize_if_needed(_decode_image(image_bytes))
    lines = recognizer(image)
    # Retry only an empty reading; do not merge two readings and duplicate a paragraph.
    if not any(str(line.get("text", "")).strip() for line in lines):
        lines = recognizer(_contrast_variant(image))
    ordered = reading_lines(lines)
    raw_text = "\n".join(line["text"] for line in ordered)
    return {
        "raw_text": raw_text,
        "lines": ordered,
        **parse_text_dates(raw_text, date_order),
        "timing_ms": {"total": _elapsed_ms(started)},
        "model": "PP-OCRv6-small",
    }


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
