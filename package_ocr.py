"""Fast package-code OCR pipeline with one bounded enhancement retry."""

import time

import cv2
import numpy as np

from package_ocr_models import recognize_package_image
from package_text_parser import (
    BATCH,
    EXPIRE_AT,
    MANUFACTURE_AT,
    confusion_positions,
    parse_package_text,
    reconcile_batch_candidate,
)


MAX_IMAGE_EDGE = 1600
MIN_IMAGE_EDGE = 32


def recognize_package_text(
    image_bytes,
    required_fields,
    batch_candidates=None,
    recognizer=recognize_package_image,
):
    started_at = time.perf_counter()
    timings = {}

    stage_started = time.perf_counter()
    image = _decode_image(image_bytes)
    image = _resize_if_needed(image)
    timings["decode"] = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    primary_lines = recognizer(image)
    timings["primary_ocr"] = _elapsed_ms(stage_started)
    primary_text = _lines_to_text(primary_lines)
    selected = parse_package_text(primary_text, required_fields)
    batch_observations = [selected.get(BATCH)] if selected.get(BATCH) else []

    missing = _missing_fields(selected, required_fields)
    timings["enhanced_ocr"] = 0
    if missing:
        stage_started = time.perf_counter()
        enhanced_lines = recognizer(_contrast_variant(image))
        timings["enhanced_ocr"] = _elapsed_ms(stage_started)
        enhanced_text = _lines_to_text(enhanced_lines)
        enhanced = parse_package_text(enhanced_text, required_fields)
        if enhanced.get(BATCH):
            batch_observations.append(enhanced[BATCH])
        combined_text = _merge_text(primary_text, enhanced_text)
        combined = parse_package_text(combined_text, required_fields)
        selected = _select_result(selected, enhanced, combined, required_fields)
        selected["raw_text"] = combined_text

    review_positions = confusion_positions(selected.get(BATCH), batch_observations)
    original_batch = selected.get(BATCH)
    corrected_batch, corrected_positions, correction = reconcile_batch_candidate(
        original_batch,
        batch_candidates or [],
    )
    if corrected_batch:
        selected[BATCH] = corrected_batch
    review_positions = sorted(set(review_positions) | set(corrected_positions))
    missing = _missing_fields(selected, required_fields)

    timings["total"] = _elapsed_ms(started_at)
    return {
        MANUFACTURE_AT: selected.get(MANUFACTURE_AT) or "",
        EXPIRE_AT: selected.get(EXPIRE_AT) or "",
        BATCH: selected.get(BATCH) or "",
        "batch_original": original_batch or "",
        "batch_correction": correction or "",
        "raw_text": selected.get("raw_text", ""),
        "missing_required_fields": missing,
        "batch_review_positions": review_positions,
        "timing_ms": timings,
        "model": "PP-OCRv6-small",
    }


def _decode_image(image_bytes):
    if not image_bytes:
        raise ValueError("Image is empty")
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Unable to decode image")
    height, width = image.shape[:2]
    if width < MIN_IMAGE_EDGE or height < MIN_IMAGE_EDGE:
        raise ValueError("Image is too small")
    return image


def _resize_if_needed(image):
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_IMAGE_EDGE:
        return image
    scale = MAX_IMAGE_EDGE / float(longest)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _contrast_variant(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _lines_to_text(lines):
    ordered = sorted(lines, key=_reading_order_key)
    return "\n".join(line["text"].strip() for line in ordered if line.get("text", "").strip())


def _reading_order_key(line):
    points = line.get("box") or []
    if not points:
        return (0, 0)
    return (
        min(float(point[1]) for point in points),
        min(float(point[0]) for point in points),
    )


def _missing_fields(result, required_fields):
    return [field for field in required_fields if not str(result.get(field) or "").strip()]


def _select_result(primary, enhanced, combined, required_fields):
    candidates = (primary, enhanced, combined)
    return max(
        enumerate(candidates),
        key=lambda item: (_hit_count(item[1], required_fields), -item[0]),
    )[1].copy()


def _hit_count(result, required_fields):
    return sum(bool(str(result.get(field) or "").strip()) for field in required_fields)


def _merge_text(*texts):
    lines = []
    for text in texts:
        for line in str(text or "").splitlines():
            normalized = line.strip()
            if normalized and normalized not in lines:
                lines.append(normalized)
    return "\n".join(lines)


def _elapsed_ms(started_at):
    return round((time.perf_counter() - started_at) * 1000)
