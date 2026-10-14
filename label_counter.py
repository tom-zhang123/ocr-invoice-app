"""Count high-brightness rectangular shipping labels in one camera frame."""
import cv2
import numpy as np


MAX_PROCESSING_SIDE = 1600


def _resize_for_processing(image):
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_PROCESSING_SIDE:
        return image, 1.0
    scale = MAX_PROCESSING_SIDE / longest
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def count_white_labels(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片无法解码，请重新拍照")
    source_height, source_width = image.shape[:2]
    image, scale = _resize_for_processing(image)
    height, width = image.shape[:2]
    image_area = height * width

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white = ((saturation <= 70) & (value >= 160)).astype(np.uint8) * 255

    kernel_size = max(3, round(min(height, width) * 0.008))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = image_area * 0.002
    maximum_area = image_area * 0.45
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 28 or box_height < 20:
            continue
        aspect_ratio = box_width / box_height
        if not 0.55 <= aspect_ratio <= 4.2:
            continue
        rectangularity = area / (box_width * box_height)
        if rectangularity < 0.70:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if not 4 <= len(polygon) <= 8:
            continue
        region = hsv[y:y + box_height, x:x + box_width]
        if float(np.mean(region[:, :, 2])) < 175:
            continue
        touches_edges = sum((x <= 2, y <= 2, x + box_width >= width - 2, y + box_height >= height - 2))
        if touches_edges >= 3:
            continue

        inverse_scale = 1 / scale
        boxes.append({
            "x": round(x * inverse_scale),
            "y": round(y * inverse_scale),
            "width": round(box_width * inverse_scale),
            "height": round(box_height * inverse_scale),
            "confidence": round(min(0.99, rectangularity), 3),
        })

    boxes.sort(key=lambda box: (box["y"], box["x"]))
    return {
        "count": len(boxes),
        "boxes": boxes,
        "image": {"width": source_width, "height": source_height},
    }
