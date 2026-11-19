"""Offline real-model acceptance for positive/negative and low-contrast text."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_text_ocr import recognize_text
from package_ocr import recognize_package_text


def main():
    results = []
    for name, background, foreground in [
        ("white_black", 255, 0), ("black_white", 0, 255),
        ("dark_gray_white", 45, 230), ("black_gray", 0, 135),
    ]:
        image = np.full((260, 1100, 3), background, np.uint8)
        color = (foreground,) * 3
        for text, y in [("GENERAL TEXT ABC123", 60), ("EXP 31/12/2028", 135), ("LOT ABC123", 210)]:
            cv2.putText(image, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        data = encoded.tobytes()
        general = recognize_text(data)
        package = recognize_package_text(data, ["expire_at", "batch"])
        assert general["raw_text"] == "GENERAL TEXT ABC123\nEXP 31/12/2028\nLOT ABC123", (name, general)
        assert general["dates"][0]["value"] == "2028-12-31", (name, general)
        assert package["expire_at"] == "2028-12-31", (name, package)
        assert package["batch"] == "ABC123", (name, package)
        results.append({"case": name, "raw_text": general["raw_text"], "date": package["expire_at"], "batch": package["batch"]})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
