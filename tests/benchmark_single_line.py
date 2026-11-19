"""Real-model speed and text parity check using generated samples only."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_text_ocr import recognize_text
from package_ocr_models import warmup_package_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", type=Path, help="Optional font containing Chinese glyphs")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    warmup_package_model()
    cases = [("asn", "AS2026111900081"), ("date", "EXP 31/12/2028"), ("text", "GENERAL TEXT ABC123")]
    if args.font:
        cases.append(("chinese", "生产日期2026年11月19日"))
    report = []
    for background, foreground in [(255, 0), (0, 255), (45, 230), (0, 135)]:
        for name, text in cases:
            image = np.full((120, 1100, 3), background, np.uint8)
            if name == "chinese":
                from PIL import Image, ImageDraw, ImageFont
                canvas = Image.fromarray(image)
                ImageDraw.Draw(canvas).text((30, 25), text, font=ImageFont.truetype(str(args.font), 38), fill=(foreground,) * 3)
                image = np.asarray(canvas).copy()
            else:
                cv2.putText(image, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (foreground,) * 3, 2, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", image)
            assert ok
            baseline_ms, line_ms = [], []
            for _ in range(args.runs):
                baseline = recognize_text(encoded.tobytes())
                fast = recognize_text(encoded.tobytes(), text_mode="identifier" if name == "asn" else "line")
                assert fast["raw_text"] == baseline["raw_text"] == text, (name, background, baseline["raw_text"], fast["raw_text"])
                assert fast["dates"] == baseline["dates"], (name, background)
                if name == "asn":
                    assert fast["timing_ms"]["primary_ocr"] == 0, (name, background, "identifier fell back to detection")
                baseline_ms.append(baseline["timing_ms"]["total"])
                line_ms.append(fast["timing_ms"]["total"])
            report.append({"case": name, "background": background, "foreground": foreground,
                           "mode": "identifier" if name == "asn" else "line",
                           "baseline_median_ms": statistics.median(baseline_ms),
                           "single_line_median_ms": statistics.median(line_ms)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
