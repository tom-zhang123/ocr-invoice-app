import unittest

import cv2
import numpy as np

from package_ocr import recognize_package_text
from package_text_parser import BATCH, EXPIRE_AT, MANUFACTURE_AT


def image_bytes():
    ok, encoded = cv2.imencode(".jpg", np.full((160, 640, 3), 255, dtype=np.uint8))
    if not ok:
        raise AssertionError("Unable to create test image")
    return encoded.tobytes()


def lines(*texts):
    return [
        {
            "box": [[0, index * 30], [500, index * 30], [500, index * 30 + 20], [0, index * 30 + 20]],
            "text": text,
            "confidence": 0.99,
        }
        for index, text in enumerate(texts)
    ]


class PackageOcrTest(unittest.TestCase):
    def test_complete_primary_result_does_not_run_enhanced_retry(self):
        calls = []

        def recognizer(_image):
            calls.append(True)
            return lines("27/06/2027", "617820476A")

        result = recognize_package_text(
            image_bytes(),
            [EXPIRE_AT, BATCH],
            recognizer=recognizer,
        )

        self.assertEqual(1, len(calls))
        self.assertEqual("2027-06-27", result[EXPIRE_AT])
        self.assertEqual("617820476A", result[BATCH])
        self.assertEqual([], result["missing_required_fields"])
        self.assertEqual(0, result["timing_ms"]["enhanced_ocr"])

    def test_missing_field_runs_one_retry_and_merges_results(self):
        calls = []

        def recognizer(_image):
            calls.append(True)
            if len(calls) == 1:
                return lines("MFG 01/04/2026", "ABC123")
            return lines("EXP 21/03/2028")

        result = recognize_package_text(
            image_bytes(),
            [MANUFACTURE_AT, EXPIRE_AT, BATCH],
            recognizer=recognizer,
        )

        self.assertEqual(2, len(calls))
        self.assertEqual("2026-04-01", result[MANUFACTURE_AT])
        self.assertEqual("2028-03-21", result[EXPIRE_AT])
        self.assertEqual("ABC123", result[BATCH])
        self.assertEqual([], result["missing_required_fields"])

    def test_asn_candidate_correction_is_returned_for_operator_review(self):
        result = recognize_package_text(
            image_bytes(),
            [EXPIRE_AT, BATCH],
            batch_candidates=["61840E316B"],
            recognizer=lambda _image: lines("03/01/2028", "618405316B"),
        )

        self.assertEqual("61840E316B", result[BATCH])
        self.assertEqual("618405316B", result["batch_original"])
        self.assertEqual("asn_batch_candidate", result["batch_correction"])
        self.assertEqual([5], result["batch_review_positions"])

    def test_invalid_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decode"):
            recognize_package_text(b"not-an-image", [EXPIRE_AT])


if __name__ == "__main__":
    unittest.main()
