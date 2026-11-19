import unittest
from unittest.mock import Mock

from general_text_ocr import recognize_text
from text_dates import parse_text_dates
from tests.test_package_ocr import image_bytes, lines


class TextDateTest(unittest.TestCase):
    def test_explicit_dates_and_roles_preserve_original_text(self):
        text = "生产日期2026年11月19日\nEXP 31/12/2028\n19 NOV 2026"
        result = parse_text_dates(text)
        self.assertEqual(["2026-11-19", "2028-12-31", "2026-11-19"], [d["value"] for d in result["dates"]])
        self.assertEqual(["manufacture_at", "expire_at", ""], [d["field"] for d in result["dates"]])
        self.assertEqual("生产日期2026-11-19\nEXP 2028-12-31\n2026-11-19", result["normalized_text"])

    def test_ambiguous_dates_are_not_silently_normalized(self):
        result = parse_text_dates("EXP 01/02/2028")
        self.assertEqual("ambiguous", result["dates"][0]["status"])
        self.assertEqual(["2028-02-01", "2028-01-02"], result["dates"][0]["candidates"])
        self.assertEqual("EXP 01/02/2028", result["normalized_text"])
        self.assertEqual("2028-02-01", parse_text_dates("01/02/2028", "dmy")["dates"][0]["value"])
        self.assertEqual("2028-01-02", parse_text_dates("01/02/2028", "mdy")["dates"][0]["value"])

    def test_invalid_dates_leap_years_and_embedded_codes(self):
        result = parse_text_dates("2027-02-29 2028-02-29 31/04/2028 ABC20281119X")
        self.assertEqual(["invalid", "valid", "invalid"], [d["status"] for d in result["dates"]])
        self.assertEqual("2028-11-19", parse_text_dates("20281119")["dates"][0]["value"])
        self.assertEqual([], parse_text_dates("LOT ABC123\n2028-11")["dates"])
        self.assertEqual([], parse_text_dates("商品编码 12345678")["dates"])

    def test_nearby_date_labels_are_not_swapped(self):
        dates = parse_text_dates("MFG 2026-11-19 EXP 2028-11-19")["dates"]
        self.assertEqual(["manufacture_at", "expire_at"], [d["field"] for d in dates])

    def test_english_month_first_and_compact_dates(self):
        dates = parse_text_dates("November 19, 2026\n31122028\n20012028\n01022028")["dates"]
        self.assertEqual(["2026-11-19", "2028-12-31", "2028-01-20", ""], [d["value"] for d in dates])
        self.assertEqual("ambiguous", dates[-1]["status"])

    def test_identifier_values_are_not_normalized_as_dates(self):
        text = "SKU: 20281119\nBATCH NO. 20281119 EXP 2028-12-31\n批次号：20281119"
        result = parse_text_dates(text)
        self.assertEqual(["2028-12-31"], [d["value"] for d in result["dates"]])
        self.assertEqual(text, result["normalized_text"])

    def test_roles_do_not_leak_to_later_dates_or_other_labels(self):
        result = parse_text_dates("MFG 2026-11-19; REVIEW 2026-11-20\n生产日期 2026-11-19，送货日期 2026-11-20\nEXP 2028-12-31 LOT 20281119")
        self.assertEqual(["manufacture_at", "", "manufacture_at", "", "expire_at"], [d["field"] for d in result["dates"]])

    def test_glued_date_labels_are_supported_without_extracting_inside_codes(self):
        result = parse_text_dates("MFG2026/11/19 EXP2028/12/31 CODEMFG20261119")
        self.assertEqual(["2026-11-19", "2028-12-31"], [d["value"] for d in result["dates"]])
        self.assertEqual(["manufacture_at", "expire_at"], [d["field"] for d in result["dates"]])


class GeneralTextOcrTest(unittest.TestCase):
    def test_confident_identifier_skips_detection_without_extracting_embedded_date(self):
        detector = Mock(side_effect=AssertionError("Detection should not run"))
        line_reader = Mock(return_value=lines("AS202611190001"))
        result = recognize_text(image_bytes(), recognizer=detector, text_mode="identifier", line_recognizer=line_reader)
        self.assertEqual("AS202611190001", result["raw_text"])
        self.assertEqual([], result["dates"])
        self.assertEqual(0, result["timing_ms"]["primary_ocr"])
        self.assertEqual({"fast_path_used": True, "fallback_reason": None, "line_confidence": 0.99}, result["line_diagnostics"])
        line_reader.assert_called_once()
        detector.assert_not_called()

    def test_low_confidence_empty_or_multiple_lines_fall_back_to_detection(self):
        low_score = lines("AS123")
        low_score[0]["confidence"] = 0.5
        for candidate, reason in [([], "no_text"), (low_score, "low_confidence"), (lines("FIRST", "SECOND"), "multiple_lines"), (lines("GENERAL TEXT ABC123"), "invalid_identifier"), (lines("GENERALTEXT"), "invalid_identifier")]:
            with self.subTest(candidate=candidate):
                detector = Mock(return_value=lines("CORRECT"))
                result = recognize_text(image_bytes(), recognizer=detector, text_mode="identifier", line_recognizer=lambda _: candidate)
                self.assertEqual("CORRECT", result["raw_text"])
                self.assertFalse(result["line_diagnostics"]["fast_path_used"])
                self.assertEqual(reason, result["line_diagnostics"]["fallback_reason"])
                detector.assert_called_once()

    def test_line_engine_error_falls_back_and_paragraph_does_not_use_line_engine(self):
        for mode in ["identifier", "line", "paragraph"]:
            with self.subTest(mode=mode):
                line_reader = Mock(side_effect=RuntimeError("Line inference failed"))
                detector = Mock(return_value=lines("CORRECT"))
                result = recognize_text(image_bytes(), recognizer=detector, text_mode=mode, line_recognizer=line_reader)
                self.assertEqual("CORRECT", result["raw_text"])
                detector.assert_called_once()
                self.assertEqual(1 if mode == "identifier" else 0, line_reader.call_count)
                if mode == "identifier":
                    self.assertEqual("line_error", result["line_diagnostics"]["fallback_reason"])
                else:
                    self.assertNotIn("line_diagnostics", result)

    def test_invalid_mode_is_rejected_before_inference(self):
        detector = Mock()
        with self.assertRaises(ValueError):
            recognize_text(image_bytes(), recognizer=detector, text_mode="invalid")
        detector.assert_not_called()

    def test_arbitrary_text_succeeds_without_any_business_field(self):
        result = recognize_text(image_bytes(), recognizer=lambda _: lines("普通文字 ABC-123", "EXP 31/12/2028"))
        self.assertEqual("普通文字 ABC-123\nEXP 31/12/2028", result["raw_text"])
        self.assertEqual("普通文字 ABC-123\nEXP 2028-12-31", result["normalized_text"])
        self.assertEqual(2, len(result["lines"]))

    def test_fragments_on_same_line_are_read_left_to_right(self):
        fragments = [
            {"text": "2028-12-31", "confidence": 0.98, "box": [[60, 0], [120, 0], [120, 20], [60, 20]]},
            {"text": "EXP", "confidence": 0.99, "box": [[0, 2], [50, 2], [50, 22], [0, 22]]},
        ]
        result = recognize_text(image_bytes(), recognizer=lambda _: fragments)
        self.assertEqual("EXP 2028-12-31", result["raw_text"])
        self.assertEqual("expire_at", result["dates"][0]["field"])

    def test_empty_result_retries_once_without_merging_text(self):
        calls = []
        def recognizer(_):
            calls.append(True)
            return [] if len(calls) == 1 else lines("普通文字")
        result = recognize_text(image_bytes(), recognizer=recognizer)
        self.assertEqual(2, len(calls))
        self.assertEqual("普通文字", result["raw_text"])
        self.assertEqual([], result["dates"])
