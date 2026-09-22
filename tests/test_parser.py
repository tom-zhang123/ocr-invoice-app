import unittest

from parser import (
    clean_sku,
    extract_header,
    parse_batch_candidates,
    reconcile_batch_candidate,
    reconcile_date_years,
    reconcile_quantities_with_total,
)


class SkuCleanupTest(unittest.TestCase):
    def test_removes_trailing_non_digit_when_remaining_gtin_is_valid(self):
        self.assertEqual("18934804041534", clean_sku("18934804041534N"))
        self.assertEqual("8934804035178", clean_sku("8934804035178*"))


class DateYearReconciliationTest(unittest.TestCase):
    def test_repairs_unique_context_year_error(self):
        row = {"exp": "2099-09-11", "mfg": "2096-06-11"}

        result = reconcile_date_years(
            row,
            {"inbound_date": "19/09/2026"},
            current_year=2035,
        )

        self.assertEqual("2029-09-11", result["exp"])
        self.assertEqual("2026-06-11", result["mfg"])
        self.assertEqual("2099-09-11", result["raw_exp"])
        self.assertEqual("context_year", result["corrections"]["exp"])

    def test_clears_ambiguous_year_for_manual_entry(self):
        row = {"exp": "2021-09-11", "mfg": ""}

        result = reconcile_date_years(row, {}, current_year=2026)

        self.assertEqual("", result["exp"])
        self.assertEqual("2021-09-11", result["raw_exp"])
        self.assertEqual("manual_required", result["review_reasons"]["exp"])

    def test_repairs_unique_two_character_year_error(self):
        row = {"exp": "9098-07-31", "mfg": ""}

        result = reconcile_date_years(row, {}, current_year=2026)

        self.assertEqual("2028-07-31", result["exp"])
        self.assertEqual("context_year", result["corrections"]["exp"])

    def test_flags_expiry_before_manufacturing(self):
        row = {"exp": "2026-01-01", "mfg": "2026-06-01"}

        result = reconcile_date_years(row, {}, current_year=2026)

        self.assertEqual("date_order", result["review_reasons"]["exp"])
        self.assertEqual("date_order", result["review_reasons"]["mfg"])

    def test_clears_invalid_date_for_manual_entry(self):
        row = {"exp": "2028-04-50", "mfg": "B"}

        result = reconcile_date_years(row, {}, current_year=2026)

        self.assertEqual("", result["exp"])
        self.assertEqual("", result["mfg"])
        self.assertEqual("2028-04-50", result["raw_exp"])
        self.assertEqual("invalid_date_cleared", result["corrections"]["exp"])
        self.assertEqual("manual_required", result["review_reasons"]["mfg"])

    def test_clears_partial_year_month_values(self):
        row = {"exp": "1901-07", "mfg": "2026-05"}

        result = reconcile_date_years(row, {}, current_year=2026)

        self.assertEqual("", result["exp"])
        self.assertEqual("", result["mfg"])


class HeaderExtractionTest(unittest.TestCase):
    def test_extracts_multi_page_marker(self):
        items = [
            {"text": "Page 2 of 2"},
            {"text": "PO667003"},
        ]

        header = extract_header(items)

        self.assertEqual("PO667003", header["po_no"])
        self.assertEqual(2, header["page_number"])
        self.assertEqual(2, header["total_pages"])


class QuantityTotalReconciliationTest(unittest.TestCase):
    def test_applies_unique_candidate_combination_matching_total(self):
        rows = [
            {
                "qty": "72",
                "field_candidates": {"qty": [
                    {"text": "7"},
                    {"text": "72"},
                ]},
                "review_reasons": {"qty": "model_disagreement"},
            },
            {
                "qty": "271",
                "field_candidates": {"qty": [
                    {"text": "27"},
                    {"text": "271"},
                ]},
                "review_reasons": {"qty": "model_disagreement"},
            },
        ]

        result = reconcile_quantities_with_total(rows, 34)

        self.assertEqual(["7", "27"], [row["qty"] for row in result])
        self.assertEqual("invoice_total_candidates", result[0]["corrections"]["qty"])
        self.assertNotIn("qty", result[0]["review_reasons"])

    def test_applies_unique_single_digit_change_matching_total(self):
        rows = [
            {"qty": "232", "review_reasons": {"qty": "model_disagreement"}},
            {"qty": "53"},
        ]

        result = reconcile_quantities_with_total(rows, 185)

        self.assertEqual("132", result[0]["qty"])
        self.assertEqual("232", result[0]["raw_qty"])
        self.assertEqual(
            "invoice_total_single_digit",
            result[0]["corrections"]["qty"],
        )
        self.assertNotIn("qty", result[0]["review_reasons"])

    def test_does_not_guess_when_multiple_single_digit_changes_match_total(self):
        rows = [
            {"qty": "232", "review_reasons": {"qty": "model_disagreement"}},
            {"qty": "260", "review_reasons": {"qty": "model_disagreement"}},
        ]

        result = reconcile_quantities_with_total(rows, 392)

        self.assertEqual(["232", "260"], [row["qty"] for row in result])


class BatchCandidateTest(unittest.TestCase):
    def test_matches_submitted_batch_candidate(self):
        candidates = parse_batch_candidates('["633B1PCHP1", "628G1PCHS1"]')

        row = reconcile_batch_candidate({"batch": "633BIPCHP1"}, candidates)

        self.assertEqual("633B1PCHP1", row["batch"])
        self.assertEqual("633BIPCHP1", row["raw_batch"])

    def test_without_candidates_preserves_old_ocr_value(self):
        row = {"batch": "RAWBATCH01"}

        result = reconcile_batch_candidate(row, [])

        self.assertEqual("RAWBATCH01", result["batch"])
        self.assertNotIn("raw_batch", result)

    def test_ambiguous_candidate_is_left_for_manual_selection(self):
        row = reconcile_batch_candidate(
            {"batch": "ABC12"},
            ["ABC11", "ABC13"],
        )

        self.assertEqual("", row["batch"])
        self.assertEqual("manual_required", row["review_reasons"]["batch"])


if __name__ == "__main__":
    unittest.main()
