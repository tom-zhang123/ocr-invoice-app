import unittest

import numpy as np

from enhanced_ocr import (
    COLUMN_BOUNDS,
    FIELD_ORDER,
    WITHOUT_BATCH_BOUNDS,
    fit_regular_row_centers,
    prepare_page_for_ocr,
    refine_column_layout,
    row_centers_to_lines,
    select_aligned_fragments,
    select_fallback_fields,
    select_quantity_candidate,
    select_sku_candidate,
    select_structured_candidate,
)


class LayoutSelectionTest(unittest.TestCase):
    def test_uses_header_positions_for_seven_column_layout(self):
        items = [
            {"text": "PALLET", "xc": 95, "yc": 138},
            {"text": "BOX", "xc": 190, "yc": 143},
            {"text": "SKU #", "xc": 334, "yc": 145},
            {"text": "Name", "xc": 628, "yc": 139},
            {"text": "aty", "xc": 872, "yc": 116},
        ]

        bounds, layout = refine_column_layout(items, 166, COLUMN_BOUNDS, "with_batch")

        self.assertEqual("without_batch", layout)
        self.assertEqual(WITHOUT_BATCH_BOUNDS, bounds)

    def test_keeps_explicit_batch_layout(self):
        items = [
            {"text": "Name", "xc": 574, "yc": 121},
            {"text": "Qty", "xc": 778, "yc": 138},
            {"text": "EXP", "xc": 875, "yc": 138},
            {"text": "MFG", "xc": 996, "yc": 140},
            {"text": "Batch", "xc": 1142, "yc": 144},
        ]

        bounds, layout = refine_column_layout(items, 156, WITHOUT_BATCH_BOUNDS, "fixed")

        self.assertEqual("with_batch", layout)
        self.assertEqual(COLUMN_BOUNDS, bounds)

    def test_uses_numeric_density_when_continuation_page_has_no_header(self):
        items = []
        for index in range(10):
            y = 200 + index * 40
            items.extend([
                {"text": f"930060513{100 + index}", "xc": 340, "yc": y},
                {"text": str(10 + index), "xc": 890, "yc": y},
                {"text": "2027.09.11", "xc": 1000, "yc": y},
                {"text": "2026.06.11", "xc": 1150, "yc": y},
            ])

        bounds, layout = refine_column_layout(items, 162, COLUMN_BOUNDS, "fixed")

        self.assertEqual("without_batch", layout)
        self.assertEqual(WITHOUT_BATCH_BOUNDS, bounds)


class RowFittingTest(unittest.TestCase):
    def test_fits_missing_first_observation_without_shifting_rows(self):
        fallback = [172 + index * 33 for index in range(10)]
        observed = [fallback[index] + (0.4 if index % 2 else -0.3) for index in range(1, 10)]

        fitted = fit_regular_row_centers(observed, 10, fallback)

        self.assertEqual(10, len(fitted))
        self.assertLess(abs(fitted[0] - fallback[0]), 2)
        self.assertLess(abs(fitted[-1] - fallback[-1]), 2)

    def test_converts_centers_to_bounded_lines(self):
        lines = row_centers_to_lines([20, 40, 60], 5, 75)

        self.assertEqual([10, 30, 50, 70], lines)

    def test_does_not_merge_fragments_on_different_baselines(self):
        fragments = [
            {"text": "16", "score": 0.99, "x1": 850, "x2": 877, "yc": 823, "y1": 807, "y2": 839},
            {"text": "28", "score": 0.99, "x1": 880, "x2": 912, "yc": 833, "y1": 814, "y2": 852},
        ]

        selected = select_aligned_fragments(fragments, target_y=833, target_x=890)

        self.assertEqual(["28"], [fragment["text"] for fragment in selected])


class CandidateSelectionTest(unittest.TestCase):
    def test_box_value_selects_an_existing_quantity_candidate(self):
        selected, _, ambiguous, _ = select_quantity_candidate(
            "54",
            0.86,
            [{"source": "cell", "text": "5", "score": 0.91}],
            box_value="54",
        )

        self.assertEqual("54", selected)
        self.assertTrue(ambiguous)

    def test_valid_cell_date_replaces_invalid_page_date(self):
        selected, _, ambiguous, _ = select_structured_candidate(
            "exp",
            "9098-07-31",
            0.95,
            [{"source": "cell", "text": "2028-07-31", "score": 0.82}],
        )

        self.assertEqual("2028-07-31", selected)
        self.assertFalse(ambiguous)

    def test_invalid_cell_date_does_not_fill_blank_value(self):
        selected, _, ambiguous, _ = select_structured_candidate(
            "mfg",
            "",
            None,
            [{"source": "cell", "text": "B", "score": 0.25}],
        )

        self.assertEqual("", selected)
        self.assertFalse(ambiguous)

    def test_removes_disagreeing_low_confidence_leading_slash_artifacts(self):
        selected, _, ambiguous, _ = select_quantity_candidate(
            "625",
            0.77,
            [{"source": "cell", "text": "525", "score": 0.68}],
        )

        self.assertEqual("25", selected)
        self.assertTrue(ambiguous)

    def test_sku_majority_overrides_disagreeing_page_reading(self):
        selected, _, ambiguous, _ = select_sku_candidate(
            "8934804041534",
            1.0,
            [
                {"source": "cell", "text": "18934804041534", "score": 0.96},
                {"source": "column", "text": "18934804041534", "score": 0.98},
            ],
        )

        self.assertEqual("18934804041534", selected)
        self.assertFalse(ambiguous)

    def test_shorter_sku_majority_cannot_replace_complete_page_barcode(self):
        selected, _, ambiguous, _ = select_sku_candidate(
            "4902397871897",
            1.0,
            [
                {"source": "cell", "text": "97871897", "score": 1.0},
                {"source": "column", "text": "97871897", "score": 1.0},
            ],
        )

        self.assertEqual("4902397871897", selected)
        self.assertTrue(ambiguous)


class FastPathTest(unittest.TestCase):
    def test_keeps_full_page_when_row_geometry_is_unavailable(self):
        image = np.full((280, 1280, 3), 255, dtype=np.uint8)
        image[165:180, 500:520] = 0

        prepared, skipped = prepare_page_for_ocr(
            image,
            header_y=150,
            table_bottom=250,
            column_bounds=COLUMN_BOUNDS,
            table_layout="fixed",
            row_lines=None,
            ocr_bottom=270,
            ocr_right=1280,
        )

        self.assertFalse(skipped)
        self.assertTrue(np.array_equal(image[:270], prepared))

    def test_masks_only_data_name_cells_and_preserves_header_and_last_row(self):
        image = np.full((280, 1280, 3), 255, dtype=np.uint8)
        image[130:140, 500:520] = 0
        image[165:180, 500:520] = 0
        image[215:230, 500:520] = 0
        image[165:180, 300:320] = 0
        image[200:202, 390:725] = 0

        prepared, skipped = prepare_page_for_ocr(
            image,
            header_y=150,
            table_bottom=250,
            column_bounds=COLUMN_BOUNDS,
            table_layout="with_batch",
            row_lines=[150, 200, 250],
            ocr_bottom=270,
            ocr_right=1280,
        )

        self.assertTrue(skipped)
        self.assertTrue(np.all(prepared[165:180, 500:520] == 255))
        self.assertTrue(np.all(prepared[130:140, 500:520] == 0))
        self.assertTrue(np.all(prepared[215:230, 500:520] == 0))
        self.assertTrue(np.all(prepared[165:180, 300:320] == 0))
        self.assertTrue(np.all(prepared[200:202, 390:725] == 0))

    def test_selects_fallback_only_for_unreliable_visible_fields(self):
        rec = {field: [] for field in FIELD_ORDER}
        conf = {field: None for field in FIELD_ORDER}
        rec["sku"] = [{"text": "18853301013946", "score": 0.98, "xc": 310}]
        conf["sku"] = 0.98
        rec["batch"] = [{"text": "WRONG001", "score": 0.95, "xc": 1100}]
        conf["batch"] = 0.95
        table_rows = [{
            "yc": 180,
            "rec": rec,
            "conf": conf,
            "review_reasons": {},
        }]
        centers = {field: [180] for field in COLUMN_BOUNDS}

        selected = select_fallback_fields(
            table_rows,
            ("sku", "box", "batch"),
            {(0, "batch")},
            COLUMN_BOUNDS,
            centers,
            ["628B18ST01"],
        )

        self.assertEqual(["batch"], selected)

    def test_low_confidence_required_field_activates_fallback(self):
        rec = {field: [] for field in FIELD_ORDER}
        conf = {field: None for field in FIELD_ORDER}
        rec["sku"] = [{"text": "18853301013946", "score": 0.70, "xc": 310}]
        conf["sku"] = 0.70
        table_rows = [{"yc": 180, "rec": rec, "conf": conf}]
        centers = {field: [180] for field in COLUMN_BOUNDS}

        selected = select_fallback_fields(
            table_rows,
            ("sku",),
            set(),
            COLUMN_BOUNDS,
            centers,
        )

        self.assertEqual(["sku"], selected)


if __name__ == "__main__":
    unittest.main()
