import unittest

from package_text_parser import (
    BATCH,
    EXPIRE_AT,
    MANUFACTURE_AT,
    confusion_positions,
    parse_batch_candidates,
    parse_package_text,
    parse_required_fields,
    reconcile_batch_candidate,
)


class PackageTextParserTest(unittest.TestCase):
    def test_six_sample_text_shapes(self):
        cases = [
            (
                "06/07/2027\n618720478A",
                [EXPIRE_AT, BATCH],
                {EXPIRE_AT: "2027-07-06", BATCH: "618720478A"},
            ),
            (
                "MAH:01 04.2026\nEXP21.03.2028\n6 091 4282 20 01:57 02 092",
                [MANUFACTURE_AT, EXPIRE_AT, BATCH],
                {
                    MANUFACTURE_AT: "2026-04-01",
                    EXPIRE_AT: "2028-03-21",
                    BATCH: "6 091 4282 20 01:57 02 092",
                },
            ),
            (
                "03/01/2028\n618405316B",
                [EXPIRE_AT, BATCH],
                {EXPIRE_AT: "2028-01-03", BATCH: "618405316B"},
            ),
            (
                "24/12/2027\n:43 61752131L\n3:43 61752131L",
                [EXPIRE_AT, BATCH],
                {EXPIRE_AT: "2027-12-24", BATCH: "61752131L"},
            ),
            (
                "DLD no. 01 07 68 0846\n07/08/2026 62194832GA\n06/02/2028 02:26",
                [MANUFACTURE_AT, EXPIRE_AT, BATCH],
                {
                    MANUFACTURE_AT: "2026-08-07",
                    EXPIRE_AT: "2028-02-06",
                    BATCH: "62194832GA",
                },
            ),
            (
                "1500\n27/06/2027\n617820476A",
                [EXPIRE_AT, BATCH],
                {EXPIRE_AT: "2027-06-27", BATCH: "617820476A"},
            ),
        ]
        for raw_text, required_fields, expected in cases:
            with self.subTest(raw_text=raw_text):
                parsed = parse_package_text(raw_text, required_fields)
                for field, value in expected.items():
                    self.assertEqual(value, parsed[field])

    def test_single_unlabelled_date_obeys_required_field(self):
        manufacture = parse_package_text("01/04/2026\nABC123", [MANUFACTURE_AT, BATCH])
        expiry = parse_package_text("01/04/2026\nABC123", [EXPIRE_AT, BATCH])

        self.assertEqual("2026-04-01", manufacture[MANUFACTURE_AT])
        self.assertIsNone(manufacture[EXPIRE_AT])
        self.assertEqual("2026-04-01", expiry[EXPIRE_AT])

    def test_invalid_and_conflicting_dates_are_not_guessed(self):
        invalid = parse_package_text("EXP 31/02/2028\nABC123", [EXPIRE_AT, BATCH])
        conflict = parse_package_text(
            "MFG 06/07/2028\nEXP 06/07/2027\nABC123",
            [MANUFACTURE_AT, EXPIRE_AT, BATCH],
        )

        self.assertIsNone(invalid[EXPIRE_AT])
        self.assertIsNone(conflict[MANUFACTURE_AT])
        self.assertIsNone(conflict[EXPIRE_AT])

    def test_unique_asn_candidate_corrects_e_five_and_marks_position(self):
        batch, positions, correction = reconcile_batch_candidate(
            "618405316B",
            ["61840E316B", "617820476A"],
        )

        self.assertEqual("61840E316B", batch)
        self.assertEqual([5], positions)
        self.assertEqual("asn_batch_candidate", correction)

    def test_ambiguous_candidate_is_not_applied(self):
        batch, positions, correction = reconcile_batch_candidate(
            "A5C123",
            ["AEC123", "ASC123"],
        )

        self.assertEqual("A5C123", batch)
        self.assertEqual([], positions)
        self.assertIsNone(correction)

    def test_variant_disagreement_marks_only_supported_confusion(self):
        self.assertEqual([1], confusion_positions("A5C", ["AEC"]))
        self.assertEqual([], confusion_positions("ABC", ["AXC"]))

    def test_contract_lists_are_validated_and_deduplicated(self):
        self.assertEqual(
            [EXPIRE_AT, BATCH],
            parse_required_fields('["expire_at","batch","batch"]'),
        )
        self.assertEqual(
            ["ABC-01", "XYZ02"],
            parse_batch_candidates('["ABC-01","abc01","XYZ02"]'),
        )
        with self.assertRaises(ValueError):
            parse_required_fields('["unknown"]')


if __name__ == "__main__":
    unittest.main()
