import unittest
from datetime import datetime

from openpyxl import load_workbook
from pydantic import ValidationError

from export_excel import ExportRequest, build_excel


class ExcelExportTest(unittest.TestCase):
    def test_exports_current_values_with_spreadsheet_types(self):
        request = ExportRequest(
            po_no="PO0066940",
            note="人工复核",
            document_total=1932,
            rows=[
                {
                    "sku": "0018853301013946",
                    "name": "",
                    "qty": "24",
                    "exp": "2028-09-07",
                    "mfg": "2026-09-07",
                    "batch": "=SUM(1,1)",
                    "pallet": "1/1",
                    "box": "24",
                    "row_confidence": 0.91,
                    "reviewed": True,
                },
                {
                    "sku": "18853301014011",
                    "qty": "7",
                    "box": "7",
                    "row_confidence": 0.46,
                    "reviewed": False,
                },
            ],
        )

        workbook = load_workbook(build_excel(request))
        sheet = workbook.active

        self.assertEqual("入库单明细", sheet.title)
        self.assertEqual("PO0066940", sheet["B2"].value)
        self.assertEqual("人工复核", sheet["B3"].value)
        self.assertEqual(31, sheet["E2"].value)
        self.assertEqual(1932, sheet["H2"].value)
        self.assertEqual("0018853301013946", sheet["B6"].value)
        self.assertEqual("s", sheet["B6"].data_type)
        self.assertEqual("", sheet["C6"].value or "")
        self.assertEqual(24, sheet["D6"].value)
        self.assertEqual(datetime(2028, 9, 7), sheet["E6"].value)
        self.assertEqual("=SUM(1,1)", sheet["G6"].value)
        self.assertEqual("s", sheet["G6"].data_type)
        self.assertEqual("1/1", sheet["H6"].value)
        self.assertEqual(0.91, sheet["J6"].value)
        self.assertEqual("0%", sheet["J6"].number_format)
        self.assertEqual("已勾选", sheet["K6"].value)
        self.assertEqual("未勾选", sheet["K7"].value)
        self.assertEqual("C6", sheet.freeze_panes)
        self.assertEqual("A5:K7", sheet.tables["InvoiceRows"].ref)

    def test_invalid_quantity_is_preserved_and_total_requires_review(self):
        request = ExportRequest(
            include_batch=False,
            rows=[{"sku": "00123456789012", "qty": "7x", "exp": "2028-02"}],
        )

        sheet = load_workbook(build_excel(request)).active

        self.assertEqual("待核实", sheet["E2"].value)
        self.assertEqual("7x", sheet["D6"].value)
        self.assertEqual("s", sheet["D6"].data_type)
        self.assertEqual("2028-02", sheet["E6"].value)
        self.assertNotIn("批次号", [cell.value for cell in sheet[5]])
        self.assertEqual("A5:J6", sheet.tables["InvoiceRows"].ref)

    def test_rejects_oversized_export(self):
        with self.assertRaises(ValidationError):
            ExportRequest(rows=[{}] * 501)


if __name__ == "__main__":
    unittest.main()
