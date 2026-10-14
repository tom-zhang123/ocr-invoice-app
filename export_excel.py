"""Export the operator's current invoice table as an Excel workbook."""
from datetime import date
from io import BytesIO
import re

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from pydantic import BaseModel, Field


class ExportRow(BaseModel):
    sku: str = Field(default="", max_length=64)
    name: str = Field(default="", max_length=256)
    qty: str = Field(default="", max_length=32)
    exp: str = Field(default="", max_length=32)
    mfg: str = Field(default="", max_length=32)
    batch: str = Field(default="", max_length=64)
    pallet: str = Field(default="", max_length=64)
    box: str = Field(default="", max_length=64)
    row_confidence: float | None = Field(default=None, ge=0, le=1)
    reviewed: bool = False


class ExportRequest(BaseModel):
    po_no: str = Field(default="", max_length=64)
    note: str = Field(default="", max_length=1000)
    document_total: int | None = Field(default=None, ge=0)
    include_batch: bool = True
    rows: list[ExportRow] = Field(min_length=1, max_length=500)


COLUMNS = (
    ("sku", "SKU", 21),
    ("name", "品名", 32),
    ("qty", "Qty", 10),
    ("exp", "EXP", 14),
    ("mfg", "MFG", 14),
    ("batch", "批次号", 20),
    ("pallet", "板号", 12),
    ("box", "箱号", 12),
)


def _set_text(cell, value):
    cell.value = str(value or "")
    cell.data_type = "s"
    cell.number_format = "@"


def _set_quantity(cell, value):
    raw = str(value).strip()
    if re.fullmatch(r"[1-9]\d*", raw) and len(raw) <= 15:
        cell.value = int(raw)
        cell.number_format = "#,##0"
        return True
    _set_text(cell, raw)
    cell.fill = PatternFill("solid", fgColor="FFF0CC")
    return False


def _set_date(cell, value):
    raw = str(value or "").strip()
    if raw and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        try:
            cell.value = date.fromisoformat(raw)
            cell.number_format = "yyyy-mm-dd"
            return
        except ValueError:
            pass
    _set_text(cell, raw)


def build_excel(request: ExportRequest) -> BytesIO:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "入库单明细"
    sheet.sheet_view.showGridLines = False

    sheet["A1"] = "入库单核实明细"
    sheet["A1"].font = Font(name="Arial", size=14, bold=True, color="263238")
    sheet["A2"] = "PO 号"
    _set_text(sheet["B2"], request.po_no)
    sheet["D2"] = "明细合计"
    sheet["G2"] = "单据合计"
    if request.document_total is not None:
        sheet["H2"] = request.document_total
        sheet["H2"].number_format = "#,##0"
    sheet["A3"] = "备注"
    _set_text(sheet["B3"], request.note)

    columns = [column for column in COLUMNS if request.include_batch or column[0] != "batch"]
    headers = ["#", *(column[1] for column in columns), "置信度", "核实"]
    for position, label in enumerate(headers, 1):
        cell = sheet.cell(5, position, label)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="334155")
        cell.alignment = Alignment(horizontal="center", vertical="center")
    sheet.row_dimensions[5].height = 25

    total = 0
    all_quantities_valid = True
    for index, row in enumerate(request.rows, 1):
        excel_row = index + 5
        sheet.cell(excel_row, 1, index)
        for position, (field, _, _) in enumerate(columns, 2):
            cell = sheet.cell(excel_row, position)
            value = getattr(row, field)
            if field == "qty":
                valid = _set_quantity(cell, value)
                all_quantities_valid = all_quantities_valid and valid
                if valid:
                    total += cell.value
            elif field in ("exp", "mfg"):
                _set_date(cell, value)
            else:
                _set_text(cell, value)

        confidence = sheet.cell(excel_row, len(columns) + 2)
        if row.row_confidence is not None:
            confidence.value = row.row_confidence
            confidence.number_format = "0%"
        status = sheet.cell(excel_row, len(columns) + 3)
        _set_text(status, "已勾选" if row.reviewed else "未勾选")
        if not row.reviewed:
            status.fill = PatternFill("solid", fgColor="FFF0CC")
        sheet.row_dimensions[excel_row].height = 21

    if all_quantities_valid:
        sheet["E2"] = total
        sheet["E2"].number_format = "#,##0"
    else:
        _set_text(sheet["E2"], "待核实")

    widths = [6, *(column[2] for column in columns), 12, 12]
    for position, width in enumerate(widths, 1):
        sheet.column_dimensions[sheet.cell(5, position).column_letter].width = width
    sheet.freeze_panes = "C6"
    last_column = sheet.cell(5, len(headers)).column_letter
    table = Table(displayName="InvoiceRows", ref=f"A5:{last_column}{len(request.rows) + 5}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    sheet.add_table(table)

    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return output
