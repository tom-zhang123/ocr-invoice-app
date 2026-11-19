"""Conservative date extraction for arbitrary OCR text; never guess date roles."""
import re
from datetime import date

_MONTHS = {name: index for index, name in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1
)}
_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"(?P<y>20\d{2}|19\d{2}|21\d{2})\s*(?:[-/.年]|\s)\s*(?P<m>\d{1,2})\s*(?:[-/.月]|\s)\s*(?P<d>\d{1,2})日?"
    r"|(?P<a>\d{1,2})\s*[-/.]\s*(?P<b>\d{1,2})\s*[-/.]\s*(?P<year>19\d{2}|20\d{2}|21\d{2})"
    r"|(?P<day>\d{1,2})[\s./-]*(?P<month>JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)[\s,./-]*(?P<english_year>19\d{2}|20\d{2}|21\d{2})"
    r"|(?P<first_month>JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)[\s./-]+(?P<first_day>\d{1,2})[\s,./-]+(?P<first_year>19\d{2}|20\d{2}|21\d{2})"
    r"|(?P<compact>(?:19|20|21)\d{6}|\d{4}(?:19|20|21)\d{2})"
    r")(?![A-Za-z0-9_])", re.IGNORECASE,
)
_LABELS = {
    "manufacture_at": re.compile(r"\b(?:MFG|MFD|MANUFACTURE|MANUFACTURING|PROD|PRODUCTION)(?:\b|(?=\d))|生产日期|生产|制造日期", re.I),
    "expire_at": re.compile(r"\b(?:EXP|EXPIRY|EXPIRATION|BBE|BEST\s+BEFORE|USE\s+BY)(?:\b|(?=\d))|有效期至|有效期|效期|到期日期|保质期至", re.I),
}
_IDENTIFIER_LABEL = re.compile(
    r"\b(?:LOT|BATCH|SKU|BARCODE|ITEM|PRODUCT\s*CODE)(?:\b|(?=\d))|批次号?|商品编码|物料号|条码", re.I,
)
_LABEL_SUFFIX = re.compile(r"(?:\s|[:：#=.\-]|DATE|NO\.?|NUMBER|CODE|日期|号|为|是|至)*$", re.I)


def parse_text_dates(text, date_order="auto"):
    """Return normalized dates and original spans, including ambiguous/invalid values."""
    if date_order not in {"auto", "dmy", "mdy"}:
        raise ValueError("date_order must be auto, dmy or mdy")
    dates = []
    normalized_lines = []
    for line_index, line in enumerate(str(text or "").splitlines()):
        replacements = []
        scan_line = line
        # Mask only known labels glued to digits, keeping original text and offsets.
        for pattern in [*_LABELS.values(), _IDENTIFIER_LABEL]:
            for label_match in pattern.finditer(line):
                if line[label_match.end():label_match.end() + 1].isdigit():
                    scan_line = scan_line[:label_match.start()] + " " * (label_match.end() - label_match.start()) + scan_line[label_match.end():]
        previous_end = 0
        for match in _PATTERN.finditer(scan_line):
            context = line[previous_end:match.start()]
            previous_end = match.end()
            label = _label_for(context)
            if label == "identifier":
                continue
            parts = match.groupdict()
            candidates = []
            if parts["y"]:
                triples = [(int(parts["y"]), int(parts["m"]), int(parts["d"]))]
            elif parts["compact"]:
                value = parts["compact"]
                triples = []
                if 1900 <= int(value[:4]) <= 2199:
                    triples.append((int(value[:4]), int(value[4:6]), int(value[6:])))
                if 1900 <= int(value[4:]) <= 2199:
                    if date_order in {"auto", "dmy"}:
                        triples.append((int(value[4:]), int(value[2:4]), int(value[:2])))
                    if date_order in {"auto", "mdy"}:
                        triples.append((int(value[4:]), int(value[:2]), int(value[2:4])))
            elif parts["month"]:
                triples = [(int(parts["english_year"]), _MONTHS[parts["month"][:3].upper()], int(parts["day"]))]
            elif parts["first_month"]:
                triples = [(int(parts["first_year"]), _MONTHS[parts["first_month"][:3].upper()], int(parts["first_day"]))]
            else:
                year, a, b = int(parts["year"]), int(parts["a"]), int(parts["b"])
                triples = []
                if date_order in {"auto", "dmy"}:
                    triples.append((year, b, a))
                if date_order in {"auto", "mdy"}:
                    triples.append((year, a, b))
            for triple in triples:
                try:
                    value = date(*triple).isoformat()
                except ValueError:
                    continue
                if value not in candidates:
                    candidates.append(value)
            status = "valid" if len(candidates) == 1 else "ambiguous" if candidates else "invalid"
            value = candidates[0] if status == "valid" else ""
            dates.append({
                "raw": line[match.start():match.end()], "value": value, "candidates": candidates,
                "status": status, "field": label, "line_index": line_index,
            })
            if value:
                replacements.append((match.start(), match.end(), value))
        for start, end, value in reversed(replacements):
            line = line[:start] + value + line[end:]
        normalized_lines.append(line)
    return {"dates": dates, "normalized_text": "\n".join(normalized_lines)}


def _label_for(context):
    labels = []
    for field, pattern in {**_LABELS, "identifier": _IDENTIFIER_LABEL}.items():
        for match in pattern.finditer(context):
            if _LABEL_SUFFIX.fullmatch(context[match.end():]):
                labels.append((match.end(), field))
    if not labels:
        return ""
    closest = max(end for end, _ in labels)
    fields = {field for end, field in labels if end == closest}
    return fields.pop() if len(fields) == 1 else ""
