"""Parse production dates, expiry dates, and batch numbers from package OCR text."""

from datetime import date
import json
import re


MANUFACTURE_AT = "manufacture_at"
EXPIRE_AT = "expire_at"
BATCH = "batch"
VALID_FIELDS = {MANUFACTURE_AT, EXPIRE_AT, BATCH}

DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:[./-]|\s+)\s*(\d{1,2})\s*"
    r"(?:[./-]|\s+)\s*((?:19|20|21)\d{2})(?!\d)"
)
COMPACT_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{2})(\d{2})\s*[./-]\s*((?:19|20|21)\d{2})(?!\d)"
)
TIME_PATTERN = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3])\s*[:.]\s*[0-5]\d(?::[0-5]\d)?(?!\d)")
LEADING_TIME_PATTERN = re.compile(
    r"^\s*(?:[01]?\d|2[0-3])\s*[:.]\s*[0-5]\d(?::[0-5]\d)?\s+"
)
INCOMPLETE_LEADING_TIME_PATTERN = re.compile(r"^\s*:\s*[0-5]\d\s+")
MANUFACTURE_LABEL_PATTERN = re.compile(
    r"\b(?:MFG|MFD|MAN|MANUFACTURE|MANUFACTURING|PROD|PRODUCED|PRODUCTION)\b",
    re.IGNORECASE,
)
EXPIRE_LABEL_PATTERN = re.compile(
    r"\b(?:EXP|EXPIRY|EXPIRATION|BBE|USE\s+BY|BEST\s+BEFORE)\b",
    re.IGNORECASE,
)
BATCH_LABEL_PATTERN = re.compile(
    r"\b(?:LOT|BATCH)(?:\s*(?:NO|NUMBER|CODE))?\s*[:#-]?\s*",
    re.IGNORECASE,
)
IGNORED_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:DATE|DD|MM|YYYY|DLD|REG|TRADEMARK|WEBSITE|WWW|BEST|BEFORE|"
    r"MANUFACTURING|EXPIRY|EXPIRATION)\b",
    re.IGNORECASE,
)
REMOVABLE_LABEL_PATTERN = re.compile(
    r"\b(?:MFG|MFD|MAN|MANUFACTURE|MANUFACTURING|PROD|PRODUCED|PRODUCTION|"
    r"EXP|EXPIRY|EXPIRATION|BBE|USE\s+BY|BEST\s+BEFORE)\b",
    re.IGNORECASE,
)
ALLOWED_BATCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:\- ]*[A-Za-z0-9]$|^[A-Za-z0-9]$")
TRAILING_YEAR_PATTERN = re.compile(r".*\b(?:19|20|21)\d{2}$")
CONFUSION_GROUPS = (
    frozenset(("O", "0")),
    frozenset(("I", "1")),
    frozenset(("1", "7")),
    frozenset(("B", "8")),
    frozenset(("S", "5")),
    frozenset(("E", "5")),
    frozenset(("Z", "2")),
    frozenset(("G", "6")),
)


def parse_required_fields(value):
    """Accept a JSON array or a delimiter-separated field list."""
    if isinstance(value, (list, tuple, set)):
        parsed = list(value)
    else:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("required_fields is required")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = re.split(r"[\s,;]+", raw)
    if not isinstance(parsed, list):
        raise ValueError("required_fields must be an array")
    fields = []
    for item in parsed:
        field = str(item).strip()
        if field not in VALID_FIELDS:
            raise ValueError(f"Unsupported required field: {field}")
        if field not in fields:
            fields.append(field)
    if not fields:
        raise ValueError("required_fields is required")
    return fields


def parse_batch_candidates(value):
    """Accept a JSON array or a delimiter-separated candidate list."""
    if isinstance(value, (list, tuple, set)):
        parsed = list(value)
    else:
        raw = str(value or "").strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = re.split(r"[\s,，;；]+", raw)
    if not isinstance(parsed, list):
        raise ValueError("batch_candidates must be an array")
    unique = {}
    for item in parsed:
        candidate = str(item).strip()
        normalized = normalize_batch(candidate)
        if not normalized:
            continue
        if len(normalized) > 64:
            raise ValueError("A batch candidate cannot exceed 64 characters")
        unique.setdefault(normalized, candidate)
    if len(unique) > 50:
        raise ValueError("At most 50 batch candidates are allowed")
    return list(unique.values())


def parse_package_text(raw_text, required_fields):
    normalized_raw = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    fields = set(required_fields)
    if not normalized_raw:
        return _empty_result(normalized_raw)

    lines = normalized_raw.splitlines()
    date_candidates = []
    for line_index, line in enumerate(lines):
        for pattern in (DATE_PATTERN, COMPACT_DATE_PATTERN):
            for match in pattern.finditer(line):
                value = _parse_date_match(match)
                if value:
                    date_candidates.append({
                        "value": value,
                        "ordinal": int(value.replace("-", "")),
                        "line_index": line_index,
                        "label": _date_label(line, match.span()),
                    })

    manufacture = _distinct_candidates(date_candidates, "manufacture")
    expire = _distinct_candidates(date_candidates, "expire")
    unknown = _distinct_candidates(date_candidates, "unknown")
    manufacture_conflict = len(manufacture) > 1
    expire_conflict = len(expire) > 1
    manufacture_at = manufacture[0] if len(manufacture) == 1 else None
    expire_at = expire[0] if len(expire) == 1 else None
    unknown = [
        candidate for candidate in unknown
        if candidate["value"] not in {
            manufacture_at and manufacture_at["value"],
            expire_at and expire_at["value"],
        }
    ]

    if not manufacture_conflict and not expire_conflict:
        if manufacture_at is None and expire_at is None and len(unknown) == 1:
            if MANUFACTURE_AT in fields and EXPIRE_AT not in fields:
                manufacture_at = unknown[0]
            else:
                expire_at = unknown[0]
        elif manufacture_at is None and expire_at is None and len(unknown) == 2:
            ordered = sorted(unknown, key=lambda item: item["ordinal"])
            if ordered[0]["ordinal"] < ordered[1]["ordinal"]:
                manufacture_at, expire_at = ordered
        elif manufacture_at is not None and expire_at is None and len(unknown) == 1:
            if unknown[0]["ordinal"] > manufacture_at["ordinal"]:
                expire_at = unknown[0]
        elif manufacture_at is None and expire_at is not None and len(unknown) == 1:
            if unknown[0]["ordinal"] < expire_at["ordinal"]:
                manufacture_at = unknown[0]

    if (
        manufacture_at is not None
        and expire_at is not None
        and manufacture_at["ordinal"] >= expire_at["ordinal"]
    ):
        manufacture_at = None
        expire_at = None

    return {
        MANUFACTURE_AT: manufacture_at and manufacture_at["value"],
        EXPIRE_AT: expire_at and expire_at["value"],
        BATCH: _parse_batch(lines, date_candidates),
        "raw_text": normalized_raw,
    }


def reconcile_batch_candidate(batch, candidates):
    normalized = normalize_batch(batch)
    if not normalized or not candidates:
        return batch, [], None
    scored = sorted(
        (_edit_distance(normalized, normalize_batch(candidate)), str(candidate).strip())
        for candidate in candidates
        if normalize_batch(candidate)
    )
    if not scored:
        return batch, [], None
    threshold = max(1, int(len(normalized) * 0.15))
    unique_best = len(scored) == 1 or scored[0][0] < scored[1][0]
    if not unique_best or scored[0][0] > threshold:
        return batch, [], None
    selected = scored[0][1]
    selected_normalized = normalize_batch(selected)
    normalized_positions = differing_positions(normalized, selected_normalized)
    selected_character_positions = [
        index for index, character in enumerate(selected) if character.isalnum()
    ]
    positions = [
        selected_character_positions[index]
        for index in normalized_positions
        if index < len(selected_character_positions)
    ]
    return selected, positions, "asn_batch_candidate" if positions else None


def confusion_positions(selected, alternatives):
    batch = str(selected or "")
    positions = set()
    for alternative in alternatives:
        candidate = str(alternative or "")
        if len(candidate) != len(batch):
            continue
        for index, (left, right) in enumerate(zip(batch.upper(), candidate.upper())):
            if left == right:
                continue
            if any(left in group and right in group for group in CONFUSION_GROUPS):
                positions.add(index)
    return sorted(positions)


def differing_positions(left, right):
    if len(left) != len(right):
        return []
    return [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]


def normalize_batch(value):
    return re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()


def _parse_batch(lines, date_candidates):
    explicit = []
    for line in lines:
        label = BATCH_LABEL_PATTERN.search(line)
        if not label:
            continue
        candidate = _remove_dates(line[label.end():])
        candidate = LEADING_TIME_PATTERN.sub("", candidate)
        value = _sanitize_batch(candidate, _contains_date(line), True, True)
        if value and value not in explicit:
            explicit.append(value)
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        return None

    date_lines = {candidate["line_index"] for candidate in date_candidates}
    candidates = []
    for index, original in enumerate(lines):
        if IGNORED_DESCRIPTION_PATTERN.search(original):
            continue
        if MANUFACTURE_LABEL_PATTERN.search(original) or EXPIRE_LABEL_PATTERN.search(original):
            continue
        cleaned = _remove_dates(original)
        cleaned = REMOVABLE_LABEL_PATTERN.sub(" ", cleaned)
        cleaned = LEADING_TIME_PATTERN.sub("", cleaned)
        near_date = index in date_lines or index - 1 in date_lines or index + 1 in date_lines
        value = _sanitize_batch(cleaned, _contains_date(original), False, near_date)
        if not value:
            continue
        compact = "".join(character for character in value if character.isalnum())
        mixed = any(character.isalpha() for character in compact) and any(
            character.isdigit() for character in compact
        )
        structured = _is_structured_numeric_batch(value)
        score = (
            (100 if near_date else 0)
            + (100 if structured else 0)
            + (50 if mixed else 0)
            + (20 if " " not in value else 0)
            + min(len(compact), 40)
        )
        if value not in {candidate["value"] for candidate in candidates}:
            candidates.append({"value": value, "score": score})
    if not candidates:
        return None
    best_score = max(candidate["score"] for candidate in candidates)
    best = [candidate["value"] for candidate in candidates if candidate["score"] == best_score]
    return best[0] if len(best) == 1 else None


def _sanitize_batch(value, line_contains_date, explicit_label, allow_structured_numeric):
    normalized = re.sub(r"\s+", " ", str(value or "").strip(" \t;,.#-()[]"))
    if not normalized or not ALLOWED_BATCH_PATTERN.fullmatch(normalized):
        return None
    if INCOMPLETE_LEADING_TIME_PATTERN.search(normalized):
        return None
    if DATE_PATTERN.fullmatch(normalized) or TIME_PATTERN.fullmatch(normalized):
        return None
    if TRAILING_YEAR_PATTERN.fullmatch(normalized):
        return None
    compact = "".join(character for character in normalized if character.isalnum())
    if len(compact) < 5 or len(compact) > 48 or not any(character.isdigit() for character in compact):
        return None
    if " " in normalized and not line_contains_date:
        tokens = normalized.split(" ")
        strong = any(sum(character.isalnum() for character in token) >= 5 for token in tokens)
        alpha_numeric = any(
            any(character.isalpha() for character in token)
            and any(character.isdigit() for character in token)
            for token in tokens
        )
        structured = allow_structured_numeric and _is_structured_numeric_batch(normalized)
        if not strong and not alpha_numeric and not structured and not explicit_label:
            return None
    return normalized


def _is_structured_numeric_batch(value):
    if not TIME_PATTERN.search(value) or _contains_date(value):
        return False
    if any(not (character.isdigit() or character.isspace() or character == ":") for character in value):
        return False
    groups = [group for group in re.split(r"\s+", value) if group]
    return len(groups) >= 5 and sum(character.isdigit() for character in value) >= 12


def _remove_dates(value):
    return COMPACT_DATE_PATTERN.sub(" ", DATE_PATTERN.sub(" ", value))


def _contains_date(value):
    return bool(DATE_PATTERN.search(value) or COMPACT_DATE_PATTERN.search(value))


def _parse_date_match(match):
    day, month, year = (int(group) for group in match.groups())
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    return parsed.isoformat()


def _date_label(line, date_span):
    manufacture_distance = _nearest_label_distance(MANUFACTURE_LABEL_PATTERN, line, date_span)
    expire_distance = _nearest_label_distance(EXPIRE_LABEL_PATTERN, line, date_span)
    if manufacture_distance is None and expire_distance is None:
        return "unknown"
    if expire_distance is None or (
        manufacture_distance is not None and manufacture_distance < expire_distance
    ):
        return "manufacture"
    if manufacture_distance is None or expire_distance < manufacture_distance:
        return "expire"
    return "unknown"


def _nearest_label_distance(pattern, line, date_span):
    distances = []
    for match in pattern.finditer(line):
        if match.end() <= date_span[0]:
            distance = date_span[0] - match.end()
        elif match.start() >= date_span[1]:
            distance = match.start() - date_span[1]
        else:
            distance = 0
        if distance <= 48:
            distances.append(distance)
    return min(distances) if distances else None


def _distinct_candidates(candidates, label):
    values = {}
    for candidate in candidates:
        if candidate["label"] == label:
            values.setdefault(candidate["value"], candidate)
    return list(values.values())


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def _empty_result(raw_text):
    return {
        MANUFACTURE_AT: None,
        EXPIRE_AT: None,
        BATCH: None,
        "raw_text": raw_text,
    }
