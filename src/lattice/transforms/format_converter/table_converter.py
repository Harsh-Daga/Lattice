"""Markdown table detection/conversion and CSV/TSV helpers."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from typing import Any


def detect_markdown_table(text: str) -> list[list[str]] | None:
    """Detect a Markdown table and return parsed rows.

    Supports standard GFM tables:
    | Header 1 | Header 2 |
    |----------|----------|
    | Cell 1   | Cell 2   |
    """
    lines = text.splitlines()
    rows: list[list[str]] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            rows.append(cells)
            in_table = True
        elif in_table and stripped:
            break

    if len(rows) >= 2:
        return rows
    return None


def markdown_to_csv(
    rows: list[list[str]],
    *,
    tsv_threshold_cols: int,
    tsv_threshold_field_len: int,
) -> str | None:
    """Convert parsed Markdown table rows to CSV."""
    if not rows:
        return None

    max_cols = max(len(r) for r in rows)
    max_field_len = max(len(c) for r in rows for c in r)
    use_tsv = max_cols > tsv_threshold_cols or max_field_len > tsv_threshold_field_len

    output = io.StringIO()
    delimiter = "\t" if use_tsv else ","
    writer = csv.writer(output, lineterminator="\n", delimiter=delimiter)

    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        writer.writerow(padded)

    return output.getvalue()


def check_tabularity(
    rows: list[dict[str, Any]],
    *,
    min_tabular_rows: int,
    key_uniformity_threshold: float,
) -> bool:
    """Return True if a list of dicts qualifies as tabular."""
    if len(rows) < min_tabular_rows:
        return False

    key_sets = [set(row.keys()) for row in rows]
    if not key_sets:
        return False

    counter = Counter(frozenset(ks) for ks in key_sets)

    most_common_count = counter.most_common(1)[0][1]
    if most_common_count / len(rows) >= key_uniformity_threshold:
        return True

    best_coverage = 0
    for schema in counter:
        schema_set = set(schema)
        coverage = sum(1 for ks in key_sets if ks <= schema_set)
        if coverage > best_coverage:
            best_coverage = coverage

    return best_coverage / len(rows) >= key_uniformity_threshold


def to_csv(
    rows: list[dict[str, Any]],
    *,
    tsv_threshold_cols: int,
    tsv_threshold_field_len: int,
) -> str | None:
    """Convert list of dicts to RFC 4180 CSV or TSV."""
    if not rows:
        return None

    key_sets = [tuple(sorted(row.keys())) for row in rows]
    header = list(Counter(key_sets).most_common(1)[0][0])

    max_field_len = max(
        (len(str(v)) for row in rows for v in row.values()),
        default=0,
    )
    use_tsv = len(header) > tsv_threshold_cols or max_field_len > tsv_threshold_field_len

    output = io.StringIO()
    delimiter = "\t" if use_tsv else ","
    writer = csv.writer(output, lineterminator="\n", delimiter=delimiter)

    writer.writerow(header)

    for row in rows:
        values = [serialize_csv_value(row.get(key, "")) for key in header]
        writer.writerow(values)

    return output.getvalue()


def serialize_csv_value(value: Any) -> str:
    """Serialize a value for CSV."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def from_csv(text: str) -> list[dict[str, Any]] | None:
    """Parse CSV/TSV text back into list of dicts."""
    first_line = text.splitlines()[0] if text else ""
    delimiter = "\t" if "\t" in first_line else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: list[dict[str, Any]] = []
    for row in reader:
        parsed_row: dict[str, Any] = {}
        for key, val in row.items():
            if key is None:
                continue
            parsed_row[key] = parse_csv_value(val)
        rows.append(parsed_row)
    return rows


def parse_csv_value(value: str) -> Any:
    """Best-effort type parsing for CSV values."""
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    first_char = value[0]
    if first_char == "-" or first_char.isdigit() or first_char in ("[", "{"):
        pass
    else:
        return value

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if first_char in ("[", "{"):
        try:
            return json.loads(value)
        except Exception:
            pass
    return value
