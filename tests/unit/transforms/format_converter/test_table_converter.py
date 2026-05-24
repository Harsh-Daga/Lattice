"""Phase 5 — Markdown table detection and CSV conversion."""

from __future__ import annotations

from lattice.transforms.format_converter.table_converter import (
    detect_markdown_table,
    markdown_to_csv,
)


def test_markdown_table_round_trip_preserves_cells() -> None:
    md = "| name | value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"
    rows = detect_markdown_table(md)
    assert rows is not None
    assert rows[0] == ["name", "value"]
    csv_text = markdown_to_csv(rows, tsv_threshold_cols=20, tsv_threshold_field_len=200)
    assert csv_text is not None
    assert "alpha" in csv_text
    assert "beta" in csv_text
