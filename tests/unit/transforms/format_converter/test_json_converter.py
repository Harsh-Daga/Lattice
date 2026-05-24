"""Phase 5 — JSON shape detection and tabular CSV export."""

from __future__ import annotations

import json

from lattice.transforms.format_converter.json_converter import DataShape, detect_shape, to_csv
from lattice.transforms.format_converter.table_converter import check_tabularity


def test_detect_shape_tabular_list_of_dicts() -> None:
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
    shape = detect_shape(
        data,
        min_tabular_rows=2,
        key_uniformity_threshold=0.8,
        max_nesting_depth=3,
    )
    assert shape == DataShape.TABULAR
    assert check_tabularity(data, min_tabular_rows=2, key_uniformity_threshold=0.8)


def test_json_list_to_csv_round_trip_keys() -> None:
    data = [{"x": "one", "y": "two"}, {"x": "three", "y": "four"}]
    csv_text = to_csv(data, tsv_threshold_cols=20, tsv_threshold_field_len=200)
    assert csv_text is not None
    parsed = list(__import__("csv").DictReader(csv_text.splitlines()))
    assert len(parsed) == 2
    assert set(parsed[0].keys()) == {"x", "y"}
    roundtrip = json.loads(json.dumps(data))
    assert roundtrip[0]["x"] == parsed[0]["x"]
