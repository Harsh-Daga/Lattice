"""JSON shape detection, YAML conversion, and embedded JSON extraction."""

from __future__ import annotations

import enum
import json
import re
from typing import Any, cast

import structlog

from lattice.transforms.format_converter.table_converter import check_tabularity, to_csv

logger = structlog.get_logger()


class DataShape(enum.Enum):
    """Classification of parsed JSON data."""

    TABULAR = "tabular"  # list[dict] with mostly identical keys
    CONFIG = "config"  # dict with nested dicts
    IRREGULAR = "irregular"  # list[dict] with mixed/inconsistent keys
    PRIMITIVE = "primitive"  # not a dict or list
    ARRAY_PRIMITIVE = "array_primitive"  # list[str|int|...] (not list[dict])


def detect_shape(
    data: Any,
    *,
    min_tabular_rows: int,
    key_uniformity_threshold: float,
    max_nesting_depth: int,
) -> DataShape:
    """Determine the shape of parsed JSON data."""
    if not isinstance(data, list):
        if isinstance(data, dict):
            return (
                DataShape.CONFIG
                if is_nested_config(data, max_nesting_depth=max_nesting_depth)
                else DataShape.IRREGULAR
            )
        return DataShape.PRIMITIVE

    if len(data) == 0:
        return DataShape.PRIMITIVE

    if all(isinstance(item, dict) for item in data):
        tabular = check_tabularity(
            data,
            min_tabular_rows=min_tabular_rows,
            key_uniformity_threshold=key_uniformity_threshold,
        )
        return DataShape.TABULAR if tabular else DataShape.IRREGULAR

    return DataShape.ARRAY_PRIMITIVE


def is_nested_config(data: dict[str, Any], *, max_nesting_depth: int, depth: int = 0) -> bool:
    """Check if a dictionary represents a config with nested structure."""
    if depth >= max_nesting_depth:
        return False

    nested_count = 0
    for value in data.values():
        if isinstance(value, dict):
            nested_count += 1
            if depth > 0:
                return True
            if nested_count >= 2:
                return True
            if is_nested_config(value, max_nesting_depth=max_nesting_depth, depth=depth + 1):
                return True
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            return False

    if depth == 0:
        return nested_count >= 1 and len(data) >= 1

    return False


def try_convert_direct(
    text: str,
    *,
    min_tabular_rows: int,
    key_uniformity_threshold: float,
    max_nesting_depth: int,
    tsv_threshold_cols: int,
    tsv_threshold_field_len: int,
    log: structlog.stdlib.BoundLogger,
) -> str | None:
    """Try converting text that starts with JSON."""
    if not text or text[0] not in ("[", "{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if parsed is not None:
        shape = detect_shape(
            parsed,
            min_tabular_rows=min_tabular_rows,
            key_uniformity_threshold=key_uniformity_threshold,
            max_nesting_depth=max_nesting_depth,
        )

        if shape == DataShape.TABULAR:
            return to_csv(
                parsed,
                tsv_threshold_cols=tsv_threshold_cols,
                tsv_threshold_field_len=tsv_threshold_field_len,
            )

        if shape == DataShape.CONFIG:
            return to_yaml(parsed, log=log)

        if isinstance(parsed, dict) and len(parsed) == 1:
            sole_value = next(iter(parsed.values()))
            if isinstance(sole_value, list) and len(sole_value) >= min_tabular_rows:
                inner_shape = detect_shape(
                    sole_value,
                    min_tabular_rows=min_tabular_rows,
                    key_uniformity_threshold=key_uniformity_threshold,
                    max_nesting_depth=max_nesting_depth,
                )
                if inner_shape == DataShape.TABULAR:
                    return to_csv(
                        sole_value,
                        tsv_threshold_cols=tsv_threshold_cols,
                        tsv_threshold_field_len=tsv_threshold_field_len,
                    )

    return None


def try_extract_and_convert(
    text: str,
    *,
    try_convert_direct_fn: Any,
) -> str | None:
    """Scan for embedded JSON blocks and convert them."""
    pattern = r"(?:^|\n)\s*(\{[\s\S]*?\}(?:\s*\n|$)|\[[\s\S]*?\](?:\s*\n|$))"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    result_parts: list[str] = []
    last_end = 0
    any_converted = False

    for match in matches:
        start, end = match.span()
        result_parts.append(text[last_end:start])
        json_text = match.group(1).strip()
        if json_text.startswith("|"):
            result_parts.append(match.group(0))
            last_end = end
            continue
        converted = try_convert_direct_fn(json_text)
        if converted and len(converted) < len(json_text):
            result_parts.append(converted)
            any_converted = True
        else:
            result_parts.append(match.group(0))
        last_end = end

    if not any_converted:
        return None

    result_parts.append(text[last_end:])
    return "".join(result_parts)


def to_yaml(data: dict[str, Any], *, log: structlog.stdlib.BoundLogger) -> str | None:
    """Convert nested dict to YAML."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        log.warning("yaml_not_installed", skipping_conversion=True)
        return None

    try:
        result = yaml.safe_dump(
            data,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
        )
        return cast("str | None", result)
    except Exception:
        return None


def from_yaml(text: str) -> Any | None:
    """Parse YAML text back to native Python."""
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return None


def deep_equal(a: Any, b: Any) -> bool:
    """Deep equality check for arbitrary structures."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, float):
        return bool(abs(a - b) < 1e-9)
    return bool(a == b)
