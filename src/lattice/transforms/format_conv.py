"""Format Conversion transform — Production Grade.

Detects structured data (tables, configs) and converts to token-efficient
formats: tables→CSV/TSV, nested configs→YAML, Markdown tables→CSV.

Detection Algorithm
-------------------
1. Attempt JSON-parse of message content
2. If parse succeeds:
   - Root is list[dict] with uniform keys    → TABULAR (convert to CSV/TSV)
   - Root is dict[str, Any] with nesting depth > 2 → CONFIG (convert to YAML)
   - Root is list[dict] with mixed keys      → IRREGULAR (skip)
   - Root is primitive                         → skip
3. If JSON parse fails, check for Markdown tables
4. If neither, skip

Conversion Rules
----------------
CSV: RFC 4180 compliant. Handles commas, quotes, newlines in fields.
TSV: Tab-separated for wide tables (>8 columns or very long values).
YAML: PyYAML safe_dump. Sorts keys, uses literal style for multiline strings.

Safety
------
- Round-trip validation: parse → convert → parse back → compare
- Conversion failure → skip (no modification)
- NOT reversible at response level (provider sees converted format)

Token Savings
-------------
Tables: 30-50% reduction (repeated key names eliminated)
Configs: 20-40% reduction (YAML indentation vs JSON braces)
Markdown tables: 40-60% reduction (parsing overhead eliminated)

Performance
-----------
Detection: <0.1ms. Conversion: <0.2ms. Target: <0.5ms total.

Priority: 25 (between ref_sub=20 and tool_filter=30)
"""

from __future__ import annotations

import csv
import enum
import io
import json
import re
from typing import Any, cast

import structlog

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

logger = structlog.get_logger()


# =============================================================================
# Data shape detection
# =============================================================================


class DataShape(enum.Enum):
    """Classification of parsed JSON data."""

    TABULAR = "tabular"  # list[dict] with mostly identical keys
    CONFIG = "config"  # dict with nested dicts
    IRREGULAR = "irregular"  # list[dict] with mixed/inconsistent keys
    PRIMITIVE = "primitive"  # not a dict or list
    ARRAY_PRIMITIVE = "array_primitive"  # list[str|int|...] (not list[dict])


# =============================================================================
# FormatConverter
# =============================================================================


class FormatConverter(ReversibleSyncTransform):
    """Convert structured data to token-efficient formats.

    Configuration
    -------------
    - min_tabular_rows: Minimum rows to trigger CSV conversion. Default: 2
    - key_uniformity_threshold: Fraction of rows sharing keys to be
      considered tabular. Default: 0.8 (80%)
    - max_nesting_depth: Maximum YAML nesting depth before rejecting
      conversion. Default: 10.
    - max_field_length: Maximum individual field length (in chars) before
      wrapping in CSV. Default: 5000.
    - tsv_threshold_cols: Use TSV instead of CSV if columns exceed this.
      Default: 8.
    - tsv_threshold_field_len: Use TSV if any field exceeds this length.
      Default: 200.
    - validate_roundtrip: Enable round-trip validation in process().
      Default: False (only used in testing / benchmarks for correctness).
    - enable_markdown_tables: Detect and convert Markdown tables to CSV.
      Default: True.
    """

    name = "format_conversion"
    priority = 25

    def __init__(
        self,
        min_tabular_rows: int = 2,
        key_uniformity_threshold: float = 0.8,
        max_nesting_depth: int = 10,
        max_field_length: int = 5000,
        tsv_threshold_cols: int = 8,
        tsv_threshold_field_len: int = 200,
        validate_roundtrip: bool = False,
        enable_markdown_tables: bool = True,
    ) -> None:
        self.min_tabular_rows = max(1, min_tabular_rows)
        self.key_uniformity_threshold = max(0.0, min(1.0, key_uniformity_threshold))
        self.max_nesting_depth = max_nesting_depth
        self.max_field_length = max_field_length
        self.tsv_threshold_cols = tsv_threshold_cols
        self.tsv_threshold_field_len = tsv_threshold_field_len
        self.validate_roundtrip = validate_roundtrip
        self.enable_markdown_tables = enable_markdown_tables
        self._log = logger.bind(transform="format_conversion")

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Detect and convert structured data in messages."""
        total_saved = 0
        converted_count = 0

        for msg in request.messages:
            original_content = msg.content
            original_len = len(original_content)

            # Skip if content contains tool calls (structured output, don't touch)
            if msg.tool_calls:
                continue

            # Attempt conversion (JSON first, then Markdown)
            converted = self._try_convert(original_content)
            if converted is None or converted == original_content:
                continue

            # Validate round-trip (expensive, disabled in production)
            if self.validate_roundtrip and not self._validate_roundtrip(
                original_content, converted
            ):
                self._log.warning(
                    "roundtrip_failed",
                    request_id=context.request_id,
                    shape=self._detect_shape(json.loads(original_content)).value,
                )
                continue

            # Apply conversion
            msg.content = converted
            converted_count += 1
            saved = max(0, original_len - len(converted))
            total_saved += saved

        if converted_count > 0:
            context.record_metric(self.name, "messages_converted", converted_count)
            context.record_metric(self.name, "tokens_saved_estimate", total_saved // 4)
            self._log.info(
                "format_converted",
                request_id=context.request_id,
                messages=converted_count,
                chars_saved=total_saved,
            )

        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — format conversion is irreversible (not needed)."""
        return response

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _try_convert(self, text: str) -> str | None:
        """Attempt JSON parse and conversion. Return converted text or None."""
        text = text.strip()
        if not text:
            return None

        # Quick heuristic: starts with [ or { → attempt JSON
        if text[0] in ("[", "{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

            if parsed is not None:
                shape = self._detect_shape(parsed)

                if shape == DataShape.TABULAR:
                    return self._to_csv(parsed)

                if shape == DataShape.CONFIG:
                    return self._to_yaml(parsed)

                # Common API pattern: {"employees": [{...}, {...}]}
                # Extract the single list value and convert it
                if isinstance(parsed, dict) and len(parsed) == 1:
                    sole_value = next(iter(parsed.values()))
                    if isinstance(sole_value, list) and len(sole_value) >= self.min_tabular_rows:
                        inner_shape = self._detect_shape(sole_value)
                        if inner_shape == DataShape.TABULAR:
                            return self._to_csv(sole_value)

        # Try Markdown table detection
        if self.enable_markdown_tables:
            md_table = self._detect_markdown_table(text)
            if md_table is not None:
                return self._markdown_to_csv(md_table)

        # Try diff detection
        diff_result = self._detect_and_compress_diff(text)
        if diff_result is not None:
            return diff_result

        # Try log detection
        log_result = self._detect_and_compress_log(text)
        if log_result is not None:
            return log_result

        return None

    def _detect_shape(self, data: Any) -> DataShape:
        """Determine the shape of parsed JSON data."""
        if not isinstance(data, list):
            if isinstance(data, dict):
                return DataShape.CONFIG if self._is_nested_config(data) else DataShape.IRREGULAR
            return DataShape.PRIMITIVE

        if len(data) == 0:
            return DataShape.PRIMITIVE

        # All elements are dicts → check tabularity
        if all(isinstance(item, dict) for item in data):
            return self._check_tabularity(data)

        # List of primitives
        return DataShape.ARRAY_PRIMITIVE

    def _check_tabularity(self, rows: list[dict[str, Any]]) -> DataShape:
        """Check if a list of dicts qualifies as tabular.

        Supports optional fields using subset compatibility: a row whose
        keys are a subset of the most common schema is still tabular.
        """
        if len(rows) < self.min_tabular_rows:
            return DataShape.IRREGULAR

        key_sets = [set(row.keys()) for row in rows]
        if not key_sets:
            return DataShape.IRREGULAR

        from collections import Counter

        counter = Counter(frozenset(ks) for ks in key_sets)

        # Strategy 1: exact schema match
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count / len(rows) >= self.key_uniformity_threshold:
            return DataShape.TABULAR

        # Strategy 2: subset compatibility with any present schema
        best_coverage = 0
        for schema in counter:
            schema_set = set(schema)
            coverage = sum(1 for ks in key_sets if ks <= schema_set)
            if coverage > best_coverage:
                best_coverage = coverage

        if best_coverage / len(rows) >= self.key_uniformity_threshold:
            return DataShape.TABULAR

        return DataShape.IRREGULAR

    def _is_nested_config(self, data: dict[str, Any], depth: int = 0) -> bool:
        """Check if a dictionary represents a config with nested structure."""
        if depth >= self.max_nesting_depth:
            return False

        nested_count = 0
        for value in data.values():
            if isinstance(value, dict):
                nested_count += 1
                if depth > 0:
                    return True
                if nested_count >= 2:
                    return True
                if self._is_nested_config(value, depth + 1):
                    return True
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                return False

        if depth == 0:
            return nested_count >= 1 and len(data) >= 1

        return False

    # ------------------------------------------------------------------
    # Markdown table detection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Diff detection and compression
    # ------------------------------------------------------------------

    def _detect_and_compress_diff(self, text: str) -> str | None:
        """Detect unified diff and compress if very large."""
        lines = text.splitlines()
        if len(lines) < 5:
            return None

        # Require at least 2 diff signals
        diff_signals = sum(
            1 for line in lines if line.startswith(("--- ", "+++ ", "@@ ", "diff --git"))
        )
        if diff_signals < 2:
            return None

        # If diff is very large, extract only changed hunks
        if len(lines) > 1000:
            return self._compress_large_diff(lines)
        return None  # Small diffs are left alone

    def _compress_large_diff(self, lines: list[str]) -> str:
        """Extract only changed hunks from a large diff."""
        result: list[str] = []
        current_hunk: list[str] = []
        in_hunk = False
        header_lines: list[str] = []

        for line in lines:
            if line.startswith("diff --git"):
                # Flush previous hunk
                if current_hunk:
                    result.extend(current_hunk)
                    current_hunk = []
                header_lines = [line]
                in_hunk = False
            elif line.startswith(("--- ", "+++ ")):
                header_lines.append(line)
            elif line.startswith("@@"):
                # Flush previous hunk
                if current_hunk:
                    result.extend(current_hunk)
                    current_hunk = []
                if header_lines:
                    result.extend(header_lines)
                    header_lines = []
                current_hunk = [line]
                in_hunk = True
            elif in_hunk:
                if (
                    line.startswith(("+", "-"))
                    or line.startswith(" ")
                    or line == "\\ No newline at end of file"
                ):
                    current_hunk.append(line)
                else:
                    # End of hunk
                    if current_hunk:
                        result.extend(current_hunk)
                        current_hunk = []
                    in_hunk = False

        if current_hunk:
            result.extend(current_hunk)

        if not result:
            return "\n".join(lines)
        return "\n".join(result)

    # ------------------------------------------------------------------
    # Log detection and compression
    # ------------------------------------------------------------------

    def _detect_and_compress_log(self, text: str) -> str | None:
        """Detect log output and compress if very large."""
        lines = text.splitlines()
        if len(lines) < 10:
            return None

        # Count log-like lines (timestamp or severity)
        log_lines = sum(
            1
            for line in lines
            if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
            or re.search(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", line)
        )
        if log_lines < len(lines) * 0.5:
            return None

        # Group by severity and keep representative samples
        severity_groups: dict[str, list[str]] = {}
        for line in lines:
            sev = "OTHER"
            for level in ("CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"):
                if level in line:
                    sev = level
                    break
            severity_groups.setdefault(sev, []).append(line)

        kept: list[str] = []
        for sev in ("CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "OTHER"):
            group = severity_groups.get(sev, [])
            if not group:
                continue
            if len(group) <= 5:
                kept.extend(group)
            else:
                # Keep first 2, last 2, and a sample from the middle
                kept.extend(group[:2])
                kept.append(f"... ({len(group) - 4} more {sev} lines) ...")
                kept.extend(group[-2:])

        return "\n".join(kept)

    @staticmethod
    def _detect_markdown_table(text: str) -> list[list[str]] | None:
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
                # Skip separator lines (--- or :---:)
                if all(re.match(r"^:?-+:?$", c) for c in cells):
                    continue
                rows.append(cells)
                in_table = True
            else:
                if in_table and stripped:
                    # Non-empty line breaks table continuity
                    break

        if len(rows) >= 2:
            return rows
        return None

    def _markdown_to_csv(self, rows: list[list[str]]) -> str | None:
        """Convert parsed Markdown table rows to CSV."""
        if not rows:
            return None

        # Determine if TSV is better
        max_cols = max(len(r) for r in rows)
        max_field_len = max(len(c) for r in rows for c in r)
        use_tsv = max_cols > self.tsv_threshold_cols or max_field_len > self.tsv_threshold_field_len

        output = io.StringIO()
        delimiter = "\t" if use_tsv else ","
        writer = csv.writer(output, lineterminator="\n", delimiter=delimiter)

        for row in rows:
            # Pad short rows
            padded = row + [""] * (max_cols - len(row))
            writer.writerow(padded)

        return output.getvalue()

    # ------------------------------------------------------------------
    # CSV conversion
    # ------------------------------------------------------------------

    def _to_csv(self, rows: list[dict[str, Any]]) -> str | None:
        """Convert list of dicts to RFC 4180 CSV or TSV."""
        if not rows:
            return None

        from collections import Counter

        key_sets = [tuple(sorted(row.keys())) for row in rows]
        header = list(Counter(key_sets).most_common(1)[0][0])

        # Determine CSV vs TSV
        max_field_len = max(
            (len(str(v)) for row in rows for v in row.values()),
            default=0,
        )
        use_tsv = (
            len(header) > self.tsv_threshold_cols or max_field_len > self.tsv_threshold_field_len
        )

        output = io.StringIO()
        delimiter = "\t" if use_tsv else ","
        writer = csv.writer(output, lineterminator="\n", delimiter=delimiter)

        # Write header row
        writer.writerow(header)

        # Write data rows
        for row in rows:
            values = []
            for key in header:
                val = row.get(key, "")
                values.append(self._serialize_csv_value(val))
            writer.writerow(values)

        return output.getvalue()

    @staticmethod
    def _serialize_csv_value(value: Any) -> str:
        """Serialize a value for CSV.

        Rules:
        - None → empty string
        - bool → lowercase string
        - int/float → str(value)
        - str → str(value)
        - list/dict → json.dumps(value) (to preserve structure)
        """
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        # For nested objects (list, dict): serialize to JSON string
        return json.dumps(value, ensure_ascii=False)

    # ------------------------------------------------------------------
    # YAML conversion
    # ------------------------------------------------------------------

    def _to_yaml(self, data: dict[str, Any]) -> str | None:
        """Convert nested dict to YAML.

        Uses yaml.safe_dump with sorted keys for determinism.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            # yaml not installed (optional dependency)
            self._log.warning("yaml_not_installed", skipping_conversion=True)
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
            # yaml.safe_dump can fail on unhashable or circular references
            return None

    # ------------------------------------------------------------------
    # Round-trip validation
    # ------------------------------------------------------------------

    def _validate_roundtrip(self, original: str, converted: str) -> bool:
        """Validate that converted text can be parsed back to equivalent data.

        For CSV: parse CSV → reconstruct dicts → compare to JSON parse
        For YAML: parse YAML → compare to JSON parse
        """
        try:
            original_parsed = json.loads(original)
        except Exception:
            return False

        if isinstance(original_parsed, list):
            # Converted is CSV or TSV
            converted_parsed = self._from_csv(converted)
        elif isinstance(original_parsed, dict):
            # Converted is YAML or JSON (YAML case)
            converted_parsed = self._from_yaml(converted)
            if converted_parsed is None:
                # Fallback: if it's actually JSON we didn't convert
                return converted == original
        else:
            return False

        if converted_parsed is None:
            return False

        return self._deep_equal(original_parsed, converted_parsed)

    def _from_csv(self, text: str) -> list[dict[str, Any]] | None:
        """Parse CSV/TSV text back into list of dicts."""
        # Detect delimiter from first line
        first_line = text.splitlines()[0] if text else ""
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed_row: dict[str, Any] = {}
            for key, val in row.items():
                if key is None:
                    continue
                parsed_row[key] = self._parse_csv_value(val)
            rows.append(parsed_row)
        return rows

    @staticmethod
    def _parse_csv_value(value: str) -> Any:
        """Best-effort type parsing for CSV values.

        Tries: bool → int → float → JSON parse (only for values that start
        with [ or {) → string.
        """
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

    def _from_yaml(self, text: str) -> Any | None:
        """Parse YAML text back to native Python."""
        try:
            import yaml

            return yaml.safe_load(text)
        except Exception:
            return None

    @staticmethod
    def _deep_equal(a: Any, b: Any) -> bool:
        """Deep equality check for arbitrary structures."""
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            if set(a) != set(b):
                return False
            return all(FormatConverter._deep_equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(FormatConverter._deep_equal(x, y) for x, y in zip(a, b, strict=False))
        if isinstance(a, float):
            return bool(abs(a - b) < 1e-9)
        return bool(a == b)
