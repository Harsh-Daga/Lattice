"""Format Conversion transform — Production Grade.

Detects structured data (tables, configs) and converts to token-efficient
formats: tables→CSV/TSV, nested configs→YAML, Markdown tables→CSV.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result
from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2
from lattice.pipeline.base import ReversibleSyncTransform
from lattice.transforms.format_converter.json_converter import (
    DataShape,
    deep_equal,
    detect_shape,
    from_yaml,
    to_yaml,
    try_convert_direct,
    try_extract_and_convert,
)
from lattice.transforms.format_converter.table_converter import (
    check_tabularity,
    detect_markdown_table,
    from_csv,
    markdown_to_csv,
    parse_csv_value,
    serialize_csv_value,
    to_csv,
)
from lattice.transport.types import Request, Response

logger = structlog.get_logger()

__all__ = ["DataShape", "FormatConverter"]


class FormatConverter(ReversibleSyncTransform):
    """Convert structured data to token-efficient formats."""

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

    def optimize(
        self,
        ir: PromptIRV2,
        _request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """IR-native: convert structured data in IR spans to token-efficient formats."""
        total_saved = 0
        converted_count = 0
        new_sections: list[SectionV2] = []

        for sec in ir.sections:
            new_spans: list[SpanV2] = []
            for span in sec.spans:
                original = span.text
                converted: str | None = None

                sec_type = sec.type.lower()
                if sec_type in ("json", "data") or original.strip().startswith(("[", "{")):
                    converted = self._try_convert(original)

                if sec_type == "table" and converted is None:
                    md_table = self._detect_markdown_table(original)
                    if md_table is not None:
                        converted = self._markdown_to_csv(md_table)

                if sec_type in ("log", "logs") and converted is None:
                    converted = self._detect_and_compress_log(original)

                if converted is None and self.enable_markdown_tables:
                    md_table = self._detect_markdown_table(original)
                    if md_table is not None:
                        converted = self._markdown_to_csv(md_table)

                if converted is None:
                    converted = self._try_convert(original)

                if converted is not None and converted != original:
                    new_spans.append(span.with_text(converted))
                    converted_count += 1
                    total_saved += max(0, len(original) - len(converted))
                else:
                    new_spans.append(span)

            new_sections.append(sec.with_spans(tuple(new_spans)))

        if converted_count > 0:
            context.record_metric(self.name, "spans_converted", converted_count)
            context.record_metric(self.name, "tokens_saved_estimate", total_saved // 4)

        return Ok(ir.with_sections(tuple(new_sections)))

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — format conversion is irreversible (not needed)."""
        return response

    def _try_convert(self, text: str) -> str | None:
        """Attempt JSON parse and conversion. Return converted text or None."""
        text = text.strip()
        if not text:
            return None

        direct = self._try_convert_direct(text)
        if direct is not None:
            return direct

        extracted = self._try_extract_and_convert(text)
        if extracted is not None:
            return extracted

        if self.enable_markdown_tables:
            md_table = self._detect_markdown_table(text)
            if md_table is not None:
                return self._markdown_to_csv(md_table)

        diff_result = self._detect_and_compress_diff(text)
        if diff_result is not None:
            return diff_result

        log_result = self._detect_and_compress_log(text)
        if log_result is not None:
            return log_result

        return None

    def _try_convert_direct(self, text: str) -> str | None:
        return try_convert_direct(
            text,
            min_tabular_rows=self.min_tabular_rows,
            key_uniformity_threshold=self.key_uniformity_threshold,
            max_nesting_depth=self.max_nesting_depth,
            tsv_threshold_cols=self.tsv_threshold_cols,
            tsv_threshold_field_len=self.tsv_threshold_field_len,
            log=self._log,
        )

    def _try_extract_and_convert(self, text: str) -> str | None:
        return try_extract_and_convert(text, try_convert_direct_fn=self._try_convert_direct)

    def _detect_shape(self, data: Any) -> DataShape:
        return detect_shape(
            data,
            min_tabular_rows=self.min_tabular_rows,
            key_uniformity_threshold=self.key_uniformity_threshold,
            max_nesting_depth=self.max_nesting_depth,
        )

    def _check_tabularity(self, rows: list[dict[str, Any]]) -> DataShape:
        if check_tabularity(
            rows,
            min_tabular_rows=self.min_tabular_rows,
            key_uniformity_threshold=self.key_uniformity_threshold,
        ):
            return DataShape.TABULAR
        return DataShape.IRREGULAR

    def _detect_and_compress_diff(self, text: str) -> str | None:
        """Detect unified diff and compress if very large."""
        lines = text.splitlines()
        if len(lines) < 5:
            return None

        diff_signals = sum(
            1 for line in lines if line.startswith(("--- ", "+++ ", "@@ ", "diff --git"))
        )
        if diff_signals < 2:
            return None

        if len(lines) > 1000:
            return self._compress_large_diff(lines)
        return None

    def _compress_large_diff(self, lines: list[str]) -> str:
        """Extract only changed hunks from a large diff."""
        result: list[str] = []
        current_hunk: list[str] = []
        in_hunk = False
        header_lines: list[str] = []

        for line in lines:
            if line.startswith("diff --git"):
                if current_hunk:
                    result.extend(current_hunk)
                    current_hunk = []
                header_lines = [line]
                in_hunk = False
            elif line.startswith(("--- ", "+++ ")):
                header_lines.append(line)
            elif line.startswith("@@"):
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
                    if current_hunk:
                        result.extend(current_hunk)
                        current_hunk = []
                    in_hunk = False

        if current_hunk:
            result.extend(current_hunk)

        if not result:
            return "\n".join(lines)
        return "\n".join(result)

    def _detect_and_compress_log(self, text: str) -> str | None:
        """Detect log output and compress if very large."""
        lines = text.splitlines()
        if len(lines) < 10:
            return None

        log_lines = sum(
            1
            for line in lines
            if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
            or re.search(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", line)
        )
        if log_lines < len(lines) * 0.5:
            return None

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
                kept.extend(group[:2])
                kept.append(f"... ({len(group) - 4} more {sev} lines) ...")
                kept.extend(group[-2:])

        return "\n".join(kept)

    @staticmethod
    def _detect_markdown_table(text: str) -> list[list[str]] | None:
        return detect_markdown_table(text)

    def _markdown_to_csv(self, rows: list[list[str]]) -> str | None:
        return markdown_to_csv(
            rows,
            tsv_threshold_cols=self.tsv_threshold_cols,
            tsv_threshold_field_len=self.tsv_threshold_field_len,
        )

    def _to_csv(self, rows: list[dict[str, Any]]) -> str | None:
        return to_csv(
            rows,
            tsv_threshold_cols=self.tsv_threshold_cols,
            tsv_threshold_field_len=self.tsv_threshold_field_len,
        )

    @staticmethod
    def _serialize_csv_value(value: Any) -> str:
        return serialize_csv_value(value)

    def _to_yaml(self, data: dict[str, Any]) -> str | None:
        return to_yaml(data, log=self._log)

    def _validate_roundtrip(self, original: str, converted: str) -> bool:
        """Validate that converted text can be parsed back to equivalent data."""
        try:
            original_parsed = json.loads(original)
        except Exception:
            return False

        if isinstance(original_parsed, list):
            converted_parsed = self._from_csv(converted)
        elif isinstance(original_parsed, dict):
            converted_parsed = self._from_yaml(converted)
            if converted_parsed is None:
                return converted == original
        else:
            return False

        if converted_parsed is None:
            return False

        return self._deep_equal(original_parsed, converted_parsed)

    def _from_csv(self, text: str) -> list[dict[str, Any]] | None:
        return from_csv(text)

    @staticmethod
    def _parse_csv_value(value: str) -> Any:
        return parse_csv_value(value)

    def _from_yaml(self, text: str) -> Any | None:
        return from_yaml(text)

    @staticmethod
    def _deep_equal(a: Any, b: Any) -> bool:
        return deep_equal(a, b)
