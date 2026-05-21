"""optimizer/ir_structure_optimizer.py — IR-native structure optimization.

Phase 4 — Port structure_optimizer to IR-native.

Operates on PromptIR Section/Span nodes with pre-computed structure metadata
from the normalizer (ir_normalizer.py):

- JSON spans: use json_shape, constant_fields, arithmetic_fields
- TABLE spans: use table_columns, constant_columns, sequence_columns
- TOOL_OUTPUT spans: use already-parsed JSON structure
- LOG spans: use log_events metadata

Why IR-native is better:
  - Normalizer already parsed structure (1 pass, not N passes per transform)
  - Compression is structural, not regex-based
  - Protection is span-aware, not threshold-based
  - Reverse is trivial: just don't modify the IR (lossless)
"""
from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result
from lattice.core.transport import Request
from lattice.ir.native_optimizer import IRNativeOptimizer
from lattice.ir.types import PromptIR, SectionType


class IRStructureOptimizer(IRNativeOptimizer):
    """IR-native structure optimizer.

    Operates on JSON / TABLE / TOOL_OUTPUT / LOGS sections.
    """

    name = "ir_structure_optimizer"
    priority = 20
    can_process_sections = {
        SectionType.JSON,
        SectionType.TABLE,
        SectionType.TOOL_OUTPUT,
        SectionType.LOGS,
    }

    def optimize_ir(
        self, ir: PromptIR, _request: Request, _context: TransformContext
    ) -> Result[PromptIR, TransformError]:
        """Factor JSON, collapse tables, group logs using IR metadata."""
        for section in ir.sections:
            if section.type not in self.can_process_sections:
                continue

            for span in section.spans:
                if span.protected:
                    continue
                if not span.compressible:
                    continue

                structure = span.structure
                if not isinstance(structure, dict):
                    continue

                if section.type in (
                    SectionType.JSON,
                    SectionType.TOOL_OUTPUT,
                ):
                    factored = _factor_json(structure, span.text)
                    if factored and factored != span.text:
                        span.text = factored
                        span.metadata["factored"] = True

                elif section.type == SectionType.TABLE:
                    collapsed = _factor_table(structure, span.text)
                    if collapsed and collapsed != span.text:
                        span.text = collapsed
                        span.metadata["collapsed"] = True

                elif section.type == SectionType.LOGS:
                    grouped = _group_logs(structure, span.text)
                    if grouped and grouped != span.text:
                        span.text = grouped
                        span.metadata["grouped"] = True

        return Ok(ir)


# ------------------------------------------------------------------
# JSON factoring
# ------------------------------------------------------------------


def _factor_json(structure: dict, text: str) -> str | None:
    """If constant_fields or arithmetic_fields exist, prepend a compact summary."""
    constant_fields = structure.get("constant_fields")
    arithmetic_fields = structure.get("arithmetic_fields")
    row_count = structure.get("row_count", 0)
    json_shape = structure.get("json_shape", [])

    if not constant_fields and not arithmetic_fields:
        return None

    lines: list[str] = [f"JSON: {row_count} rows"]
    if json_shape:
        lines.append(f"  keys: {', '.join(json_shape)}")

    if constant_fields:
        lines.append("  constant:")
        for item in constant_fields:
            if isinstance(item, dict):
                field = item.get("field", "")
                value = item.get("value", "")
            else:
                field, value = str(item), ""
            lines.append(f"    {field} = {value}")

    if arithmetic_fields:
        lines.append("  arithmetic:")
        for item in arithmetic_fields:
            formula = item if isinstance(item, str) else item.get("formula", "")
            lines.append(f"    {formula}")

    lines.append("  ... (full data follows)")
    lines.append(text)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Table factoring
# ------------------------------------------------------------------


def _factor_table(structure: dict, text: str) -> str | None:
    """If constant_columns + sequence_columns exist, prepend a compact summary."""
    table_columns = structure.get("table_columns", [])
    constant_cols = structure.get("constant_columns", {})
    sequence_cols = structure.get("sequence_columns", {})
    row_count = structure.get("row_count", 0)

    if not constant_cols and not sequence_cols:
        return None

    lines: list[str] = [
        f"Table: {row_count} rows, columns: {', '.join(table_columns)}"
    ]

    if constant_cols:
        lines.append("  constant:")
        for col, val in constant_cols.items():
            lines.append(f"    {col} = {val}")

    if sequence_cols:
        lines.append("  ranges:")
        for col, seq in sequence_cols.items():
            formula = seq if isinstance(seq, str) else seq.get("formula", "")
            lines.append(f"    {col}: {formula}")

    lines.append("")
    lines.append(text)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Log grouping
# ------------------------------------------------------------------


def _group_logs(structure: dict, text: str) -> str | None:
    """If log_events or grouped_events exist, prepend a grouped summary."""
    grouped = structure.get("grouped_events", {})
    if not grouped:
        return None

    lines: list[str] = ["Logs (grouped by severity):"]
    for severity, count in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {severity}: {count} events")

    lines.append("")
    lines.append(text)
    return "\n".join(lines)
