"""IR Serializer — converts canonical PromptIR back into LLM-readable text.

The serializer produces human/model-readable output, NOT opaque placeholders.
Every optimization is presented as structured text the LLM can reason over.

Principles:
- Never emit opaque placeholders (<d_36>, <g_1>, <ref_17>) without a manifest
- Use descriptive labels (ERROR_MODULE_NOT_FOUND, UUID_DUP_1) when aliasing
- Present table data as columnar summary with formulas for sequences
- Group diagnostic repetition with counts + examples preserved
- Always include ALIAS MAP or DICTIONARY headers when using abbreviations
"""

from __future__ import annotations

from typing import Any

from lattice.ir.types import PromptIR, Section, SectionType


def serialize_ir_to_text(ir: PromptIR) -> str:
    """Serialize a PromptIR into a single LLM-readable text string."""
    parts: list[str] = []

    for section in ir.sections:
        text = _serialize_section(section)
        if text.strip():
            parts.append(text)

    return "\n\n".join(parts)


def _serialize_section(section: Section) -> str:
    serializer = _SECTION_SERIALIZERS.get(section.type, _serialize_default)
    return serializer(section)


def _serialize_default(section: Section) -> str:
    return "\n".join(s.text for s in section.spans)


def _serialize_system(section: Section) -> str:
    return "\n".join(s.text for s in section.spans)


def _serialize_task(section: Section) -> str:
    lines = ["TASK:"]
    for span in section.spans:
        lines.append(span.text)
    return "\n".join(lines)


def _serialize_instruction(section: Section) -> str:
    lines = ["INSTRUCTIONS:"]
    for span in section.spans:
        lines.append(span.text)
    return "\n".join(lines)


def _serialize_constraints(section: Section) -> str:
    lines = ["CONSTRAINTS:"]
    for span in section.spans:
        lines.append(f"- {span.text.strip()}")
    return "\n".join(lines)


def _serialize_context(section: Section) -> str:
    return "\n".join(s.text for s in section.spans)


def _serialize_data(section: Section) -> str:
    return "\n".join(s.text for s in section.spans)


def _serialize_output_format(section: Section) -> str:
    lines = ["OUTPUT FORMAT:"]
    for span in section.spans:
        lines.append(span.text)
    return "\n".join(lines)


def _serialize_tool_output(section: Section) -> str:
    for span in section.spans:
        if span.structure:
            return _serialize_structured_tool_output(span, section)
    return "\n".join(s.text for s in section.spans)


def _serialize_json(section: Section) -> str:
    for span in section.spans:
        if span.structure and span.structure.get("json_shape"):
            return _serialize_json_factored(span)
    return "\n".join(s.text for s in section.spans)


def _serialize_table(section: Section) -> str:
    for span in section.spans:
        if span.structure:
            return _serialize_columnar_table(span)
    return "\n".join(s.text for s in section.spans)


def _serialize_logs(section: Section) -> str:
    for span in section.spans:
        if span.structure and span.structure.get("log_events"):
            return _serialize_log_events(span)
    return "\n".join(s.text for s in section.spans)


def _serialize_code(section: Section) -> str:
    return "\n".join(s.text for s in section.spans)


def _serialize_error(section: Section) -> str:
    lines = []
    for span in section.spans:
        if span.role.value == "cause" and span.structure.get("causal_relations"):
            lines.append(_serialize_causal_span(span))
        else:
            lines.append(span.text)
    return "\n".join(lines)


def _serialize_stack_trace(section: Section) -> str:
    return "\n".join(s.text for s in section.spans)


# ---------------------------------------------------------------------------
# Structured serializers
# ---------------------------------------------------------------------------


def _serialize_structured_tool_output(span: Any, section: Section) -> str:
    structure = span.structure
    lines = []

    if section.type == SectionType.ERROR:
        lines.append("ERROR DATA:")
    else:
        lines.append("TOOL OUTPUT:")

    if structure.get("json_shape"):
        lines.append(f"  keys: {', '.join(structure['json_shape'])}")

    if structure.get("row_count"):
        lines.append(f"  total entries: {structure['row_count']}")

    if structure.get("constant_fields"):
        lines.append("  constant fields:")
        for cf in structure["constant_fields"]:
            lines.append(f"    {cf['field']} = {cf['value']}")

    if structure.get("arithmetic_fields"):
        lines.append("  computed fields:")
        for af in structure["arithmetic_fields"]:
            lines.append(f"    {af['formula']}")

    lines.append("")
    lines.append(span.text)
    return "\n".join(lines)


def _serialize_json_factored(span: Any) -> str:
    structure = span.structure
    lines = ["JSON DATA:"]

    keys = structure.get("json_shape", [])
    if keys:
        lines.append(f"  keys: {', '.join(keys)}")

    if structure.get("row_count"):
        lines.append(f"  rows: {structure['row_count']}")

    if structure.get("constant_fields"):
        for cf in structure["constant_fields"]:
            lines.append(f"  {cf['field']} = {cf['value']} (all rows)")

    if structure.get("arithmetic_fields"):
        for af in structure["arithmetic_fields"]:
            lines.append(f"  {af['formula']}")

    lines.append("")
    if len(span.text) > 500:
        lines.append(span.text[:250])
        lines.append("... [truncated] ...")
        lines.append(span.text[-200:])
    else:
        lines.append(span.text)

    return "\n".join(lines)


def _serialize_columnar_table(span: Any) -> str:
    structure = span.structure
    columns = structure.get("table_columns", [])
    lines = []

    title = structure.get("table_name", "TABLE:")
    lines.append(title)
    if columns:
        lines.append(f"  columns: {', '.join(columns)}")

    if structure.get("row_count"):
        lines.append(f"  row count: {structure['row_count']}")

    if structure.get("constant_columns"):
        lines.append("  constant columns:")
        for col, val in structure["constant_columns"].items():
            lines.append(f"    {col} = {val} (all rows)")

    if structure.get("sequence_columns"):
        lines.append("  sequence columns:")
        for col, info in structure["sequence_columns"].items():
            step = info.get("step", 0)
            start = info.get("start", 0)
            if step == 1:
                lines.append(f"    {col} = {start}..{start + (structure['row_count'] - 1) * step}")
            elif step == 0:
                lines.append(f"    {col} = {start} (constant)")
            else:
                lines.append(f"    {col} = {start} + n*{step}")

    return "\n".join(lines)


def _serialize_log_events(span: Any) -> str:
    events = span.structure.get("log_events", [])
    severity_counts = span.structure.get("severity_counts", {})

    lines = [f"LOG DATA: {len(events)} events"]

    if severity_counts:
        sev_parts = [f"{k}: {v}" for k, v in sorted(severity_counts.items())]
        lines.append(f"  severity distribution: {', '.join(sev_parts)}")

    errors = [e for e in events if e.get("severity") in ("ERROR", "CRITICAL", "FATAL")]
    if errors:
        lines.append(f"  errors ({len(errors)}):")
        for e in errors[:5]:
            lines.append(f"    [{e.get('timestamp', '?')}] {e.get('message', '')[:100]}")
        if len(errors) > 5:
            lines.append(f"    ... and {len(errors) - 5} more")

    if len(events) > 20:
        lines.append("")
        lines.append(span.text[:300])
        lines.append("... [truncated] ...")

    return "\n".join(lines)


def _serialize_causal_span(span: Any) -> str:
    relations = span.structure.get("causal_relations", [])
    lines = ["CAUSAL GRAPH:"]
    for i, rel in enumerate(relations):
        lines.append(f"  {rel['cause']}")
        lines.append(f"    -> {rel['effect']}")
        if i < len(relations) - 1:
            lines.append("")
    return "\n".join(lines)


_SECTION_SERIALIZERS = {
    SectionType.SYSTEM: _serialize_system,
    SectionType.TASK: _serialize_task,
    SectionType.INSTRUCTION: _serialize_instruction,
    SectionType.CONSTRAINTS: _serialize_constraints,
    SectionType.CONTEXT: _serialize_context,
    SectionType.DATA: _serialize_data,
    SectionType.OUTPUT_FORMAT: _serialize_output_format,
    SectionType.TOOL_OUTPUT: _serialize_tool_output,
    SectionType.TOOL_SCHEMA: _serialize_default,
    SectionType.JSON: _serialize_json,
    SectionType.TABLE: _serialize_table,
    SectionType.LOGS: _serialize_logs,
    SectionType.CODE: _serialize_code,
    SectionType.ERROR: _serialize_error,
    SectionType.STACK_TRACE: _serialize_stack_trace,
}
