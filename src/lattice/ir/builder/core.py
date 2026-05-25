"""IR builder entry points."""

from __future__ import annotations

import json as _json

from lattice.ir.builder._patterns import _FENCE_START, _JSON_LINE, _MD_TABLE_LINE
from lattice.ir.builder.analyze import (
    _classify_roles,
    _classify_text_section,
    _derive_protection,
    _extract_entities_and_numbers,
)
from lattice.ir.builder.partition import (
    _classify_json_section,
    _extract_code_block,
    _extract_diff_block,
    _extract_json_block,
    _extract_log_block,
    _extract_stack_trace_block,
    _extract_table_block,
    _is_diff_block,
    _is_diff_header,
    _is_log_block,
    _is_log_line,
    _is_stack_trace_block,
    _is_stack_trace_start,
    _is_table_block,
    _make_section,
    _partition_tool_output,
)
from lattice.ir.types import PromptIR, Section, SectionType
from lattice.transport.types import Message, Request


def build_ir(request: Request) -> PromptIR:
    """Build canonical PromptIR from a Request.

    This is the primary entry point. The IR is consumed by the scheduler,
    safety guards, and transforms. ``build_ir`` is a pure function: it does
    not mutate ``request`` — callers that want IR-summary metadata stored
    back on the request should use :func:`compile_request_ir` (which runs
    build → normalize → store).
    """
    sections: list[Section] = []
    span_counter = 0

    for msg in request.messages:
        msg_sections, span_counter = _partition_message(msg, span_counter)
        sections.extend(msg_sections)

    _extract_entities_and_numbers(sections)
    _classify_roles(sections)
    _derive_protection(sections)

    return PromptIR(sections=sections)


def compile_request_ir(request: Request) -> PromptIR:
    """Full IR compile path: build → normalize → store summary metadata.

    Replaces the deleted ``core.compiler.PromptCompiler.compile`` from
    Phase 1. The IR summary stored on ``request.metadata`` reflects the
    **post-normalize** state, matching the pre-refactor behavior that
    ``core/pipeline.py`` relies on when reading ``_lattice_protected_spans``.
    """
    from lattice.ir.normalizer import normalize_ir

    ir = normalize_ir(build_ir(request))
    _store_ir_metadata(request, ir)
    return ir


def _store_ir_metadata(request: Request, ir: PromptIR) -> None:
    """Store IR summary in request metadata for scheduler and safety guards.

    Previously lived in core/compiler.py; inlined here as part of Phase 1.
    """
    request.metadata["_lattice_ir_summary"] = ir.summary()
    request.metadata["_lattice_protected_spans"] = ir.protected_span_ids()

    section_types = ir.section_types
    if "error" in section_types or "stack_trace" in section_types:
        request.metadata["_lattice_has_errors"] = True

    if ir.metadata.get("has_causal_chains"):
        request.metadata["_lattice_has_causal"] = True
        request.metadata["_lattice_causal_count"] = ir.metadata.get("causal_span_count", 0)


def _partition_message(msg: Message, span_counter: int) -> tuple[list[Section], int]:
    """Partition a single message into typed sections."""
    role = msg.role
    if hasattr(role, "value"):
        role = role.value
    content = msg.content

    if not content.strip():
        return [], span_counter

    if role == "system":
        return _partition_text_section(content, SectionType.SYSTEM, span_counter)

    if role == "tool":
        return _partition_tool_output(content, span_counter)

    if msg.tool_call_id:
        return _partition_tool_output(content, span_counter)

    return _partition_user_message(content, msg, span_counter)


def _partition_user_message(
    content: str, msg: Message, span_counter: int
) -> tuple[list[Section], int]:
    """Partition a user/assistant message."""
    sections: list[Section] = []
    remaining = content

    code_fences = list(_FENCE_START.finditer(remaining))
    if code_fences:
        i = 0
        in_code = False
        last_pos = 0
        code_lines: list[str] = []
        code_lang = ""  # noqa: F841 — used in code fence parsing flow

        for m in code_fences:
            if not in_code:
                prefix = remaining[last_pos : m.start()].strip()
                if prefix:
                    result_sections, span_counter = _partition_text_section(
                        prefix, SectionType.DATA, span_counter
                    )
                    sections.extend(result_sections)
                in_code = True
                last_pos = m.end()
            else:
                code_lines.extend(remaining[last_pos : m.start()].splitlines())
                code_text = "\n".join(code_lines)
                if code_text.strip():
                    sections.append(_make_section(SectionType.CODE, code_text, span_counter))
                    span_counter += 1
                in_code = False
                code_lines = []
                last_pos = m.end()
            i += 1

        if in_code:
            code_lines.extend(remaining[last_pos:].splitlines())
            code_text = "\n".join(code_lines)
            if code_text.strip():
                sections.append(_make_section(SectionType.CODE, code_text, span_counter))
                span_counter += 1
        else:
            trailer = remaining[last_pos:].strip()
            if trailer:
                result_sections, span_counter = _partition_text_section(
                    trailer, SectionType.DATA, span_counter
                )
                sections.extend(result_sections)
    else:
        return _partition_text_section(content, SectionType.DATA, span_counter)

    return sections, span_counter


def _partition_text_section(
    text: str, fallback_type: SectionType, span_counter: int
) -> tuple[list[Section], int]:
    """Partition free-form text into typed sections."""
    lines = text.splitlines()
    if not lines or not any(line.strip() for line in lines):
        return [], span_counter

    sections: list[Section] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if _FENCE_START.match(line):
            i, code_sec = _extract_code_block(lines, i, span_counter)
            if code_sec:
                sections.append(code_sec)
                span_counter += 1
            continue

        if _is_stack_trace_block(lines, i):
            i, trace_sec = _extract_stack_trace_block(lines, i, span_counter)
            sections.append(trace_sec)
            span_counter += 1
            continue

        if _is_log_block(lines, i):
            i, log_sec = _extract_log_block(lines, i, span_counter)
            sections.append(log_sec)
            span_counter += 1
            continue

        if _JSON_LINE.match(line) or line.startswith("{"):
            try:
                block_text = _extract_json_block(lines, i)
            except ValueError:
                i += 1
                continue
            parsed = _json.loads(block_text)
            sec_type = _classify_json_section(parsed)
            sections.append(_make_section(sec_type, block_text, span_counter))
            span_counter += 1
            i += len(block_text.splitlines())
            continue

        if _is_table_block(lines, i):
            i, table_sec = _extract_table_block(lines, i, span_counter)
            sections.append(table_sec)
            span_counter += 1
            continue

        if _is_diff_block(lines, i):
            i, diff_sec = _extract_diff_block(lines, i, span_counter)
            sections.append(diff_sec)
            span_counter += 1
            continue

        text_lines = [line]
        i += 1
        while i < len(lines):
            nl = lines[i].strip()
            if not nl:
                i += 1
                continue
            if _FENCE_START.match(nl):
                break
            if _JSON_LINE.match(nl):
                break
            if _is_stack_trace_start(nl):
                break
            if _MD_TABLE_LINE.match(nl):
                break
            if _is_log_line(nl):
                break
            if _is_diff_header(nl):
                break
            text_lines.append(nl)
            i += 1

        text_block = "\n".join(text_lines).strip()
        if text_block:
            stype = _classify_text_section(text_block, fallback_type)
            sections.append(_make_section(stype, text_block, span_counter))
            span_counter += 1

    return sections, span_counter


