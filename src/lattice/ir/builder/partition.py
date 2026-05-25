"""Message partitioning helpers."""

from __future__ import annotations

import json as _json
from typing import Any

from lattice.ir.builder._patterns import (
    _DIFF_HDR,
    _EXCEPTION_LINE,
    _JAVA_FRAME,
    _LOG_LEVEL,
    _LOG_TIMESTAMP,
    _MD_TABLE_LINE,
    _MD_TABLE_SEP,
    _PYTHON_FRAME,
    _TRACEBACK_START,
)
from lattice.ir.types import Section, SectionType, Span, SpanRole


def _partition_tool_output(content: str, span_counter: int) -> tuple[list[Section], int]:
    """Partition tool output content."""
    stripped = content.strip()
    if not stripped:
        return [], span_counter

    if stripped[0] in ("[", "{"):
        try:
            parsed = _json.loads(stripped)
        except _json.JSONDecodeError:
            return [
                _make_section(SectionType.TOOL_OUTPUT, stripped, span_counter)
            ], span_counter + 1

        sec_type = _classify_json_section(parsed)
        return [_make_section(sec_type, stripped, span_counter)], span_counter + 1

    if _is_log_block(content.splitlines(), 0):
        return [_make_section(SectionType.LOGS, stripped, span_counter)], span_counter + 1

    return [_make_section(SectionType.TOOL_OUTPUT, stripped, span_counter)], span_counter + 1


def _make_section(sec_type: SectionType, text: str, span_counter: int) -> Section:
    span = Span(
        span_id=f"s{span_counter:03d}",
        text=text,
        role=SpanRole.DATA,
        section_type=sec_type,
    )
    return Section(type=sec_type, spans=[span])


def _extract_code_block(
    lines: list[str], start_idx: int, span_counter: int
) -> tuple[int, Section | None]:
    """Extract a fenced code block."""
    lang = lines[start_idx].strip()
    lang = lang[3:].strip() if lang.startswith("```") else ""
    block_lines = []
    i = start_idx + 1
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            i += 1
            break
        block_lines.append(lines[i])
        i += 1

    text = "\n".join(block_lines).strip()
    if not text:
        return i, None

    return i, _make_section(SectionType.CODE, text, span_counter)


def _is_stack_trace_block(lines: list[str], idx: int) -> bool:
    line = lines[idx].strip()
    return bool(
        _TRACEBACK_START.match(line)
        or (_PYTHON_FRAME.match(line) and idx < len(lines) - 1)
        or (_JAVA_FRAME.search(line) and idx < len(lines) - 1)
    )


def _is_stack_trace_start(line: str) -> bool:
    return bool(_TRACEBACK_START.match(line))


def _extract_stack_trace_block(
    lines: list[str], idx: int, span_counter: int
) -> tuple[int, Section]:
    block_lines = [lines[idx]]
    i = idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            block_lines.append("")
            i += 1
            break
        if not (
            stripped.startswith(" ")
            or stripped.startswith("\t")
            or _PYTHON_FRAME.match(stripped)
            or _JAVA_FRAME.search(stripped)
            or _EXCEPTION_LINE.match(stripped)
        ):
            break
        block_lines.append(lines[i])
        i += 1

    return i, _make_section(SectionType.STACK_TRACE, "\n".join(block_lines).strip(), span_counter)


def _is_log_line(line: str) -> bool:
    stripped = line.strip()
    return bool(_LOG_TIMESTAMP.search(stripped) and _LOG_LEVEL.search(stripped))


def _is_log_block(lines: list[str], idx: int) -> bool:
    line = lines[idx].strip()
    return bool(_LOG_TIMESTAMP.search(line) and _LOG_LEVEL.search(line))


def _extract_log_block(lines: list[str], idx: int, span_counter: int) -> tuple[int, Section]:
    block_lines = [lines[idx]]
    i = idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _is_log_line(stripped) or (
            stripped.startswith(" ") and _is_log_line(lines[i - 1].strip())
        ):
            block_lines.append(lines[i])
            i += 1
            continue
        break

    return i, _make_section(SectionType.LOGS, "\n".join(block_lines).strip(), span_counter)


def _extract_json_block(lines: list[str], idx: int) -> str:
    text = "\n".join(lines[idx:])
    first_ch = lines[idx].strip()[0]
    closing = "]" if first_ch == "[" else "}"
    depth = 0
    chars: list[str] = []
    for ch in text:
        chars.append(ch)
        if ch == first_ch:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                break

    if depth != 0:
        raise ValueError(f"Unbalanced JSON at line {idx}: remaining depth {depth}")

    return "".join(chars)


def _classify_json_section(parsed: Any) -> SectionType:
    if isinstance(parsed, dict):
        if any(k in parsed for k in ("is_error", "tool", "content")):
            return SectionType.TOOL_OUTPUT
        if any(k in parsed for k in ("error", "errors", "failures", "warnings")):
            return SectionType.ERROR
        return SectionType.JSON
    if isinstance(parsed, list) and len(parsed) > 0:
        if isinstance(parsed[0], dict):
            if any(k in parsed[0] for k in ("id", "name", "status", "latency")):
                return SectionType.JSON
        return SectionType.DATA
    return SectionType.DATA


def _is_table_block(lines: list[str], idx: int) -> bool:
    if idx + 2 >= len(lines):
        return False
    return bool(
        _MD_TABLE_LINE.match(lines[idx].strip()) and _MD_TABLE_SEP.match(lines[idx + 1].strip())
    )


def _extract_table_block(lines: list[str], idx: int, span_counter: int) -> tuple[int, Section]:
    block_lines = [lines[idx], lines[idx + 1]]
    i = idx + 2
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            break
        if _MD_TABLE_LINE.match(stripped):
            block_lines.append(lines[i])
            i += 1
            continue
        break

    return i, _make_section(SectionType.TABLE, "\n".join(block_lines).strip(), span_counter)


def _is_diff_header(line: str) -> bool:
    return bool(_DIFF_HDR.match(line))


def _is_diff_block(lines: list[str], idx: int) -> bool:
    return _is_diff_header(lines[idx].strip())


def _extract_diff_block(lines: list[str], idx: int, span_counter: int) -> tuple[int, Section]:
    block_lines = [lines[idx]]
    i = idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _is_diff_header(stripped):
            break
        if stripped.startswith(("diff ", "---", "+++", "@@", "+", "-", " ")):
            block_lines.append(lines[i])
            i += 1
            continue
        break

    return i, _make_section(SectionType.DATA, "\n".join(block_lines).strip(), span_counter)
