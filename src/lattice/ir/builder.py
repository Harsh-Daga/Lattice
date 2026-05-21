"""IR Builder — converts Request messages into canonical PromptIR.

The builder performs three sequential passes:
1. Partition — segment each message into typed sections
2. Analyze — split sections into spans with role classification
3. Protect — mark spans as protected or compressible based on semantic role

This produces a structured IR that subsequent transforms can operate on safely.
"""

from __future__ import annotations

import json as _json
import re
from typing import Any

from lattice.core.transport import Message, Request
from lattice.ir.types import (
    PromptIR,
    Section,
    SectionType,
    Span,
    SpanRole,
)

_INDENT = re.compile(r"^\s+")
_FENCE_START = re.compile(r"^```(\w+)?$")
_JSON_LINE = re.compile(r"^\s*[\[\{]")
_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_LOG_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_LOG_LEVEL = re.compile(r"\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b")
_TRACEBACK_START = re.compile(r"^Traceback\s*\(", re.MULTILINE)
_PYTHON_FRAME = re.compile(r'^\s+File\s+"([^"]+)",\s+line\s+(\d+)', re.MULTILINE)
_JAVA_FRAME = re.compile(r"\bat\s+(\S+)\s*\(([^)]+):(\d+)\)")
_EXCEPTION_LINE = re.compile(r"^\w+(?:Error|Exception|Warning|Fault)(?::\s*.*)?$", re.MULTILINE)
_DIFF_HDR = re.compile(r"^(---|\+\+\+|diff\s+--git|index\s+\w+)", re.MULTILINE)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_URL_RE = re.compile(r"https?://[^\s)>]+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:/[\w\.\-]+)+")
_ERROR_KEYWORDS = frozenset(
    {
        "error",
        "exception",
        "failure",
        "fail",
        "crash",
        "timeout",
        "refused",
        "denied",
        "abort",
        "panic",
        "fatal",
        "critical",
        "segfault",
        "oom",
    }
)
_ROOT_CAUSE_KEYWORDS = frozenset(
    {
        "root cause",
        "the cause was",
        "the reason is",
        "determined that",
        "because",
        "therefore",
        "consequently",
        "due to",
        "caused by",
        "triggered by",
        "resulting in",
        "leading to",
    }
)
_TASK_KEYWORDS = frozenset(
    {
        "analyze",
        "debug",
        "fix",
        "investigate",
        "explain",
        "compare",
        "optimize",
        "refactor",
        "implement",
        "review",
        "test",
        "deploy",
        "configure",
        "migrate",
        "upgrade",
        "resolve",
    }
)
_CONSTRAINT_KEYWORDS = frozenset(
    {
        "must",
        "required",
        "mandatory",
        "essential",
        "critical",
        "shall",
        "should not",
        "must not",
        "cannot",
        "do not",
        "never",
        "always",
        "ensure",
        "guarantee",
        "preserve",
        "keep",
    }
)
_FORMAT_KEYWORDS = frozenset(
    {
        "json",
        "yaml",
        "csv",
        "markdown",
        "table",
        "code block",
        "output format",
        "return format",
        "respond in",
    }
)
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "about",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "also",
        "not",
    }
)


def build_ir(request: Request) -> PromptIR:
    """Build canonical PromptIR from a Request.

    This is the primary entry point. The IR is stored in content_profiler
    metadata and consumed by the scheduler, safety guards, and transforms.

    Also stores a summary of the IR back on request.metadata so downstream
    transforms can read protected-span/causal info without recompiling.
    """
    sections: list[Section] = []
    span_counter = 0

    for msg in request.messages:
        msg_sections, span_counter = _partition_message(msg, span_counter)
        sections.extend(msg_sections)

    _extract_entities_and_numbers(sections)
    _classify_roles(sections)
    _derive_protection(sections)

    ir = PromptIR(sections=sections)
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


def _classify_text_section(text: str, fallback: SectionType) -> SectionType:
    lower = text.lower()

    if any(k in lower for k in _TASK_KEYWORDS):
        if any(k in lower for k in _FORMAT_KEYWORDS):
            return SectionType.INSTRUCTION
        return SectionType.TASK

    if any(k in lower for k in _CONSTRAINT_KEYWORDS):
        return SectionType.CONSTRAINTS

    if any(k in lower for k in _FORMAT_KEYWORDS):
        return SectionType.OUTPUT_FORMAT

    if any(k in lower for k in _ERROR_KEYWORDS) or (any(k in lower for k in _ROOT_CAUSE_KEYWORDS)):
        return SectionType.ERROR

    return fallback


def _extract_entities_and_numbers(sections: list[Section]) -> None:
    """Extract entities (UUIDs, URLs, paths) and numbers from spans."""
    for section in sections:
        for span in section.spans:
            text = span.text
            span.entities = _UUID_RE.findall(text) + _URL_RE.findall(text)
            span.numbers = _NUMBER_RE.findall(text)
            span.keys = _extract_json_keys(text)


def _extract_json_keys(text: str) -> list[str]:
    """Extract JSON keys from text without parsing."""
    return re.findall(r'"(\w[\w\.\d_]*)"\s*:', text)


def _classify_roles(sections: list[Section]) -> None:
    """Classify the semantic role of each span."""
    for section in sections:
        for span in section.spans:
            span.role = _classify_span_role(span, section.type)


def _classify_span_role(span: Span, sec_type: SectionType) -> SpanRole:
    lower = span.text.lower()

    if sec_type == SectionType.CODE:
        return SpanRole.DATA

    if sec_type in (SectionType.STACK_TRACE, SectionType.ERROR):
        if any(k in lower for k in _ROOT_CAUSE_KEYWORDS):
            return SpanRole.CAUSE
        return SpanRole.DIAGNOSTIC

    if _has_meaningful_entities(span):
        return SpanRole.ENTITY

    if any(k in lower for k in _CONSTRAINT_KEYWORDS):
        return SpanRole.CONSTRAINT

    if any(k in lower for k in _ROOT_CAUSE_KEYWORDS):
        return SpanRole.CAUSE

    word_count = len(span.text.split())
    if word_count > 10 and _entity_density(span) > 0.2:
        return SpanRole.DATA

    content_words = set(w.lower() for w in span.text.split()) - _STOP_WORDS
    if not content_words:
        return SpanRole.BOILERPLATE

    if any(k in lower for k in _TASK_KEYWORDS):
        return SpanRole.REASONING

    if sec_type == SectionType.TASK or sec_type == SectionType.INSTRUCTION:
        return SpanRole.REASONING

    if sec_type == SectionType.CONTEXT:
        return SpanRole.DATA

    return SpanRole.DATA


def _has_meaningful_entities(span: Span) -> bool:
    return len(span.entities) > 0 or len(span.numbers) > 3


def _entity_density(span: Span) -> float:
    words = span.text.split()
    if not words:
        return 0.0
    entity_count = len(span.entities) + len(span.numbers) * 0.5 + len(span.keys) * 0.5
    return min(entity_count / len(words), 1.0)


def _derive_protection(sections: list[Section]) -> None:
    """Mark protected and compressible spans."""
    for section in sections:
        for span in section.spans:
            span.protected = _should_protect(span, section)
            span.compressible = _should_compress(span, section)


def _should_protect(span: Span, section: Section) -> bool:
    sec_type = section.type

    if sec_type in (
        SectionType.ERROR,
        SectionType.STACK_TRACE,
    ):
        return True

    if sec_type == SectionType.CONSTRAINTS:
        return True

    if sec_type == SectionType.OUTPUT_FORMAT:
        return True

    if span.role == SpanRole.CAUSE:
        return True

    if span.role == SpanRole.CONSTRAINT:
        return True

    if span.role == SpanRole.DIAGNOSTIC:
        return True

    if span.role == SpanRole.ENTITY and span.numbers:
        return True

    if section.protected_count == 0 and len(section.spans) <= 2:
        if sec_type in (SectionType.TASK, SectionType.INSTRUCTION):
            return True

    if span.role == SpanRole.COUNT:
        return True

    return False


def _should_compress(span: Span, section: Section) -> bool:
    if span.protected:
        return False

    if span.role == SpanRole.BOILERPLATE:
        return True

    sec_type = section.type

    if sec_type == SectionType.CODE:
        return is_repeated_template(span.text)

    if sec_type in (SectionType.JSON, SectionType.TABLE):
        return True

    if sec_type == SectionType.LOGS:
        return True

    return span.role == SpanRole.DATA and _entity_density(span) < 0.1


def is_repeated_template(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    trimmed = [ln.strip() for ln in lines if ln.strip()]
    unique = set(trimmed)
    return len(unique) / max(len(trimmed), 1) < 0.4
