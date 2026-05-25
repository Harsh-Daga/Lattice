"""Span analysis and protection."""

from __future__ import annotations

import re

from lattice.ir.builder._patterns import (
    _CONSTRAINT_KEYWORDS,
    _ERROR_KEYWORDS,
    _FORMAT_KEYWORDS,
    _NUMBER_RE,
    _ROOT_CAUSE_KEYWORDS,
    _STOP_WORDS,
    _TASK_KEYWORDS,
    _URL_RE,
    _UUID_RE,
)
from lattice.ir.types import Section, SectionType, Span, SpanRole


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
