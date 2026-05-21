"""IR Normalizer — canonicalizes PromptIR before compression passes.

Normalization runs BEFORE compression transforms. It canonicalizes:
1. JSON sections — sort keys, detect repeated shape, arithmetic sequences, constant fields
2. Table sections — detect column structure, constant columns, sequence columns
3. Log sections — parse into structured event entries
4. Constraint lifting — extract must/required/format constraints into dedicated section
5. Causal chain extraction — detect cause-effect markers and build chain metadata

All normalization is lossless — content is preserved and structure metadata is
attached to spans for downstream transforms and serializers to use.
"""

from __future__ import annotations

import json as _json
import re
from collections import Counter
from typing import Any

from lattice.ir.types import (
    PromptIR,
    Section,
    SectionType,
    Span,
    SpanRole,
)

_CAUSAL_MARKERS = re.compile(
    r"\b(caused by|triggered by|due to|because|resulting in|leading to|"
    r"as a result|consequently|therefore|hence|thus|so)\b",
    re.IGNORECASE,
)
_CAUSAL_PATTERNS = [
    re.compile(r"(.+?)\s+(?:caused|triggered)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(
        r"(.+?),\s+(?:causing|triggering|leading to|resulting in)\s+(.+?)(?:\.|$)", re.IGNORECASE
    ),
    re.compile(r"(?:due to|because)\s+(.+?),\s+(.+?)(?:\.|$)", re.IGNORECASE),
]


def normalize_ir(ir: PromptIR) -> PromptIR:
    """Apply all normalizations to an IR."""
    ir = normalize_json_sections(ir)
    ir = normalize_table_sections(ir)
    ir = normalize_log_sections(ir)
    ir = lift_constraints(ir)
    ir = extract_causal_chains(ir)
    return ir


# ---------------------------------------------------------------------------
# JSON normalization
# ---------------------------------------------------------------------------


def normalize_json_sections(ir: PromptIR) -> PromptIR:
    for section in ir.sections:
        if section.type not in (SectionType.JSON, SectionType.TOOL_OUTPUT, SectionType.ERROR):
            continue
        for span in section.spans:
            if not span.text.strip():
                continue
            if span.text.strip()[0] not in ("[", "{"):
                continue
            try:
                parsed = _json.loads(span.text)
            except (_json.JSONDecodeError, ValueError):
                continue

            _attach_json_structure(parsed, span.structure)
            if isinstance(parsed, list) and len(parsed) >= 3:
                _detect_json_patterns(parsed, span)
    return ir


def _attach_json_structure(parsed: Any, structure: dict[str, Any]) -> None:
    if isinstance(parsed, dict):
        structure["keys"] = sorted(parsed.keys())
        structure["key_count"] = len(parsed)
        nested = {}
        for k, v in parsed.items():
            if isinstance(v, (dict, list)):
                nested[k] = type(v).__name__
            elif isinstance(v, (int, float)):
                nested[k] = f"number={v}" if k != "id" else f"range={v}"
            elif isinstance(v, str):
                nested[k] = f"len={len(v)}" if len(v) > 20 else v
            else:
                nested[k] = str(type(v).__name__)
        structure["value_types"] = nested
    elif isinstance(parsed, list):
        structure["item_count"] = len(parsed)
        structure["item_types"] = list(dict(Counter(type(v).__name__ for v in parsed[:20])).keys())


def _detect_json_patterns(items: list[dict], span: Span) -> None:
    if not items or not all(isinstance(it, dict) for it in items):
        return
    if not items:
        return

    first_keys = list(items[0].keys())
    if any(set(d.keys()) != set(first_keys) for d in items[1:]):
        return

    structure = span.structure
    structure["json_shape"] = first_keys
    structure["row_count"] = len(items)

    constant_fields = []
    for key in first_keys:
        values = [d.get(key) for d in items]
        unique = set(str(v) for v in values)
        if len(unique) == 1:
            constant_fields.append({"field": key, "value": str(values[0])})
    if constant_fields:
        structure["constant_fields"] = constant_fields

    arithmetic_fields = []
    for key in first_keys:
        values = [d.get(key) for d in items]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(numeric) == len(values) and len(numeric) >= 3:
                d = numeric[1] - numeric[0]
                if all(numeric[i] - numeric[i - 1] == d for i in range(2, len(numeric))):
                    arithmetic_fields.append(
                        {
                            "field": key,
                            "start": numeric[0],
                            "step": d,
                            "formula": f"{key} = {numeric[0]} + n*{d}",
                        }
                    )
    if arithmetic_fields:
        structure["arithmetic_fields"] = arithmetic_fields


# ---------------------------------------------------------------------------
# Table normalization
# ---------------------------------------------------------------------------


def normalize_table_sections(ir: PromptIR) -> PromptIR:
    for section in ir.sections:
        if section.type != SectionType.TABLE:
            continue
        for span in section.spans:
            lines = span.text.splitlines()
            if len(lines) < 3:
                continue
            header_line = lines[0].strip()
            if not header_line.startswith("|"):
                continue
            headers = [h.strip() for h in header_line.split("|") if h.strip()]
            sep_line = lines[1].strip()
            if not re.match(r"^\|[\s\-:|]+\|$", sep_line):
                continue

            data_lines = [ln.strip() for ln in lines[2:] if ln.strip() and ln.startswith("|")]
            if not data_lines:
                continue

            rows = []
            for dl in data_lines:
                cells = [c.strip() for c in dl.split("|") if c.strip()]
                rows.append(dict(zip(headers, cells)))

            span.structure.update(
                {
                    "table_columns": headers,
                    "row_count": len(rows),
                    "rows": rows,
                }
            )
            _detect_table_patterns(rows, span)
    return ir


def _detect_table_patterns(rows: list[dict], span: Span) -> None:
    if not rows:
        return
    structure = span.structure
    columns = structure.get("table_columns", [])
    if not columns:
        return

    for col in columns:
        values = [r.get(col, "") for r in rows]
        unique = set(values)
        if len(unique) == 1:
            structure.setdefault("constant_columns", {})[col] = values[0]
        elif all(
            v.replace(".", "", 1).isdigit()
            for v in values
            if v and not v.replace("-", "", 1).replace(".", "", 1).isdigit()
        ):
            continue
        else:
            numeric_vals = _try_parse_numeric_column(values)
            if numeric_vals and len(numeric_vals) >= 3:
                d = numeric_vals[1] - numeric_vals[0]
                if all(
                    numeric_vals[j] - numeric_vals[j - 1] == d for j in range(2, len(numeric_vals))
                ):
                    structure.setdefault("sequence_columns", {})[col] = {
                        "start": numeric_vals[0],
                        "step": d,
                    }


def _try_parse_numeric_column(values: list[str]) -> list[int] | None:
    result: list[int] = []
    for v in values:
        try:
            result.append(int(float(v)))
        except ValueError:
            return None
    return result


# ---------------------------------------------------------------------------
# Log normalization
# ---------------------------------------------------------------------------


def normalize_log_sections(ir: PromptIR) -> PromptIR:
    for section in ir.sections:
        if section.type != SectionType.LOGS:
            continue
        for span in section.spans:
            text = span.text
            raw_lines = text.splitlines()
            events = []
            current_event: dict[str, Any] | None = None

            for line in raw_lines:
                stripped = line.strip()
                if not stripped:
                    if current_event:
                        events.append(current_event)
                        current_event = None
                    continue

                ts_match = re.search(
                    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)", stripped
                )
                if ts_match:
                    if current_event:
                        events.append(current_event)
                    severity = "UNKNOWN"
                    for sev in ("CRITICAL", "FATAL", "ERROR", "WARN", "INFO", "DEBUG"):
                        if sev in stripped:
                            severity = sev
                            break
                    current_event = {
                        "timestamp": ts_match.group(1),
                        "severity": severity,
                        "message": stripped[ts_match.end() :].strip(),
                        "raw": stripped,
                    }
                elif current_event is not None:
                    current_event["message"] += " " + stripped
                    current_event["raw"] += "\n" + stripped

            if current_event is not None:
                events.append(current_event)

            if events:
                span.structure["log_events"] = events
                span.structure["event_count"] = len(events)

                severity_counts = Counter(e.get("severity", "UNKNOWN") for e in events)
                span.structure["severity_counts"] = dict(severity_counts)
    return ir


# ---------------------------------------------------------------------------
# Constraint lifting
# ---------------------------------------------------------------------------


def lift_constraints(ir: PromptIR) -> PromptIR:
    """Extract constraint and format spans into dedicated CONSTRAINT section."""
    constraint_spans: list[Span] = []
    format_spans: list[Span] = []

    for section in ir.sections:
        remaining: list[Span] = []
        for span in section.spans:
            role_val = span.role.value if hasattr(span.role, "value") else span.role
            if role_val == SpanRole.CONSTRAINT.value or _is_format_requirement(span.text):
                lower = span.text.lower()
                if any(
                    k in lower for k in ("json", "yaml", "csv", "markdown", "format", "return as")
                ):
                    format_spans.append(span)
                else:
                    constraint_spans.append(span)
            else:
                remaining.append(span)
        section.spans = remaining

    combined_constraints: list[Span] = []
    if constraint_spans:
        combined_constraints.extend(constraint_spans)
    if format_spans:
        combined_constraints.extend(format_spans)

    if combined_constraints:
        ir.sections = [
            Section(type=SectionType.CONSTRAINTS, spans=combined_constraints),
            *ir.sections,
        ]

    return ir


def _is_format_requirement(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(r"\breturn\s+(?:in\s+)?json\b", lower)
        or re.search(r"\boutput\s+(?:as|in|format)\b", lower)
        or re.search(r"\bformat\s+(?:as|should|must)\b", lower)
        or re.search(r"\brespond\s+(?:with|in)\s+(?:json|yaml|csv|markdown)\b", lower)
        or re.search(r"\buse\s+(?:json|yaml|csv|markdown)\s+(?:format|output)\b", lower)
    )


# ---------------------------------------------------------------------------
# Causal chain extraction
# ---------------------------------------------------------------------------


def extract_causal_chains(ir: PromptIR) -> PromptIR:
    """Detect cause-effect relationships and store as structured metadata."""
    for section in ir.sections:
        causal_spans: list[Span] = []
        for span in section.spans:
            text = span.text
            if not _CAUSAL_MARKERS.search(text.lower()):
                continue

            relations = []
            for pattern in _CAUSAL_PATTERNS:
                for m in pattern.finditer(text):
                    left = m.group(1).strip()
                    right = m.group(2).strip()
                    if left and right:
                        relations.append(
                            {
                                "cause": _extract_event_label(left),
                                "effect": _extract_event_label(right),
                                "cause_text": left[:120],
                                "effect_text": right[:120],
                            }
                        )

            if relations:
                span.structure["causal_relations"] = relations
                if span.role not in (SpanRole.CAUSE, SpanRole.DIAGNOSTIC):
                    span.role = SpanRole.CAUSE
                span.protected = True
                causal_spans.append(span)

        if causal_spans:
            ir.metadata["has_causal_chains"] = True
            ir.metadata["causal_span_count"] = ir.metadata.get("causal_span_count", 0) + len(
                causal_spans
            )

    return ir


def _extract_event_label(text: str) -> str:
    text = text.strip().rstrip(".")
    if len(text) > 40:
        words = text.split()
        mid = []
        for w in words:
            mid.append(w)
            if len(" ".join(mid)) >= 35:
                break
        return " ".join(mid) + "..."
    return text
