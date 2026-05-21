"""Constraint Lifting — OBSERVABILITY_ONLY transform.

Extracts buried requirements and formatting constraints from prompts into
explicit, labeled sections. This is a quality transform — it improves
model comprehension without reducing tokens.

The transform reads the PromptIR built by content_profiler and lifts
constraint/format spans into labeled sections in the serialized output.
"""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response
from lattice.ir.primitives import PromptIRV2

_CONSTRAINT_RE = re.compile(
    r"(?:\b(?:must|required|mandatory|essential|shall|should|should\s+not|must\s+not"
    r"|cannot|can\s+not|do\s+not|don't|never|always|ensure|guarantee|preserve|keep"
    r"|retain|maintain)\b)",
    re.IGNORECASE,
)
_FORMAT_RE = re.compile(
    r"(?:\b(?:return|respond|output|format)\s+(?:as|in|with|using)?\s*(?:json|yaml|csv|markdown|table|xml)\b"
    r"|\bjson\s+(?:format|output|response)\b"
    r"|\boutput\s+(?:should|must|format)\b)",
    re.IGNORECASE,
)
_LENGTH_RE = re.compile(
    r"\b(?:brief|concise|short|summarize|at\s+most|no\s+more\s+than|within"
    r"|limit\s+(?:to|of)|maximum|minimum|at\s+least|no\s+less\s+than"
    r"|exactly|precisely)\b",
    re.IGNORECASE,
)
_PRESERVE_RE = re.compile(
    r"\b(?:preserve|keep|retain|maintain|include|do\s+not\s+(?:drop|remove|omit|skip|exclude"
    r"|delete))\b",
    re.IGNORECASE,
)


class ConstraintLiftingTransform(ReversibleSyncTransform):
    name = "constraint_lifting"
    priority = 6
    transform_class = TransformClass.OBSERVABILITY_ONLY

    # ------------------------------------------------------------------
    # IR-native optimize() — v2 path
    # ------------------------------------------------------------------

    def optimize(
        self,
        ir: PromptIRV2,
        _request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """IR-native: lift constraints from span text and prepend CONSTRAINTS block."""
        lifted_count = 0
        new_sections = []
        for sec in ir.sections:
            new_spans = []
            for span in sec.spans:
                if len(span.text) < 20 or span.protected:
                    new_spans.append(span)
                    continue
                lifted = _lift_constraints(span.text)
                if lifted != span.text:
                    new_spans.append(span.with_text(lifted))
                    lifted_count += 1
                else:
                    new_spans.append(span)
            new_sections.append(sec.with_spans(tuple(new_spans)))
        context.record_metric(self.name, "spans_lifted", lifted_count)
        return Ok(ir.with_sections(tuple(new_sections)))

    # ------------------------------------------------------------------
    # Legacy process()
    # ------------------------------------------------------------------

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        lifted_count = 0

        for msg in request.messages:
            if msg.role not in ("user", "system"):
                continue
            content = msg.content
            if not content or len(content) < 20:
                continue

            lifted = _lift_constraints(content)
            if lifted != content:
                msg.content = lifted
                lifted_count += 1

        context.record_metric(self.name, "messages_lifted", lifted_count)
        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        combined = "\n".join(m.content for m in request.messages)
        return bool(_CONSTRAINT_RE.search(combined) or _FORMAT_RE.search(combined))


def _lift_constraints(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    constraint_lines: list[str] = []
    format_lines: list[str] = []
    length_lines: list[str] = []
    preserve_lines: list[str] = []
    remaining: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            remaining.append(line)
            continue

        scored = False

        if _FORMAT_RE.search(stripped):
            format_lines.append(line)
            scored = True

        if _CONSTRAINT_RE.search(stripped):
            if _PRESERVE_RE.search(stripped):
                preserve_lines.append(line)
                scored = True
            elif _LENGTH_RE.search(stripped):
                length_lines.append(line)
                scored = True
            else:
                constraint_lines.append(line)
                scored = True

        if not scored:
            remaining.append(line)

    seen: set[str] = set()
    all_constraints: list[str] = []
    for lst in (format_lines, preserve_lines, constraint_lines, length_lines):
        for ln in lst:
            key = ln.strip()
            if key and key not in seen:
                seen.add(key)
                all_constraints.append(ln)
    if not all_constraints or len(remaining) >= len(lines) * 0.7:
        return text

    parts: list[str] = []

    if remaining:
        parts.append("\n".join(remaining))

    parts.append("CONSTRAINTS:")
    for line in all_constraints:
        stripped = line.strip()
        parts.append(f"- {stripped}")

    return "\n".join(parts)
