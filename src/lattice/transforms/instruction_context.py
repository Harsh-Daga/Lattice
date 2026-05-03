"""Instruction/Context Separation — OBSERVABILITY_ONLY.

Separates mixed prompts into labeled sections (TASK, CONSTRAINTS, CONTEXT,
DATA, OUTPUT FORMAT). Does NOT remove or compress content — purely structural
reorganization that improves model comprehension.

This is a quality transform, not a compression transform.
"""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response

_HEADER_MARKERS = re.compile(
    r"^(#+\s+.*|Task:|Goal:|Objective:|Instructions?:|Guidelines?:|Context:|Background:"
    r"|Data:|Output Format:|Format:|Constraints?:|Rules?:|Requirements?:)",
    re.MULTILINE | re.IGNORECASE,
)

_CONSTRAINT_KEYWORDS = re.compile(
    r"\b(must|required|mandatory|shall|should not|must not|cannot|do not|never|always)\b",
    re.IGNORECASE,
)

_FORMAT_MARKERS = re.compile(
    r"json format|json output|return json|respond in json|table format|markdown|csv|yaml|code block",
    re.IGNORECASE,
)


class InstructionContextSeparator(ReversibleSyncTransform):
    name = "instruction_context_sep"
    priority = 8
    transform_class = TransformClass.OBSERVABILITY_ONLY

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        for msg in request.messages:
            content = msg.content
            if msg.role in ("user", "system"):
                separated = _separate_sections(content)
                if separated and len(separated.splitlines()) > 3:
                    new_messages.append(Message(role=msg.role, content=separated))
                else:
                    new_messages.append(msg)
            else:
                new_messages.append(msg)

        return Ok(
            Request(
                model=request.model,
                messages=new_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata,
            )
        )

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        combined = "\n".join(m.content for m in request.messages)
        return len(combined) > 200 and len(combined.splitlines()) > 5


def _separate_sections(text: str) -> str:
    """Separate mixed prompt into labeled sections."""
    if _HEADER_MARKERS.search(text):
        return text

    lines = text.splitlines()
    if len(lines) < 5:
        return text

    task_lines: list[str] = []
    context_lines: list[str] = []
    data_lines: list[str] = []
    format_lines: list[str] = []
    constraint_lines: list[str] = []

    in_data = False
    in_context = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("`") or stripped.startswith("│"):
            data_lines.append(line)
            in_data = True
            continue

        if re.search(r"\{|\}", stripped) and len(stripped) > 20:
            data_lines.append(line)
            in_data = True
            continue

        if re.match(r"^\|.*\|$", stripped):
            data_lines.append(line)
            in_data = True
            continue

        if _CONSTRAINT_KEYWORDS.search(stripped) or re.search(
            r"\b(format|output|style|length|limit)\b", stripped, re.IGNORECASE
        ):
            constraint_lines.append(line)
            continue

        if _FORMAT_MARKERS.search(stripped):
            format_lines.append(line)
            continue

        if in_context or re.search(
            r"\b(example|sample|previously|earlier|above|below)\b", stripped, re.IGNORECASE
        ):
            context_lines.append(line)
            in_context = True
            continue

        if in_data:
            data_lines.append(line)
        else:
            task_lines.append(line)

    sections: list[str] = []

    if task_lines:
        sections.append("TASK:\n" + "\n".join(task_lines))

    if context_lines:
        sections.append("\nCONTEXT:\n" + "\n".join(context_lines))

    if data_lines:
        sections.append("\nDATA:\n" + "\n".join(data_lines))

    if constraint_lines:
        sections.append("\nCONSTRAINTS:\n" + "\n".join(constraint_lines))

    if format_lines:
        sections.append("\nOUTPUT FORMAT:\n" + "\n".join(format_lines))

    if len(sections) > 1:
        return "\n".join(sections)

    return text
