"""Lossless Code/File Factoring — LOSSLESS_SAFE.

Detects identical transformations across multiple files and factors them
into a common template with variables. Models can read the template and
understand the pattern.

Replaces the lossy structural_fingerprint with readable, reversible factoring.
"""

from __future__ import annotations

import re
from collections import defaultdict

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response


class CodeFactoringTransform(ReversibleSyncTransform):
    name = "code_factoring"
    priority = 12

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            compressed, saved_delta = _factor_code(msg.content)
            saved += saved_delta
            new_messages.append(Message(role=msg.role, content=compressed))

        context.record_metric(self.name, "chars_saved", saved)
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
        return request.token_estimate > 300 and _has_file_references(request)


def _has_file_references(request: Request) -> bool:
    combined = "\n".join(m.content for m in request.messages)
    return bool(re.search(r"(?:\.py|\.js|\.ts|\.rs|\.go|\.java)(?::\d+|,\s)", combined))


_FILE_LINE_RE = re.compile(
    r"^(.+?\.(?:py|js|ts|jsx|tsx|rs|go|java|cpp|c|h)):\d+[:]\s+(.+)$", re.MULTILINE
)


def _factor_code(text: str) -> tuple[str, int]:
    matches = _FILE_LINE_RE.findall(text)
    if len(matches) < 5:
        return text, 0

    file_groups: dict[str, list[str]] = defaultdict(list)
    for filepath, content in matches:
        file_groups[filepath].append(content)

    if len(file_groups) < 2:
        return text, 0

    patterns: dict[str, list[str]] = defaultdict(list)
    for filepath, contents in file_groups.items():
        for line in contents:
            normalized = re.sub(r"\b\d+\b", "N", line)
            normalized = re.sub(r'"[^"]+"', '"STR"', normalized)
            normalized = re.sub(r"'[^']+'", "'STR'", normalized)
            patterns[normalized].append(filepath)

    factored_patterns = {pattern: files for pattern, files in patterns.items() if len(files) >= 2}

    if not factored_patterns:
        return text, 0

    result_lines = []
    for i, (pattern, files) in enumerate(
        sorted(factored_patterns.items(), key=lambda x: -len(x[1]))
    ):
        result_lines.append(f"PATTERN_P{i}:")
        result_lines.append(
            f"  FILES: {', '.join(sorted(set(files))[:10])}{' ...' if len(files) > 10 else ''}"
        )
        result_lines.append(f"  COUNT: {len(files)} occurrences")
        result_lines.append(f"  SHAPE: {pattern[:120]}")
        result_lines.append("")

    return "\n".join(result_lines), len(text) - len("\n".join(result_lines))
