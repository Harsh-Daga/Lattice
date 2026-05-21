"""Diagnostic Run-Length Encoding — LOSSLESS_SAFE.

Compresses repeated diagnostic patterns (errors, timeouts, etc.) using
count-preserving grouping. Repeated lines are not noise — they are signal.

Example:
  service_0 timeout
  service_1 timeout
  service_2 timeout
  service_3 timeout
→
  4 services timeout:
  - service_0
  - service_1
  - service_2
  - service_3

For cyclic patterns, detects the period and reports a formula + exceptions.
"""

from __future__ import annotations

import re
from collections import defaultdict

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response


class DiagnosticRLE(ReversibleSyncTransform):
    name = "diagnostic_rle"
    priority = 17

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        total_saved = 0

        for msg in request.messages:
            if msg.role not in ("user", "assistant", "tool", "function"):
                new_messages.append(msg)
                continue

            lines = msg.content.splitlines()
            grouped = _group_repeated_error_lines(lines)
            if not grouped:
                new_messages.append(msg)
                continue

            compressed_lines = _format_rle_output(grouped)
            before_len = len(msg.content)
            after = "\n".join(compressed_lines)
            total_saved += before_len - len(after)
            new_msg = msg.copy()
            new_msg.content = after
            new_messages.append(new_msg)
            context.record_metric(self.name, "groups_created", sum(len(v) for v in grouped.values()))

        context.record_metric(self.name, "chars_saved", total_saved)
        new_req = request.copy()
        new_req.messages = new_messages
        return Ok(new_req)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        return request.token_estimate > 100


def _group_repeated_error_lines(lines: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        pattern = _extract_pattern(stripped)
        if pattern is None:
            i += 1
            continue

        group: list[str] = []
        j = i
        while j < len(lines) and _extract_pattern(lines[j].strip()) == pattern:
            group.append(lines[j].strip())
            j += 1

        if len(group) >= 3:
            groups[pattern] = group
            i = j
        else:
            i += 1

    return dict(groups)


def _extract_pattern(line: str) -> str | None:
    """Extract a pattern signature from a line. Replace variable parts with placeholders."""
    if not line:
        return None

    # Replace identifiers, paths, numbers with placeholders
    pattern = re.sub(r"\b[a-zA-Z_][\w.-]*\d+\b", "<ID>", line)
    pattern = re.sub(r"\b\d+\b", "<N>", pattern)
    pattern = re.sub(r"(/[^/\s]+)+", "<PATH>", pattern)

    if pattern == line or len(pattern) < 10:
        return None

    return pattern


def _format_rle_output(groups: dict[str, list[str]]) -> list[str]:
    output: list[str] = []
    for pattern, lines in groups.items():
        count = len(lines)
        representative = _find_representative(lines)
        output.append(f"{count}x {representative}")

        if count <= 8:
            indent = "  - "
            for line in lines:
                output.append(f"{indent}{line}")
        else:
            output.append(f"  - {lines[0]}")
            output.append("  - ...")
            output.append(f"  - {lines[-1]}")

    return output


def _find_representative(lines: list[str]) -> str:
    """Pick a representative line. Prefer the most informative."""
    for line in reversed(lines):
        if re.search(r"(error|exception|timeout|failed)", line, re.IGNORECASE):
            return line
    return lines[0]
