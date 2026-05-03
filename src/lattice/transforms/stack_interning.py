"""Stack Trace Interning — LOSSLESS_SAFE.

Interning repeated stack trace patterns. Stack traces share massive
boilerplate; this interning preserves the causal structure while saving
tokens.

Example:
  Traceback...
  File app.py:10 -> db.py:22 -> client.py:44
  ... (same pattern repeats)
→
  STACK_PATTERNS:
  S1: app.py:10 -> db.py:22 -> client.py:44
  ERRORS:
  E1 uses S1: ModuleNotFoundError
  E2 uses S1: TimeoutError
"""

from __future__ import annotations

import re
from collections import defaultdict

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response

_STACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
_JAVA_FRAME_RE = re.compile(r"at\s+(\S+)\s*\(([^)]+):(\d+)\)")


class StackTraceInterning(ReversibleSyncTransform):
    name = "stack_interning"
    priority = 26

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            compressed, saved_delta = _intern_stack_traces(msg.content)
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
        return request.token_estimate > 100 and _has_stack_traces(request)


def _has_stack_traces(request: Request) -> bool:
    combined = "\n".join(m.content for m in request.messages)
    return bool(
        re.search(r"Traceback|File\s+\".+?\",\s+line\s+\d+", combined)
        or re.search(r"at\s+\S+\s*\([^)]+:\d+\)", combined)
    )


def _intern_stack_traces(text: str) -> tuple[str, int]:
    trace_blocks = _extract_trace_blocks(text)
    if len(trace_blocks) < 2:
        return text, 0

    pattern_map: dict[str, str] = {}
    pattern_to_errors: dict[str, list[str]] = defaultdict(list)

    for block_text, error_type in trace_blocks:
        pattern = _extract_frame_pattern(block_text)
        if pattern:
            if pattern not in pattern_map:
                pattern_map[pattern] = f"S{len(pattern_map) + 1}"
            pattern_to_errors[pattern_map[pattern]].append(error_type or "unknown_error")

    if len(pattern_map) < 2:
        return text, 0

    result_lines = ["STACK_PATTERNS:"]
    for pattern, sid in sorted(pattern_map.items(), key=lambda x: x[1]):
        result_lines.append(f"  {sid}: {pattern}")
    result_lines.append("ERRORS:")
    for sid, errors in sorted(pattern_to_errors.items()):
        compact_errors = _compact_errors(errors)
        result_lines.append(f"  {compact_errors} uses {sid}")

    return "\n".join(result_lines), len(text) - len("\n".join(result_lines))


def _extract_trace_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []

    error_pattern = re.compile(
        r"((?:Traceback\s*\(.*?\):\s*\n|)"
        r"(?:File\s+\".+?\",\s+line\s+\d+.*?\n)"
        r"(?:File\s+\".+?\",\s+line\s+\d+.*?\n)*"
        r"\w+(?:Error|Exception|Warning|Fault)(?::\s*.*?)?)",
        re.DOTALL,
    )

    for match in error_pattern.finditer(text):
        block = match.group()
        error_type_match = re.search(r"(\w+(?:Error|Exception))", block)
        error_type = error_type_match.group(1) if error_type_match else "Error"
        blocks.append((block, error_type))

    return blocks


def _extract_frame_pattern(block: str) -> str:
    frames = _STACK_FRAME_RE.findall(block)
    if frames:
        short_frames = [f"{_shorten_path(f)}:{ln}" for f, ln in frames]
        return " → ".join(short_frames)

    java_frames = _JAVA_FRAME_RE.findall(block)
    if java_frames:
        short_frames = [f"{cls}:{ln}" for cls, _, ln in java_frames]
        return " → ".join(short_frames)

    return ""


def _shorten_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if len(parts) > 3:
        return ".../" + "/".join(parts[-2:])
    return "/".join(parts)


def _compact_errors(errors: list[str]) -> str:
    if len(errors) == 1:
        return errors[0]
    counter: defaultdict[str, int] = defaultdict(int)
    for e in errors:
        counter[e] += 1
    parts = [f"{v}x{e}" for e, v in counter.items()]
    return ", ".join(parts)
