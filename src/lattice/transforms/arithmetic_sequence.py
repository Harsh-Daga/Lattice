"""Arithmetic Sequence Compression — LOSSLESS_SAFE.

Detects and compresses monotonic numeric sequences using formulas.
Tables/logs with regular increments (IDs, timestamps, salaries, latencies)
are compressed into arithmetic expressions that models can read.

Example:
  ID 0 salary 100000
  ID 1 salary 101000
  ...99...
→ 100 rows: id = 0..99, salary = 100000 + id * 1000
"""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response


class ArithmeticSequenceCompressor(ReversibleSyncTransform):
    name = "arithmetic_sequence"
    priority = 18

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            numbers = _extract_number_columns(msg.content)
            if not numbers:
                new_messages.append(msg)
                continue

            compressed, saved_delta = _compress_sequences(msg.content, numbers)
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
        return request.token_estimate > 200


def _extract_number_columns(text: str) -> list[list[int]]:
    lines = text.splitlines()
    columns: list[list[int]] = []
    for line in lines:
        nums = re.findall(r"\b\d+\b", line)
        if len(nums) >= 2:
            for ci, num in enumerate(nums):
                while len(columns) <= ci:
                    columns.append([])
                columns[ci].append(int(num))

    if len(columns) < 2:
        return []

    min_len = min(len(c) for c in columns)
    if min_len < 5:
        return []

    return [c[:min_len] for c in columns]


def _is_sequence(values: list[int]) -> int | None:
    """Return the common difference if values form an arithmetic sequence."""
    if len(values) < 3 or values[0] == 0:
        return None
    d = values[1] - values[0]
    for i in range(2, len(values)):
        if values[i] - values[i - 1] != d:
            return None
    return d


def _compress_sequences(text: str, columns: list[list[int]]) -> tuple[str, int]:
    sequences: list[tuple[int, int, int]] = []
    for col in columns:
        d = _is_sequence(col)
        if d is not None:
            sequences.append((col[0], col[-1], d))

    if not sequences:
        return text, 0

    result = text
    for start, end, diff in sequences:
        orig = f"{start}..{end}"
        compressed = f"values = {start} + n*{diff} for n in 0..{(end - start) // diff}"
        result = result.replace(orig, compressed, 1)

    return result, len(text) - len(result)
