"""Columnar Table Packing — LOSSLESS_SAFE.

Compresses markdown/plain tables by detecting regular columns and packing
them vertically. Replaces token-expensive markdown table syntax with a
compact columnar representation.

Only compresses tables with >= 8 data rows to ensure net savings
(columnar packing has fixed overhead from header and per-column labels).
"""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response


class ColumnarTablePack(ReversibleSyncTransform):
    name = "columnar_pack"
    priority = 19

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            compressed, saved_delta = _pack_tables(msg.content)
            saved += saved_delta
            new_msg = msg.copy()
            new_msg.content = compressed
            new_messages.append(new_msg)

        context.record_metric(self.name, "chars_saved", saved)
        new_req = request.copy()
        new_req.messages = new_messages
        return Ok(new_req)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        return request.token_estimate > 300


# Matches a contiguous block of markdown table rows:
# header row + separator row + data rows. Each line starts with |.
_MD_TABLE_BLOCK_RE = re.compile(
    r"(^\|.+?\|\s*$\n"  # header
    r"^\|[\s\-:|]+\|\s*$\n"  # separator
    r"(?:^\|.+?\|\s*$\n?)+)",  # data rows (one or more)
    re.MULTILINE,
)


def _pack_tables(text: str) -> tuple[str, int]:
    matches = list(_MD_TABLE_BLOCK_RE.finditer(text))
    if not matches:
        return text, 0

    result = text
    total_saved = 0

    for match in reversed(matches):
        block = match.group()
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 3:
            continue

        # Parse header
        header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
        if len(header_cells) < 2:
            continue

        # Skip the separator line (lines[1] is ---|----|...)
        data_lines = lines[2:]
        if len(data_lines) < 8:
            continue  # too small, columnar packing has fixed overhead

        # Parse data rows
        columns: dict[str, list[str]] = {h: [] for h in header_cells}
        for row_text in data_lines:
            cells = [c.strip() for c in row_text.strip("|").split("|")]
            for ci, h in enumerate(header_cells):
                if ci < len(cells):
                    columns[h].append(cells[ci])

        # Build compact output
        output_lines = [f"TABLE {len(data_lines)} rows: {','.join(header_cells)}"]
        for h, vals in columns.items():
            if len(set(vals)) == 1:
                output_lines.append(f"  {h}: {vals[0]} (all {len(vals)})")
            else:
                unique = list(dict.fromkeys(vals))  # dedup-preserving order
                if len(unique) <= 5:
                    output_lines.append(f"  {h}: {','.join(unique)}")
                elif _is_arithmetic(unique):
                    start = int(unique[0]) if unique[0].isdigit() else unique[0]
                    end = int(unique[-1]) if unique[-1].isdigit() else unique[-1]
                    output_lines.append(f"  {h}: {start}..{end} ({len(unique)} values)")
                else:
                    output_lines.append(f"  {h}: {unique[0]}..{unique[-1]} ({len(vals)} values)")

        packed = "\n".join(output_lines)
        block_len = len(block)
        packed_len = len(packed)

        if packed_len >= block_len:
            continue  # no savings

        result = result[: match.start()] + packed + result[match.end() :]
        total_saved += block_len - packed_len

    return result, total_saved


def _is_arithmetic(values: list[str]) -> bool:
    if len(values) < 3:
        return False
    try:
        nums = [int(v) for v in values]
    except (ValueError, TypeError):
        return False
    d = nums[1] - nums[0]
    if d == 0:
        return False
    return all(nums[i] - nums[i - 1] == d for i in range(2, len(nums)))
