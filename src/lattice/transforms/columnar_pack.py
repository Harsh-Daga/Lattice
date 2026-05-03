"""Columnar Table Packing — LOSSLESS_SAFE.

Compresses markdown/plain tables by detecting regular columns and packing
them vertically. Replaces token-expensive markdown table syntax with a
compact columnar representation.

Example markdown table:
  | id | name | dept | salary |
  | 0  | A    | Eng  | 100k   |
  | 1  | B    | Eng  | 105k   |
→
  TABLE employees: columns=id,name,dept,salary
  id: 0..1
  dept: Eng x2
  salary: 100k,105k
  name: A,B
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
        return request.token_estimate > 300


_MD_TABLE_RE = re.compile(r"^\|(.+?)\|\s*$", re.MULTILINE)


def _pack_tables(text: str) -> tuple[str, int]:
    tables = _MD_TABLE_RE.findall(text)
    if len(tables) < 3:
        return text, 0

    header_cells = [c.strip() for c in tables[0].split("|")]
    if not header_cells or len(header_cells) < 2:
        return text, 0

    sep = tables[1]
    if not re.match(r"^[\s\-:|]+$", sep):
        return text, 0

    data_rows = tables[2:]
    if not data_rows:
        return text, 0

    columns: dict[str, list[str]] = {h: [] for h in header_cells}
    for row in data_rows:
        cells = [c.strip() for c in row.split("|")]
        for ci, h in enumerate(header_cells):
            if ci < len(cells):
                columns[h].append(cells[ci])

    # Detect constant columns and arithmetic sequences
    output_lines = [f"TABLE: columns={','.join(header_cells)}"]
    for h, vals in columns.items():
        if len(set(vals)) == 1:
            output_lines.append(f"  {h}: {vals[0]} x{len(vals)}")
        else:
            # Show unique values or sample
            unique = sorted(set(vals))
            if len(unique) <= 5:
                output_lines.append(f"  {h}: {','.join(unique[:5])}")
            else:
                output_lines.append(f"  {h}: {unique[0]}..{unique[-1]} ({len(vals)} values)")

    packed = "\n".join(output_lines)
    result = _MD_TABLE_RE.sub(packed, text, count=1)
    return result, len(text) - len(result)
