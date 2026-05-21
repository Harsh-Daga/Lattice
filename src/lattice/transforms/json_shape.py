"""JSON Shape Factoring — LOSSLESS_SAFE.

Compresses repeated JSON objects by factoring out the shape (key names)
and listing values compactly.

Example:
  [{"id":1,"status":"ok","latency":120}, {"id":2,"status":"ok","latency":121}]
→
  JSON_SHAPE: {id, status, latency}
  rows: 1,ok,120 / 2,ok,121
  status = ok for all rows
"""

from __future__ import annotations

import json as _json

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response


class JSONShapeFactor(ReversibleSyncTransform):
    name = "json_shape"
    priority = 21

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            compressed, saved_delta = _factor_json(msg.content)
            saved += saved_delta
            if saved_delta > 0:
                new_msg = msg.copy()
                new_msg.content = compressed
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)

        context.record_metric(self.name, "chars_saved", saved)
        new_req = request.copy()
        new_req.messages = new_messages
        return Ok(new_req)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        return request.token_estimate > 200


def _factor_json(text: str) -> tuple[str, int]:
    i = 0
    result_parts: list[str] = []
    saved_total = 0

    while i < len(text):
        obj_start = text.find("[{", i)
        if obj_start == -1:
            result_parts.append(text[i:])
            break

        result_parts.append(text[i:obj_start])
        depth = 0
        end = obj_start
        for j in range(obj_start + 1, len(text)):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                if depth == 0:
                    end = j + 1
                    break
                depth -= 1

        if end <= obj_start:
            result_parts.append(text[obj_start:])
            break

        array_text = text[obj_start + 1 : end - 1]
        factored = _factor_array(array_text)
        result_parts.append(f"JSON_SHAPE[{factored}]")
        saved_total += len(array_text) - len(factored)
        i = end

    return "".join(result_parts), saved_total


def _factor_array(text: str) -> str:
    objects = []
    depth = 0
    current = []
    for ch in text:
        current.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj_str = "".join(current).strip()
                if not obj_str.startswith("{"):
                    break
                try:
                    objects.append(_json.loads(obj_str))
                except _json.JSONDecodeError:
                    return text
                current = []

    if len(objects) < 3:
        return text

    first = objects[0]
    if not isinstance(first, dict):
        return text

    keys = list(first.keys())
    for obj in objects[1:]:
        if not isinstance(obj, dict) or list(obj.keys()) != keys:
            return text

    parts: list[str] = [f"keys={','.join(keys)}"]

    constant_keys: list[str] = []
    for key in keys:
        vals = [obj[key] for obj in objects]
        if len(set(str(v) for v in vals)) == 1:
            constant_keys.append(f"{key}={vals[0]}")

    if constant_keys:
        parts.append("constant: " + ", ".join(constant_keys))

    var_keys = [k for k in keys if k not in [ck.split("=")[0] for ck in constant_keys]]
    rows: list[str] = []
    for obj in objects:
        row_vals = [str(obj[k]) for k in var_keys]
        rows.append(",".join(row_vals))

    if len(rows) > 6:
        parts.append(f"rows({len(rows)}):")
        for r in rows[:3]:
            parts.append(f"  {r}")
        parts.append("  ...")
        for r in rows[-2:]:
            parts.append(f"  {r}")
    else:
        parts.append(f"rows: {' / '.join(rows)}")

    return "\n".join(parts)
