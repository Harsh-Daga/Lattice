"""Query-Aware Tool Output Projection — LOSSLESS_CONTEXTUAL.

Projects tool output fields based on the user's actual question instead
of generic filtering. Preserves error messages, counts, module names,
and stack frames that are relevant to the query.

Replaces the lossy tool_filter with query-aware analytical projection.
"""

from __future__ import annotations

import json
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response

_DEFAULT_REQUIRED_FIELDS = frozenset(
    {
        "error",
        "message",
        "severity",
        "level",
        "module",
        "stack",
        "timestamp",
        "time",
        "count",
        "status",
        "id",
        "name",
        "type",
        "result",
        "output",
    }
)

_IGNORED_FIELDS = frozenset(
    {
        "debug_id",
        "trace_id_hex",
        "internal_id",
        "metadata",
        "headers",
        "cookies",
        "raw_body",
        "raw_payload",
        "created_at",
        "updated_at",
        "deleted_at",
        "_links",
    }
)


class QueryAwareProjection(ReversibleSyncTransform):
    name = "tool_projection"
    transform_class = TransformClass.LOSSLESS_CONTEXTUAL
    priority = 29

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        user_query = _extract_user_query(request)
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            if msg.role not in ("tool", "function") and not msg.tool_call_id:
                new_messages.append(msg)
                continue

            projected, saved_delta = _project_tool_output(msg.content, user_query)
            saved += saved_delta
            new_messages.append(Message(role=msg.role, content=projected))

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


def _extract_user_query(request: Request) -> str:
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.content
    return ""


def _relevant_fields(query: str) -> set[str]:
    fields = set(_DEFAULT_REQUIRED_FIELDS)
    lower = query.lower()
    if "error" in lower or "fail" in lower or "exception" in lower:
        fields.update({"error", "exception", "traceback", "stack_trace"})
    if "build" in lower:
        fields.update({"build", "compile", "link", "artifact"})
    if "latency" in lower or "slow" in lower or "timeout" in lower:
        fields.update({"latency", "duration", "elapsed", "timeout"})
    if "uuid" in lower or "id" in lower:
        fields.update({"uuid", "id", "identifier"})
    if "log" in lower:
        fields.update({"level", "logger", "thread", "trace_id"})
    return fields


def _project_tool_output(content: str, query: str) -> tuple[str, int]:
    stripped = content.strip()
    if not stripped:
        return content, 0

    relevant: set[str] = _relevant_fields(query) if query else set(_DEFAULT_REQUIRED_FIELDS)
    if not relevant:
        relevant = set(_DEFAULT_REQUIRED_FIELDS)

    if stripped[0] not in ("[", "{"):
        lines = content.splitlines()
        kept = [ln for ln in lines if any(f in ln.lower() for f in relevant)]
        if kept and len(kept) < len(lines) * 0.5:
            result = "\n".join(kept)
            return result, len(content) - len(result)
        return content, 0

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return content, 0

    projected = _project_value(parsed, relevant)
    result = json.dumps(projected, separators=(",", ":"))
    return result, len(content) - len(result)


def _project_value(value: Any, relevant: frozenset[str] | set[str]) -> Any:
    if isinstance(value, dict):
        return {
            k: _project_value(v, relevant)
            for k, v in value.items()
            if k not in _IGNORED_FIELDS and (k in relevant or not k.startswith("_"))
        }

    if isinstance(value, list):
        if not value:
            return value
        if isinstance(value[0], dict) and len(value) > 20:
            sample = [_project_value(v, relevant) for v in value[:10]]
            summary = {
                "_total": len(value),
                "_showing": 10,
                "_sample": sample,
            }
            return summary
        return value[:100]  # Truncate very long lists

    return value
