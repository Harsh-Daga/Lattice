"""Query-Aware Tool Output Projection — LOSSLESS_CONTEXTUAL.

Projects tool output fields based on the user's actual question instead
of generic filtering. Preserves error messages, counts, module names,
and stack frames that are relevant to the query.

Critical invariant: if a Request contains tool output, the transformed
prompt MUST preserve a tool-output marker so the LLM knows the data is
already available and should NOT generate a new tool call.
"""

from __future__ import annotations

import json
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.ir.primitives import PromptIRV2
from lattice.transport.types import Request, Response

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

_TOOL_OUTPUT_HEADER = (
    "\n[TOOL OUTPUT PROVIDED — data is already available; do NOT re-invoke the tool]\n"
)


class QueryAwareProjection(ReversibleSyncTransform):
    name = "tool_projection"
    transform_class = TransformClass.LOSSLESS_CONTEXTUAL
    priority = 29

    # ------------------------------------------------------------------
    # IR-native optimize() — v2 path
    # ------------------------------------------------------------------

    def optimize(
        self,
        ir: PromptIRV2,
        _request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """IR-native: project tool-output spans based on query relevance."""
        user_query = _extract_user_query(_request)
        total_saved = 0
        tool_output_seen = False
        new_sections = []
        for sec in ir.sections:
            new_spans = []
            for span in sec.spans:
                if span.protected or not span.text:
                    new_spans.append(span)
                    continue
                projected, saved = _project_tool_output(span.text, user_query)
                if projected != span.text and projected.strip():
                    # Prepend header once per transformed span
                    if saved > 0 and not tool_output_seen:
                        projected = _TOOL_OUTPUT_HEADER + projected
                        tool_output_seen = True
                    elif saved > 0 and saved > len(span.text) * 0.05:
                        projected = _TOOL_OUTPUT_HEADER + projected
                    new_spans.append(span.with_text(projected))
                    total_saved += saved
                else:
                    new_spans.append(span)
            new_sections.append(sec.with_spans(tuple(new_spans)))
        context.record_metric(self.name, "chars_saved", total_saved)
        return Ok(ir.with_sections(tuple(new_sections)))

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
        return value[:100]

    return value
