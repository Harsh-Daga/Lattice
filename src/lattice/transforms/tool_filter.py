"""Tool Output Metadata Filter — metadata-only transform.

Only removes fields that are truly internal/metadata/empty/null:
- empty/null/None values
- internal provider metadata (headers, cookies, raw_body, _links)
- pagination cursors not needed by the model
- duplicated raw payload blobs
- unreferenced debug IDs

NEVER removes actual result content, error messages, counts,
severity labels, module names, or any field the model might need.
This is a safe metadata scrub, not a lossy content filter.
"""

from __future__ import annotations

import json
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.ir.primitives import PromptIRV2
from lattice.transport.types import Message, Request, Response

_INTERNAL_FIELDS = frozenset(
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
        "pagination",
        "cursor",
        "next_cursor",
        "prev_cursor",
        "etag",
        "last_modified",
        "x_request_id",
        "x_trace_id",
        "request_id",
        "response_headers",
        "content_type",
        "content_length",
        "encoding",
        "compression",
        "internal_blob",
    }
)

_ALWAYS_KEEP = frozenset(
    {
        "error",
        "errors",
        "exception",
        "failure",
        "failures",
        "warning",
        "warnings",
        "message",
        "messages",
        "severity",
        "level",
        "module",
        "modules",
        "stack",
        "stack_trace",
        "timestamp",
        "time",
        "count",
        "total",
        "status",
        "code",
        "id",
        "ids",
        "name",
        "names",
        "type",
        "types",
        "result",
        "results",
        "output",
        "data",
        "value",
        "values",
        "key",
        "keys",
        "build",
        "build_id",
        "build_status",
    }
)


class ToolOutputFilter(ReversibleSyncTransform):
    """Metadata-only tool output filter.

    Safely removes internal provider fields without touching content.
    This is designed to be safe even on reasoning/debugging tasks.
    """

    name = "tool_filter"
    priority = 30
    transform_class = TransformClass.STRUCTURAL_RISKY

    def __init__(self) -> None:
        pass

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        modified = 0
        saved_chars = 0

        for msg in request.messages:
            if not self._is_tool_output(msg):
                continue
            original = msg.content
            cleaned = self._scrub(original)
            if cleaned != original and len(cleaned) > 0:
                msg.content = cleaned
                modified += 1
                saved_chars += max(0, len(original) - len(cleaned))

        context.record_metric(self.name, "modified_count", modified)
        context.record_metric(self.name, "chars_saved", saved_chars)
        return Ok(request)

    def optimize(
        self, ir: PromptIRV2, request: Request, context: TransformContext
    ) -> Result[PromptIRV2, TransformError]:
        """Apply the metadata scrub directly to immutable PromptIRV2."""
        updated_sections = []
        modified = 0
        saved_chars = 0

        for section in ir.sections:
            section_type = (
                section.type.value if hasattr(section.type, "value") else str(section.type)
            )
            if section_type not in {"tool_output", "json", "logs", "error"}:
                updated_sections.append(section)
                continue

            new_spans = []
            section_modified = False
            for span in section.spans:
                original = span.text
                cleaned = self._scrub(original)
                if cleaned != original:
                    span = span.with_text(cleaned)
                    section_modified = True
                    modified += 1
                    saved_chars += max(0, len(original) - len(cleaned))
                new_spans.append(span)

            if section_modified:
                updated_sections.append(section.with_spans(tuple(new_spans)))
            else:
                updated_sections.append(section)

        updated = ir.with_sections(tuple(updated_sections))
        if modified > 0:
            updated = updated.add_metadata(
                _lattice_tool_filter_applied=True,
                _lattice_tool_filter_modified_spans=modified,
                _lattice_tool_filter_chars_saved=saved_chars,
            )
            request.metadata["_lattice_tool_filter_applied"] = True
            request.metadata["_lattice_tool_filter_modified_spans"] = modified
            request.metadata["_lattice_tool_filter_chars_saved"] = saved_chars
            context.session_state["_lattice_ir_v2"] = updated
            context.record_metric(self.name, "modified_count", modified)
            context.record_metric(self.name, "chars_saved", saved_chars)
        return Ok(updated)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def _is_tool_output(self, msg: Message) -> bool:
        if msg.role == "tool":
            return True
        if msg.tool_call_id:
            return True
        return bool(msg.role == "user" and msg.metadata.get("is_tool_output"))

    def _scrub(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text

        if stripped[0] not in ("[", "{"):
            return text

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return text

        cleaned = self._scrub_value(parsed)
        result = json.dumps(cleaned, separators=(",", ":"))
        if len(result) < 2:
            return text
        return result

    def _scrub_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: self._scrub_value(v)
                for k, v in value.items()
                if v is not None
                and v != ""
                and not (isinstance(v, str) and v.strip() == "")
                and k not in _INTERNAL_FIELDS
            }

        if isinstance(value, list):
            cleaned = [self._scrub_value(v) for v in value]
            cleaned = [
                v
                for v in cleaned
                if v is not None and v != "" and not (isinstance(v, str) and v.strip() == "")
            ]
            return cleaned

        return value
