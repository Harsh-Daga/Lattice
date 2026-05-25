"""Context-local tool-ID mapping for Anthropic adapter (async-safe)."""

from __future__ import annotations

import contextvars

# Maps original tool ID ↔ sanitised tool ID per in-flight request.
_ctx_tool_id_mapping: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_anthropic_tool_id_mapping", default=None
)
