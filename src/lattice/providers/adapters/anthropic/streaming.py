"""Anthropic adapter — full Messages API support.

Implements every nuance discovered in FreeRouter ``provider.ts`` and
LiteLLM ``anthropic/chat/transformation.py``:

1. **OAuth mode** — detects ``sk-ant-oat`` tokens and sends the full
   Claude Code header stack (``Authorization: Bearer``, ``anthropic-beta``,
   ``user-agent``, etc.).
2. **System prompt blocks** — supports ``cache_control`` on system blocks
   (required for Claude Code OAuth) and plain-string fallback.
3. **Tool ID sanitisation** — validates ``^[a-zA-Z0-9_-]+$`` before sending;
   bidirectional mapping so response tool IDs are faithfully restored.
4. **Tool result merging** — consecutive ``role: tool`` messages collapse
   into a single user message with an array of ``tool_result`` blocks.
5. **Tool argument parsing** — explicitly ``json.loads`` string arguments
   (OpenAI tool_calls often have ``arguments`` as a JSON string).
6. **Thinking config** — supports ``adaptive`` (Opus 4.6+) and ``enabled``
   with ``budget_tokens``; temperature is **omitted** when thinking is on.
7. **max_tokens arithmetic** — when thinking is enabled, budget is added
   to the user's ``max_tokens`` so Claude can both think and answer.
 8. **Streaming** — full state machine via
    :mod:`AnthropicStreamState <lattice.providers.stream_state>`:
   text deltas, tool_use start + argument streaming, and **thinking skipped**.
9. **Finish reason mapping** — ``tool_use`` → ``tool_calls``,
   ``end_turn`` → ``stop``, ``max_tokens`` → ``length``.

Thread Safety
-------------
Tool-ID mappings are stored in a :class:`contextvars.ContextVar` so
concurrent async requests never collide.

References
----------
- LiteLLM ``litellm/llms/anthropic/chat/transformation.py`` (~2100 LOC).
- Anthropic Messages API docs (2024-06-01 version).
"""

from __future__ import annotations

from typing import Any

from lattice.providers.adapters.anthropic.deserialization import AnthropicDeserializeMixin
from lattice.providers.stream_state import AnthropicStreamState


class AnthropicStreamMixin:
    """Anthropic adapter segment."""

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize Anthropic SSE to OpenAI delta format.

        **Deprecated for streaming tool_use** — use
        :meth:`normalize_sse_stream` to get a stateful machine.
        This method handles text-only streams correctly.
        """
        etype = chunk.get("type", "")
        delta_text = ""
        finish: str | None = None

        if etype == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                delta_text = delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                # skip thinking in legacy path
                return None
        elif etype == "message_delta":
            stop_reason = chunk.get("delta", {}).get("stop_reason")
            if stop_reason:
                self._stop_reason = stop_reason
                finish = AnthropicDeserializeMixin._map_finish_reason(stop_reason)
        elif etype == "message_stop":
            finish = AnthropicDeserializeMixin._map_finish_reason(self._stop_reason or "stop")
            if finish:
                return {"choices": [{"delta": {}, "finish_reason": finish}]}
            return None
        else:
            return None

        if not delta_text and finish is None:
            return None

        return {
            "choices": [
                {
                    "delta": {"content": delta_text} if delta_text else {},
                    "finish_reason": finish,
                }
            ]
        }

    def normalize_sse_stream(self, model: str) -> AnthropicStreamState:
        """Return a fresh stream state machine for Anthropic SSE chunks."""
        return AnthropicStreamState(model=model)

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def extract_content(self, msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, list):
            return AnthropicDeserializeMixin._extract_content_from_blocks(content)
        return str(content) if content else ""
