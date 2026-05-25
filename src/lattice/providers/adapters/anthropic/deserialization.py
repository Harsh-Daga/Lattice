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

import json
from typing import Any

from lattice.providers.adapters.anthropic.context import _ctx_tool_id_mapping
from lattice.providers.tool_sanitizer import (
    restore_tool_call_ids,
)
from lattice.transport.types import Response


class AnthropicDeserializeMixin:
    """Anthropic adapter segment."""

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """Anthropic JSON → internal Response."""
        content_blocks = data.get("content", [])
        content = self._extract_content_from_blocks(content_blocks)
        thinking = self._extract_thinking_from_blocks(content_blocks)
        usage = self._extract_usage(data)
        tool_calls = self._extract_tool_calls(content_blocks)

        # Restore original tool IDs
        mapping = _ctx_tool_id_mapping.get()
        if mapping and tool_calls:
            tool_calls = restore_tool_call_ids(tool_calls, mapping)

        resp = Response(
            content=content,
            role=data.get("role", "assistant"),
            model=data.get("model", ""),
            usage=usage,
            finish_reason=self._map_finish_reason(data.get("stop_reason")),
            tool_calls=tool_calls or None,
        )
        if thinking:
            resp.metadata["thinking"] = thinking
        return resp

    @staticmethod
    def _extract_content_from_blocks(content: list[dict[str, Any]]) -> str:
        """Concatenate only ``text`` blocks; skip ``tool_use`` and ``thinking``."""
        parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _extract_thinking_from_blocks(content: list[dict[str, Any]]) -> str:
        """Extract thinking content from blocks for metadata preservation."""
        parts: list[str] = []
        for block in content:
            if block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
            elif block.get("type") == "redacted_thinking":
                parts.append("[redacted_thinking]")
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
        u = data.get("usage", {})
        prompt_tokens = u.get("input_tokens", u.get("prompt_tokens", 0))
        completion_tokens = u.get("output_tokens", u.get("completion_tokens", 0))
        cache_read = u.get("cache_read_input_tokens", 0)
        cache_create = u.get("cache_creation_input_tokens", 0)
        usage: dict[str, int] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read
            usage["cached_tokens"] = cache_read
        if cache_create:
            usage["cache_creation_input_tokens"] = cache_create
        return usage

    @staticmethod
    def _extract_tool_calls(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract ``tool_use`` blocks as OpenAI-style tool_calls."""
        calls: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "tool_use":
                args = block.get("input", {})
                calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    }
                )
        return calls

    @staticmethod
    def _map_finish_reason(reason: str | None) -> str | None:
        if reason is None:
            return None
        mappings = {
            "tool_use": "tool_calls",
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        return mappings.get(reason, reason)

    # ------------------------------------------------------------------
    # Streaming normalization
    # ------------------------------------------------------------------

