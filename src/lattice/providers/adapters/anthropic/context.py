from __future__ import annotations

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


import contextvars

# =============================================================================
# Context-local storage (async-safe)
# =============================================================================

# Maps original tool ID ↔ sanitised tool ID per in-flight request.
_ctx_tool_id_mapping: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_anthropic_tool_id_mapping", default=None
)

