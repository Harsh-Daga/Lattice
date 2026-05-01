"""Anthropic streaming state machine.

Normalises Anthropic SSE events to OpenAI ``chat.completion.chunk`` deltas
while tracking complex state such as:

- ``thinking`` blocks (skipped entirely from output)
- ``tool_use`` blocks (turned into OpenAI ``tool_calls`` deltas)
- ``text_delta`` blocks (normal content)
- ``finish_reason`` resolution (``tool_use`` → ``tool_calls``, ``end_turn`` → ``stop``)

This is a **pure, allocation-light** state machine — it does no I/O.
The caller feeds it one parsed JSON chunk at a time.

References
----------
- FreeRouter provider.ts:421-524 — real implementation tested with
  Claude Code production traffic.
- Anthropic Messages API docs — ``content_block_start``,
  ``content_block_delta``, ``content_block_stop``, ``message_delta``,
  ``message_stop``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ============================================================================
# State types
# ============================================================================

BlockType = Literal["text", "thinking", "tool_use", None]


@dataclass(slots=True)
class StreamingToolCall:
    """Partially-built tool call emitted as OpenAI delta."""

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""
    done: bool = False


# ============================================================================
# Result types
# ============================================================================

@dataclass(frozen=True, slots=True)
class StreamResult:
    """What the caller should emit after processing one chunk."""

    # None → no client-visible event (internal state change)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    # True when the stream is completely finished
    done: bool = False
    # OpenAI-style finish_reason
    finish_reason: str | None = None
    # Provider-specific metadata accumulated during the stream (e.g. thinking)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnthropicStreamState:
    """Tracks Anthropic SSE events and produces OpenAI-style deltas.

    Usage
    -----
    .. code-block:: python

        sm = AnthropicStreamState(model="claude-sonnet-4")
        for event_json in sse_events:
            result = sm.process(event_json)
            for chunk in result.chunks:
                yield chunk
            if result.done:
                break
    """

    model: str
    # --- State ---
    _block_type: BlockType = None
    _inside_thinking: bool = False
    _thinking_buffer: str = ""
    _current_tool_index: int = -1
    _tool_calls: list[StreamingToolCall] = field(default_factory=list)
    _stop_reason: str | None = None
    # Tracking whether we have emitted *any* tool_call start
    _tool_started: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, event: dict[str, Any]) -> StreamResult:
        """Process one Anthropic event and return what to emit.

        Parameters
        ----------
        event:
            The JSON payload of a ``data:`` line (already parsed).

        Returns
        -------
        StreamResult with zero or more OpenAI-compatible chunks,
        and an optional done flag.
        """
        etype = event.get("type", "")

        if etype == "content_block_start":
            return self._on_content_block_start(event)
        if etype == "content_block_delta":
            return self._on_content_block_delta(event)
        if etype == "content_block_stop":
            return self._on_content_block_stop(event)
        if etype == "message_delta":
            return self._on_message_delta(event)
        if etype == "message_stop":
            return self._on_message_stop(event)

        # Ignored: "ping", "message_start", "message_delta" (already handled above)
        return StreamResult()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_content_block_start(self, event: dict[str, Any]) -> StreamResult:
        block = event.get("content_block", {})
        btype: BlockType = block.get("type")
        self._block_type = btype

        if btype == "thinking":
            self._inside_thinking = True
            self._thinking_buffer = ""
            return StreamResult()

        if btype == "tool_use":
            self._inside_thinking = False
            self._current_tool_index += 1
            self._tool_started = True
            tc = StreamingToolCall(
                index=self._current_tool_index,
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments="",
            )
            self._tool_calls.append(tc)
            chunk = self._make_chunk(
                tool_calls=[
                    {
                        "index": tc.index,
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": "",
                        },
                    }
                ]
            )
            return StreamResult(chunks=[chunk])

        # Default: text, "redacted_thinking", etc
        self._inside_thinking = False
        return StreamResult()

    def _on_content_block_delta(self, event: dict[str, Any]) -> StreamResult:
        delta = event.get("delta", {})
        dtype = delta.get("type")

        # Accumulate thinking text (not emitted to preserve privacy by default)
        if self._inside_thinking:
            if dtype == "thinking_delta":
                self._thinking_buffer += delta.get("thinking", "")
            return StreamResult()

        # Streamed tool argument build-up
        if self._block_type == "tool_use" and dtype == "input_json_delta":
            partial = delta.get("partial_json", "")
            tc = self._tool_calls[self._current_tool_index]
            tc.arguments += partial
            chunk = self._make_chunk(
                tool_calls=[
                    {
                        "index": tc.index,
                        "function": {
                            "arguments": partial,
                        },
                    }
                ]
            )
            return StreamResult(chunks=[chunk])

        # Regular text delta
        if dtype == "text_delta":
            text = delta.get("text", "")
            if text:
                return StreamResult(chunks=[self._make_chunk({"content": text})])
            return StreamResult()

        # Thinking delta — skip even outside "thinking" block
        if dtype == "thinking_delta":
            return StreamResult()

        # Signature / json / citation deltas — skip or handle later
        return StreamResult()

    def _on_content_block_stop(self, _event: dict[str, Any]) -> StreamResult:
        if self._block_type == "thinking":
            self._inside_thinking = False
        self._block_type = None
        return StreamResult()

    def _on_message_delta(self, event: dict[str, Any]) -> StreamResult:
        stop_reason = event.get("delta", {}).get("stop_reason")
        if stop_reason:
            self._stop_reason = stop_reason
        # Capture usage if present in message_delta
        usage = event.get("usage", {})
        meta: dict[str, Any] = {}
        if usage:
            meta["usage"] = self._normalize_usage(usage)
        return StreamResult(metadata=meta)

    def _on_message_stop(self, event: dict[str, Any]) -> StreamResult:
        finish = self._resolve_finish_reason()
        chunk = self._make_chunk(delta={}, finish_reason=finish)
        meta: dict[str, Any] = {}
        if self._thinking_buffer:
            meta["thinking"] = self._thinking_buffer
        # Capture usage if present in message_stop
        usage = event.get("usage", {})
        if usage:
            meta["usage"] = self._normalize_usage(usage)
        return StreamResult(chunks=[chunk], done=True, finish_reason=finish, metadata=meta)

    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
        prompt = usage.get("input_tokens", 0)
        completion = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        out: dict[str, int] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if cache_read:
            out["cache_read_input_tokens"] = cache_read
            out["cached_tokens"] = cache_read
        if cache_create:
            out["cache_creation_input_tokens"] = cache_create
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        delta: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible delta chunk."""
        d: dict[str, Any] = {}
        if delta:
            d.update(delta)
        if tool_calls:
            d["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{id(self):x}",
            "object": "chat.completion.chunk",
            "created": 0,  # caller can fill in
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": d,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _resolve_finish_reason(self) -> str | None:
        """Map Anthropic stop_reason to OpenAI finish_reason."""
        sr = self._stop_reason
        if sr == "tool_use":
            return "tool_calls"
        if sr == "end_turn":
            return "stop"
        if sr == "max_tokens":
            return "length"
        if sr == "stop_sequence":
            return "stop"
        # Fallbacks
        if self._tool_started:
            return "tool_calls"
        return sr or "stop"
