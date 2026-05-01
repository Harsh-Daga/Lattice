"""Tests for lattice.providers.stream_state."""

from __future__ import annotations

from lattice.providers.stream_state import (
    AnthropicStreamState,
    StreamingToolCall,
)

# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_defaults(self) -> None:
        sm = AnthropicStreamState(model="claude-3-sonnet")
        assert sm.model == "claude-3-sonnet"
        assert sm._block_type is None
        assert sm._inside_thinking is False
        assert sm._current_tool_index == -1
        assert sm._tool_calls == []
        assert sm._stop_reason is None


# =============================================================================
# Thinking blocks (must be completely skipped)
# =============================================================================


class TestThinkingSkipped:
    def test_thinking_delta(self) -> None:
        sm = AnthropicStreamState(model="claude")
        # Start thinking block
        r1 = sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            }
        )
        assert r1.chunks == []
        assert sm._inside_thinking is True

        # Thinking delta -- must be swallowed
        r2 = sm.process(
            {
                "type": "content_block_delta",
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "I should reason about this...",
                },
            }
        )
        assert r2.chunks == []

        # Stop thinking
        r3 = sm.process({"type": "content_block_stop"})
        assert r3.chunks == []
        assert sm._inside_thinking is False

    def test_text_delta_after_thinking(self) -> None:
        sm = AnthropicStreamState(model="claude")
        # thinking block
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            }
        )
        sm.process({"type": "content_block_stop"})

        # text block
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "text"},
            }
        )
        r = sm.process(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        )
        assert len(r.chunks) == 1
        assert r.chunks[0]["choices"][0]["delta"]["content"] == "Hello"


# =============================================================================
# Text streaming
# =============================================================================


class TestTextStreaming:
    def test_single_text_delta(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "text"},
            }
        )
        r = sm.process(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        )
        assert len(r.chunks) == 1
        delta = r.chunks[0]["choices"][0]["delta"]
        assert delta["content"] == "Hello"
        assert r.chunks[0]["choices"][0]["finish_reason"] is None

    def test_empty_text_delta(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "text"},
            }
        )
        r = sm.process(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": ""},
            }
        )
        assert r.chunks == []

    def test_message_stop(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm._stop_reason = "end_turn"
        r = sm.process({"type": "message_stop"})
        assert r.done is True
        assert r.finish_reason == "stop"
        assert len(r.chunks) == 1
        assert r.chunks[0]["choices"][0]["finish_reason"] == "stop"


# =============================================================================
# Tool_use streaming
# =============================================================================


class TestToolUseStreaming:
    def test_tool_use_start(self) -> None:
        sm = AnthropicStreamState(model="claude")
        r = sm.process(
            {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": "tool_abc123",
                    "name": "get_weather",
                },
            }
        )
        assert len(r.chunks) == 1
        delta = r.chunks[0]["choices"][0]["delta"]
        tc = delta["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["id"] == "tool_abc123"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == ""
        assert sm._current_tool_index == 0
        assert sm._tool_started is True

    def test_input_json_delta(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": "tool_abc",
                    "name": "calc",
                },
            }
        )
        r = sm.process(
            {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"a": 1',
                },
            }
        )
        assert len(r.chunks) == 1
        delta = r.chunks[0]["choices"][0]["delta"]
        tc = delta["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["function"]["arguments"] == '{"a": 1'

        # Second fragment
        r2 = sm.process(
            {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": ', "b": 2}',
                },
            }
        )
        assert (
            r2.chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == ', "b": 2}'
        )

    def test_tool_use_stop(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "x", "name": "y"},
            }
        )
        r = sm.process({"type": "content_block_stop"})
        assert r.chunks == []
        assert sm._block_type is None

    def test_multiple_tools(self) -> None:
        sm = AnthropicStreamState(model="claude")
        # Tool 1
        sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "t1", "name": "a"},
            }
        )
        sm.process({"type": "content_block_stop"})
        # Tool 2
        r = sm.process(
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "t2", "name": "b"},
            }
        )
        assert r.chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
        assert sm._current_tool_index == 1


# =============================================================================
# Finish reason mapping
# =============================================================================


class TestFinishReasonMapping:
    def test_end_turn(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm._stop_reason = "end_turn"
        r = sm.process({"type": "message_stop"})
        assert r.finish_reason == "stop"

    def test_tool_use(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm._stop_reason = "tool_use"
        sm._tool_started = True
        r = sm.process({"type": "message_stop"})
        assert r.finish_reason == "tool_calls"

    def test_max_tokens(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm._stop_reason = "max_tokens"
        r = sm.process({"type": "message_stop"})
        assert r.finish_reason == "length"

    def test_stop_sequence(self) -> None:
        sm = AnthropicStreamState(model="claude")
        sm._stop_reason = "stop_sequence"
        r = sm.process({"type": "message_stop"})
        assert r.finish_reason == "stop"

    def test_unknown(self) -> None:
        sm = AnthropicStreamState(model="claude")
        r = sm.process({"type": "message_stop"})
        assert r.finish_reason == "stop"


# =============================================================================
# Ignored events
# =============================================================================


class TestIgnoredEvents:
    def test_ping(self) -> None:
        sm = AnthropicStreamState(model="claude")
        assert sm.process({"type": "ping"}).chunks == []

    def test_message_start(self) -> None:
        sm = AnthropicStreamState(model="claude")
        assert sm.process({"type": "message_start"}).chunks == []

    def test_unknown_type(self) -> None:
        sm = AnthropicStreamState(model="claude")
        assert sm.process({"type": "weird"}).chunks == []


# =============================================================================
# StreamingToolCall dataclass
# =============================================================================


class TestStreamingToolCall:
    def test_defaults(self) -> None:
        tc = StreamingToolCall(index=0)
        assert tc.id == ""
        assert tc.name == ""
        assert tc.arguments == ""
        assert tc.done is False
