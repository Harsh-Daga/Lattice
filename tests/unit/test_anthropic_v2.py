"""Tests for lattice.providers.anthropic (full rewrite).

Covers every P0 gap identified in the gap analysis.
"""

from __future__ import annotations

import pytest

from lattice.core.transport import Message, Request
from lattice.providers.anthropic import AnthropicAdapter
from lattice.providers.tool_sanitizer import ANTHROPIC_TOOL_ID_PATTERN

# =============================================================================
# Routing
# =============================================================================


class TestRouting:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    @pytest.mark.parametrize(
        "model",
        ["anthropic/claude-3-sonnet", "claude-3-opus", "anthropic/claude-sonnet-4-5"],
    )
    def test_supports(self, model: str) -> None:
        assert self.a.supports(model)

    @pytest.mark.parametrize("model", ["gpt-4", "ollama/llama3", "openai/gpt-4o"])
    def test_does_not_support(self, model: str) -> None:
        assert not self.a.supports(model)

    def test_chat_endpoint(self) -> None:
        assert (
            self.a.chat_endpoint("claude-3", "https://api.anthropic.com")
            == "https://api.anthropic.com/v1/messages"
        )


# =============================================================================
# Auth headers
# =============================================================================


class TestAuth:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_api_key(self) -> None:
        h = self.a.auth_headers("sk-ant-api03-test")
        assert h["anthropic-version"] == "2023-06-01"
        assert h["x-api-key"] == "sk-ant-api03-test"
        assert "Authorization" not in h

    def test_oauth_headers(self) -> None:
        h = self.a.auth_headers("sk-ant-oat-test")
        assert h["Authorization"] == "Bearer sk-ant-oat-test"
        assert "anthropic-beta" in h
        assert "claude-code-20250219" in h["anthropic-beta"]
        assert h["user-agent"] == "claude-cli/2.1.2 (external, cli)"
        assert h["x-app"] == "cli"
        assert h["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in h

    def test_no_key(self) -> None:
        h = self.a.auth_headers(None)
        assert h == {"anthropic-version": "2023-06-01"}

    def test_is_oauth(self) -> None:
        assert self.a.is_oauth("sk-ant-oat-xxx") is True
        assert self.a.is_oauth("sk-ant-api-xxx") is False
        assert self.a.is_oauth(None) is False


# =============================================================================
# Tool ID sanitisation
# =============================================================================


class TestToolIdSanitization:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_sanitises_invalid_tool_ids(self) -> None:
        req = Request(
            model="claude-3-sonnet",
            messages=[Message(role="user", content="Hello")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "call:with:colons",
                        "description": "test",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        body = self.a.serialize_request(req)
        tools = body["tools"]
        # After _remap_tools, "function" key is replaced with "name"
        assert tools[0]["name"] == "call_with_colons"
        # Mapping stored (raw -> sanitized name)
        assert req.metadata.get("_anthropic_tool_id_mapping") == {
            "call:with:colons": "call_with_colons"
        }

    def test_skips_valid_tool_ids(self) -> None:
        req = Request(
            model="claude-3-sonnet",
            messages=[Message(role="user", content="Hello")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "test",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        body = self.a.serialize_request(req)
        tools = body["tools"]
        # After _remap_tools, function key is replaced by "name"
        assert tools[0]["name"] == "get_weather"
        # No mapping because no sanitisation needed
        assert not req.metadata.get("_anthropic_tool_id_mapping")

    def test_all_free_router_invalid_patterns(self) -> None:
        """Every pattern FreeRouter tests (e2e-tool-id-sanitization.ts)."""
        adapter = AnthropicAdapter()
        invalid = [
            "call:with:colons",
            "call.with.dots",
            "call/with/slashes",
            "call@with@at",
            "call with spaces",
            "call#with#hash",
        ]
        for raw in invalid:
            req = Request(
                model="claude-3",
                messages=[Message(role="user", content="Hi")],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": raw,
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
            body = adapter.serialize_request(req)
            name = body["tools"][0]["name"]
            assert ANTHROPIC_TOOL_ID_PATTERN.match(name), (
                f"sanitised {raw!r} -> {name!r} still invalid"
            )


# =============================================================================
# Tool call argument parsing (string → JSON object)
# =============================================================================


class TestToolArgParsing:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_tool_call_arguments_as_string(self) -> None:
        """OpenAI tool_calls have arguments as a JSON string."""
        req = Request(
            model="claude-3",
            messages=[
                Message(role="user", content="What's the weather?"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                ),
            ],
        )
        body = self.a.serialize_request(req)
        # After _pop_system: user message first, then assistant with tool_use
        msg = body["messages"][1]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1  # only tool_use (empty text skipped)
        tool_block = msg["content"][0]
        assert tool_block["input"] == {"city": "Paris"}

    def test_tool_call_arguments_as_object(self) -> None:
        """Some callers already send arguments as a dict."""
        req = Request(
            model="claude-3",
            messages=[
                Message(role="user", content="Yo"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                ),
            ],
        )
        body = self.a.serialize_request(req)
        msg = body["messages"][1]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1  # only tool_use block, empty text skipped
        tool_block = msg["content"][0]
        assert tool_block["input"] == {"city": "Paris"}


# =============================================================================
# Tool result merging
# =============================================================================


class TestToolResultMerging:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_consecutive_tool_results_merged(self) -> None:
        req = Request(
            model="claude-3",
            messages=[
                Message(role="user", content="Run tools"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "a"}}
                    ],
                ),
                Message(role="tool", content="result1", tool_call_id="c1"),
                Message(role="tool", content="result2", tool_call_id="c2"),
            ],
        )
        body = self.a.serialize_request(req)
        msgs = body["messages"]
        # After _remap_messages: user msg (0), assistant with tool_use (1),
        # user with merged tool_results (2)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Run tools"
        assert msgs[1]["role"] == "assistant"
        merged = msgs[2]
        assert merged["role"] == "user"
        content = merged["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "c1"
        assert content[1]["tool_use_id"] == "c2"

    def test_single_tool_result(self) -> None:
        req = Request(
            model="claude-3",
            messages=[
                Message(role="user", content="Run tool"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "a"}}
                    ],
                ),
                Message(role="tool", content="result", tool_call_id="c1"),
                Message(role="user", content="Thanks"),
            ],
        )
        body = self.a.serialize_request(req)
        msgs = body["messages"]
        # user (0), assistant with tool_use (1), merged tool_results (2), user (3)
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        content = msgs[2]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["tool_use_id"] == "c1"
        # Regular user message after
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "Thanks"


# =============================================================================
# Thinking config
# =============================================================================


class TestThinkingConfig:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_opus_46_adaptive(self) -> None:
        req = Request(model="claude-opus-4-6", messages=[])
        thinking = self.a._get_thinking_config(req.model)
        assert thinking == {"type": "adaptive"}

    def test_sonnet_enabled(self) -> None:
        req = Request(model="claude-sonnet-4-5", messages=[])
        thinking = self.a._get_thinking_config(req.model)
        assert thinking == {"type": "enabled", "budget_tokens": 4096}

    def test_no_thinking(self) -> None:
        req = Request(model="claude-haiku", messages=[])
        assert self.a._get_thinking_config(req.model) is None

    def test_max_tokens_with_thinking(self) -> None:
        # When thinking budget exceeds user max_tokens, we must bump max_tokens
        # above the budget (Anthropic API requirement).
        req = Request(model="claude-opus-4-6", messages=[], max_tokens=1000)
        thinking = {"type": "enabled", "budget_tokens": 4096}
        result = self.a._compute_max_tokens(req, thinking)
        assert result > 4096
        assert result >= self.a._default_max_tokens(req.model)

    def test_max_tokens_adaptive_no_budget(self) -> None:
        req = Request(model="claude-opus-4-6", messages=[], max_tokens=2000)
        thinking = {"type": "adaptive"}
        assert self.a._compute_max_tokens(req, thinking) == 2000

    def test_max_tokens_default_when_unspecified(self) -> None:
        # When user does not specify max_tokens, we apply a model-aware default
        # that also satisfies the thinking budget constraint.
        req = Request(model="claude-opus-4-6", messages=[])
        thinking = {"type": "enabled", "budget_tokens": 4096}
        result = self.a._compute_max_tokens(req, thinking)
        assert result is not None
        assert result > 4096
        assert result >= self.a._default_max_tokens(req.model)

    def test_max_tokens_default_without_thinking(self) -> None:
        req = Request(model="claude-haiku", messages=[])
        assert self.a._compute_max_tokens(req, None) == 4096

    def test_max_tokens_user_value_above_budget(self) -> None:
        req = Request(model="claude-sonnet-4-5", messages=[], max_tokens=10000)
        thinking = {"type": "enabled", "budget_tokens": 4096}
        assert self.a._compute_max_tokens(req, thinking) == 10000

    def test_temperature_omitted_when_thinking(self) -> None:
        req = Request(
            model="claude-sonnet-4-5",
            messages=[Message(role="user", content="Hi")],
            temperature=0.7,
        )
        body = self.a.serialize_request(req)
        assert "temperature" not in body
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_temperature_present_without_thinking(self) -> None:
        req = Request(
            model="claude-haiku",
            messages=[Message(role="user", content="Hi")],
            temperature=0.7,
        )
        body = self.a.serialize_request(req)
        assert body["temperature"] == 0.7


# =============================================================================
# Deserialization
# =============================================================================


class TestDeserialization:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_text_response(self) -> None:
        data = {
            "content": [{"type": "text", "text": "Hello world"}],
            "role": "assistant",
            "model": "claude-3-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        resp = self.a.deserialize_response(data)
        assert resp.content == "Hello world"
        assert resp.role == "assistant"
        assert resp.model == "claude-3-sonnet"
        assert resp.finish_reason == "stop"
        assert resp.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }

    def test_tool_use_response(self) -> None:
        data = {
            "content": [
                {"type": "text", "text": "I'll check that."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
            "role": "assistant",
            "model": "claude-3-sonnet",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        resp = self.a.deserialize_response(data)
        assert resp.content == "I'll check that."
        assert resp.finish_reason == "tool_calls"
        tc = resp.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "tu_1"
        assert tc[0]["type"] == "function"
        assert tc[0]["function"]["name"] == "get_weather"
        assert tc[0]["function"]["arguments"] == '{"city": "Paris"}'

    def test_finish_reason_mapping(self) -> None:
        adapter = AnthropicAdapter()
        assert adapter._map_finish_reason("tool_use") == "tool_calls"
        assert adapter._map_finish_reason("end_turn") == "stop"
        assert adapter._map_finish_reason("max_tokens") == "length"
        assert adapter._map_finish_reason("stop_sequence") == "stop"
        assert adapter._map_finish_reason("custom") == "custom"
        assert adapter._map_finish_reason(None) is None


# =============================================================================
# Streaming (state machine)
# =============================================================================


class TestStreaming:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_stream_state_factory(self) -> None:
        sm = self.a.normalize_sse_stream("claude-3-sonnet")
        from lattice.providers.stream_state import AnthropicStreamState

        assert isinstance(sm, AnthropicStreamState)

    def test_thinking_skipped_in_stream(self) -> None:
        """State machine must emit NOTHING for thinking blocks."""
        sm = self.a.normalize_sse_stream("claude")
        # thinking block start
        r1 = sm.process({"type": "content_block_start", "content_block": {"type": "thinking"}})
        assert r1.chunks == []
        # text that should be skipped because inside thinking
        r2 = sm.process(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "secret"}}
        )
        assert r2.chunks == []
        # stop thinking
        sm.process({"type": "content_block_stop"})
        # start text block
        sm.process(
            {"type": "content_block_start", "content_block": {"type": "text"}}
        )
        # now text delta should be emitted
        r3 = sm.process(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}
        )
        assert r3.chunks[0]["choices"][0]["delta"]["content"] == "Hello"


# =============================================================================
# System prompt
# =============================================================================


class TestSystemPrompt:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_system_extracted(self) -> None:
        req = Request(
            model="claude-3",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ],
        )
        body = self.a.serialize_request(req)
        assert body["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in body["messages"])

    def test_multiple_system_collapsed(self) -> None:
        req = Request(
            model="claude-3",
            messages=[
                Message(role="system", content="A"),
                Message(role="system", content="B"),
                Message(role="user", content="Hello"),
            ],
        )
        body = self.a.serialize_request(req)
        # _pop_system extracts the first system message.
        # _remap_messages then drops any remaining system/developer messages.
        assert "system" in body
        assert body["system"] == "A"
        # Second system message is dropped (Anthropic supports max one)


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def setup_method(self) -> None:
        self.a = AnthropicAdapter()

    def test_empty_content(self) -> None:
        resp = self.a.extract_content({"content": ""})
        assert resp == ""

    def test_content_as_list(self) -> None:
        resp = self.a.extract_content(
            {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "id": "x", "name": "y"},
                ]
            }
        )
        assert resp == "Hello"

    def test_no_tools_body_no_tools_key(self) -> None:
        req = Request(model="claude-3", messages=[Message(role="user", content="Hi")])
        body = self.a.serialize_request(req)
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_stream_true(self) -> None:
        req = Request(
            model="claude-3",
            messages=[Message(role="user", content="Hi")],
            stream=True,
        )
        body = self.a.serialize_request(req)
        assert body["stream"] is True
