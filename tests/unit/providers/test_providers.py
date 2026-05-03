"""Unit tests for LATTICE provider adapters.

Coverage goals:
1. Each adapter: serialization round-trip
2. ProviderRegistry: resolution logic
3. DirectHTTPProvider: mocked HTTP interactions (respx)
4. ConnectionPoolManager: lifecycle
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from httpx import Response as HttpxResponse

from lattice.core.transport import Message, Request
from lattice.providers import (
    AI21Adapter,
    AnthropicAdapter,
    AzureAdapter,
    BedrockAdapter,
    ConnectionPoolManager,
    DirectHTTPProvider,
    OllamaAdapter,
    OpenAIAdapter,
    ProviderRegistry,
)
from lattice.providers.base import _pop_system, _remap_tool_choice, _remap_tools
from lattice.providers.gemini import GeminiAdapter, VertexAdapter
from lattice.providers.openai_compatible import (
    FireworksAdapter,
    GroqAdapter,
    OpenRouterAdapter,
    TogetherAdapter,
)
from lattice.providers.transport import RateLimitTracker

# =============================================================================
# Helpers
# =============================================================================


def _make_request(model: str = "gpt-4", messages: list[Message] | None = None) -> Request:
    return Request(
        model=model,
        messages=messages or [Message(role="user", content="Hello")],
    )


# =============================================================================
# ProviderRegistry
# =============================================================================


class TestProviderRegistry:
    def test_resolve_ollama_prefix(self) -> None:
        reg = ProviderRegistry()
        adapter = reg.resolve("ollama/llama3")
        assert isinstance(adapter, OllamaAdapter)

    def test_resolve_anthropic_prefix(self) -> None:
        reg = ProviderRegistry()
        adapter = reg.resolve("anthropic/claude-3-opus")
        assert isinstance(adapter, AnthropicAdapter)

    def test_resolve_bare_model_raises(self) -> None:
        reg = ProviderRegistry()
        from lattice.core.errors import ProviderError

        with pytest.raises(ProviderError):
            reg.resolve("gpt-4")

    def test_resolve_unknown_raises(self) -> None:
        from lattice.core.errors import ProviderError

        reg = ProviderRegistry([OllamaAdapter()])  # no OpenAI fallback
        with pytest.raises(ProviderError):
            reg.resolve("gpt-4")

    def test_list_adapters(self) -> None:
        reg = ProviderRegistry()
        names = reg.list_adapters()
        assert "ollama" in names
        assert "anthropic" in names
        assert "openai" in names


# =============================================================================
# OpenAIAdapter
# =============================================================================


class TestOpenAIAdapter:
    def test_supports_bare_model(self) -> None:
        a = OpenAIAdapter()
        assert a.supports("gpt-4") is False
        assert a.supports("claude-3") is False  # no catch-all

    def test_supports_prefix(self) -> None:
        a = OpenAIAdapter()
        assert a.supports("openai/gpt-4") is True
        assert a.supports("azure/gpt-4o") is True
        assert a.supports("ollama/llama3") is False

    def test_serialize_request_minimal(self) -> None:
        a = OpenAIAdapter()
        req = _make_request(model="gpt-4")
        body = a.serialize_request(req)
        assert body["model"] == "gpt-4"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert "temperature" not in body

    def test_serialize_request_full(self) -> None:
        a = OpenAIAdapter()
        req = Request(
            model="gpt-4",
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="Hi"),
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stream=True,
            stop=["\n"],
            tools=[{"type": "function", "function": {"name": "weather"}}],
            tool_choice="auto",
        )
        body = a.serialize_request(req)
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert body["top_p"] == 0.9
        assert body["stream"] is True
        assert body["stop"] == ["\n"]
        assert body["tools"]
        assert body["tool_choice"] == "auto"

    def test_deserialize_response(self) -> None:
        a = OpenAIAdapter()
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = a.deserialize_response(data)
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4"
        assert resp.finish_reason == "stop"
        assert resp.usage["total_tokens"] == 15

    def test_deserialize_response_normalizes_cached_tokens(self) -> None:
        a = OpenAIAdapter()
        resp = a.deserialize_response(
            {
                "model": "gpt-5",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 72},
                },
            }
        )
        assert resp.usage["cached_tokens"] == 72

    def test_deserialize_response_reasoning_field(self) -> None:
        a = OpenAIAdapter()
        data = {
            "choices": [{"message": {"role": "assistant", "content": "", "reasoning": "I think"}}],
            "model": "deepseek",
        }
        resp = a.deserialize_response(data)
        assert resp.content == "I think"

    def test_deserialize_response_reasoning_content(self) -> None:
        a = OpenAIAdapter()
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Let me think...",
                    }
                }
            ],
            "model": "o3-mini",
        }
        resp = a.deserialize_response(data)
        assert resp.content == "Let me think..."
        assert resp.metadata.get("reasoning") == "Let me think..."

    def test_deserialize_response_refusal(self) -> None:
        a = OpenAIAdapter()
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "refusal": "I can't help with that.",
                    }
                }
            ],
            "model": "gpt-4o",
        }
        resp = a.deserialize_response(data)
        assert resp.metadata.get("refusal") == "I can't help with that."

    def test_serialize_request_reasoning_effort(self) -> None:
        a = OpenAIAdapter()
        req = Request(
            model="o3-mini",
            messages=[Message(role="user", content="Solve this")],
            metadata={"reasoning_effort": "high"},
        )
        body = a.serialize_request(req)
        assert body["reasoning_effort"] == "high"

    def test_serialize_request_prompt_cache_controls(self) -> None:
        a = OpenAIAdapter()
        req = Request(
            model="gpt-5",
            messages=[Message(role="user", content="Use the stable prefix.")],
            metadata={
                "prompt_cache_key": "session:abc123",
                "prompt_cache_retention": "24h",
            },
        )
        body = a.serialize_request(req)
        assert body["prompt_cache_key"] == "session:abc123"
        assert body["prompt_cache_retention"] == "24h"

    def test_normalize_sse_chunk(self) -> None:
        a = OpenAIAdapter()
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        out = a.normalize_sse_chunk(chunk)
        assert out == chunk

    def test_normalize_sse_chunk_empty_choices(self) -> None:
        a = OpenAIAdapter()
        chunk = {"choices": [], "usage": {"prompt_tokens": 1}}
        assert a.normalize_sse_chunk(chunk) is None


class TestAzureAdapter:
    def test_serialize_request_prompt_cache_controls_without_model_body(self) -> None:
        a = AzureAdapter()
        req = Request(
            model="azure/gpt-5",
            messages=[Message(role="user", content="Hello")],
            extra_body={
                "prompt_cache_key": "tenant:42",
                "prompt_cache_retention": "24h",
            },
        )
        body = a.serialize_request(req)
        assert "model" not in body
        assert body["prompt_cache_key"] == "tenant:42"
        assert body["prompt_cache_retention"] == "24h"


# =============================================================================
# OllamaAdapter
# =============================================================================


class TestOllamaAdapter:
    def test_supports_prefix(self) -> None:
        a = OllamaAdapter()
        assert a.supports("ollama/llama3") is True
        assert a.supports("ollama/glm-5.1:cloud") is True
        assert a.supports("gpt-4") is False

    def test_strip_prefix(self) -> None:
        a = OllamaAdapter()
        assert a._strip_prefix("ollama/llama3") == "llama3"
        assert a._strip_prefix("llama3") == "llama3"

    def test_serialize_request(self) -> None:
        a = OllamaAdapter()
        req = _make_request(model="ollama/llama3")
        body = a.serialize_request(req)
        assert body["model"] == "llama3"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["stream"] is False
        assert "options" not in body

    def test_serialize_request_with_params(self) -> None:
        a = OllamaAdapter()
        req = Request(
            model="ollama/llama3",
            messages=[Message(role="user", content="Test")],
            temperature=0.5,
            top_p=0.9,
            max_tokens=200,
            stream=True,
        )
        body = a.serialize_request(req)
        opts = body["options"]
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["num_predict"] == 200

    def test_deserialize_response(self) -> None:
        a = OllamaAdapter()
        data = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Greetings!"},
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 3,
        }
        resp = a.deserialize_response(data)
        assert resp.content == "Greetings!"
        assert resp.model == "llama3"
        assert resp.finish_reason == "stop"
        assert resp.usage["prompt_tokens"] == 12
        assert resp.usage["completion_tokens"] == 3

    def test_normalize_sse_chunk(self) -> None:
        a = OllamaAdapter()
        chunk = {"message": {"role": "assistant", "content": "H"}, "done": False}
        out = a.normalize_sse_chunk(chunk)
        assert out is not None
        assert out["choices"][0]["delta"]["content"] == "H"

    def test_normalize_sse_chunk_done(self) -> None:
        a = OllamaAdapter()
        chunk = {"message": {"role": "assistant", "content": ""}, "done": True}
        out = a.normalize_sse_chunk(chunk)
        assert out is not None
        assert out["choices"][0]["finish_reason"] == "stop"

    def test_normalize_sse_chunk_empty(self) -> None:
        a = OllamaAdapter()
        chunk = {"message": {"role": "assistant", "content": ""}, "done": False}
        assert a.normalize_sse_chunk(chunk) is None


# =============================================================================
# AnthropicAdapter
# =============================================================================


class TestAnthropicAdapter:
    def test_supports_prefix(self) -> None:
        a = AnthropicAdapter()
        assert a.supports("anthropic/claude-3-opus") is True
        assert a.supports("claude-3-sonnet") is True
        assert a.supports("gpt-4") is False

    def test_auth_headers(self) -> None:
        a = AnthropicAdapter()
        h = a.auth_headers("sk-ant-xxx")
        assert h["x-api-key"] == "sk-ant-xxx"
        assert h["anthropic-version"] == "2023-06-01"

    def test_serialize_request_system_extracted(self) -> None:
        a = AnthropicAdapter()
        req = Request(
            model="claude-3-haiku",
            messages=[
                Message(role="system", content="Be nice."),
                Message(role="user", content="Hello"),
            ],
            temperature=0.7,
            max_tokens=100,
        )
        body = a.serialize_request(req)
        assert body["system"] == "Be nice."
        assert all(m["role"] != "system" for m in body["messages"])
        assert body["messages"][0]["role"] == "user"
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert "top_p" not in body

    def test_serialize_request_uses_cache_arbitrage_annotations(self) -> None:
        a = AnthropicAdapter()
        req = Request(
            model="anthropic/claude-3-5-sonnet",
            messages=[
                Message(role="system", content="Stable operating rules."),
                Message(role="user", content="Hello"),
            ],
            metadata={
                "_cache_arbitrage": {
                    "annotations": {
                        "provider": "anthropic",
                        "cache": {
                            "mode": "explicit_breakpoint",
                            "default_ttl_seconds": 3600,
                        },
                    }
                }
            },
        )
        body = a.serialize_request(req)
        assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_serialize_request_tool_message_remap(self) -> None:
        a = AnthropicAdapter()
        req = Request(
            model="claude-3-opus",
            messages=[
                Message(role="user", content="What's the weather?"),
                Message(
                    role="tool",
                    content="Sunny, 25°C",
                    tool_call_id="call_123",
                ),
            ],
        )
        body = a.serialize_request(req)
        tool_msg = body["messages"][1]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_123"
        assert tool_msg["content"][0]["content"] == "Sunny, 25°C"

    def test_deserialize_response_content_blocks(self) -> None:
        a = AnthropicAdapter()
        data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus",
            "content": [
                {"type": "text", "text": "Hello!"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        resp = a.deserialize_response(data)
        assert resp.content == "Hello!"
        assert resp.model == "claude-3-opus"
        assert resp.finish_reason == "stop"  # mapped from end_turn
        assert resp.usage["total_tokens"] == 15

    def test_deserialize_response_tool_use(self) -> None:
        a = AnthropicAdapter()
        data = {
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "NYC"}},
            ],
            "model": "claude-3-opus",
        }
        resp = a.deserialize_response(data)
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["type"] == "function"
        assert resp.tool_calls[0]["function"]["name"] == "get_weather"

    def test_normalize_sse_text_delta(self) -> None:
        a = AnthropicAdapter()
        chunk = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        out = a.normalize_sse_chunk(chunk)
        assert out is not None
        assert out["choices"][0]["delta"]["content"] == "Hello"

    def test_normalize_sse_message_stop(self) -> None:
        a = AnthropicAdapter()
        chunk = {"type": "message_stop"}
        out = a.normalize_sse_chunk(chunk)
        assert out is not None
        assert out["choices"][0]["finish_reason"] == "stop"

    def test_normalize_sse_ping_ignored(self) -> None:
        a = AnthropicAdapter()
        assert a.normalize_sse_chunk({"type": "ping"}) is None


# =============================================================================
# Shared helpers
# =============================================================================


class TestPopSystem:
    def test_no_system(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        system, rest = _pop_system(msgs)
        assert system is None
        assert rest == msgs

    def test_system_first_extracted(self) -> None:
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]
        system, rest = _pop_system(msgs)
        assert system == "Be helpful."
        assert len(rest) == 1
        assert rest[0]["role"] == "user"


class TestRemapTools:
    def test_basic(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        out = _remap_tools(tools)
        assert out[0]["name"] == "weather"
        assert "input_schema" in out[0]

    def test_empty(self) -> None:
        assert _remap_tools([]) == []


class TestRemapToolChoice:
    def test_none_auto_none(self) -> None:
        assert _remap_tool_choice(None) is None
        assert _remap_tool_choice("auto") == "auto"

    def test_function_to_tool(self) -> None:
        tc = {"type": "function", "function": {"name": "weather"}}
        out = _remap_tool_choice(tc)
        assert out == {"type": "tool", "name": "weather"}


# =============================================================================
# ConnectionPoolManager
# =============================================================================


class TestConnectionPoolManager:
    @pytest.mark.asyncio
    async def test_get_client_creates_new(self) -> None:
        mgr = ConnectionPoolManager()
        client = mgr.get_client("openai", "https://api.openai.com")
        assert isinstance(client, httpx.AsyncClient)
        assert mgr.pool_count == 1
        await mgr.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses(self) -> None:
        mgr = ConnectionPoolManager()
        c1 = mgr.get_client("openai", "https://api.openai.com")
        c2 = mgr.get_client("openai", "https://api.openai.com")
        assert c1 is c2
        await mgr.close()

    @pytest.mark.asyncio
    async def test_close_clears(self) -> None:
        mgr = ConnectionPoolManager()
        mgr.get_client("openai", "https://api.openai.com")
        await mgr.close()
        assert mgr.pool_count == 0


# =============================================================================
# DirectHTTPProvider (mocked HTTP)
# =============================================================================


class TestDirectHTTPProviderMocked:
    @pytest.mark.asyncio
    @respx.mock
    async def test_completion_openai(self) -> None:
        """DirectHTTPProvider makes a real HTTP call and gets back our Response."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=HttpxResponse(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "42"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            )
        )

        provider = DirectHTTPProvider(
            default_api_key="fake",
            provider_base_urls={
                "openai": "https://api.openai.com",
                "groq": "https://api.groq.com/openai",
            },
        )
        resp = await provider.completion(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "What is 6*7?"}],
        )
        assert resp.content == "42"
        assert resp.model == "gpt-4"
        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_completion_ollama(self) -> None:
        route = respx.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=HttpxResponse(
                200,
                json={
                    "model": "llama3",
                    "message": {"role": "assistant", "content": "Hi!"},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                },
            )
        )

        provider = DirectHTTPProvider(
            provider_base_urls={"ollama": "http://127.0.0.1:11434"},
        )
        resp = await provider.completion(
            model="ollama/llama3",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert resp.content == "Hi!"
        assert route.called
        # Verify the model prefix was stripped in the payload
        request_json = route.calls[0].request.content
        assert request_json is not None
        payload = json.loads(request_json)
        assert payload["model"] == "llama3"

    @pytest.mark.asyncio
    @respx.mock
    async def test_completion_timeout(self) -> None:
        from lattice.core.errors import ProviderTimeoutError

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        provider = DirectHTTPProvider(
            timeout=0.01,
            default_api_key="fake",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        with pytest.raises(ProviderTimeoutError):
            await provider.completion(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "Timeout"}],
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_completion_http_error(self) -> None:
        from lattice.core.errors import ProviderError

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=HttpxResponse(401, json={"error": "invalid key"})
        )

        provider = DirectHTTPProvider(
            default_api_key="fake",
            provider_base_urls={
                "openai": "https://api.openai.com",
                "groq": "https://api.groq.com/openai",
            },
        )
        with pytest.raises(ProviderError) as exc_info:
            await provider.completion(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "Auth"}],
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @respx.mock
    async def test_completion_retry_on_429(self) -> None:
        """DirectHTTPProvider retries on 429 with exponential backoff."""

        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                HttpxResponse(429, headers={"retry-after": "0"}, text="rate limited"),
                HttpxResponse(
                    200,
                    json={
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "OK"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                ),
            ]
        )

        provider = DirectHTTPProvider(
            default_api_key="fake",
            provider_base_urls={
                "openai": "https://api.openai.com",
                "groq": "https://api.groq.com/openai",
            },
        )
        resp = await provider.completion(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "Retry me"}],
        )
        assert resp.content == "OK"
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_completion_explicit_provider(self) -> None:
        """Explicit provider_name routes to the correct adapter."""
        route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=HttpxResponse(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Groq!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )

        provider = DirectHTTPProvider(
            default_api_key="fake",
            provider_base_urls={
                "openai": "https://api.openai.com",
                "groq": "https://api.groq.com/openai",
            },
        )
        resp = await provider.completion(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": "Hello"}],
            provider_name="groq",
        )
        assert resp.content == "Groq!"
        assert route.called
        # Verify model is passed through as-is (no aliases)
        payload = json.loads(route.calls[0].request.content)
        assert payload["model"] == "llama-3.1-70b-versatile"


# =============================================================================
# Model name mapping — prefix stripping only
# =============================================================================


class TestMapModelName:
    def test_openai_prefix_stripped(self) -> None:
        a = OpenAIAdapter()
        assert a.map_model_name("openai/gpt-4") == "gpt-4"
        assert a.map_model_name("azure/gpt-4o") == "gpt-4o"
        assert a.map_model_name("gpt-4") == "gpt-4"  # bare name unchanged

    def test_groq_prefix_stripped(self) -> None:
        a = GroqAdapter()
        assert a.map_model_name("groq/llama-3.1-70b") == "llama-3.1-70b"
        assert a.map_model_name("llama-3.1-8b") == "llama-3.1-8b"

    def test_together_prefix_stripped(self) -> None:
        a = TogetherAdapter()
        assert a.map_model_name("together/llama-3.1-70b") == "llama-3.1-70b"

    def test_fireworks_prefix_stripped(self) -> None:
        a = FireworksAdapter()
        assert a.map_model_name("fireworks/llama-3.1-70b") == "llama-3.1-70b"

    def test_openrouter_prefix_stripped(self) -> None:
        a = OpenRouterAdapter()
        assert a.map_model_name("openrouter/anthropic/claude-3") == "anthropic/claude-3"

    def test_ai21_prefix_stripped(self) -> None:
        a = AI21Adapter()
        assert a.map_model_name("ai21/jamba-1.5-mini") == "jamba-1.5-mini"


# =============================================================================
# Retry config
# =============================================================================


class TestRetryConfig:
    def test_openai_default(self) -> None:
        a = OpenAIAdapter()
        cfg = a.retry_config()
        assert cfg["max_retries"] == 3
        assert 429 in cfg["retry_on"]

    def test_groq_higher_retries(self) -> None:
        a = GroqAdapter()
        cfg = a.retry_config()
        assert cfg["max_retries"] == 5

    def test_anthropic_retry(self) -> None:
        a = AnthropicAdapter()
        cfg = a.retry_config()
        assert cfg["max_retries"] >= 3


# =============================================================================
# Extra headers
# =============================================================================


class TestExtraHeaders:
    def test_openai_empty(self) -> None:
        a = OpenAIAdapter()
        req = _make_request()
        assert a.extra_headers(req) == {}

    def test_openrouter_referer(self) -> None:
        a = OpenRouterAdapter()
        auth = a.auth_headers("sk-test")
        assert auth["HTTP-Referer"] == "https://lattice.dev"
        assert auth["X-Title"] == "LATTICE"


# =============================================================================
# Rate limit tracker
# =============================================================================


class TestRateLimitTracker:
    def test_parse_headers(self) -> None:
        tracker = RateLimitTracker()
        headers = httpx.Headers(
            {
                "x-ratelimit-limit": "100",
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "60",
            }
        )
        tracker.update("groq", headers)
        assert tracker.is_throttled("groq") is True
        assert tracker.retry_after("groq") is None

    def test_retry_after_header(self) -> None:
        tracker = RateLimitTracker()
        headers = httpx.Headers({"retry-after": "5"})
        tracker.update("openai", headers)
        assert tracker.retry_after("openai") == 5.0

    def test_unknown_provider_not_throttled(self) -> None:
        tracker = RateLimitTracker()
        assert tracker.is_throttled("unknown") is False


# =============================================================================
# AI21 Adapter
# =============================================================================


class TestAI21Adapter:
    def test_supports_prefix(self) -> None:
        a = AI21Adapter()
        assert a.supports("ai21/jamba-1.5-mini") is True
        assert a.supports("gpt-4") is False

    def test_chat_endpoint(self) -> None:
        a = AI21Adapter()
        assert a.chat_endpoint("jamba-1.5-mini", "https://api.ai21.com/studio/v1") == (
            "https://api.ai21.com/studio/v1/chat/completions"
        )

    def test_map_model_name_strips_prefix(self) -> None:
        a = AI21Adapter()
        assert a.map_model_name("ai21/jamba-1.5-mini") == "jamba-1.5-mini"

    def test_serialize_request(self) -> None:
        a = AI21Adapter()
        req = _make_request(model="ai21/jamba-1.5-mini")
        body = a.serialize_request(req)
        assert body["model"] == "ai21/jamba-1.5-mini"


# =============================================================================
# Cache-aware provider serialization
# =============================================================================


class TestBedrockAdapter:
    def test_serialize_request_adds_cache_point_from_plan(self) -> None:
        a = BedrockAdapter()
        req = Request(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[
                Message(role="system", content="Stable system prompt."),
                Message(role="user", content="Hello"),
            ],
            metadata={
                "_cache_arbitrage": {
                    "annotations": {
                        "provider": "bedrock",
                        "cache": {"mode": "explicit_breakpoint"},
                    }
                }
            },
        )
        body = a.serialize_request(req)
        assert body["system"] == [
            {"text": "Stable system prompt."},
            {"cachePoint": {"type": "default"}},
        ]

    def test_deserialize_response_preserves_cache_usage(self) -> None:
        a = BedrockAdapter()
        resp = a.deserialize_response(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 10,
                    "totalTokens": 110,
                    "cacheReadInputTokens": 64,
                    "cacheWriteInputTokens": 32,
                },
            }
        )
        assert resp.usage["cached_tokens"] == 64
        assert resp.usage["cache_creation_input_tokens"] == 32

    def test_normalize_sse_chunk_preserves_cache_usage(self) -> None:
        a = BedrockAdapter()
        chunk = a.normalize_sse_chunk(
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 100,
                        "outputTokens": 10,
                        "totalTokens": 110,
                        "cacheReadInputTokens": 64,
                        "cacheWriteInputTokens": 32,
                    }
                }
            }
        )
        assert chunk is not None
        assert chunk["usage"]["cached_tokens"] == 64
        assert chunk["usage"]["cache_creation_input_tokens"] == 32


class TestGeminiAdapter:
    def test_serialize_request_passes_explicit_cached_content(self) -> None:
        a = GeminiAdapter()
        req = Request(
            model="gemini/gemini-2.5-pro",
            messages=[Message(role="user", content="Hello")],
            metadata={"gemini_cached_content": "cachedContents/abc123"},
        )
        body = a.serialize_request(req)
        assert body["cachedContent"] == "cachedContents/abc123"


class TestVertexAdapter:
    def test_serialize_request_passes_explicit_cached_content(self) -> None:
        a = VertexAdapter()
        req = Request(
            model="vertex/gemini-2.5-pro",
            messages=[Message(role="user", content="Hello")],
            extra_body={"cachedContent": "projects/p/locations/us/cachedContents/abc"},
        )
        body = a.serialize_request(req)
        assert body["cachedContent"] == "projects/p/locations/us/cachedContents/abc"
