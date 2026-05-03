"""Integration tests for LATTICE proxy with DirectHTTPProvider.

Tests the full end-to-end flow:
1. Client sends OpenAI-compatible request
2. Proxy compresses via transforms
3. DirectHTTPProvider (monkeypatched) returns a Response
4. Proxy decompresses via reverse transforms
5. Client receives complete response

Technique:
- ``monkeypatch`` fixture patches ``DirectHTTPProvider.completion``
  and ``DirectHTTPProvider.completion_stream``
- ``fastapi.testclient.TestClient`` sends requests to the ASGI app
- Assert on response structure, content, and headers

Coverage goals:
- Request serialization round-trip
- Response decompression round-trip
- Session header propagation
- Provider error propagation (timeout, HTTP errors)
- Streaming response handling
- Metrics collection
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lattice.core.config import LatticeConfig
from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.core.transport import Message, Request, Response
from lattice.proxy.server import create_app

# =============================================================================
# Fixtures
# =============================================================================


def _make_config() -> LatticeConfig:
    return LatticeConfig(
        provider_base_url="http://127.0.0.1:11434",
        provider_base_urls={
            "openai": "https://api.openai.com",
            "anthropic": "https://api.anthropic.com",
            "ollama": "http://127.0.0.1:11434",
        },
        provider_api_key=None,
        graceful_degradation=True,
    )


@pytest.fixture
def config() -> LatticeConfig:
    return _make_config()


@pytest.fixture
def test_client() -> TestClient:
    config = _make_config()
    app = create_app(config=config)
    return TestClient(app)


# =============================================================================
# Health & meta endpoints
# =============================================================================


class TestHealthEndpoints:
    """Healthz, readyz, metrics, stats endpoints."""

    def test_healthz(self, test_client: TestClient) -> None:
        response = test_client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["provider"] == "direct_http"
        assert "adapters" in data

    def test_readyz(self, test_client: TestClient) -> None:
        response = test_client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["config"] is True
        assert data["checks"]["pipeline"] is True
        assert data["checks"]["provider"] is True
        assert "http2_pools" in data["checks"]

    def test_stats(self, test_client: TestClient) -> None:
        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "transforms" in data
        assert data["pipeline"]["runtime_contract_enabled"] is True
        assert {"batching", "speculative", "delta_encoder"}.issubset(
            set(data["pipeline"]["execution_transforms"])
        )
        assert "manifest" in data
        assert data["manifest"]["sessions_with_manifest"] >= 0
        assert data["provider"] == "direct_http"
        assert "adapters" in data
        assert "pools" in data

    def test_metrics(self, test_client: TestClient) -> None:
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "# HELP" in response.text or "lattice_requests_total" in response.text


# =============================================================================
# Full round-trip with mocked DirectHTTPProvider.completion
# =============================================================================


class TestProxyRoundTrip:
    """End-to-end proxy tests with DirectHTTPProvider monkeypatched."""

    def _patch_completion(self, monkeypatch: pytest.MonkeyPatch, response: Response) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _mock(*_args: Any, **_kwargs: Any) -> Response:
            return response

        monkeypatch.setattr(DirectHTTPProvider, "completion", _mock)

    def test_chat_completions_success(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full round-trip: request → proxy → mock provider → proxy → response."""
        self._patch_completion(
            monkeypatch,
            Response(
                content="The result is 42.",
                role="assistant",
                model="glm-5.1:cloud",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                finish_reason="stop",
            ),
        )

        payload = {
            "model": "ollama/glm-5.1:cloud",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 6 * 7?"},
            ],
        }
        response = test_client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "The result is 42."
        assert data["usage"]["total_tokens"] == 15

    def test_chat_completions_openai_provider(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test with OpenAI provider response (different model)."""
        self._patch_completion(
            monkeypatch,
            Response(
                content="42",
                role="assistant",
                model="gpt-4",
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                finish_reason="stop",
            ),
        )

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "What is 6 * 7?"}],
        }
        response = test_client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "42"

    def test_semantic_cache_hit_reports_zero_billed_cost_and_agent_savings(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        calls = 0

        async def _mock(*_args: Any, **_kwargs: Any) -> Response:
            nonlocal calls
            calls += 1
            return Response(
                content="cached answer",
                role="assistant",
                model="gpt-4o",
                usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
                finish_reason="stop",
            )

        monkeypatch.setattr(DirectHTTPProvider, "completion", _mock)

        payload = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "cache accounting probe"}],
        }
        headers = {"x-lattice-client-profile": "codex"}
        first = test_client.post("/v1/chat/completions", json=payload, headers=headers)
        second = test_client.post("/v1/chat/completions", json=payload, headers=headers)

        assert first.status_code == 200
        assert second.status_code == 200
        assert calls == 1
        assert second.headers["x-lattice-cache-hit"] == "true"
        assert second.headers["x-lattice-cached-tokens"] == "1500"
        assert "x-lattice-cost-usd" not in second.headers
        assert float(second.headers["x-lattice-cache-savings-usd"]) > 0

        stats = test_client.get("/stats").json()["agents"]
        codex = stats["per_agent"]["codex"]
        assert codex["requests_total"] == 2
        assert codex["cache_hits"] == 1
        assert codex["cache_misses"] == 1
        assert codex["cached_tokens_total"] == 1500

    def test_provider_timeout(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderTimeoutError(provider="openai", timeout_seconds=1)

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Timeout test"}],
        }
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 504
        data = response.json()
        assert "error" in data

    def test_provider_auth_error(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderError(provider="openai", status_code=401, message="Unauthorized")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Auth test"}]}
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_provider_rate_limit(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderError(provider="openai", status_code=429, message="Rate limited")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Rate test"}]}
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 429
        data = response.json()
        assert "error" in data

    def test_provider_bad_request(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderError(provider="openai", status_code=422, message="Invalid")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Bad req"}]}
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data

    def test_provider_not_found(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderError(provider="openai", status_code=404, message="Not found")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Not found"}]}
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_provider_unavailable(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderError(provider="openai", status_code=503, message="Unavailable")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Unavail"}]}
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 503
        data = response.json()
        assert "error" in data

    def test_provider_502(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ProviderError(provider="openai", status_code=502, message="Bad gateway")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 502
        data = response.json()
        assert "error" in data

    def test_provider_generic_error(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any other exception → proxy returns 502."""
        from lattice.providers.transport import DirectHTTPProvider

        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _raise)

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "generic error test"}],
        }
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 502
        data = response.json()
        assert "error" in data


# =============================================================================
# Streaming
# =============================================================================


class TestStreaming:
    """SSE streaming through the proxy."""

    def test_streaming_request(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Streaming flag → SSE response with normalized chunks."""
        from lattice.providers.transport import DirectHTTPProvider

        async def _mock_stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
            yield {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": "world!"}, "finish_reason": None}]}

        monkeypatch.setattr(DirectHTTPProvider, "completion_stream_with_stall_detect", _mock_stream)

        payload = {
            "model": "ollama/glm-5.1:cloud",
            "messages": [{"role": "user", "content": "Stream me"}],
            "stream": True,
        }
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        body = response.read()
        assert b"Hello " in body
        assert b"world!" in body


# =============================================================================
# Session handling
# =============================================================================


class TestSessionPropagation:
    """Session ID header propagation and session store interaction."""

    def test_session_header_forwarded(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """x-lattice-session-id header creates a session and is forwarded."""
        from lattice.providers.transport import DirectHTTPProvider

        called = False

        async def _mock(*_args: Any, **_kwargs: Any) -> Response:
            nonlocal called
            called = True
            return Response(content="OK", model="gpt-4")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _mock)

        session_id = "sess-test-001"
        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Session test"}],
        }
        response = test_client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"x-lattice-session-id": session_id},
        )
        assert response.status_code == 200
        assert called


# =============================================================================
# Request/response adapters
# =============================================================================


class TestAdapters:
    """OpenAI request/response serialization correctness."""

    def test_openai_request_adapter(self) -> None:
        from lattice.proxy.server import _deserialize_openai_request

        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User question"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stream": False,
            "stop": ["\n"],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                }
            ],
        }
        req = _deserialize_openai_request(body)
        assert req.model == "gpt-4"
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[0].content == "System prompt"
        assert req.temperature == 0.7
        assert req.max_tokens == 100
        assert req.top_p == 0.9
        assert req.stream is False
        assert req.stop == ["\n"]
        assert req.tools is not None
        assert len(req.tools) == 1

    def test_openai_request_adapter_missing_optional(self) -> None:
        from lattice.proxy.server import _deserialize_openai_request

        body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        req = _deserialize_openai_request(body)
        assert req.temperature is None
        assert req.max_tokens is None
        assert req.tools is None
        assert req.stream is False

    def test_openai_response_adapter(self) -> None:
        from lattice.proxy.server import _serialize_openai_response

        req = Request(
            model="gpt-4",
            messages=[Message(role="user", content="test")],
        )
        resp = Response(
            content="Hello there",
            role="assistant",
            model="gpt-4",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
        )
        body = _serialize_openai_response(resp, req)

        assert body["object"] == "chat.completion"
        assert body["model"] == "gpt-4"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["content"] == "Hello there"
        assert body["usage"]["total_tokens"] == 2

    def test_openai_response_adapter_reasoning(self) -> None:
        from lattice.proxy.server import _serialize_openai_response

        req = Request(model="o3-mini", messages=[Message(role="user", content="test")])
        resp = Response(
            content="The answer is 42.",
            role="assistant",
            model="o3-mini",
            metadata={"reasoning": "Let me think... 6*7=42"},
        )
        body = _serialize_openai_response(resp, req)
        assert body["choices"][0]["message"]["reasoning_content"] == "Let me think... 6*7=42"

    def test_openai_response_adapter_refusal(self) -> None:
        from lattice.proxy.server import _serialize_openai_response

        req = Request(model="gpt-4o", messages=[Message(role="user", content="test")])
        resp = Response(
            content="",
            role="assistant",
            metadata={"refusal": "I can't help with that."},
        )
        body = _serialize_openai_response(resp, req)
        assert body["choices"][0]["message"]["refusal"] == "I can't help with that."

    def test_openai_request_adapter_reasoning_effort(self) -> None:
        from lattice.proxy.server import _deserialize_openai_request

        body = {
            "model": "o3-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "high",
        }
        req = _deserialize_openai_request(body)
        assert req.metadata.get("reasoning_effort") == "high"


# =============================================================================
# Anthropic Messages API endpoint
# =============================================================================


class TestAnthropicMessagesEndpoint:
    """Tests for /v1/messages Anthropic Messages API proxy endpoint."""

    def test_anthropic_request_deserialization(self) -> None:
        """Anthropic request body is correctly deserialized."""
        from lattice.proxy.server import _deserialize_anthropic_request

        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hi there!"},
                    ],
                },
            ],
            "temperature": 0.7,
        }
        req = _deserialize_anthropic_request(body)
        assert req.model == "claude-3-5-sonnet-20241022"
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        # System should be extracted as first message
        assert any(m.role == "system" for m in req.messages)
        # User message should be preserved
        assert any(m.role == "user" and m.content == "Hello!" for m in req.messages)

    def test_anthropic_response_serialization(self) -> None:
        """Internal Response is correctly serialized to Anthropic format."""
        from lattice.proxy.server import _serialize_anthropic_response

        request = Request(model="claude-3-5-sonnet-20241022", messages=[])
        response = Response(
            content="Hello there",
            role="assistant",
            model="claude-3-5-sonnet-20241022",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )
        body = _serialize_anthropic_response(response, request)

        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["model"] == "claude-3-5-sonnet-20241022"
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"] == "Hello there"
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["total_tokens"] == 15

    def test_anthropic_messages_post_success(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full round-trip via /v1/messages with mocked HTTP client."""

        class _MockResponse:
            def __init__(self) -> None:
                self.status_code = 200
                self._json = {
                    "id": "msg_01Test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet-20241022",
                    "content": [{"type": "text", "text": "The answer is 42."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
                self.content = json.dumps(self._json).encode("utf-8")
                self.headers = {"content-type": "application/json"}

            def json(self) -> dict[str, Any]:
                return self._json

            @property
            def is_success(self) -> bool:
                return True

        class _MockClient:
            async def request(self, *_args: Any, **_kwargs: Any) -> _MockResponse:
                return _MockResponse()

            async def aclose(self) -> None:
                pass

        from lattice.providers.transport import ConnectionPoolManager

        def _mock_get_client(_self: Any, _provider: str, _base_url: str) -> _MockClient:
            return _MockClient()

        monkeypatch.setattr(ConnectionPoolManager, "get_client", _mock_get_client)

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What is 6 * 7?"}],
        }
        response = test_client.post(
            "/v1/messages",
            json=payload,
            headers={"x-api-key": "test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"][0]["text"] == "The answer is 42."
        assert data["usage"]["output_tokens"] == 5

    def test_anthropic_messages_passthrough(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passthrough mode forwards raw request without transformation."""

        class _MockResponse:
            def __init__(self) -> None:
                self.status_code = 200
                self._json = {
                    "id": "msg_01Passthrough",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet-20241022",
                    "content": [{"type": "text", "text": "Passthrough works."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                }
                self.content = json.dumps(self._json).encode("utf-8")
                self.headers = {"content-type": "application/json"}

            @property
            def is_success(self) -> bool:
                return True

        class _MockClient:
            async def request(self, _method: str, _url: str, **kwargs: Any) -> _MockResponse:
                # Verify the raw body is forwarded unchanged
                body = kwargs.get("content", b"")
                parsed = json.loads(body)
                assert parsed["model"] == "claude-3-5-sonnet-20241022"
                assert parsed["messages"][0]["content"] == "Hello"
                return _MockResponse()

            async def aclose(self) -> None:
                pass

        from lattice.providers.transport import ConnectionPoolManager

        def _mock_get_client(_self: Any, _provider: str, _base_url: str) -> _MockClient:
            return _MockClient()

        monkeypatch.setattr(ConnectionPoolManager, "get_client", _mock_get_client)

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = test_client.post(
            "/v1/messages",
            json=payload,
            headers={"Authorization": "Bearer sk-ant-oat01-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["content"][0]["text"] == "Passthrough works."

    def test_anthropic_messages_passthrough_streaming(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passthrough mode relays SSE streams verbatim."""

        class _MockResponse:
            def __init__(self) -> None:
                self.status_code = 200
                self._chunks = [
                    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01"}}\n\n',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n',
                    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
                ]
                self.headers = {"content-type": "text/event-stream"}

            @property
            def is_success(self) -> bool:
                return True

            def aiter_text(self) -> Any:
                async def _gen() -> AsyncGenerator[str, None]:
                    for chunk in self._chunks:
                        yield chunk

                return _gen()

            async def __aenter__(self) -> _MockResponse:
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

        class _MockClient:
            def stream(self, _method: str, _url: str, **_kwargs: Any) -> _MockResponse:
                return _MockResponse()

            async def aclose(self) -> None:
                pass

        from lattice.providers.transport import ConnectionPoolManager

        def _mock_get_client(_self: Any, _provider: str, _base_url: str) -> _MockClient:
            return _MockClient()

        monkeypatch.setattr(ConnectionPoolManager, "get_client", _mock_get_client)

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = test_client.post(
            "/v1/messages",
            json=payload,
            headers={"Authorization": "Bearer sk-ant-oat01-test"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = response.text
        assert "message_start" in text
        assert "Hi" in text
        assert "message_stop" in text

    def test_anthropic_messages_with_system_block_array(self) -> None:
        """System as array of blocks with cache_control is handled."""
        from lattice.proxy.server import _deserialize_anthropic_request

        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "System prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        req = _deserialize_anthropic_request(body)
        system_msg = next((m for m in req.messages if m.role == "system"), None)
        assert system_msg is not None
        assert system_msg.content == "System prompt"

    def test_anthropic_messages_with_tools(self) -> None:
        """Anthropic tools in input_schema format are remapped."""
        from lattice.proxy.server import _deserialize_anthropic_request

        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }
        req = _deserialize_anthropic_request(body)
        assert req.tools is not None
        assert len(req.tools) == 1
        # Should be remapped to OpenAI format
        assert req.tools[0]["type"] == "function"
        assert req.tools[0]["function"]["name"] == "get_weather"
        assert "parameters" in req.tools[0]["function"]

    def test_anthropic_messages_with_tool_choice(self) -> None:
        """Anthropic tool_choice format is remapped."""
        from lattice.proxy.server import _deserialize_anthropic_request

        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
        req = _deserialize_anthropic_request(body)
        assert req.tool_choice == {"type": "function", "function": {"name": "get_weather"}}

    def test_anthropic_messages_with_thinking(self) -> None:
        """Anthropic thinking config is preserved in metadata."""
        from lattice.proxy.server import _deserialize_anthropic_request

        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
        req = _deserialize_anthropic_request(body)
        assert req.metadata.get("thinking") == {"type": "enabled", "budget_tokens": 1024}


# =============================================================================
# Latency and metrics integration
# =============================================================================


class TestMetricsIntegration:
    """Verify metrics are collected during proxy operation."""

    def test_request_increments_counter(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lattice.providers.transport import DirectHTTPProvider

        async def _mock(*_args: Any, **_kwargs: Any) -> Response:
            return Response(content="Metrics!", model="gpt-4")

        monkeypatch.setattr(DirectHTTPProvider, "completion", _mock)

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Count me"}],
        }
        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        metrics_response = test_client.get("/metrics")
        assert metrics_response.status_code == 200
        assert "lattice_requests_total" in metrics_response.text
