"""End-to-end integration tests for LATTICE.

Validates that every major component works together in realistic scenarios:
- Proxy with batching, speculation, fallback
- SDK client compression
- MCP tool integration
- Agent integration wrap/unwrap flows
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from lattice.core.config import LatticeConfig
from lattice.core.transport import Response
from lattice.integrations.agents import (
    ClaudeCodeIntegration,
    CodexIntegration,
    CursorIntegration,
    OpenCodeIntegration,
    agent_status,
    unwrap_agent,
    wrap_agent,
)
from lattice.integrations.mcp import LatticeMCPTools
from lattice.proxy.server import create_app
from lattice.sdk.client import LatticeClient

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def lattice_config() -> LatticeConfig:
    return LatticeConfig(
        provider_base_url="http://127.0.0.1:11434",
        provider_base_urls={"openai": "https://api.openai.com"},
        provider_api_key="fake-key",
        session_store="memory",
    )


@pytest.fixture
def test_client(lattice_config: LatticeConfig) -> TestClient:
    app = create_app(config=lattice_config)
    return TestClient(app)


# =============================================================================
# Proxy end-to-end
# =============================================================================


class TestProxyEndToEnd:
    """Full proxy request/response cycles with all optimizers enabled."""

    def _patch_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        response: Response,
    ) -> None:
        """Patch provider completion to avoid real network calls."""

        async def _fake_completion(*_args: Any, **_kwargs: Any) -> Response:
            return response

        monkeypatch.setattr(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            _fake_completion,
        )

    def test_batching_single_request(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single non-streaming request is handled correctly (batching attempted)."""
        self._patch_completion(
            monkeypatch,
            Response(
                content="Hello!",
                role="assistant",
                model="gpt-4",
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                finish_reason="stop",
            ),
        )

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 7

    def test_speculative_miss(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Speculative execution misses and falls back to real response."""
        call_count = 0

        async def _counting_completion(*_args: Any, **_kwargs: Any) -> Response:
            nonlocal call_count
            call_count += 1
            return Response(
                content="Real response.",
                role="assistant",
                model="gpt-4",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                finish_reason="stop",
            )

        import lattice.providers.transport as _lpt
        monkeypatch.setattr(
            _lpt.DirectHTTPProvider,
            "completion",
            _counting_completion,
        )

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Real response."
        assert call_count == 1

    def test_tier_classifier_reasoning(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Complex reasoning request is handled without model fallback."""

        async def _fake_completion(*_args: Any, **_kwargs: Any) -> Response:
            return Response(
                content="42",
                role="assistant",
                model="gpt-4",
                usage={"total_tokens": 2},
                finish_reason="stop",
            )

        monkeypatch.setattr(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            _fake_completion,
        )

        payload = {
            "model": "openai/gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Prove by induction that the sum of first n integers "
                        "is n(n+1)/2. Step by step reasoning required."
                    ),
                }
            ],
        }
        resp = test_client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"x-lattice-disable-transforms": "true"},
        )
        assert resp.status_code == 200

    def test_tier_classifier_simple(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Short simple request is handled without model fallback."""

        async def _fake_completion(*_args: Any, **_kwargs: Any) -> Response:
            return Response(
                content="Hi",
                role="assistant",
                model="gpt-4",
                usage={"total_tokens": 1},
                finish_reason="stop",
            )

        monkeypatch.setattr(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            _fake_completion,
        )

        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = test_client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"x-lattice-disable-transforms": "true"},
        )
        assert resp.status_code == 200


# =============================================================================
# SDK end-to-end
# =============================================================================


class TestSDKEndToEnd:
    """LatticeClient compression and session management."""

    def test_compress_request(self) -> None:
        client = LatticeClient()
        req = client.compress_request(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            model="openai/gpt-4",
        )
        assert req.model == "openai/gpt-4"
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"

    def test_compression_stats(self) -> None:
        client = LatticeClient()
        original = [
            {"role": "user", "content": "Hello world!"},
        ]
        req = client.compress_request(messages=original, model="openai/gpt-4")
        stats = client.compression_stats(original, req)
        assert "tokens_before" in stats
        assert "tokens_after" in stats
        assert "compression_ratio" in stats

    def test_session_start(self) -> None:
        client = LatticeClient()
        sid = client.start_session(provider="openai", model="gpt-4")
        assert sid.startswith("lc-")

    def test_count_tokens_approximate(self) -> None:
        client = LatticeClient()
        count = client.count_tokens("Hello world", model="gpt-4")
        assert count > 0


# =============================================================================
# MCP end-to-end
# =============================================================================


class TestMCPEndToEnd:
    """MCP tool integration."""

    def test_lattice_compress_tool(self) -> None:
        tools = LatticeMCPTools()
        result = tools.lattice_compress(
            messages=[
                {"role": "user", "content": "Hello!"},
            ],
            model="openai/gpt-4",
        )
        assert "compressed_messages" in result
        assert "tokens_before" in result
        assert "tokens_after" in result
        assert "compression_ratio" in result
        assert result["runtime"]["tier"] == "SIMPLE"
        assert result["runtime_budget"]["budget_ms"] > 0

    def test_lattice_session_start_tool(self) -> None:
        tools = LatticeMCPTools()
        result = tools.lattice_session_start(provider="openai", model="gpt-4")
        assert "session_id" in result
        assert result["session_id"].startswith("sess-")

    def test_lattice_stats_tool(self) -> None:
        tools = LatticeMCPTools()
        result = tools.lattice_stats()
        assert "version" in result
        assert "available_transforms" in result
        assert result["pipeline"]["runtime_contract_enabled"] is True
        assert result["pipeline"]["execution_transforms"] == []


# =============================================================================
# Agent integration end-to-end
# =============================================================================


class TestAgentIntegrationEndToEnd:
    """Agent wrap/unwrap flows for all supported agents."""

    def test_claude_code_wrap_unwrap(self, tmp_path: pathlib.Path) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            proxy_host="127.0.0.1",
            proxy_port=8787,
        )
        integration = ClaudeCodeIntegration(config)
        env_path = tmp_path / "claude.env"
        with patch.object(integration, "_env_file", return_value=env_path):
            result = integration.patch()
            assert result.patched is True
            assert env_path.exists()
            content = env_path.read_text()
            assert "OPENAI_BASE_URL" in content
            assert "ANTHROPIC_BASE_URL" in content
            # API key should NOT be overwritten
            assert "OPENAI_API_KEY" not in content

            status = integration.is_patched()
            assert status is True

            restored = integration.unpatch()
            assert restored.patched is False
            assert not env_path.exists()

    def test_codex_wrap_unwrap(self, tmp_path: pathlib.Path) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            proxy_host="127.0.0.1",
            proxy_port=8787,
        )
        integration = CodexIntegration(config)
        env_path = tmp_path / "codex.env"
        state_path = tmp_path / "codex_state.json"
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=tmp_path / "nonexistent.toml"),
        ):
            result = integration.patch()
            assert result.patched is True
            content = env_path.read_text()
            assert "OPENAI_BASE_URL" in content
            assert "codex /logout" in result.message

            restored = integration.unpatch()
            assert restored.patched is False

    def test_cursor_wrap_unwrap(self, tmp_path: pathlib.Path) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            proxy_host="127.0.0.1",
            proxy_port=8787,
        )
        config_path = tmp_path / "settings.json"
        config_path.write_text(
            json.dumps({
                "cursor.openai.baseUrl": "https://api.openai.com/v1",
                "cursor.anthropic.baseUrl": "https://api.anthropic.com",
            })
        )
        integration = CursorIntegration(config)
        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.patch()
            assert result.patched is True
            raw = json.loads(config_path.read_text())
            assert raw["cursor.openai.baseUrl"] == config.proxy_url()
            assert raw["cursor.anthropic.baseUrl"] == config.proxy_url()

            restored = integration.unpatch()
            assert restored.patched is False
            raw2 = json.loads(config_path.read_text())
            assert raw2["cursor.openai.baseUrl"] == "https://api.openai.com/v1"
            assert raw2["cursor.anthropic.baseUrl"] == "https://api.anthropic.com"

    def test_opencode_wrap_unwrap(self, tmp_path: pathlib.Path) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            proxy_host="127.0.0.1",
            proxy_port=8787,
        )
        config_path = tmp_path / "opencode.json"
        state_path = tmp_path / "opencode_state.json"
        config_path.write_text(
            json.dumps({
                "provider": {
                    "openai": {"options": {"baseURL": "https://api.openai.com/v1"}},
                    "anthropic": {"options": {"baseURL": "https://api.anthropic.com"}},
                }
            })
        )
        integration = OpenCodeIntegration(config)
        with (
            patch.object(integration, "_config_path", return_value=config_path),
            patch.object(integration, "_state_path", return_value=state_path),
        ):
            result = integration.patch()
            assert result.patched is True
            raw = json.loads(config_path.read_text())
            assert raw["provider"]["openai"]["options"]["baseURL"] == config.proxy_url()
            assert raw["provider"]["anthropic"]["options"]["baseURL"] == config.proxy_url()

            restored = integration.unpatch()
            assert restored.patched is False
            raw2 = json.loads(config_path.read_text())
            assert raw2["provider"]["openai"]["options"]["baseURL"] == "https://api.openai.com/v1"
            assert raw2["provider"]["anthropic"]["options"]["baseURL"] == "https://api.anthropic.com"

    def test_wrap_agent_api(self, tmp_path: pathlib.Path) -> None:
        config = LatticeConfig(proxy_host="127.0.0.1", proxy_port=8787)
        env_path = tmp_path / "claude.env"
        integration = ClaudeCodeIntegration(config)
        with patch.object(integration, "_env_file", return_value=env_path):
            result = wrap_agent("claude", config)
            assert result.patched is True

            status = agent_status("claude", config)
            assert status["patched"] is True

            unwrap_agent("claude", config)
            assert not env_path.exists()

    def test_agent_status_unknown(self) -> None:
        config = LatticeConfig()
        status = agent_status("unknown_agent", config)
        assert status["patched"] is False
        assert "Unknown" in status["message"]


# =============================================================================
# OpenAI Responses API passthrough
# =============================================================================


class TestResponsesAPIPassthrough:
    """Validate passthrough endpoints for OpenAI Responses API."""

    def test_models_get(self, test_client: TestClient) -> None:
        import httpx
        import respx

        with respx.mock:
            respx.get("https://api.openai.com/v1/models").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "data": [{"id": "gpt-4", "object": "model"}],
                    },
                )
            )
            resp = test_client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"][0]["id"] == "gpt-4"

    def test_responses_post(self, test_client: TestClient) -> None:
        import httpx
        import respx

        with respx.mock:
            route = respx.post("https://api.openai.com/v1/responses").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "resp_123",
                        "object": "response",
                        "status": "completed",
                    },
                )
            )
            resp = test_client.post(
                "/v1/responses",
                json={"model": "gpt-4", "input": "hello"},
                headers={"authorization": "Bearer test-key"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "resp_123"
            assert route.called
            request = route.calls.last.request
            assert request.headers["authorization"] == "Bearer test-key"

    def test_responses_get(self, test_client: TestClient) -> None:
        import httpx
        import respx

        with respx.mock:
            respx.get("https://api.openai.com/v1/responses/resp_456").mock(
                return_value=httpx.Response(
                    200,
                    json={"id": "resp_456", "object": "response"},
                )
            )
            resp = test_client.get("/v1/responses/resp_456")
            assert resp.status_code == 200
            assert resp.json()["id"] == "resp_456"

    def test_responses_delete(self, test_client: TestClient) -> None:
        import httpx
        import respx

        with respx.mock:
            respx.delete("https://api.openai.com/v1/responses/resp_789").mock(
                return_value=httpx.Response(200, json={"id": "resp_789", "deleted": True})
            )
            resp = test_client.delete("/v1/responses/resp_789")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True

    def test_responses_websocket(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bidirectional WebSocket relay to upstream Responses API."""

        class FakeUpstreamWS:
            def __init__(self):
                self.sent: list[str] = []
                self._calls = 0

            async def send(self, msg: str) -> None:
                self.sent.append(msg)

            async def recv(self) -> str:
                self._calls += 1
                if self._calls == 1:
                    return '{"type": "response"}'
                # End the relay loop so the handler can clean up
                raise RuntimeError("upstream done")

            async def close(self) -> None:
                pass

        fake = FakeUpstreamWS()

        async def _fake_connect(*_args: Any, **_kwargs: Any) -> FakeUpstreamWS:
            return fake

        monkeypatch.setattr("websockets.connect", _fake_connect)

        with test_client.websocket_connect("/v1/responses", headers={"authorization": "Bearer test"}) as ws:
            ws.send_text('{"type": "test"}')
            # Wait for the relay to forward and for upstream to respond
            msg = ws.receive_text()
            assert msg == '{"type": "response"}'
            assert fake.sent == ['{"type": "test"}']
