"""Integration tests for LATTICE proxy session persistence.

Tests cover:
- /lattice/session/start creates a session with manifest
- /lattice/session/append updates session messages
- /lattice/session/{id} retrieves session metadata
- /v1/chat/completions persists session after turn 1
- Session headers are returned in responses
"""

from __future__ import annotations

import json

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.transport import Response
from lattice.protocol.framing import BinaryFramer
from lattice.proxy.server import create_app


@pytest.fixture
def app():
    config = LatticeConfig(
        session_store="memory",
        provider_base_url="http://localhost:9999",
        provider_base_urls={"openai": "http://localhost:9999"},
        provider_api_key="fake",
    )
    return create_app(config)


@pytest.fixture
async def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


class TestLatticeSessionEndpoints:
    def test_session_start(self, client) -> None:
        resp = client.post(
            "/lattice/session/start",
            json={
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                "provider": "openai",
                "model": "openai/gpt-4",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"].startswith("lattice-")
        assert "anchor_version" in data
        assert "anchor_hash" in data
        assert len(data["anchor_hash"]) == 64
        assert data["manifest"]["session_id"].startswith("lattice-")
        assert data["manifest"]["segment_count"] >= 1
        assert data["cache_plan"]["provider"] == "openai"
        assert "expected_cached_tokens" in data["cache_plan"]

    def test_session_start_with_tools(self, client) -> None:
        resp = client.post(
            "/lattice/session/start",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "openai",
                "model": "openai/gpt-4",
                "tools": [{"name": "search"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["cache_plan"]["provider"] == "openai"

    def test_session_append(self, client) -> None:
        # Start session
        start = client.post(
            "/lattice/session/start",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "openai",
                "model": "openai/gpt-4",
            },
        )
        session_id = start.json()["session_id"]

        # Append
        resp = client.post(
            "/lattice/session/append",
            json={
                "session_id": session_id,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["message_count"] == 2
        assert data["anchor_version"] == 1
        assert data["manifest"]["anchor_version"] == 1
        assert data["cache_plan"]["provider"] == "openai"

    def test_session_append_missing_id(self, client) -> None:
        resp = client.post(
            "/lattice/session/append",
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "missing_session_id"

    def test_session_append_not_found(self, client) -> None:
        resp = client.post(
            "/lattice/session/append",
            json={
                "session_id": "nonexistent",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404
        assert resp.json()["error"] == "session_not_found"

    def test_session_get(self, client) -> None:
        start = client.post(
            "/lattice/session/start",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "anthropic",
                "model": "anthropic/claude-sonnet",
            },
        )
        session_id = start.json()["session_id"]

        resp = client.get(f"/lattice/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["provider"] == "anthropic"
        assert data["model"] == "anthropic/claude-sonnet"
        assert data["message_count"] == 1
        assert data["manifest"]["metadata"]["provider"] == "anthropic"
        assert data["cache_plan"]["provider"] == "anthropic"

    def test_session_get_not_found(self, client) -> None:
        resp = client.get("/lattice/session/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["error"] == "session_not_found"


class TestChatCompletionsSessionHeaders:
    def test_response_includes_session_header(self, client) -> None:
        # Even without a real provider, the proxy should create a session
        # and return its ID in headers. We mock the provider call to avoid
        # needing a real backend.
        import httpx
        import respx

        with respx.mock:
            respx.post("http://localhost:9999/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 123,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    },
                )
            )
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 200
            assert "x-lattice-session-id" in resp.headers
            session_id = resp.headers["x-lattice-session-id"]
            assert session_id.startswith("lattice-")

    def test_existing_session_reused(self, client) -> None:
        import httpx
        import respx

        with respx.mock:
            respx.post("http://localhost:9999/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 123,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    },
                )
            )
            # Turn 1
            resp1 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            session_id = resp1.headers["x-lattice-session-id"]

            # Turn 2 with same session
            resp2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4",
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "how are you"},
                    ],
                },
                headers={"x-lattice-session-id": session_id},
            )
            assert resp2.headers["x-lattice-session-id"] == session_id


class TestNativeGatewayEndpoint:
    def test_gateway_json_roundtrip(self, client, monkeypatch) -> None:
        async def _fake_completion(_self, *_args, **_kwargs):
            return Response(content="native-ok", model="openai/gpt-4")

        monkeypatch.setattr(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            _fake_completion,
        )

        resp = client.post(
            "/lattice/gateway",
            json={
                "model": "openai/gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "native-ok"

    def test_gateway_binary_roundtrip(self, client, monkeypatch) -> None:
        async def _fake_completion(_self, *_args, **_kwargs):
            return Response(content="native-binary-ok", model="openai/gpt-4")

        monkeypatch.setattr(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            _fake_completion,
        )

        framer = BinaryFramer()
        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        raw = framer.encode_request(json.dumps(payload).encode("utf-8"))[0].to_bytes()
        resp = client.post(
            "/lattice/gateway",
            content=raw,
            headers={"content-type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        frame = framer.decode_frame(resp.content)
        body = json.loads(frame.payload.decode("utf-8"))
        assert body["choices"][0]["message"]["content"] == "native-binary-ok"
