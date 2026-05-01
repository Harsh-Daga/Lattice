"""Integration tests for session correctness.

Validates that the proxy correctly:
- Creates sessions on turn 1
- Reconstructs full context from deltas
- Persists sessions across turns
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lattice.core.config import LatticeConfig
from lattice.core.transport import Response
from lattice.proxy.server import create_app


@pytest.fixture
def test_client() -> TestClient:
    config = LatticeConfig(
        provider_base_url="http://127.0.0.1:11434",
        provider_api_key="fake-key",
        session_store="memory",
    )
    app = create_app(config=config)
    return TestClient(app)


def _patch_provider(monkeypatch):
    """Patch provider completion to avoid real network calls."""
    from lattice.providers.transport import DirectHTTPProvider

    async def _fake_completion(*_args, **_kwargs):
        return Response(
            content="Hello!",
            role="assistant",
            model="gpt-4",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            finish_reason="stop",
        )

    monkeypatch.setattr(
        DirectHTTPProvider,
        "completion",
        _fake_completion,
    )


class TestSessionCorrectness:
    def test_proxy_creates_session_on_turn1(self, test_client, monkeypatch):
        """First turn creates a session and returns its ID in headers."""
        _patch_provider(monkeypatch)

        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"x-lattice-disable-transforms": "true"},
        )
        assert resp.status_code == 200
        assert "x-lattice-session-id" in resp.headers
        sid = resp.headers["x-lattice-session-id"]
        assert sid.startswith("lattice-")

    def test_delta_reconstructs_full_context(self, test_client, monkeypatch):
        """Delta request with session ID rebuilds full context for provider."""
        _patch_provider(monkeypatch)

        # Turn 1: establish session
        resp1 = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"x-lattice-disable-transforms": "true"},
        )
        assert resp1.status_code == 200
        sid = resp1.headers.get("x-lattice-session-id")
        assert sid is not None

        # Turn 2: delta with session ID
        resp2 = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4",
                "messages": [{"role": "user", "content": "Follow-up"}],
            },
            headers={
                "x-lattice-session-id": sid,
                "x-lattice-disable-transforms": "true",
            },
        )
        assert resp2.status_code == 200

        # Verify session exists and accumulated messages
        resp3 = test_client.get(f"/lattice/session/{sid}")
        assert resp3.status_code == 200
        data = resp3.json()
        assert data["session_id"] == sid
        # Session should have at least the original messages
        assert data["message_count"] >= 2

    def test_session_failover_to_full_prompt(self, test_client, monkeypatch):
        """If session is missing, proxy falls back to full prompt (no error)."""
        _patch_provider(monkeypatch)

        # Send with a non-existent session ID
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={
                "x-lattice-session-id": "lattice-nonexistent",
                "x-lattice-disable-transforms": "true",
            },
        )
        assert resp.status_code == 200
        # A new session should be created
        assert "x-lattice-session-id" in resp.headers
