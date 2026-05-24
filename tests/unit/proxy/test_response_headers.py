"""LatticeHeaderMiddleware emits x-lattice-* response headers."""

from __future__ import annotations

import respx
from fastapi.testclient import TestClient
from httpx import Response

from lattice.core.config import LatticeConfig
from lattice.proxy.server import create_app


def test_healthz_has_no_required_chat_headers() -> None:
    """Health probes do not require provider credentials."""
    client = TestClient(create_app(LatticeConfig()))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_startupz_registered() -> None:
    client = TestClient(create_app(LatticeConfig()))
    response = client.get("/startupz")
    assert response.status_code == 200
    assert response.json()["status"] == "started"


@respx.mock
def test_chat_completion_lattice_headers_from_middleware() -> None:
    """Routing headers on /v1/chat/completions are applied by middleware."""
    config = LatticeConfig(
        provider_base_url="http://127.0.0.1:11434",
        provider_base_urls={"ollama": "http://127.0.0.1:11434"},
    )
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=Response(
            200,
            json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "hi"},
                "done": True,
            },
        )
    )
    client = TestClient(create_app(config))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-lattice-disable-transforms": "true"},
        json={
            "model": "ollama/llama3.2",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    assert response.headers.get("x-lattice-model")
    assert response.headers.get("x-lattice-compression") is not None
    assert response.headers.get("x-lattice-framing") == "json"
