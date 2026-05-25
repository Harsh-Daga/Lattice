"""All six contract-required x-lattice-* headers on a successful compat route."""

from __future__ import annotations

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from lattice.core.config import LatticeConfig
from lattice.proxy.server import create_app

REQUIRED_HEADERS = {
    "x-lattice-compression",
    "x-lattice-session-id",
    "x-lattice-delta",
    "x-lattice-cost-usd",
    "x-lattice-provider",
    "x-lattice-transforms-applied",
}


@respx.mock
def test_all_six_lattice_headers_present_on_chat_completion() -> None:
    """Successful /v1/chat/completions must emit every api-surface header."""
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
    seen = {name.lower() for name in response.headers}
    missing = REQUIRED_HEADERS - seen
    assert not missing, (
        f"Missing required headers: {missing}\n"
        f"Present lattice: {sorted(h for h in seen if h.startswith('x-lattice'))}"
    )


@pytest.mark.contract
def test_headers_on_live_proxy_chat(proxy_subprocess: int) -> None:
    """Live proxy returns lattice headers (status may vary without upstream keys)."""
    url = f"http://127.0.0.1:{proxy_subprocess}/v1/chat/completions"
    import httpx

    response = httpx.post(
        url,
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout=15,
    )
    assert response.status_code in (200, 401, 403, 502)
    if response.status_code != 200:
        pytest.skip("upstream not configured for live header probe")
    seen = {name.lower() for name in response.headers}
    missing = REQUIRED_HEADERS - seen
    assert not missing, f"Missing headers on live proxy: {missing}"
