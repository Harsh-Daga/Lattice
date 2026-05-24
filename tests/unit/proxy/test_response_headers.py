"""LatticeHeaderMiddleware emits x-lattice-* response headers."""

from __future__ import annotations

from fastapi.testclient import TestClient

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
