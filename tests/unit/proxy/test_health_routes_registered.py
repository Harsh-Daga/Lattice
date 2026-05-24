"""Health and metrics routes are registered on the proxy app."""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.proxy.server import create_app


def test_health_routes_in_app() -> None:
    app = create_app(LatticeConfig())
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    for path in ("/healthz", "/readyz", "/startupz", "/metrics", "/stats"):
        assert path in paths, f"missing route {path}"
