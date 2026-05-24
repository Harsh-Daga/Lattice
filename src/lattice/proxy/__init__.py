"""LATTICE proxy server components.

Usage::

    from lattice.proxy.server import create_app
    app = create_app()

    from lattice.proxy import HealthManager
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lattice.core.config import LatticeConfig
from lattice.providers.transport import DirectHTTPProvider
from lattice.proxy.health import HealthManager

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = [
    "DirectHTTPProvider",
    "HealthManager",
    "create_app",
]


def create_app(config: LatticeConfig | None = None) -> FastAPI:
    """Lazy import to avoid a gateway.compat ↔ proxy.server cycle."""
    from lattice.proxy.server import create_app as _create_app

    return _create_app(config)
