"""LATTICE proxy server components.

This module contains the FastAPI proxy application that receives
OpenAI-compatible requests, runs LATTICE compression transforms, and
routes to LLM providers via DirectHTTPProvider (our own transport).

Usage:
    from lattice.proxy.server import create_app
    app = create_app()

    # Or from CLI:
    uvicorn lattice.proxy.server:create_app --factory
"""

from lattice.providers.transport import DirectHTTPProvider
from lattice.proxy.server import create_app

__all__ = [
    "DirectHTTPProvider",
    "create_app",
]
