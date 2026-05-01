"""Claude runtime helpers for LATTICE lace."""

from __future__ import annotations

DEFAULT_API_URL = "https://api.anthropic.com"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Claude integrations."""
    return f"http://127.0.0.1:{port}"


def build_launch_env(port: int, backend: str = "openai") -> dict[str, str]:
    """Build environment variables for Claude through the local proxy."""
    del backend
    url = proxy_base_url(port)
    return {"ANTHROPIC_BASE_URL": url}
