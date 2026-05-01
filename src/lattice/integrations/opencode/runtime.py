"""OpenCode runtime helpers for LATTICE lace.

OpenCode's JSON config (``~/.config/opencode/opencode.json``) contains a
``provider`` dict where each provider has its own ``options.baseURL``.
``lattice init opencode`` mutates that file so **every** supported
provider routes through LATTICE with an ``x-lattice-provider`` header.

The runtime env var here is only a fallback for providers that read
``OPENAI_BASE_URL`` instead of their config block.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build environment variables for OpenCode through the local proxy.

    In practice OpenCode reads per-provider ``baseURL`` from its JSON
    config file (written by ``lattice init``).  The env var here is a
    fallback for providers that don't read the config block.
    """
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    env["OPENAI_BASE_URL"] = base_url
    return env
