"""Copilot runtime helpers for LATTICE lace."""

from __future__ import annotations

import os
from collections.abc import Mapping


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build environment variables for Copilot BYOK through the local proxy."""
    env = dict(environ or os.environ)
    env["COPILOT_PROVIDER_TYPE"] = "openai"
    env["COPILOT_PROVIDER_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["COPILOT_PROVIDER_WIRE_API"] = "completions"
    return env
