"""OpenCode install-time helpers for LATTICE init.

OpenCode supports a ``provider`` dict where each provider gets its own
``options`` block.  We inject every LATTICE-supported provider so that
regardless of which one the user has selected in OpenCode, traffic routes
through LATTICE.  The proxy reads ``x-lattice-provider`` to know which
upstream provider to use.

Example of what we write::

    {
      "provider": {
        "openai": {
          "options": {
            "baseURL": "http://127.0.0.1:8787/v1",
            "headers": {"x-lattice-provider": "openai"}
          }
        },
        "anthropic": { ... },
        ...
      }
    }
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from lattice.providers.capabilities import get_capability_registry


def _opencode_config_path() -> Path:
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _make_provider_options(port: int, provider: str) -> dict[str, Any]:
    """Build the options dict for a single OpenCode provider."""
    return {
        "options": {
            "baseURL": f"http://127.0.0.1:{port}/v1",
            "headers": {"x-lattice-provider": provider},
        }
    }


def _snapshot_providers(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Deep-copy the existing ``provider`` section so we can restore it."""
    existing = payload.get("provider")
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    return None


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for OpenCode.

    OpenCode reads per-provider ``baseURL`` from its JSON config, so
    env vars are only a fallback for providers that don't read config.
    """
    del backend
    return {"OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1"}


def apply_provider_scope(port: int = 8787) -> dict[str, Any] | None:
    """Apply OpenCode provider-scope configuration.

    Injects a ``provider.*.options`` block for every LATTICE-supported
    provider that is OpenAI-compatible.
    """
    path = _opencode_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = {}

    previous = _snapshot_providers(payload)

    providers: dict[str, Any] = {}
    for provider in get_capability_registry().list_providers():
        providers[provider] = _make_provider_options(port, provider)

    payload["provider"] = providers
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "target": "opencode",
        "kind": "json-providers",
        "path": str(path),
        "previous": previous,
    }


def revert_provider_scope(mutation: dict[str, Any]) -> None:
    """Revert OpenCode provider-scope configuration."""
    path_str = mutation.get("path")
    if not path_str:
        return
    path = Path(str(path_str))
    if not path.exists():
        return

    payload: dict[str, Any] = json.loads(path.read_text())
    previous = mutation.get("previous")

    if previous is None:
        payload.pop("provider", None)
    else:
        payload["provider"] = previous

    path.write_text(json.dumps(payload, indent=2) + "\n")
