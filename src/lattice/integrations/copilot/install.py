"""Copilot install-time helpers for LATTICE init."""

from __future__ import annotations

import json
from pathlib import Path

from lattice.integrations.copilot.runtime import build_launch_env


def _copilot_config_path() -> Path:
    return Path.home() / ".copilot" / "config.json"


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Copilot."""
    del backend
    return build_launch_env(port=port, environ={})


def apply_provider_scope(port: int = 8787) -> dict[str, str] | None:
    """Apply Copilot provider-scope configuration."""
    del port
    path = _copilot_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text())
    hooks = payload.get("hooks")
    hooks_map = dict(hooks) if isinstance(hooks, dict) else {}
    # We store a marker that lattice init ran
    hooks_map["lattice_init"] = True
    payload["hooks"] = hooks_map
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"target": "copilot", "kind": "json-hooks", "path": str(path)}


def revert_provider_scope(mutation: dict[str, str]) -> None:
    """Revert Copilot provider-scope configuration."""
    path_str = mutation.get("path")
    if not path_str:
        return
    path = Path(str(path_str))
    if not path.exists():
        return
    payload: dict[str, object] = json.loads(path.read_text())
    hooks = payload.get("hooks")
    hooks_map = dict(hooks) if isinstance(hooks, dict) else {}
    hooks_map.pop("lattice_init", None)
    if not hooks_map:
        payload.pop("hooks", None)
    else:
        payload["hooks"] = hooks_map
    path.write_text(json.dumps(payload, indent=2) + "\n")
