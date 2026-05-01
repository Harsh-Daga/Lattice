"""Claude install-time helpers for LATTICE init."""

from __future__ import annotations

import json
from pathlib import Path

from lattice.core.config import LatticeConfig

_CLAUDE_MARKER_START = "# --- LATTICE init provider ---"
_CLAUDE_MARKER_END = "# --- end LATTICE init provider ---"


def _claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Claude."""
    del backend
    return {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}


def apply_provider_scope(port: int = 8787) -> dict[str, object] | None:
    """Apply Claude provider-scope configuration when requested."""
    path = _claude_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text())
    env = payload.get("env")
    env_map = dict(env) if isinstance(env, dict) else {}
    url = f"http://127.0.0.1:{port}"
    previous = {name: env_map.get(name) for name in ("ANTHROPIC_BASE_URL",)}
    env_map["ANTHROPIC_BASE_URL"] = url
    payload["env"] = env_map
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"target": "claude", "kind": "json-env", "path": str(path), "previous": previous}


def revert_provider_scope(mutation: dict[str, object]) -> None:
    """Revert Claude provider-scope configuration."""
    path_str = mutation.get("path")
    if not path_str:
        return
    path = Path(str(path_str))
    if not path.exists():
        return
    payload: dict[str, object] = json.loads(path.read_text())
    env = payload.get("env")
    env_map = dict(env) if isinstance(env, dict) else {}
    previous = mutation.get("previous", {})
    for name in ("ANTHROPIC_BASE_URL",):
        if previous.get(name) is None:
            env_map.pop(name, None)
        else:
            env_map[name] = previous[name]  # type: ignore[literal-required]
    payload["env"] = env_map
    path.write_text(json.dumps(payload, indent=2) + "\n")
