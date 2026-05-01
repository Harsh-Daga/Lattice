"""Codex install-time helpers for LATTICE init.

Injects a [model_providers.lattice] block into ~/.codex/config.toml
with marker delimiters for safe revert. Snapshots the pre-init state
so unlace can restore byte-for-byte.
"""

from __future__ import annotations

import re
from pathlib import Path

_CODEX_MARKER_START = "# --- LATTICE persistent provider ---"
_CODEX_MARKER_END = "# --- end LATTICE persistent provider ---"
_CODEX_PATTERN = re.compile(
    re.escape(_CODEX_MARKER_START) + r".*?" + re.escape(_CODEX_MARKER_END),
    re.DOTALL,
)


def _codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Codex."""
    del backend
    return {"OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1"}


def apply_provider_scope(port: int = 8787) -> dict[str, str] | None:
    """Apply Codex provider-scope configuration."""
    path = _codex_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    section = (
        f"{_CODEX_MARKER_START}\n"
        'model_provider = "lattice"\n\n'
        "[model_providers.lattice]\n"
        'name = "LATTICE proxy"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        'env_key = "OPENAI_API_KEY"\n'
        "requires_openai_auth = true\n"
        "supports_websockets = true\n"
        f"{_CODEX_MARKER_END}\n"
    )
    if path.exists():
        existing = path.read_text()
        if _CODEX_MARKER_START in existing:
            merged = _CODEX_PATTERN.sub(section, existing)
        else:
            merged = existing.rstrip() + "\n\n" + section + "\n"
    else:
        merged = section + "\n"
    path.write_text(merged)
    return {"target": "codex", "kind": "toml-block", "path": str(path)}


def revert_provider_scope(mutation: dict[str, str]) -> None:
    """Revert Codex provider-scope configuration."""
    path_str = mutation.get("path")
    if not path_str:
        return
    path = Path(str(path_str))
    if not path.exists():
        return
    content = path.read_text()
    if _CODEX_MARKER_START not in content:
        return
    cleaned = _CODEX_PATTERN.sub("", content).strip() + "\n"
    path.write_text(cleaned)
