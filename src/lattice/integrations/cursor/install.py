"""Cursor install-time helpers for LATTICE init.

Cursor is GUI-based — we print setup instructions rather than mutating files.
"""

from __future__ import annotations


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Cursor."""
    from lattice.integrations.cursor.runtime import build_proxy_targets

    del backend
    targets = build_proxy_targets(port)
    return {
        "OPENAI_BASE_URL": targets.openai_base_url,
        "ANTHROPIC_BASE_URL": targets.anthropic_base_url,
    }


def apply_provider_scope(port: int = 8787) -> dict[str, str] | None:
    """Cursor has no config file to patch — return instructions marker."""
    del port
    return {"target": "cursor", "kind": "instructions", "path": ""}


def revert_provider_scope(_mutation: dict[str, str]) -> None:
    """No-op for Cursor."""
    pass
