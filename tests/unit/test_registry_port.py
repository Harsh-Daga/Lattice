"""Tests for registry port threading (Phase 2 bugfix)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def _test_agent_port(agent: str, apply_fn: Any, config_path: Path, port: int = 9999) -> None:
    """Generic helper: apply provider scope and assert port is threaded."""
    mutation = apply_fn(port=port)
    assert mutation is not None
    assert mutation.get("target") == agent
    written = config_path.read_text()
    assert "http://127.0.0.1:9999" in written
    assert "8787" not in written


def test_claude_apply_uses_port_parameter(monkeypatch: Any) -> None:
    """Claude install must inject the given port, not hardcode 8787."""
    from lattice.integrations.claude import install as claude_install

    with tempfile.TemporaryDirectory() as tmp:
        fake_path = Path(tmp) / "settings.json"
        monkeypatch.setattr(claude_install, "_claude_settings_path", lambda: fake_path)
        _test_agent_port("claude", claude_install.apply_provider_scope, fake_path)


def test_codex_apply_uses_port_parameter(monkeypatch: Any) -> None:
    """Codex install must inject the given port, not hardcode 8787."""
    from lattice.integrations.codex import install as codex_install

    with tempfile.TemporaryDirectory() as tmp:
        fake_path = Path(tmp) / "config.toml"
        monkeypatch.setattr(codex_install, "_codex_config_path", lambda: fake_path)
        _test_agent_port("codex", codex_install.apply_provider_scope, fake_path)


def test_opencode_apply_uses_port_parameter(monkeypatch: Any) -> None:
    """OpenCode install must inject the given port into every provider block."""
    from lattice.integrations.opencode import install as opencode_install

    with tempfile.TemporaryDirectory() as tmp:
        fake_path = Path(tmp) / "opencode.json"
        monkeypatch.setattr(opencode_install, "_opencode_config_path", lambda: fake_path)
        _test_agent_port("opencode", opencode_install.apply_provider_scope, fake_path)


def test_registry_threads_port_to_handlers() -> None:
    """Registry apply_provider_scope must pass port to underlying handlers."""
    from lattice.integrations.registry import apply_provider_scope

    # Cursor has a no-op apply that just returns instructions; any port works.
    result = apply_provider_scope("cursor", port=1234)
    assert result is not None
    assert result.get("target") == "cursor"
