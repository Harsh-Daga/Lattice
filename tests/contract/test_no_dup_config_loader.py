"""LatticeConfig loading is centralized in core/config (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path

# YAML/TOML loads allowed outside core/config.py (not LatticeConfig entry points).
_ALLOWLIST = (
    "src/lattice/core/config.py",
    "src/lattice/transforms/format_converter/json_converter.py",
    "src/lattice/integrations/agents/env_builder.py",
    "src/lattice/providers/credentials.py",
)


def test_yaml_safe_load_only_in_allowlisted_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-l", "yaml\\.safe_load", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 1:
        return
    paths = [p.strip() for p in proc.stdout.splitlines() if p.strip()]
    offenders = [p for p in paths if p not in _ALLOWLIST]
    assert not offenders, f"yaml.safe_load outside allowlist: {offenders}"
