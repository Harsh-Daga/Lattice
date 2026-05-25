"""Session domain type has one home (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_one_session_class_in_state() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-c", "^class Session\\b", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    total = sum(int(line.split(":")[-1]) for line in proc.stdout.splitlines() if line)
    assert total == 1


def test_session_import_path() -> None:
    from lattice.state.session import Session

    assert Session.__module__ == "lattice.state.session"
