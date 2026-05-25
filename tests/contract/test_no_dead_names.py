"""Stale misleading names must not appear in src/ (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_no_stale_names_in_src_and_tests() -> None:
    root = Path(__file__).resolve().parents[2]
    pattern = (
        r"MILV|BatchAccumulator|strategy_selector|constraint_lifting|"
        r"information_theoretic_selector"
    )
    proc = subprocess.run(
        [
            "rg",
            pattern,
            "src/",
            "--glob",
            "!src/lattice/core/config.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout or proc.stderr
