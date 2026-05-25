"""Cost estimation has one implementation (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_one_cost_estimator_class() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-c", "^class CostEstimator\\b", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    total = sum(int(line.split(":")[-1]) for line in proc.stdout.splitlines() if line)
    assert total == 1


def test_no_duplicate_cost_function_names() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            "rg",
            "-n",
            "^def (estimate_cost|compute_cost|calc_cost)\\b",
            "src/lattice",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout or proc.stderr
