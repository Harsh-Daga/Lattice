"""Candidate scoring has one formula home (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_composite_score_defined_once() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-c", "^def composite_score\\b", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    total = sum(int(line.split(":")[-1]) for line in proc.stdout.splitlines() if line)
    assert total == 1


def test_candidate_scorer_imports_canonical_formula() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            "rg",
            "-n",
            "from lattice.ir.scoring import composite_score",
            "src/lattice/ir/transform.py",
            "src/lattice/ir/primitives.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.count("composite_score") >= 2
