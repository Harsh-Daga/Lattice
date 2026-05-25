"""Candidate scoring has one formula home (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import count_lines_matching, repo_root, src_lattice


def test_composite_score_defined_once() -> None:
    root = repo_root(__file__)
    n = count_lines_matching(src_lattice(root), re.compile(r"^def composite_score\b"))
    assert n == 1


def test_candidate_scorer_imports_canonical_formula() -> None:
    root = repo_root(__file__)
    transform = (root / "src/lattice/ir/transform.py").read_text(encoding="utf-8")
    primitives = (root / "src/lattice/ir/primitives.py").read_text(encoding="utf-8")
    assert "from lattice.ir.scoring import composite_score" in transform
    assert "from lattice.ir.scoring import composite_score" in primitives
