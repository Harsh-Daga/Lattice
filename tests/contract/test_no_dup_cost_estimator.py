"""Cost estimation has one implementation (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import count_lines_matching, repo_root, src_lattice


def test_one_cost_estimator_class() -> None:
    root = repo_root(__file__)
    n = count_lines_matching(src_lattice(root), re.compile(r"^class CostEstimator\b"))
    assert n == 1


def test_no_duplicate_cost_function_names() -> None:
    root = repo_root(__file__)
    n = count_lines_matching(
        src_lattice(root),
        re.compile(r"^def (estimate_cost|compute_cost|calc_cost)\b"),
    )
    assert n == 0
