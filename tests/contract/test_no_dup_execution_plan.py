"""Single ExecutionPlan class (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import count_lines_matching, repo_root, src_lattice


def test_one_execution_plan_class() -> None:
    root = repo_root(__file__)
    n = count_lines_matching(src_lattice(root), re.compile(r"^class ExecutionPlan\b"))
    assert n == 1


def test_planner_reexports_ir_execution_plan() -> None:
    from lattice.ir.primitives import ExecutionPlan as IrPlan
    from lattice.planner import ExecutionPlan as PlannerPlan

    assert IrPlan is PlannerPlan
