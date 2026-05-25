"""Single ExecutionPlan class (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_one_execution_plan_class() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-c", "^class ExecutionPlan\\b", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    total = sum(int(line.split(":")[-1]) for line in proc.stdout.splitlines() if line)
    assert total == 1


def test_planner_reexports_ir_execution_plan() -> None:
    from lattice.ir.primitives import ExecutionPlan as IrPlan
    from lattice.planner import ExecutionPlan as PlannerPlan

    assert IrPlan is PlannerPlan
