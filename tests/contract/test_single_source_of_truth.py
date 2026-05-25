"""SSOT registry paths exist; core Phase 13 invariants hold."""

from __future__ import annotations

import subprocess
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src/lattice"
_ROOT = Path(__file__).resolve().parents[2]

_PHASE_13_REQUIRED = (
    "src/lattice/ir/primitives.py",
    "src/lattice/ir/scoring.py",
    "src/lattice/pipeline/post_transform_guard.py",
    "src/lattice/pipeline/checks.py",
    "src/lattice/pipeline/request_coalescer.py",
    "src/lattice/pipeline/_generated_factories.py",
    "src/lattice/runtime/validation_engine.py",
    "src/lattice/transforms/registry.py",
    "src/lattice/telemetry/cost_estimator.py",
    "src/lattice/state/session.py",
    "src/lattice/utils/token_count.py",
    "src/lattice/proxy/middleware.py",
    "src/lattice/gateway/compat/headers.py",
)


def test_phase_13_ssot_paths_exist() -> None:
    """Honesty-pass registry rows that must exist on disk today."""
    missing = [p for p in _PHASE_13_REQUIRED if not (_ROOT / p).is_file()]
    assert not missing, missing


def test_ssot_execution_plan_and_scoring_homes() -> None:
    assert (_SRC / "ir/primitives.py").is_file()
    assert (_SRC / "ir/scoring.py").is_file()
    assert not (_SRC / "planner/execution_plan.py").exists()


def test_check_internal_no_duplication_script_passes() -> None:
    proc = subprocess.run(
        ["bash", "scripts/check_internal_no_duplication.sh"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
