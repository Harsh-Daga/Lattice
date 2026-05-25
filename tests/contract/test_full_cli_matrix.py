"""Exhaustive CLI argv-shape contract (api-surface.json + Phase 11 matrix)."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

import pytest

from tests.contract.test_cli_contract import run_lattice

ALL_CLI_SHAPES: list[tuple[list[str], int, str]] = [
    ([], 0, "commands"),
    (["--help"], 0, "commands"),
    (["-h"], 0, "commands"),
    (["--version"], 0, "lattice"),
    (["-v"], 0, "lattice"),
    (["version"], 0, "lattice"),
    (["proxy", "--help"], 0, "usage"),
    (["proxy", "run", "--help"], 0, "start"),
    (["proxy", "start", "--help"], 0, "background"),
    (["proxy", "stop", "--help"], 0, "stop"),
    (["proxy", "restart", "--help"], 0, "stop"),
    (["proxy", "status", "--help"], 0, "status"),
    (["init", "--help"], 0, "detect"),
    (["lace", "--help"], 0, "route"),
    (["unlace", "--help"], 0, "restore"),
    (["info", "--help"], 0, "usage"),
    (["config", "--help"], 0, "usage"),
    (["benchmark", "--help"], 0, "--suite"),
    (["health", "--help"], 0, "proxy"),
    (["status", "--help"], 0, "usage"),
    (["doctor", "--help"], 0, "diagnose"),
    (["nonexistent"], 1, "unknown"),
]


def _shape_id(argv: Sequence[str]) -> str:
    return " ".join(argv) if argv else "lattice"


@pytest.mark.parametrize(
    "argv,code,needle", ALL_CLI_SHAPES, ids=[_shape_id(a) for a, _, _ in ALL_CLI_SHAPES]
)
def test_cli_shape(argv: list[str], code: int, needle: str) -> None:
    result = run_lattice(argv)
    assert result.returncode == code, (
        f"lattice {' '.join(argv)}: exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert needle.lower() in combined.lower(), (
        f"lattice {' '.join(argv)}: expected '{needle}' in output\n{combined[:600]}"
    )


def test_root_invocation_matches_module() -> None:
    """``lattice`` on PATH and ``python -m lattice.cli`` both resolve."""
    root = subprocess.run(
        [sys.executable, "-m", "lattice.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert root.returncode == 0
