"""Phase 10 — lattice benchmark wraps benchmarks/evals/cli.py."""

from __future__ import annotations

import subprocess
import sys


def _run_benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lattice.cli", "benchmark", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_benchmark_invokes_real_cli() -> None:
    """`lattice benchmark --help` shows the real benchmark CLI, not the redirect stub."""
    result = _run_benchmark("--help")
    assert result.returncode == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "has moved" not in combined
    assert "--suite" in combined


def test_benchmark_no_v2_flag() -> None:
    """`--use-v2-pipeline` must not be accepted."""
    result = _run_benchmark("--use-v2-pipeline")
    assert result.returncode != 0
    err = (result.stderr or "").lower()
    assert "unrecognized" in err or "unknown" in err
