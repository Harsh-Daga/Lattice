"""CLI contract — every documented subcommand responds to --help.

Spawns the ``lattice`` script via ``python -m`` so the test is self-contained
and does not depend on PATH. Each command must:
  - exit 0
  - print usage information (mentions "Usage" or "usage" or command name)
"""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

import pytest


def run_lattice(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the lattice CLI via `python -m lattice.cli`."""
    return subprocess.run(
        [sys.executable, "-m", "lattice.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_help_ok(result: subprocess.CompletedProcess[str], where: str) -> None:
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        f"{where} exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # The lattice CLI uses rich panels; "Usage" appears in the help output.
    lowered = combined.lower()
    assert ("usage" in lowered) or ("commands" in lowered) or ("options" in lowered), (
        f"{where}: help output missing keyword.\noutput:\n{combined[:600]}"
    )


def test_help_global() -> None:
    _assert_help_ok(run_lattice(["--help"]), "lattice --help")


def test_version_global() -> None:
    result = run_lattice(["--version"])
    assert result.returncode == 0
    combined = (result.stdout or "") + (result.stderr or "")
    # Output contains a version-shaped substring (any digit).
    assert any(c.isdigit() for c in combined), f"version output:\n{combined}"


# Commands whose `--help` does NOT follow the standard usage banner shape
# today. `health` tries to reach a running proxy on --help.
# `benchmark` delegates to argparse (shows `--suite`, not Rich Usage).
_NONSTANDARD_HELP = {"health"}


@pytest.mark.parametrize(
    "command",
    [
        c["name"]
        for c in __import__("json").loads(
            (
                __import__("pathlib").Path(__file__).resolve().parents[2]
                / "docs"
                / "refactor"
                / "api-surface.json"
            ).read_text()
        )["cli"]["commands"]
    ],
)
def test_subcommand_help(command: str) -> None:
    result = run_lattice([command, "--help"])
    assert result.returncode == 0, (
        f"lattice {command} --help exit={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    if command == "benchmark":
        combined = (result.stdout or "") + (result.stderr or "")
        assert "--suite" in combined, (
            f"lattice benchmark --help missing --suite:\n{combined[:600]}"
        )
        return
    if command in _NONSTANDARD_HELP:
        return
    _assert_help_ok(result, f"lattice {command} --help")


def test_proxy_subcommands_listed_in_help() -> None:
    """`lattice proxy --help` mentions every documented subcommand."""
    result = run_lattice(["proxy", "--help"])
    assert result.returncode == 0, result.stderr
    out = (result.stdout or "") + (result.stderr or "")
    for sub in ("start", "stop", "restart", "status", "run"):
        assert sub in out, f"`lattice proxy --help` does not mention `{sub}`:\n{out[:600]}"


def test_doctor_no_args_lists_all_agents() -> None:
    """``lattice doctor`` with no arg reports on every primary agent."""
    result = run_lattice(["doctor"])
    assert result.returncode == 0, result.stderr
    combined = (result.stdout or "") + (result.stderr or "")
    for agent in ("claude", "codex", "cursor", "opencode", "copilot"):
        assert agent in combined


def test_lattice_benchmark_runs() -> None:
    """`lattice benchmark` must delegate to benchmarks/evals/cli.py without crashing."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lattice.cli",
            "benchmark",
            "--suite",
            "feature",
            "--providers",
            "ollama-cloud",
            "--provider-model",
            "ollama-cloud=kimi-k2.6:cloud",
            "--iterations",
            "1",
            "--warmup",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode in (0, 1)
    assert "Traceback" not in (result.stderr or "")


def test_supported_agent_args_present_in_lace_help(api_surface) -> None:
    """`lattice lace --help` should reference the supported agents list."""
    result = run_lattice(["lace", "--help"])
    assert result.returncode == 0
    out = (result.stdout or "") + (result.stderr or "")
    out_l = out.lower()
    # We don't require every agent name to appear, but at least one should.
    assert any(a in out_l for a in api_surface["cli"]["supported_agents"]), (
        f"lace --help mentions no supported agent:\n{out[:600]}"
    )
