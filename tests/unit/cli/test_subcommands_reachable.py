"""CLI subcommands respond to --help without error."""

from __future__ import annotations

import subprocess

import pytest


@pytest.mark.parametrize(
    "cmd",
    [
        ["lattice", "--version"],
        ["lattice", "version"],
        ["lattice", "--help"],
        ["lattice", "proxy", "--help"],
        ["lattice", "proxy", "run", "--help"],
        ["lattice", "proxy", "start", "--help"],
        ["lattice", "proxy", "stop", "--help"],
        ["lattice", "proxy", "restart", "--help"],
        ["lattice", "proxy", "status", "--help"],
        ["lattice", "init", "--help"],
        ["lattice", "lace", "--help"],
        ["lattice", "unlace", "--help"],
        ["lattice", "info", "--help"],
        ["lattice", "config", "--help"],
        ["lattice", "benchmark", "--help"],
        ["lattice", "health", "--help"],
        ["lattice", "status", "--help"],
        ["lattice", "doctor", "--help"],
    ],
)
def test_subcommand_help(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{cmd}: stderr={result.stderr}"
