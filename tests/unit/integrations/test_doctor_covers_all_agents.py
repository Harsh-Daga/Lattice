"""Doctor CLI covers every primary agent."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("agent", ["claude", "codex", "cursor", "opencode", "copilot"])
def test_doctor_runs_for_each(agent: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lattice.cli", "doctor", agent],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{agent}: stderr={result.stderr}"
    combined = (result.stdout or "") + (result.stderr or "")
    assert agent in combined
    assert "Unknown agent" not in combined
