"""Streaming chunk buffers are not duplicated before Phase 27 canonical home."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Non-streaming buffers allowed until pipeline/streaming/chunk_buffer.py lands.
_ALLOWLIST_SUFFIXES = (
    "integrations/tunnel.py",  # ReplayBuffer for tunnel replay
)


def test_no_duplicate_streaming_buffer_classes() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-n", "^class \\w*Buffer\\b", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 1:
        return
    offenders: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        path = line.split(":", 1)[0]
        if any(path.endswith(suffix) for suffix in _ALLOWLIST_SUFFIXES):
            continue
        offenders.append(line)
    assert not offenders, f"unexpected Buffer classes: {offenders}"
