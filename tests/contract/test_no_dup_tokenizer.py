"""Token counting goes through utils/token_count (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_tiktoken_import_only_in_token_count_module() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-l", "^import tiktoken|^from tiktoken", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    paths = {p.strip() for p in proc.stdout.splitlines() if p.strip()}
    assert paths == {"src/lattice/utils/token_count.py"}, (
        f"tiktoken import outside token_count: {paths}"
    )


def test_count_tokens_defined_once() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-c", "^def count_tokens\\b", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    total = sum(int(line.split(":")[-1]) for line in proc.stdout.splitlines() if line)
    assert total == 1
