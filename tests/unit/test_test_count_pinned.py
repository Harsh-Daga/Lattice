"""Pin the authoritative pytest collection count (Phase 11)."""

from __future__ import annotations

import re
import subprocess
import sys

EXPECTED_TEST_COUNT = 2042


def test_test_count_matches_expected() -> None:
    """Collection count must match README badge and AGENTS.md."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    combined = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"(\d+)\s+tests?\s+collected", combined)
    assert match is not None, f"could not parse collection count:\n{combined[-800:]}"
    count = int(match.group(1))
    assert count == EXPECTED_TEST_COUNT, (
        f"Test count drifted: expected {EXPECTED_TEST_COUNT}, got {count}. "
        "Update EXPECTED_TEST_COUNT and the README + AGENTS.md badge in the same PR."
    )
