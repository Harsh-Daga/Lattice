"""Cold-start latency gate — implemented in Phase 16."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Footprint harness deferred to Phase 16 (FORWARD_PLAN.md §3.2)",
)


def test_first_request_within_cold_start_budget() -> None:
    """Placeholder: first served request within 1.5 s default install."""
