"""2 GB VPS footprint gate — implemented in Phase 16."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Footprint harness deferred to Phase 16 (FORWARD_PLAN.md §3.2)",
)


def test_proxy_rss_under_ulimit_on_2gb_vps() -> None:
    """Placeholder: proxy-only under tight ulimit."""
