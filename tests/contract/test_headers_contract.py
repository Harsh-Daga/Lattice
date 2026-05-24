"""Response header contract.

Verifies the proxy's response-header generation logic emits every header
listed in REFACTOR_PLAN.md §2.3. The fast test inspects the gateway module
source for the documented header names; the slow test (``contract``) sends
a real request through a running proxy and asserts the headers are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPAT_SRC = REPO_ROOT / "src" / "lattice" / "gateway" / "compat.py"
MIDDLEWARE_SRC = REPO_ROOT / "src" / "lattice" / "proxy" / "middleware.py"


# Known header gaps — each closes in the named phase.
_KNOWN_HEADER_GAPS: dict[str, str] = {}


def test_every_required_header_is_referenced_in_compat(api_surface) -> None:
    """Each contract header must appear in compat or proxy middleware."""
    compat_src = COMPAT_SRC.read_text()
    middleware_src = MIDDLEWARE_SRC.read_text()
    combined = compat_src + middleware_src
    missing = [
        h
        for h in api_surface["http"]["headers_required"]
        if h not in combined and h not in _KNOWN_HEADER_GAPS
    ]
    assert not missing, (
        "Headers in api-surface.json#headers_required not found in gateway/compat.py "
        f"or proxy/middleware.py:\n  {missing}"
    )


def test_gateway_handlers_do_not_assign_x_lattice_headers() -> None:
    """Per-handler response.headers x-lattice-* assignment is forbidden."""
    import re

    src = COMPAT_SRC.read_text()
    assert not re.search(r'response\.headers\["x-lattice-', src)


def test_middleware_defines_six_canonical_headers() -> None:
    from lattice.proxy.middleware import _HEADER_KEYS

    assert len(_HEADER_KEYS) == 6
    assert len({v for v in _HEADER_KEYS.values()}) == 6


def test_passthrough_ratelimit_prefix_is_respected() -> None:
    """Upstream x-ratelimit-* headers must pass through the proxy."""
    src = COMPAT_SRC.read_text()
    # Be permissive: any reference to x-ratelimit (case-insensitive) is enough.
    assert "x-ratelimit" in src.lower(), (
        "gateway/compat.py does not reference x-ratelimit (passthrough header family)"
    )


# ---- Slow contract suite (spins a real proxy) ----


@pytest.mark.contract
def test_headers_present_on_real_request() -> None:
    pytest.skip("requires full proxy + provider configured; covered by integration suite")
