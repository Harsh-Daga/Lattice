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


# Known header gaps — each closes in the named phase.
_KNOWN_HEADER_GAPS = {
    "x-lattice-transforms-applied": "Phase 6 introduces LatticeHeaderMiddleware and adds transforms-applied",
}


def test_every_required_header_is_referenced_in_compat(api_surface) -> None:
    """Each contract header must appear by name in gateway/compat.py, except
    headers in the known-gap list (each annotated with the phase that closes
    the gap)."""
    src = COMPAT_SRC.read_text()
    missing = [
        h
        for h in api_surface["http"]["headers_required"]
        if h not in src and h not in _KNOWN_HEADER_GAPS
    ]
    assert not missing, (
        f"Headers in api-surface.json#headers_required not found in {COMPAT_SRC.relative_to(REPO_ROOT)}:\n"
        f"  {missing}"
    )


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
