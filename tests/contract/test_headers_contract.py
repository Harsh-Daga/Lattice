"""Response header contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPAT_PKG = REPO_ROOT / "src" / "lattice" / "gateway" / "compat"
MIDDLEWARE_SRC = REPO_ROOT / "src" / "lattice" / "proxy" / "middleware.py"

_KNOWN_HEADER_GAPS: dict[str, str] = {}


def _compat_source_bundle() -> str:
    parts: list[str] = []
    for path in sorted(COMPAT_PKG.rglob("*.py")):
        parts.append(path.read_text())
    return "\n".join(parts)


def test_every_required_header_is_referenced_in_compat(api_surface) -> None:
    combined = _compat_source_bundle() + MIDDLEWARE_SRC.read_text()
    missing = [
        h
        for h in api_surface["http"]["headers_required"]
        if h not in combined and h not in _KNOWN_HEADER_GAPS
    ]
    assert not missing, (
        "Headers in api-surface.json#headers_required not found in gateway/compat/ "
        f"or proxy/middleware.py:\n  {missing}"
    )


def test_gateway_handlers_do_not_assign_x_lattice_headers() -> None:
    src = _compat_source_bundle()
    assert not re.search(r'response\.headers\["x-lattice-', src)


def test_middleware_defines_six_canonical_headers() -> None:
    from lattice.proxy.middleware import _HEADER_KEYS

    assert len(_HEADER_KEYS) == 6
    assert len({v for v in _HEADER_KEYS.values()}) == 6


def test_passthrough_ratelimit_prefix_is_respected() -> None:
    src = _compat_source_bundle()
    assert "x-ratelimit" in src.lower(), (
        "gateway/compat/ does not reference x-ratelimit (passthrough header family)"
    )


@pytest.mark.contract
def test_headers_present_on_real_request() -> None:
    pytest.skip("requires full proxy + provider configured; covered by integration suite")
