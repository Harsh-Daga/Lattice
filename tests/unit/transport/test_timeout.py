"""Timeout resolver."""

from __future__ import annotations

from lattice.transport.timeout import TimeoutResolver


def test_resolve_defaults() -> None:
    tr = TimeoutResolver(default_seconds=90.0)
    policy = tr.resolve()
    assert policy.read == 90.0
    assert policy.for_attempt(2) == 90.0
