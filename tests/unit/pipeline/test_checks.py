"""Shared structural checks (Phase 12)."""

from __future__ import annotations

from lattice.pipeline import checks


def test_numbers_preserved_passes_when_overlap_high() -> None:
    r = checks.numbers_preserved("item 1 and 2", "item 1 and 2 done")
    assert r.passed


def test_uuids_preserved_detects_loss() -> None:
    uid = "550e8400-e29b-41d4-a716-446655440000"
    r = checks.uuids_preserved(f"id {uid}", "id missing")
    assert not r.passed
