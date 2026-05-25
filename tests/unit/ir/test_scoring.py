"""Canonical composite_score (Phase 12)."""

from __future__ import annotations

from lattice.ir.scoring import composite_score


def test_composite_score_matches_quality_dominant() -> None:
    s = composite_score({"quality_estimate": 0.9, "tokens_before": 100, "tokens_after": 50})
    assert s.composite >= 0.9


def test_composite_score_quality_floor_penalty() -> None:
    s = composite_score(
        {"quality_estimate": 0.5, "quality_floor": 0.85, "tokens_before": 10, "tokens_after": 10}
    )
    assert "quality" in s.reason
