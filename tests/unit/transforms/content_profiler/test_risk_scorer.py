"""Phase 5 — semantic risk scoring via content_profiler bridge."""

from __future__ import annotations

from lattice.transforms.content_profiler import compute_risk_score, score_request_risk
from lattice.transport.types import Message, Request


def test_compute_risk_score_returns_bounded_total() -> None:
    request = Request(
        messages=[
            Message(
                role="user",
                content='{"id": 1, "token": "secret-api-key-abc123", "status": "ok"}',
            )
        ]
    )
    risk = compute_risk_score(request)
    assert 0.0 <= risk.total <= 100.0
    assert risk.level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_score_request_risk_matches_compute_risk_score() -> None:
    request = Request(messages=[Message(role="user", content="Hello world.")])
    assert score_request_risk(request).total == compute_risk_score(request).total
