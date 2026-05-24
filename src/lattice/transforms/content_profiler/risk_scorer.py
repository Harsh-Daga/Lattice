"""Semantic risk scoring for content profiling."""

from __future__ import annotations

from lattice.safety.risk_scoring import SemanticRiskScore, compute_risk_score
from lattice.transport.types import Request


def score_request_risk(request: Request) -> SemanticRiskScore:
    """Compute semantic risk score for the request."""
    return compute_risk_score(request)


__all__ = ["SemanticRiskScore", "compute_risk_score", "score_request_risk"]
