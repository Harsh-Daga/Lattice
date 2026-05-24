"""Semantic risk scoring for content profiling."""

from __future__ import annotations

from lattice.transport.types import Request
from lattice.utils.validation import SemanticRiskScore, compute_risk_score


def score_request_risk(request: Request) -> SemanticRiskScore:
    """Compute semantic risk score for the request."""
    return compute_risk_score(request)


__all__ = ["SemanticRiskScore", "compute_risk_score", "score_request_risk"]
