"""Canonical candidate scoring — single formula for beam search and IR primitives."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from lattice.ir.primitives import CandidateScore


@dataclasses.dataclass(frozen=True, slots=True)
class ScoringWeights:
    cost: float = 0.5
    cache: float = 0.2
    transport: float = 0.2
    latency: float = 1.0


DEFAULT_WEIGHTS = ScoringWeights()


def composite_score(
    metrics: Mapping[str, Any],
    *,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    quality_floor: float | None = None,
) -> CandidateScore:
    """Compute canonical score from metric mapping."""
    tokens_before = int(metrics.get("tokens_before", 1) or 1)
    tokens_after = int(metrics.get("tokens_after", tokens_before) or tokens_before)
    latency_ms = float(metrics.get("latency_ms", 0.0) or 0.0)
    quality_estimate = float(metrics.get("quality_estimate", 1.0) or 1.0)
    cache_gain = float(metrics.get("cache_gain", 0.0) or 0.0)
    transport_gain = float(metrics.get("transport_gain", 0.0) or 0.0)
    semantic_risk = float(metrics.get("semantic_risk", 0.0) or 0.0)
    instability = float(metrics.get("instability", 0.0) or 0.0)
    floor = float(
        quality_floor if quality_floor is not None else metrics.get("quality_floor", 0.85) or 0.85
    )

    cost_reduction = max(0.0, (tokens_before - tokens_after) / max(1, tokens_before))
    latency_cost = latency_ms / 1000.0

    if quality_estimate < floor:
        return CandidateScore(
            composite=quality_estimate,
            quality=quality_estimate,
            cost_reduction=round(cost_reduction, 4),
            cache_gain=round(cache_gain, 4),
            transport_gain=round(transport_gain, 4),
            semantic_risk=round(semantic_risk, 4),
            latency_ms=round(latency_ms, 3),
            instability_penalty=round(instability, 4),
            reason=f"quality {quality_estimate:.2f} < floor {floor:.2f}",
        )

    composite = (
        quality_estimate
        + weights.cost * cost_reduction
        + weights.cache * cache_gain
        + weights.transport * transport_gain
        - semantic_risk
        - weights.latency * latency_cost
        - instability
    )

    return CandidateScore(
        composite=round(composite, 4),
        quality=quality_estimate,
        cost_reduction=round(cost_reduction, 4),
        cache_gain=round(cache_gain, 4),
        transport_gain=round(transport_gain, 4),
        semantic_risk=round(semantic_risk, 4),
        latency_ms=round(latency_ms, 3),
        instability_penalty=round(instability, 4),
        reason="all_signals_preserved",
    )
