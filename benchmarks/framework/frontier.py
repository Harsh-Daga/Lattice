"""Frontier scoring for LATTICE benchmark evaluation.

Computes frontier_score from quality_score and compression_ratio with
hard gates for safety, negative savings, and placeholder leakage.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class FrontierScore:
    quality: float = 0.0
    compression: float = 0.0
    frontier_score: float = 0.0
    passed_quality_gate: bool = False
    passed_savings_gate: bool = False
    rollback_reason: str | None = None
    placeholder_leakage: bool = False

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "compression": self.compression,
            "frontier_score": self.frontier_score,
            "passed_quality_gate": self.passed_quality_gate,
            "passed_savings_gate": self.passed_savings_gate,
            "rollback_reason": self.rollback_reason,
            "placeholder_leakage": self.placeholder_leakage,
        }


def compute_frontier(
    quality_score: float,
    compression_ratio: float,
    *,
    placeholder_leakage: bool = False,
    task_class: str = "",
) -> FrontierScore:
    quality_gate_thresholds = {
        "reasoning": 0.92,
        "debugging": 0.90,
        "analysis": 0.88,
        "structured": 0.87,
        "retrieval": 0.85,
        "summarization": 0.85,
        "simple": 0.80,
    }
    quality_threshold = quality_gate_thresholds.get(task_class, 0.85)

    frontier = quality_score - (0.35 * max(0.0, -compression_ratio)) + (0.20 * compression_ratio)

    passed_quality = quality_score >= quality_threshold
    passed_savings = compression_ratio >= 0.0

    rollback_reason: str | None = None
    if placeholder_leakage:
        rollback_reason = "placeholder_leakage"
    elif not passed_quality:
        rollback_reason = f"quality_{quality_score:.2f}_below_{quality_threshold}"
    elif not passed_savings:
        rollback_reason = "negative_savings"

    return FrontierScore(
        quality=quality_score,
        compression=compression_ratio,
        frontier_score=round(frontier, 4),
        passed_quality_gate=passed_quality,
        passed_savings_gate=passed_savings,
        rollback_reason=rollback_reason,
        placeholder_leakage=placeholder_leakage,
    )
