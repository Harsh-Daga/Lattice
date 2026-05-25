"""RequestClassifier — unified request intelligence layer.

Wraps the existing task_classifier and content_profiler into a single
decision point that produces the ExecutionPlan inputs.

Phase 1 — Request Intelligence.
"""

from __future__ import annotations

from typing import Any

from lattice.planner.task_classifier import TaskClass, classify_task
from lattice.safety.risk_scoring import SemanticRiskScore
from lattice.transport.types import Request


class RequestClassifier:
    """Unified classifier: task, risk, provider, session, content shape."""

    def __init__(self) -> None:
        self._risk_scorer = SemanticRiskScore()

    def classify(self, request: Request) -> dict[str, Any]:
        """Return a flat classification dict used to build ExecutionPlan.

        Returns:
            dict with keys: task_class, execution_tier, risk_level, risk_total,
            score, confidence, signals, quality_floor, budget_ms, is_conservative.
        """
        task = classify_task(request)

        # Compute semantic risk from text
        from lattice.safety.risk_scoring import compute_risk_score

        risk = compute_risk_score(request)

        # Map execution tier to budget
        budget_ms = _tier_to_budget_ms(task.execution_tier.value)

        # Quality floor based on task class
        quality_floor = _task_quality_floor(task.task_class.value)

        # Adjust for conservative mode
        if task.is_conservative:
            quality_floor = max(quality_floor, 0.92)
            if task.task_class == TaskClass.DEBUGGING:
                # Debugging: no lossy transforms, so quality floor is very high
                quality_floor = 0.95

        return {
            "task_class": task.task_class.value,
            "execution_tier": task.execution_tier.value,
            "risk_level": risk.level,
            "risk_total": risk.total,
            "score": task.score,
            "confidence": task.confidence,
            "signals": task.signals,
            "quality_floor": quality_floor,
            "budget_ms": budget_ms,
            "is_conservative": task.is_conservative,
            "reasoning_heavy": task.reasoning_heavy,
            "structured_heavy": task.structured_heavy,
            "debug_heavy": task.debug_heavy,
        }


def _tier_to_budget_ms(tier: str) -> float:
    """Map execution tier to latency budget."""
    from lattice.planner.plan_types import TIER_BUDGETS_MS

    return TIER_BUDGETS_MS.get(tier, 100.0)


def _task_quality_floor(task_class: str) -> float:
    """Quality floor per task class."""
    floors: dict[str, float] = {
        TaskClass.REASONING.value: 0.92,
        TaskClass.DEBUGGING.value: 0.90,
        TaskClass.ANALYSIS.value: 0.88,
        TaskClass.STRUCTURED.value: 0.87,
        TaskClass.RETRIEVAL.value: 0.85,
        TaskClass.SUMMARIZATION.value: 0.85,
        TaskClass.SIMPLE.value: 0.80,
    }
    return floors.get(task_class, 0.85)


__all__ = ["RequestClassifier"]
