"""Transform scheduler — RATS.

Consumes SIG, task classification, and risk score to decide:
- Which transforms are eligible
- In what order they should run
- When to stop (budget exhaustion, risk escalation)

RATS decides **what may run**.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lattice.core.task_classifier import TaskClass, TaskClassification
from lattice.utils.validation import (
    SemanticRiskScore,
    TransformSafetyBucket,
    get_transform_safety_bucket,
)

# Tier-based policy matrix
_ALLOWED_BUCKETS: dict[str, set[TransformSafetyBucket]] = {
    "SIMPLE": {TransformSafetyBucket.SAFE, TransformSafetyBucket.CONDITIONAL},
    "MEDIUM": {TransformSafetyBucket.SAFE, TransformSafetyBucket.CONDITIONAL},
    "COMPLEX": {TransformSafetyBucket.SAFE, TransformSafetyBucket.CONDITIONAL},
    "REASONING": {TransformSafetyBucket.SAFE, TransformSafetyBucket.CONDITIONAL},
    "REASONING_SAFE": {TransformSafetyBucket.SAFE},
}

# Transforms always blocked on REASONING/REASONING_SAFE tiers.
_REASONING_DISABLED: frozenset[str] = frozenset(
    {
        "message_dedup",
        "rate_distortion",
        "hierarchical_summary",
        "semantic_compress",
    }
)

# Per-task transform matrix — which transforms are allowed per task class.
# True = allowed by default, False = blocked, None = allowed only if safe mode off.
_TASK_TRANSFORM_MATRIX: dict[str, dict[str, bool | None]] = {
    # Retrieval / simple Q&A
    TaskClass.RETRIEVAL.value: {
        "rate_distortion": False,
        "hierarchical_summary": False,
        "structural_fingerprint": False,
        "message_dedup": None,
        "semantic_compress": False,
    },
    TaskClass.SIMPLE.value: {
        "rate_distortion": False,
        "hierarchical_summary": False,
    },
    # Debugging / root cause
    TaskClass.DEBUGGING.value: {
        "rate_distortion": False,
        "semantic_compress": False,
        "hierarchical_summary": False,
        "structural_fingerprint": False,
        "message_dedup": False,
        "grammar_compress": False,
    },
    # Reasoning
    TaskClass.REASONING.value: {
        "rate_distortion": False,
        "semantic_compress": False,
        "hierarchical_summary": False,
        "structural_fingerprint": False,
        "message_dedup": False,
    },
    # Structured data / tables
    TaskClass.STRUCTURED.value: {
        "rate_distortion": False,
        "hierarchical_summary": False,
    },
}

# Per-task quality thresholds
_TASK_QUALITY_THRESHOLDS: dict[str, float] = {
    TaskClass.REASONING.value: 0.92,
    TaskClass.DEBUGGING.value: 0.90,
    TaskClass.ANALYSIS.value: 0.88,
    TaskClass.STRUCTURED.value: 0.87,
    TaskClass.RETRIEVAL.value: 0.85,
    TaskClass.SUMMARIZATION.value: 0.85,
    TaskClass.SIMPLE.value: 0.80,
}

# Per-task compression limits (max compression ratio allowed)
_TASK_COMPRESSION_LIMITS: dict[str, float] = {
    TaskClass.REASONING.value: 0.10,
    TaskClass.DEBUGGING.value: 0.15,
    TaskClass.ANALYSIS.value: 0.25,
    TaskClass.STRUCTURED.value: 0.30,
    TaskClass.RETRIEVAL.value: 0.40,
    TaskClass.SUMMARIZATION.value: 0.35,
    TaskClass.SIMPLE.value: 0.50,
}


@dataclasses.dataclass(slots=True)
class TransformScheduleEntry:
    """A scheduled transform with its permission and rationale."""

    transform_name: str
    bucket: TransformSafetyBucket = TransformSafetyBucket.SAFE
    allowed: bool = True
    reason: str = ""
    priority: int = 50


@dataclasses.dataclass(slots=True)
class SchedulerDecision:
    """Complete scheduling output for a request."""

    task_classification: TaskClassification
    risk_score: SemanticRiskScore | None = None
    schedule: list[TransformScheduleEntry] = dataclasses.field(default_factory=list)
    blocked_transforms: list[str] = dataclasses.field(default_factory=list)
    allowed_transforms: list[str] = dataclasses.field(default_factory=list)
    protected_span_count: int = 0
    total_budget_ms: float = 20.0
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_classification.to_dict(),
            "risk_level": self.risk_score.level if self.risk_score else "unknown",
            "risk_total": self.risk_score.total if self.risk_score else 0.0,
            "blocked": self.blocked_transforms,
            "allowed": self.allowed_transforms,
            "protected_spans": self.protected_span_count,
            "budget_ms": self.total_budget_ms,
            "budget_exhausted": self.budget_exhausted,
            "schedule": [
                {
                    "name": e.transform_name,
                    "bucket": e.bucket.value,
                    "allowed": e.allowed,
                    "reason": e.reason,
                }
                for e in self.schedule
            ],
        }


def decide_schedule(
    transform_names: list[str],
    task: TaskClassification,
    risk: SemanticRiskScore | None = None,
    protected_span_count: int = 0,
    total_budget_ms: float = 20.0,
) -> SchedulerDecision:
    """Produce a schedule of transforms for a request.

    Rules:
    - SAFE transforms: always allowed unless they touch protected spans
    - CONDITIONAL: allowed at LOW/MEDIUM risk, blocked at HIGH+, blocked on DEBUGGING/REASONING
    - DANGEROUS: only at LOW risk, never on DEBUGGING/REASONING
    - Order: SAFE → CONDITIONAL → DANGEROUS (within priority)

    Args:
        transform_names: All registered transform names in priority order.
        task: Task classification from RATS.
        risk: Semantic risk score from SIG.
        protected_span_count: Number of protected spans from SIG.
        total_budget_ms: Total budget from runtime contract.
    """
    risk_level = risk.level if risk else "UNKNOWN"
    tier = task.execution_tier.value
    # REASONING/DEBUGGING task classes always get conservative treatment
    if task.is_conservative and tier not in ("REASONING", "REASONING_SAFE"):
        tier = "REASONING"

    schedule: list[TransformScheduleEntry] = []
    blocked: list[str] = []
    allowed: list[str] = []

    for name in transform_names:
        bucket = get_transform_safety_bucket(name)
        entry = TransformScheduleEntry(
            transform_name=name,
            bucket=bucket,
        )

        # Tier-based gating: only buckets allowed for this tier can run
        tier_buckets = _ALLOWED_BUCKETS.get(tier, {TransformSafetyBucket.SAFE})

        # Per-task transform matrix check
        task_class_value = task.task_class.value
        matrix = _TASK_TRANSFORM_MATRIX.get(task_class_value, {})
        if name in matrix:
            matrix_decision = matrix[name]
            if matrix_decision is False:
                entry.allowed = False
                entry.reason = f"{name}_blocked_for_{task_class_value}"
            elif matrix_decision is None and task.is_conservative:
                entry.allowed = False
                entry.reason = f"{name}_blocked_conservative_{task_class_value}"
            # if True, fall through to normal gating

        if entry.allowed is False and entry.reason:
            # Already blocked by matrix check
            pass
        elif tier in ("REASONING", "REASONING_SAFE") and name in _REASONING_DISABLED:
            entry.allowed = False
            entry.reason = "reasoning_tier_disabled"

        elif bucket not in tier_buckets:
            entry.allowed = False
            entry.reason = f"bucket_{bucket.value}_not_allowed_at_tier_{tier}"

        elif bucket == TransformSafetyBucket.SAFE:
            entry.allowed = True
            entry.reason = "safe_transform"

        elif bucket == TransformSafetyBucket.CONDITIONAL:
            if risk_level in ("HIGH", "CRITICAL"):
                entry.allowed = False
                entry.reason = f"conditional_blocked_at_{risk_level.lower()}_risk"
            else:
                entry.allowed = True
                entry.reason = "conditional_allowed"

        elif bucket == TransformSafetyBucket.DANGEROUS:
            if risk_level != "LOW":
                entry.allowed = False
                entry.reason = f"dangerous_blocked_at_{risk_level.lower()}_risk"
            else:
                entry.allowed = True
                entry.reason = "dangerous_allowed_at_low_risk"

        else:
            entry.allowed = True
            entry.reason = "unknown_bucket"

        schedule.append(entry)
        if entry.allowed:
            allowed.append(name)
        else:
            blocked.append(name)

    # Sort: SAFE first, CONDITIONAL second, DANGEROUS last (within each group, respect original priority)
    bucket_order = {
        TransformSafetyBucket.SAFE: 0,
        TransformSafetyBucket.CONDITIONAL: 1,
        TransformSafetyBucket.DANGEROUS: 2,
    }
    schedule.sort(key=lambda e: (bucket_order.get(e.bucket, 0), e.priority))

    decision = SchedulerDecision(
        task_classification=task,
        risk_score=risk,
        schedule=schedule,
        blocked_transforms=blocked,
        allowed_transforms=allowed,
        protected_span_count=protected_span_count,
        total_budget_ms=total_budget_ms,
    )

    return decision
