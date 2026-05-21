"""Transform scheduler — RATS.

Selects the optimal minimal subset of transforms per request.
Uses task classification, risk score, and transform value ranking
to cap at 8 transforms max per request.

RATS decides **what to run**, not just what to block.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lattice.core.task_classifier import TaskClass, TaskClassification
from lattice.core.transform_registry import is_transform_name_known
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
    }
)

# Per-task transform matrix — final conservative gating.
_TASK_TRANSFORM_MATRIX: dict[str, dict[str, bool | None]] = {
    TaskClass.REASONING.value: {
        "rate_distortion": False,
        "message_dedup": False,
        "context_selector": False,
        "information_theoretic_selector": False,
        # tool_filter and reference_sub are reversible; allow them.
        # The REASONING value ranking still controls selection priority.
    },
    TaskClass.DEBUGGING.value: {
        "rate_distortion": False,
        "message_dedup": False,
        "context_selector": False,
        "information_theoretic_selector": False,
        # reference_sub and tool_filter are reversible; allow them.
    },
    TaskClass.STRUCTURED.value: {
        "rate_distortion": False,
    },
    TaskClass.ANALYSIS.value: {
        "rate_distortion": False,
    },
    TaskClass.RETRIEVAL.value: {
        "rate_distortion": False,
    },
    TaskClass.SUMMARIZATION.value: {},
    TaskClass.SIMPLE.value: {
        "rate_distortion": False,
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

# Per-task compression limits (max compression ratio allowed per-transform).
_TASK_COMPRESSION_LIMITS: dict[str, float] = {
    TaskClass.REASONING.value: 0.10,
    TaskClass.DEBUGGING.value: 0.40,
    TaskClass.ANALYSIS.value: 0.25,
    TaskClass.STRUCTURED.value: 0.30,
    TaskClass.RETRIEVAL.value: 0.40,
    TaskClass.SUMMARIZATION.value: 0.35,
    TaskClass.SIMPLE.value: 0.50,
}

# Hard cap: max transforms per request regardless of task class.
# A smaller pipeline is more predictable and less prone to quality degradation
# from cascading transform interactions. This cap includes 3 core transforms
# (content_profiler, runtime_contract, strategy_selector), leaving 7-9 slots
# for ranked transforms.
HARD_MAX_TRANSFORMS = 12

# Always-run core transforms (foundational — must run first).
_CORE_TRANSFORMS = frozenset(
    {
        "content_profiler",
        "runtime_contract",
        "cache_arbitrage",
        "prefix_optimizer",
        "strategy_selector",
    }
)

# Transform value ranking by task class. The scheduler picks the highest-value
# transforms first, truncated by HARD_MAX_TRANSFORMS. This must be a superset
# of transforms that COULD be useful for the task class — the scheduler cap at 8
# ensures minimal execution. Transforms blocked by _TASK_TRANSFORM_MATRIX will
# never appear here because decide_schedule() gates them earlier.
_HIGH_VALUE_MATRIX: dict[str, list[str]] = {
    # Structured work: format conversion, JSON shaping, columnar packing
    TaskClass.STRUCTURED.value: [
        "format_conversion",
        "columnar_pack",
        "json_shape",
        "reference_sub",
        "output_cleanup",
    ],
    # Debugging: keep diagnostic signal, never lossy
    TaskClass.DEBUGGING.value: [
        "reference_sub",
        "tool_projection",
        "tool_filter",
        "columnar_pack",
        "constraint_lifting",
        "causal_chain",
        "diagnostic_rle",
        "output_cleanup",
    ],
    # Reasoning: preserve chain-of-thought, no lossy compression.
    # Allow reversible/safe transforms (reference_sub, tool_projection, columnar_pack)
    # since they don't compromise reasoning integrity. message_dedup only removes
    # exact-duplicate message pairs, which is safe even for reasoning.
    TaskClass.REASONING.value: [
        "constraint_lifting",
        "causal_chain",
        "reference_sub",
        "tool_projection",
        "tool_filter",
        "columnar_pack",
        "diagnostic_rle",
        "output_cleanup",
    ],
    # Analysis: balanced
    TaskClass.ANALYSIS.value: [
        "reference_sub",
        "columnar_pack",
        "extractive_compress",
        "json_shape",
        "tool_projection",
        "format_conversion",
        "output_cleanup",
    ],
    # Retrieval: reference/path compression, tool output filtering, format conversion
    TaskClass.RETRIEVAL.value: [
        "reference_sub",
        "columnar_pack",
        "json_shape",
        "tool_filter",
        "tool_projection",
        "format_conversion",
        "context_selector",
        "output_cleanup",
    ],
    # Summarization: extractive + dictionary/grammar
    TaskClass.SUMMARIZATION.value: [
        "extractive_compress",
        "reference_sub",
        "output_cleanup",
    ],
    # Simple: allow all SAFE+CONDITIONAL transforms that make sense for
    # straightforward requests. The HARD_MAX_TRANSFORMS=8 cap prevents
    # churn. Unranked transforms are safer than blocking useful ones.
    TaskClass.SIMPLE.value: [
        "reference_sub",
        "format_conversion",
        "tool_filter",
        "tool_projection",
        "columnar_pack",
        "json_shape",
        "cache_arbitrage",
        "prefix_optimizer",
        "context_selector",
        "message_dedup",
        "output_cleanup",
    ],
}


@dataclasses.dataclass(slots=True)
class TransformScheduleEntry:
    transform_name: str
    bucket: TransformSafetyBucket = TransformSafetyBucket.SAFE
    allowed: bool = True
    reason: str = ""
    priority: int = 50


@dataclasses.dataclass(slots=True)
class SchedulerDecision:
    task_classification: TaskClassification
    risk_score: SemanticRiskScore | None = None
    schedule: list[TransformScheduleEntry] = dataclasses.field(default_factory=list)
    blocked_transforms: list[str] = dataclasses.field(default_factory=list)
    allowed_transforms: list[str] = dataclasses.field(default_factory=list)
    protected_span_count: int = 0
    total_budget_ms: float = 20.0
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        # Map allowed transforms to parent optimizers for the optimizer architecture
        _transform_to_optimizer: dict[str, str] = {
            "json_shape": "structure_optimizer",
            "format_conversion": "structure_optimizer",
            "columnar_pack": "structure_optimizer",
            "reference_sub": "reference_optimizer",
            "path_prefix": "reference_optimizer",
            "tool_projection": "tool_optimizer",
            "tool_filter": "tool_optimizer",
            "output_cleanup": "tool_optimizer",
            "context_selector": "context_optimizer",
            "rate_distortion": "context_optimizer",
            "extractive_compress": "context_optimizer",
            "diagnostic_rle": "diagnostic_optimizer",
            "message_dedup": "context_optimizer",
        }
        allowed_optimizers = sorted(
            {
                name
                for name in (_transform_to_optimizer.get(t) for t in self.allowed_transforms)
                if name is not None
            }
        )
        # representation_optimizer is the global orchestrator. If any
        # constituent optimizers are allowed, the orchestrator must run too.
        if allowed_optimizers and "representation_optimizer" not in allowed_optimizers:
            allowed_optimizers.insert(0, "representation_optimizer")
        return {
            "task_class": self.task_classification.to_dict(),
            "risk_level": self.risk_score.level if self.risk_score else "unknown",
            "risk_total": self.risk_score.total if self.risk_score else 0.0,
            "blocked": self.blocked_transforms,
            "allowed": self.allowed_transforms,
            "allowed_optimizers": allowed_optimizers,
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
    """Select the optimal transform subset for a request.

    Algorithm (selector, not gate-keeper):
    1. Always allow core transforms (content_profiler, runtime_contract, strategy_selector).
    2. From remaining candidates, filter through safety matrix + risk gauntlet.
    3. Rank surviving candidates by per-task-class value.
    4. Select top-N until HARD_MAX_TRANSFORMS is reached.
    5. All unselected transforms are blocked (exceeds max).

    This replaces the old "allow everything not blocked" approach with
    "choose the best minimal subset."
    """
    risk_level = risk.level if risk else "UNKNOWN"
    tier = task.execution_tier.value
    if task.is_conservative and tier not in ("REASONING", "REASONING_SAFE"):
        tier = "REASONING"

    task_class_value = task.task_class.value
    tier_buckets = _ALLOWED_BUCKETS.get(tier, {TransformSafetyBucket.SAFE})
    matrix = _TASK_TRANSFORM_MATRIX.get(task_class_value, {})
    value_ranking = _HIGH_VALUE_MATRIX.get(task_class_value, [])

    # Build a value rank lookup: lower index = higher priority
    rank_of: dict[str, int] = {name: i for i, name in enumerate(value_ranking)}

    entries: dict[str, TransformScheduleEntry] = {}

    for name in transform_names:
        bucket = get_transform_safety_bucket(name)
        entry = TransformScheduleEntry(transform_name=name, bucket=bucket)

        # Compatibility-only regex transforms are not part of the canonical
        # scheduler surface. Keep them out of the production plan even if
        # legacy callers still mention them.
        if not is_transform_name_known(name):
            entry.allowed = False
            entry.reason = "compatibility_only_transform"
            entries[name] = entry
            continue

        # Core transforms always allowed.
        if name in _CORE_TRANSFORMS:
            entry.allowed = True
            entry.reason = "core_transform"
            entries[name] = entry
            continue

        # Safety matrix check
        if name in matrix:
            matrix_decision = matrix[name]
            if matrix_decision is False:
                entry.allowed = False
                entry.reason = f"{name}_blocked_for_{task_class_value}"
                entries[name] = entry
                continue
            elif matrix_decision is None and task.is_conservative:
                entry.allowed = False
                entry.reason = f"{name}_blocked_conservative_{task_class_value}"
                entries[name] = entry
                continue

        # Reasoning tier: always block certain transforms
        if tier in ("REASONING", "REASONING_SAFE") and name in _REASONING_DISABLED:
            entry.allowed = False
            entry.reason = "reasoning_tier_disabled"
            entries[name] = entry
            continue

        # Safety bucket gating
        if bucket not in tier_buckets:
            entry.allowed = False
            entry.reason = f"bucket_{bucket.value}_not_allowed_at_tier_{tier}"
            entries[name] = entry
            continue

        if bucket == TransformSafetyBucket.CONDITIONAL:
            if risk_level in ("HIGH", "CRITICAL"):
                entry.allowed = False
                entry.reason = f"conditional_blocked_at_{risk_level.lower()}_risk"
                entries[name] = entry
                continue
            entry.allowed = True
            entry.reason = "conditional_allowed"
        elif bucket == TransformSafetyBucket.DANGEROUS:
            if risk_level != "LOW":
                entry.allowed = False
                entry.reason = f"dangerous_blocked_at_{risk_level.lower()}_risk"
                entries[name] = entry
                continue
            entry.allowed = True
            entry.reason = "dangerous_allowed_at_low_risk"
        else:
            entry.allowed = True
            entry.reason = "safe_transform"

        # Reputation override
        if entry.allowed:
            from lattice.core.transform_reputation import get_reputation_registry

            rep = get_reputation_registry()
            stats = rep.stats(name)
            if stats.sample_count > 4 and stats.rollback_rate > 0.25:
                entry.allowed = False
                entry.reason = f"reputation_rollback_rate_{stats.rollback_rate:.2f}"

        entries[name] = entry

    # Sort candidates by value ranking: highest-value first, then by safety
    # bucket (SAFE > CONDITIONAL > DANGEROUS), then by original priority.
    bucket_order = {
        TransformSafetyBucket.SAFE: 0,
        TransformSafetyBucket.CONDITIONAL: 1,
        TransformSafetyBucket.DANGEROUS: 2,
    }

    def sort_key(name: str) -> tuple[int, int, int]:
        entry = entries.get(name)
        if entry is None:
            return (999, 0, 999)
        # Core transforms always come first regardless of rank.
        if name in _CORE_TRANSFORMS:
            return (-10, 0, 0)
        rank = rank_of.get(name, 100)
        bucket_val = bucket_order.get(entry.bucket, 0)
        return (0, rank, bucket_val)

    sorted_names = sorted(entries, key=sort_key)

    # Hard cap: only keep HARD_MAX_TRANSFORMS allowed entries.
    # Core transforms are always allowed. Everything else is ranked by
    # task-specific value, then bucket, then priority. Unranked transforms
    # are allowed if they pass safety gates — the cap prevents churn.
    allowed_count = 0
    for name in sorted_names:
        entry = entries[name]
        if not entry.allowed:
            continue
        if name in _CORE_TRANSFORMS:
            allowed_count += 1
            continue
        if allowed_count >= HARD_MAX_TRANSFORMS:
            entry.allowed = False
            entry.reason = f"exceeds_hard_max_{HARD_MAX_TRANSFORMS}"
        else:
            allowed_count += 1

    blocked: list[str] = []
    allowed: list[str] = []
    schedule: list[TransformScheduleEntry] = []
    for name in sorted_names:
        entry = entries[name]
        schedule.append(entry)
        if entry.allowed:
            allowed.append(name)
        else:
            blocked.append(name)

    return SchedulerDecision(
        task_classification=task,
        risk_score=risk,
        schedule=schedule,
        blocked_transforms=blocked,
        allowed_transforms=allowed,
        protected_span_count=protected_span_count,
        total_budget_ms=total_budget_ms,
    )
