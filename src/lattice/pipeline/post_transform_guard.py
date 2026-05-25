"""Rule-based post-transform validation (no model calls)."""

from __future__ import annotations

import dataclasses
from typing import Any

from lattice.pipeline import checks as structural_checks
from lattice.planner.task_classifier import TaskClass, TaskClassification
from lattice.transforms.reputation import get_reputation_registry

__all__ = [
    "PostTransformGuardResult",
    "evaluate_post_transform",
    "should_trigger_post_transform_guard",
]


@dataclasses.dataclass(slots=True)
class PostTransformGuardResult:
    passed: bool = True
    score: float = 1.0
    reason: str = ""
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 3),
            "reason": self.reason,
            "details": self.details,
        }


_TASK_THRESHOLDS: dict[str, float] = {
    TaskClass.REASONING.value: 0.92,
    TaskClass.DEBUGGING.value: 0.90,
    TaskClass.ANALYSIS.value: 0.88,
    TaskClass.STRUCTURED.value: 0.87,
    TaskClass.RETRIEVAL.value: 0.85,
    TaskClass.SUMMARIZATION.value: 0.85,
    TaskClass.SIMPLE.value: 0.80,
}


def should_trigger_post_transform_guard(
    transform_name: str,
    task: TaskClassification,
    compression_ratio: float,
    placeholder_aliasing_used: bool = False,
    tool_output_filtered: bool = False,
) -> bool:
    if compression_ratio > 0.30:
        return True
    if task.task_class in (TaskClass.REASONING, TaskClass.DEBUGGING):
        if placeholder_aliasing_used or compression_ratio > 0.10:
            return True
    if tool_output_filtered and task.task_class == TaskClass.DEBUGGING:
        return True
    rep = get_reputation_registry()
    if rep.is_high_risk(transform_name):
        return True
    return False


def evaluate_post_transform(
    before_text: str,
    after_text: str,
    task: TaskClassification,
    *,
    placeholder_aliasing_used: bool = False,
) -> PostTransformGuardResult:
    score = 1.0
    reasons: list[str] = []

    num = structural_checks.numbers_preserved(before_text, after_text)
    if not num.passed:
        penalty = (0.7 - num.score) * 0.3
        score -= penalty
        reasons.append(f"numbers_lost_{penalty:.2f}")

    uuid = structural_checks.uuids_preserved(before_text, after_text)
    if not uuid.passed:
        score -= 0.02 * (1.0 - uuid.score)
        reasons.append("uuid_lost")

    url = structural_checks.urls_preserved(before_text, after_text)
    if not url.passed:
        score -= 0.02 * (1.0 - url.score)
        reasons.append(url.detail)

    if task.task_class in (TaskClass.DEBUGGING, TaskClass.REASONING):
        rc = structural_checks.root_cause_phrases_preserved(before_text, after_text)
        if not rc.passed:
            score -= 0.08
            reasons.append("root_cause_lost")

    err = structural_checks.error_signals_preserved(before_text, after_text, task.task_class.value)
    if not err.passed:
        score -= 0.10
        reasons.append("error_signal_halved")

    pl = structural_checks.placeholder_leakage(
        before_text, after_text, placeholder_aliasing_used=placeholder_aliasing_used
    )
    if not pl.passed:
        score -= 0.20
        reasons.append("opaque_placeholders_no_manifest")

    score = max(0.0, min(1.0, score))
    threshold = _TASK_THRESHOLDS.get(task.task_class.value, 0.85)
    passed = score >= threshold

    return PostTransformGuardResult(
        passed=passed,
        score=score,
        reason="; ".join(reasons) if reasons else "all_checks_pass",
        details={
            "threshold": threshold,
            "task_class": task.task_class.value,
            "score": score,
        },
    )
