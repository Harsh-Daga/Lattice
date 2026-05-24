"""MILV — Model-in-the-Loop Validation.

Lightweight runtime judge that validates transform safety by checking
whether the model's output under the transformed prompt is semantically
equivalent to what it would produce under the original prompt.

Only triggered for high-risk transforms or when quality thresholds are
at risk. Not a heavy judge — uses fast rule-based heuristics by default.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from lattice.planner.task_classifier import TaskClass, TaskClassification


@dataclasses.dataclass(slots=True)
class MILVResult:
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


# Per-task thresholds from scheduler
_TASK_THRESHOLDS: dict[str, float] = {
    TaskClass.REASONING.value: 0.92,
    TaskClass.DEBUGGING.value: 0.90,
    TaskClass.ANALYSIS.value: 0.88,
    TaskClass.STRUCTURED.value: 0.87,
    TaskClass.RETRIEVAL.value: 0.85,
    TaskClass.SUMMARIZATION.value: 0.85,
    TaskClass.SIMPLE.value: 0.80,
}


def should_trigger_milv(
    transform_name: str,
    task: TaskClassification,
    compression_ratio: float,
    placeholder_aliasing_used: bool = False,
    tool_output_filtered: bool = False,
) -> bool:
    """Should MILV be triggered for this transform execution?"""
    if compression_ratio > 0.30:
        return True

    if task.task_class in (TaskClass.REASONING, TaskClass.DEBUGGING):
        if placeholder_aliasing_used or compression_ratio > 0.10:
            return True

    if tool_output_filtered and task.task_class == TaskClass.DEBUGGING:
        return True

    from lattice.transforms.reputation import get_reputation_registry

    rep = get_reputation_registry()
    if rep.is_high_risk(transform_name):
        return True

    return False


def validate_transform(
    before_text: str,
    after_text: str,
    task: TaskClassification,
    *,
    placeholder_aliasing_used: bool = False,
) -> MILVResult:
    """Lightweight rule-based validation of a transform's output.

    Checks: same counts preserved? critical entities preserved?
    root cause preserved? same structural elements?
    """
    score = 1.0
    reasons: list[str] = []

    # 1. Count preservation
    before_numbers = set(re.findall(r"\b\d+\b", before_text))
    after_numbers = set(re.findall(r"\b\d+\b", after_text))
    number_overlap = len(before_numbers & after_numbers) / max(len(before_numbers), 1)
    if number_overlap < 0.7 and before_numbers:
        penalty = (0.7 - number_overlap) * 0.3
        score -= penalty
        reasons.append(f"numbers_lost_{penalty:.2f}")

    # 2. Entity preservation (UUIDs, paths, URLs)
    uuids = re.findall(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        before_text,
        re.IGNORECASE,
    )
    for uuid in uuids:
        if uuid not in after_text:
            score -= 0.02
            reasons.append("uuid_lost")

    urls = re.findall(r"https?://[^\s)]+", before_text, re.IGNORECASE)
    url_lost = sum(1 for u in urls if u not in after_text)
    score -= url_lost * 0.02
    if url_lost:
        reasons.append(f"urls_lost_{url_lost}")

    # 3. Root cause preservation (debugging/reasoning)
    if task.task_class in (TaskClass.DEBUGGING, TaskClass.REASONING):
        root_cause_patterns = [
            r"root cause",
            r"the cause was",
            r"the reason is",
            r"determined that",
        ]
        for pattern in root_cause_patterns:
            before_match = re.search(pattern, before_text, re.IGNORECASE)
            after_match = re.search(pattern, after_text, re.IGNORECASE)
            if before_match and not after_match:
                score -= 0.08
                reasons.append("root_cause_lost")
                break

    # 4. Error message preservation
    error_phrases = re.findall(r"\b(error|exception|failure|warning)\b", before_text, re.IGNORECASE)
    error_after = re.findall(r"\b(error|exception|failure|warning)\b", after_text, re.IGNORECASE)
    if error_phrases and len(error_after) < len(error_phrases) * 0.5:
        score -= 0.10
        reasons.append("error_signal_halved")

    # 5. Placeholder leakage check
    if placeholder_aliasing_used:
        opaque = re.findall(r"<(?:d_|g_|ref_)\d+>", after_text)
        if opaque and not re.search(r"ALIAS MAP", after_text, re.IGNORECASE):
            score -= 0.20
            reasons.append("opaque_placeholders_no_manifest")

    score = max(0.0, min(1.0, score))
    threshold = _TASK_THRESHOLDS.get(task.task_class.value, 0.85)
    passed = score >= threshold

    return MILVResult(
        passed=passed,
        score=score,
        reason="; ".join(reasons) if reasons else "all_checks_pass",
        details={
            "threshold": threshold,
            "task_class": task.task_class.value,
            "score": score,
        },
    )
