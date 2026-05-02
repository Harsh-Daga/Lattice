"""Task classification — RATS input layer.

Deterministic hybrid scoring classifier that assigns both a task class
and an execution tier. Hard overrides ensure debugging/reasoning prompts
never get lossy transforms.

RATS decides **what may run**.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any

from lattice.core.transport import Request


class TaskClass(enum.Enum):
    SIMPLE = "simple"
    RETRIEVAL = "retrieval"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    STRUCTURED = "structured"


class ExecutionTier(enum.Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"
    REASONING_SAFE = "REASONING_SAFE"


@dataclasses.dataclass(slots=True)
class TaskClassification:
    task_class: TaskClass = TaskClass.SIMPLE
    execution_tier: ExecutionTier = ExecutionTier.SIMPLE
    score: int = 0
    confidence: float = 0.5
    signals: list[str] = dataclasses.field(default_factory=list)
    hard_override: bool = False
    preferred_strategy: str = "balanced"
    reasoning_heavy: bool = False
    structured_heavy: bool = False
    debug_heavy: bool = False
    budget_ms: float = 20.0

    @property
    def is_conservative(self) -> bool:
        return self.execution_tier in (
            ExecutionTier.REASONING,
            ExecutionTier.REASONING_SAFE,
        ) or self.task_class in (TaskClass.DEBUGGING, TaskClass.REASONING)

    @property
    def requires_safe_mode(self) -> bool:
        return self.confidence < 0.7 and self.score >= 40

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class.value,
            "execution_tier": self.execution_tier.value,
            "score": self.score,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
            "hard_override": self.hard_override,
            "preferred_strategy": self.preferred_strategy,
            "reasoning_heavy": self.reasoning_heavy,
            "structured_heavy": self.structured_heavy,
            "debug_heavy": self.debug_heavy,
            "is_conservative": self.is_conservative,
            "budget_ms": self.budget_ms,
        }


def classify_task(request: Request) -> TaskClassification:
    """Hybrid score classifier with hard overrides for REASONING."""
    text = "\n".join(msg.content or "" for msg in request.messages)
    lowered = text.lower()
    score = 0
    signals: list[str] = []
    has_debugging_cues = False
    has_reasoning_cues = False
    structured_heavy = False
    has_code = "```" in text

    # ---- Weighted signal scoring ----

    # Reasoning cues (high weight)
    reasoning_patterns = [
        (r"\bwhy\b", 12),
        (r"\broot cause\b", 15),
        (r"\banalyze\b", 10),
        (r"\binfer\b", 10),
        (r"\bdeduce\b", 10),
        (r"\bprove\b", 10),
        (r"\breason\b.*\bstep\b", 12),
        (r"\bexplain\b.*\bwhy\b", 10),
        (r"\bthink\b.*\bcarefully\b", 10),
    ]
    for pat, weight in reasoning_patterns:
        if re.search(pat, lowered):
            score += weight
            signals.append(f"reasoning:{pat[:20]}")
            has_reasoning_cues = True

    # Debugging/crash/log cues (high weight)
    debug_patterns = [
        (r"\berror\b", 8),
        (r"\bfailure\b", 10),
        (r"\bfailed\b", 10),
        (r"\btraceback\b", 10),
        (r"\bexception\b", 8),
        (r"\bcrash\b", 10),
        (r"\bbug\b", 6),
        (r"\blog\b", 3),
        (r"\bdebug\b", 8),
        (r"\boutage\b", 8),
    ]
    for pat, weight in debug_patterns:
        if re.search(pat, lowered):
            score += weight
            signals.append(f"debug:{pat[:20]}")
            has_debugging_cues = True

    # Aggregation/comparison cues
    agg_patterns = [
        (r"\bcount\b", 5),
        (r"\bcompare\b", 5),
        (r"\bdistribution\b", 8),
        (r"\bpattern\b", 4),
        (r"\btrend\b", 5),
        (r"\bgroup by\b", 5),
        (r"\bcorrelation\b", 6),
        (r"\bstatistic\b", 6),
    ]
    for pat, weight in agg_patterns:
        if re.search(pat, lowered):
            score += weight
            signals.append(f"analysis:{pat[:20]}")

    # Structured content
    has_json = "{" in text or "[" in text
    has_table = "|" in text
    structured_heavy = has_json or has_code or has_table
    if structured_heavy:
        score += 10
        signals.append("structured_input")

    # Long input
    words = len(lowered.split())
    if words > 500:
        score += 15
        signals.append("long_input")
    elif words > 200:
        score += 5
        signals.append("medium_input")

    # Code present
    if has_code:
        score += 10
        signals.append("code_present")

    # Retrieval signals
    retrieval_patterns = [
        (r"\bfind\b", 5),
        (r"\blookup\b", 5),
        (r"\bsearch\b", 4),
        (r"\bselect\b", 3),
        (r"\blist\b", 3),
        (r"\bshow\b", 2),
        (r"\bretrieve\b", 5),
    ]
    for pat, weight in retrieval_patterns:
        if re.search(pat, lowered):
            score += weight
            signals.append(f"retrieval:{pat[:20]}")

    # Log-heavy bonus
    log_lines = len(re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", lowered))
    if log_lines > 5:
        score += min(log_lines * 2, 20)
        signals.append(f"log_lines={log_lines}")
        has_debugging_cues = True

    # ---- Hard overrides ----
    hard_override = False
    task_class = TaskClass.SIMPLE
    execution_tier = ExecutionTier.SIMPLE

    # Compute category-specific scores for override logic
    debug_score = sum(weight for pat, weight in debug_patterns if re.search(pat, lowered))
    reason_score = sum(weight for pat, weight in reasoning_patterns if re.search(pat, lowered))

    # Hard rule 1: root cause or explicit why-fail → REASONING (highest priority)
    if re.search(r"\broot cause\b|\bwhy did this fail\b|\bwhy.*\bfail\b", lowered):
        task_class = TaskClass.REASONING
        execution_tier = ExecutionTier.REASONING
        hard_override = True
        signals.append("hard_override:root_cause_why")

    # Hard rule 2: both debugging AND reasoning cues scored significantly → REASONING
    elif has_debugging_cues and has_reasoning_cues and debug_score > 10 and reason_score > 10:
        task_class = TaskClass.DEBUGGING
        execution_tier = ExecutionTier.REASONING
        hard_override = True
        signals.append("hard_override:debugging+reasoning")

    # Hard rule 3: traceback or heavy log analysis → REASONING
    elif has_debugging_cues and log_lines > 5:
        task_class = TaskClass.DEBUGGING
        execution_tier = ExecutionTier.REASONING
        hard_override = True
        signals.append("hard_override:log_heavy_debugging")

    # Hard rule 4: significant debugging cues alone → REASONING
    # (no reasoning words needed — debugging tasks require lossless handling)
    elif has_debugging_cues and debug_score > 10:
        task_class = TaskClass.DEBUGGING
        execution_tier = ExecutionTier.REASONING
        hard_override = True
        signals.append("hard_override:debugging_detected")

    # Score-based classification
    elif score >= 70:
        task_class = TaskClass.REASONING
        execution_tier = ExecutionTier.REASONING
    elif score >= 45:
        task_class = TaskClass.ANALYSIS
        execution_tier = ExecutionTier.MEDIUM
    elif score >= 10 or any("retrieval" in s for s in signals):
        task_class = TaskClass.RETRIEVAL
        execution_tier = ExecutionTier.SIMPLE
    elif words > 200:
        task_class = TaskClass.SUMMARIZATION
        execution_tier = ExecutionTier.MEDIUM
    else:
        task_class = TaskClass.SIMPLE
        execution_tier = ExecutionTier.SIMPLE

    # Override for structured-but-simple
    if structured_heavy and task_class == TaskClass.SIMPLE:
        task_class = TaskClass.STRUCTURED
        execution_tier = ExecutionTier.MEDIUM

    # Confidence
    confidence = min(1.0, score / 100)
    if confidence < 0.7 and score >= 40:
        execution_tier = ExecutionTier.REASONING_SAFE

    # Budget based on tier
    budget_ms = 20.0
    if execution_tier in (ExecutionTier.REASONING, ExecutionTier.REASONING_SAFE):
        budget_ms = 50.0
    elif execution_tier == ExecutionTier.COMPLEX:
        budget_ms = 30.0

    return TaskClassification(
        task_class=task_class,
        execution_tier=execution_tier,
        score=score,
        confidence=round(confidence, 2),
        signals=signals[:20],
        hard_override=hard_override,
        preferred_strategy="conservative"
        if execution_tier in (ExecutionTier.REASONING, ExecutionTier.REASONING_SAFE)
        else "balanced",
        reasoning_heavy=has_reasoning_cues or execution_tier == ExecutionTier.REASONING,
        structured_heavy=structured_heavy,
        debug_heavy=has_debugging_cues,
        budget_ms=budget_ms,
    )
