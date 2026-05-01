"""Task classification — RATS input layer.

Deterministic task classification used by the scheduler to decide
which transforms are appropriate for a given workload.

RATS decides **what may run**.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

from lattice.core.transport import Request


class TaskClass(enum.Enum):
    """Deterministic classification of the request's primary task."""

    RETRIEVAL = "retrieval"  # lookup, search, find
    SUMMARIZATION = "summarization"  # condense, summarize, shorten
    ANALYSIS = "analysis"  # analyze, compare, evaluate
    DEBUGGING = "debugging"  # fix, error, log, stack, trace
    REASONING = "reasoning"  # deduce, infer, solve, prove


@dataclasses.dataclass(slots=True)
class TaskClassification:
    """Complete task classification output."""

    task_class: TaskClass = TaskClass.RETRIEVAL
    confidence: float = 0.5
    signals: list[str] = dataclasses.field(default_factory=list)
    preferred_strategy: str = "conservative"
    reasoning_heavy: bool = False
    structured_heavy: bool = False
    debug_heavy: bool = False
    budget_ms: float = 20.0

    @property
    def is_conservative(self) -> bool:
        return self.task_class in (TaskClass.DEBUGGING, TaskClass.REASONING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class.value,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
            "preferred_strategy": self.preferred_strategy,
            "reasoning_heavy": self.reasoning_heavy,
            "structured_heavy": self.structured_heavy,
            "debug_heavy": self.debug_heavy,
            "is_conservative": self.is_conservative,
            "budget_ms": self.budget_ms,
        }


def classify_task(request: Request) -> TaskClassification:
    """Classify a request's primary task type from its content.

    Uses lightweight heuristics — deterministic, explainable, fast.
    """
    text = "\n".join(msg.content or "" for msg in request.messages)
    lowered = text.lower()

    signals: list[str] = []
    reason_bonus = 0.0
    debug_bonus = 0.0
    analysis_bonus = 0.0
    summary_bonus = 0.0
    retrieval_bonus = 0.0

    # Reasoning signals
    reasoning_patterns = [
        (r"\bwhy\b", 5),
        (r"\banalyze\b", 5),
        (r"\broot cause\b", 8),
        (r"\bdeduce\b", 8),
        (r"\binfer\b", 8),
        (r"\bsolve\b", 5),
        (r"\bprove\b", 10),
        (r"\breason\b.*\bstep\b", 10),
        (r"\bexplain\b.*\bwhy\b", 8),
        (r"\bthink\b.*\bcarefully\b", 8),
    ]
    for pattern, weight in reasoning_patterns:
        if __import__("re").search(pattern, lowered):
            reason_bonus += weight
            signals.append(f"reasoning:{pattern[:20]}")

    # Debugging signals
    debug_patterns = [
        (r"\berror\b", 4),
        (r"\bexception\b", 6),
        (r"\btraceback\b", 8),
        (r"\blog\b", 2),
        (r"\bfailure\b", 5),
        (r"\bbug\b", 5),
        (r"\bcrash\b", 7),
        (r"\bdebug\b", 8),
        (r"\bstall\b", 5),
        (r"\btimeout\b", 4),
        (r"\boutage\b", 6),
    ]
    for pattern, weight in debug_patterns:
        if __import__("re").search(pattern, lowered):
            debug_bonus += weight
            signals.append(f"debug:{pattern[:20]}")

    # Analysis signals
    analysis_patterns = [
        (r"\bcompare\b", 4),
        (r"\bdifference\b", 3),
        (r"\btrend\b", 4),
        (r"\bpattern\b", 3),
        (r"\bcorrelation\b", 5),
        (r"\bstatistic\b", 5),
    ]
    for pattern, weight in analysis_patterns:
        if __import__("re").search(pattern, lowered):
            analysis_bonus += weight
            signals.append(f"analysis:{pattern[:20]}")

    # Summarization signals
    words = len(lowered.split())
    if words > 500:
        summary_bonus += 15
        signals.append("summarization:long_form")
    if words > 200:
        summary_bonus += 5

    # Retrieval signals
    retrieval_patterns = [
        (r"\bfind\b", 3),
        (r"\blookup\b", 5),
        (r"\bsearch\b", 4),
        (r"\bselect\b", 2),
        (r"\blist\b", 2),
        (r"\bshow\b", 1),
    ]
    for pattern, weight in retrieval_patterns:
        if __import__("re").search(pattern, lowered):
            retrieval_bonus += weight
            signals.append(f"retrieval:{pattern[:20]}")

    # Structured content bonus for analysis/debugging
    has_json = "{" in text or "[" in text
    has_code = "```" in text
    has_table = "|" in text
    structured_heavy = has_json or has_code or has_table

    # Log-heavy = debugging
    log_lines = len(__import__("re").findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", lowered))
    if log_lines > 5:
        debug_bonus += log_lines * 1.5
        signals.append(f"debug:log_lines={log_lines}")

    # Determine dominant class
    scores = {
        TaskClass.REASONING: reason_bonus,
        TaskClass.DEBUGGING: debug_bonus,
        TaskClass.ANALYSIS: analysis_bonus,
        TaskClass.SUMMARIZATION: summary_bonus,
        TaskClass.RETRIEVAL: retrieval_bonus or 1.0,
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    dominant_class = ranked[0][0]
    dominant_score = ranked[0][1]
    _unused = ranked[1][1] if len(ranked) > 1 else 0.0
    total_score = sum(scores.values()) or 1.0

    # Conservative tie-breaking
    if debug_bonus > 5 and reason_bonus > 5:
        dominant_class = TaskClass.DEBUGGING if log_lines > 5 else TaskClass.REASONING

    # Budget based on complexity
    budget_ms = 20.0
    if dominant_class in (TaskClass.DEBUGGING, TaskClass.REASONING):
        budget_ms = 50.0
    elif dominant_class == TaskClass.ANALYSIS:
        budget_ms = 30.0

    return TaskClassification(
        task_class=dominant_class,
        confidence=round(min(dominant_score / total_score, 1.0), 2),
        signals=signals[:20],
        preferred_strategy="conservative"
        if dominant_class in (TaskClass.DEBUGGING, TaskClass.REASONING)
        else "balanced",
        reasoning_heavy=reason_bonus > 10,
        structured_heavy=structured_heavy,
        debug_heavy=debug_bonus > 10,
        budget_ms=budget_ms,
    )
