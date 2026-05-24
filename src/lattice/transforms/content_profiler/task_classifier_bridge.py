"""Bridge from content profiler into planner task classification."""

from __future__ import annotations

from lattice.planner.task_classifier import TaskClassification, classify_task
from lattice.transport.types import Request


def bridge_task_classification(request: Request) -> TaskClassification:
    """Wrap planner classify_task with content-profiler-specific overrides if any."""
    return classify_task(request)


__all__ = ["TaskClassification", "bridge_task_classification", "classify_task"]
