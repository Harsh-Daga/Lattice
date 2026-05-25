"""Validation facade — delegates to existing modules (no duplicate logic)."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.ir.primitives import Candidate
from lattice.ir.validation import ValidationResult, validate_candidate
from lattice.pipeline.guardrails import ValidationOutcome, check_blank_output
from lattice.pipeline.post_transform_guard import (
    PostTransformGuardResult,
    evaluate_post_transform,
)
from lattice.planner.task_classifier import TaskClassification
from lattice.transport.types import Response

__all__ = [
    "validate_ir_candidate",
    "validate_post_transform",
    "validate_output_blank",
]


def validate_ir_candidate(
    candidate: Candidate,
    context: TransformContext,
    *,
    quality_floor: float = 0.85,
    optimizer_name: str = "pipeline",
) -> ValidationResult:
    return validate_candidate(
        candidate,
        context,
        quality_floor=quality_floor,
        optimizer_name=optimizer_name,
    )


def validate_post_transform(
    before_text: str,
    after_text: str,
    task: TaskClassification,
    *,
    placeholder_aliasing_used: bool = False,
) -> PostTransformGuardResult:
    return evaluate_post_transform(
        before_text,
        after_text,
        task,
        placeholder_aliasing_used=placeholder_aliasing_used,
    )


def validate_output_blank(
    response: Response,
    *,
    baseline_output: str = "",
) -> ValidationOutcome:
    baseline = baseline_output or response.content
    return check_blank_output(baseline, response.content)
