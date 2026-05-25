"""Execution-time pipeline package — canonical public surface.

After Phase 3 removed ``core/pipeline.py`` and ``core/pipeline_v2_wrapper.py``,
the circular-import constraint that forced this ``__init__.py`` to stay empty
is gone. Re-export the canonical entry points so callers can write::

    from lattice.pipeline import (
        Pipeline,
        PipelineTransformRegistry,
        ReversibleSyncTransform,
        TransformClass,
        build_default_pipeline,
        build_benchmark_pipeline,
        pipeline_summary,
        OptimizationPolicy,
        Allow,
        Skip,
        Reject,
        GuardAction,
        SafetyDecision,
        ValidationOutcome,
        PostTransformGuardResult,
        AutoContinuation,
        ContinuationResult,
        RequestCoalescer,
        CoalescedResult,
        CoalescedRequest,
        RepresentationOptimizer,
    )
"""

from lattice.pipeline.auto_continuation import AutoContinuation, ContinuationResult
from lattice.pipeline.base import ReversibleSyncTransform, TransformClass
from lattice.pipeline.factory import (
    build_benchmark_pipeline,
    build_default_pipeline,
    pipeline_summary,
)
from lattice.pipeline.guardrails import (
    GuardAction,
    SafetyDecision,
    ValidationOutcome,
    check_blank_output,
    check_critical_signal_loss,
    check_entity_preservation,
    check_expansion_guard,
    check_format_preservation,
    check_negative_savings,
    check_placeholder_leakage,
)
from lattice.pipeline.policy import Allow, OptimizationPolicy, Reject, Skip
from lattice.pipeline.post_transform_guard import (
    PostTransformGuardResult,
    evaluate_post_transform,
    should_trigger_post_transform_guard,
)
from lattice.pipeline.representation_optimizer import RepresentationOptimizer
from lattice.pipeline.request_coalescer import CoalescedRequest, CoalescedResult, RequestCoalescer
from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry

__all__ = [
    # runner
    "Pipeline",
    "PipelineTransformRegistry",
    "ReversibleSyncTransform",
    "TransformClass",
    # factory
    "build_default_pipeline",
    "build_benchmark_pipeline",
    "pipeline_summary",
    # policy
    "OptimizationPolicy",
    "Allow",
    "Skip",
    "Reject",
    # guardrails
    "GuardAction",
    "SafetyDecision",
    "ValidationOutcome",
    "check_expansion_guard",
    "check_entity_preservation",
    "check_format_preservation",
    "check_critical_signal_loss",
    "check_placeholder_leakage",
    "check_negative_savings",
    "check_blank_output",
    # post-transform guard
    "PostTransformGuardResult",
    "should_trigger_post_transform_guard",
    "evaluate_post_transform",
    # auto continuation
    "AutoContinuation",
    "ContinuationResult",
    # request coalescer
    "RequestCoalescer",
    "CoalescedResult",
    "CoalescedRequest",
    # representation optimizer
    "RepresentationOptimizer",
]
