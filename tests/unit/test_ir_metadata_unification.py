"""Tests for canonical IR metadata unification."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.ir.primitives import PromptIRV2
from lattice.ir.quality import estimate_transport_gain
from lattice.pipeline.representation_optimizer import _get_allowed_optimizers
from lattice.pipeline.representation_optimizer import _get_quality_floor as structure_quality_floor
from lattice.transforms.content_profiler import ContentProfiler
from lattice.transport.types import Message, Request


def _run_content_profiler(request: Request, context: TransformContext):
    return ContentProfiler().optimize(PromptIRV2(), request, context)


def test_content_profiler_embeds_semantic_metadata_into_ir() -> None:
    request = Request(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content='{"id": 1, "status": "ok"}'),
        ],
        model="gpt-4",
    )
    context = TransformContext(provider="openai", model="gpt-4")

    result = _run_content_profiler(request, context)
    assert is_ok(result)
    ir_v2 = unwrap(result)

    assert "protocol" in dict(ir_v2.metadata)
    assert "_lattice_segment_summary" in dict(ir_v2.metadata)
    assert "_prefix_manifest" in dict(ir_v2.metadata)
    assert "_lattice_protocol_manifest" in dict(ir_v2.metadata)
    assert "_lattice_optimizer_schedule" in dict(ir_v2.metadata)
    assert request.metadata["_lattice_ir_v2_summary"]["sections"] >= 0


def test_optimizer_reads_allowed_optimizers_from_ir_metadata_only() -> None:
    request = Request(
        messages=[Message(role="user", content="Please analyze this request.")],
        model="gpt-4",
    )
    context = TransformContext(provider="openai", model="gpt-4")

    result = _run_content_profiler(request, context)
    assert is_ok(result)

    # Remove the side-channel schedule so the optimizer must fall back to IR metadata.
    context.session_state.pop("_lattice_optimizer_schedule", None)
    context.session_state.pop("_lattice_execution_plan", None)
    context.session_state.pop("_lattice_segment_summary", None)

    allowed = _get_allowed_optimizers(context)
    assert allowed


def test_transport_gain_reads_provider_from_ir_metadata() -> None:
    request = Request(
        messages=[Message(role="user", content="Hello there.")],
        model="gpt-4",
    )
    context = TransformContext(provider="openai", model="gpt-4")

    result = _run_content_profiler(request, context)
    assert is_ok(result)

    context.session_state.pop("_lattice_provider", None)
    context.session_state.pop("session_id", None)
    context.session_state.pop("_lattice_prev_request_hash", None)

    gain = estimate_transport_gain(request, request.copy(), context, [])
    assert gain >= 0.08


def test_quality_floor_reads_execution_plan_from_ir_metadata() -> None:
    request = Request(
        messages=[Message(role="user", content="Hello there.")],
        model="gpt-4",
    )
    context = TransformContext(provider="openai", model="gpt-4")

    result = _run_content_profiler(request, context)
    assert is_ok(result)

    context.session_state.pop("_lattice_execution_plan", None)
    context.session_state.pop("_lattice_task_classification", None)

    assert structure_quality_floor(context) > 0.0
