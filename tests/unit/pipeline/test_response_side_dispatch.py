"""Pipeline dispatch for response-side transforms (registry is_response_side)."""

from __future__ import annotations

from unittest.mock import MagicMock

from lattice.core.context import TransformContext
from lattice.core.result import is_ok
from lattice.ir.primitives import ExecutionPlan
from lattice.pipeline.runner import Pipeline
from lattice.transforms.registry import get_transform_spec, is_response_side
from lattice.transport.types import Message, Request, Response


def test_output_cleanup_marked_response_side_in_registry() -> None:
    spec = get_transform_spec("output_cleanup")
    assert spec is not None
    assert spec.is_response_side is True
    assert is_response_side("output_cleanup") is True
    assert is_response_side("reference_sub") is False


def test_compress_skips_output_cleanup_on_request_path() -> None:
    pipeline = Pipeline()
    req = Request(messages=[Message(role="user", content="Summarize this log file.")])
    ctx = TransformContext()
    result = pipeline.compress(req, ctx)
    assert is_ok(result)
    assert "output_cleanup" not in ctx.transforms_applied


def test_reverse_runs_output_cleanup_via_registry() -> None:
    pipeline = Pipeline()
    ctx = TransformContext()
    response = Response(content="Sure! Here is the answer you asked for.\n\n42")
    plan = ExecutionPlan(
        transforms=("content_profiler", "output_cleanup"),
        latency_budget_ms=1000.0,
        quality_floor=0.85,
    )

    mock_oc = MagicMock()
    mock_oc.reverse.side_effect = lambda r, _c: Response(content=r.content.strip())
    pipeline.registry.register_instance("output_cleanup", mock_oc)

    out = pipeline.reverse(response, ctx, plan=plan)
    mock_oc.reverse.assert_called_once()
    assert out.content == "Sure! Here is the answer you asked for.\n\n42"


def test_process_skips_response_side_transforms_in_plan() -> None:
    pipeline = Pipeline()
    req = Request(messages=[Message(role="user", content="Hello.")])
    ctx = TransformContext()
    plan = ExecutionPlan(
        transforms=("output_cleanup", "runtime_contract"),
        latency_budget_ms=1000.0,
        quality_floor=0.85,
    )
    mock_oc = MagicMock()
    pipeline.registry.register_instance("output_cleanup", mock_oc)
    pipeline.registry.register_instance(
        "runtime_contract",
        pipeline.registry.get("runtime_contract"),
    )

    result = pipeline.process(req, plan, ctx)
    assert is_ok(result)
    mock_oc.process.assert_not_called()
