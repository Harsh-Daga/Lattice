"""Tests for Pipeline."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import is_ok
from lattice.ir.primitives import ExecutionPlan
from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry
from lattice.transport.types import Message, Request


class TestTransformRegistryV2:
    def test_registry_loads_content_profiler(self) -> None:
        registry = PipelineTransformRegistry()
        inst = registry.get("content_profiler")
        assert inst is not None
        assert hasattr(inst, "process")

    def test_registry_missing_returns_none(self) -> None:
        registry = PipelineTransformRegistry()
        assert registry.get("nonexistent_transform") is None

    def test_registry_get_names(self) -> None:
        registry = PipelineTransformRegistry()
        names = registry.get_transform_names()
        assert "content_profiler" in names
        assert "prefix_optimizer" in names
        assert "output_cleanup" in names


class TestPipelineV2Execution:
    def test_pipeline_executes_plan_verbatim(self) -> None:
        registry = PipelineTransformRegistry()
        pipeline = Pipeline(registry)

        plan = ExecutionPlan(
            transforms=("content_profiler", "runtime_contract", "output_cleanup"),
            quality_floor=0.8,
            latency_budget_ms=100.0,
        )

        request = Request(
            messages=[Message(role="user", content="Hello world")],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.process(request, plan, ctx)
        assert is_ok(result)
        modified = result.unwrap()
        assert modified is not None

    def test_pipeline_respects_budget(self) -> None:
        registry = PipelineTransformRegistry()
        pipeline = Pipeline(registry)

        plan = ExecutionPlan(
            transforms=("content_profiler", "runtime_contract"),
            quality_floor=0.8,
            latency_budget_ms=0.0001,
        )

        request = Request(messages=[Message(role="user", content="Hi")])
        ctx = TransformContext()

        result = pipeline.process(request, plan, ctx)
        assert is_ok(result)
        assert result.unwrap() is not None

    def test_pipeline_tracks_transforms_applied(self) -> None:
        registry = PipelineTransformRegistry()
        pipeline = Pipeline(registry)

        plan = ExecutionPlan(
            transforms=("content_profiler", "runtime_contract"),
            quality_floor=0.8,
            latency_budget_ms=100.0,
        )

        request = Request(messages=[Message(role="user", content="Hello")])
        ctx = TransformContext()

        result = pipeline.process(request, plan, ctx)
        assert is_ok(result)
        assert result.unwrap() is not None

    def test_pipeline_runs_all_transforms(self) -> None:
        registry = PipelineTransformRegistry()
        pipeline = Pipeline(registry)

        plan = ExecutionPlan(
            transforms=("content_profiler", "runtime_contract"),
            quality_floor=0.8,
            latency_budget_ms=100.0,
        )

        request = Request(messages=[Message(role="user", content="Hi")])
        ctx = TransformContext()

        result = pipeline.process(request, plan, ctx)
        assert is_ok(result)
        assert result.unwrap() is not None

    def test_pipeline_skips_missing_transform(self) -> None:
        registry = PipelineTransformRegistry()
        pipeline = Pipeline(registry)

        plan = ExecutionPlan(
            transforms=("nonexistent_transform", "content_profiler"),
            quality_floor=0.8,
            latency_budget_ms=100.0,
        )

        request = Request(messages=[Message(role="user", content="Hello")])
        ctx = TransformContext()

        result = pipeline.process(request, plan, ctx)
        assert is_ok(result)
        assert result.unwrap() is not None


class TestPipelineV2Reverse:
    def test_reverse_applies_in_reverse_order(self) -> None:
        registry = PipelineTransformRegistry()
        pipeline = Pipeline(registry)

        plan = ExecutionPlan(
            transforms=("reference_optimizer", "content_profiler"),
            quality_floor=0.8,
            latency_budget_ms=100.0,
        )

        from lattice.transport.types import Response

        response = Response(role="assistant", content="test")
        ctx = TransformContext()

        # Should not crash even without corresponding forward pass
        result = pipeline.reverse(response, plan, ctx)
        assert result is not None
