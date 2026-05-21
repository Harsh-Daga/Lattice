"""End-to-end integration test for the v2 refactored architecture.

Verifies the full new flow:
1. content_profiler builds canonical PromptIRV2 + metadata
2. UnifiedPlanner produces ExecutionPlan
3. Pipeline executes plan verbatim
4. CandidateScorer scores results (single source)
"""

from __future__ import annotations

import asyncio

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.core.unified_planner import UnifiedPlanner, profile_from_legacy
from lattice.ir.primitives import (
    Candidate,
    PromptIRV2,
    SectionV2,
    SpanV2,
)
from lattice.pipeline.factory import build_optimizer_pipeline
from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry
from lattice.transport.types import Message, Request


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestRefoundationEndToEnd:
    """Full v2 architecture integration test."""

    def test_content_profiler_builds_ir(self) -> None:
        from lattice.transforms.content_profiler import ContentProfiler

        profiler = ContentProfiler()
        request = Request(
            messages=[_req('[{"status":"ok","value":42}]')],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = profiler.process(request, ctx)
        assert is_ok(result)
        modified = unwrap(result)

        # Canonical IRV2 must have been compiled and stored
        assert "_lattice_ir_v2" in modified.metadata, (
            f"IRV2 not in metadata. Keys: {list(modified.metadata.keys())}"
        )
        assert "_lattice_segments" in ctx.session_state
        # Content profiler stores task class information
        assert "content_profiler" in ctx.session_state
        assert "_lattice_schedule" in ctx.session_state

    def test_unified_planner_reasoning_tier(self) -> None:
        """Debug input should trigger high-quality plan."""
        from lattice.transforms.content_profiler import ContentProfiler

        ctx = TransformContext()
        request = Request(
            messages=[_req("Error: KeyError in line 42 of app.py")],
            model="gpt-4",
        )

        profiler = ContentProfiler()
        profiler.process(request, ctx)

        legacy = ctx.session_state.get("_lattice_schedule", {}).get("task_class")
        profile = profile_from_legacy(legacy)
        assert profile is not None

        planner = UnifiedPlanner()
        plan = planner.plan(request, profile)

        assert "context_optimizer" not in plan.transforms
        assert plan.quality_floor >= 0.80

    def test_pipeline_v2_executes_plan(self) -> None:
        from lattice.transforms.content_profiler import ContentProfiler

        ctx = TransformContext()
        request = Request(
            messages=[_req("Hello world")],
            model="gpt-4",
        )

        profiler = ContentProfiler()
        profiler.process(request, ctx)

        legacy = ctx.session_state.get("_lattice_schedule", {}).get("task_class")
        profile = profile_from_legacy(legacy)

        planner = UnifiedPlanner()
        plan = planner.plan(request, profile)

        pipeline = Pipeline()
        result = pipeline.process(request, plan, ctx)

        assert is_ok(result)
        modified = unwrap(result)
        assert modified is not None

    def test_ir_optimizer_exists_in_registry(self) -> None:
        registry = PipelineTransformRegistry()
        inst = registry.get("ir_structure_optimizer")
        assert inst is not None
        assert hasattr(inst, "process")
        assert hasattr(inst, "can_process")

    def test_new_components_all_green(self) -> None:
        """Smoke-test all new components work together without error."""
        from lattice.ir.transform import CandidateScorer

        span = SpanV2(span_id="s1", text='{"x":1}')
        section = SectionV2(type="json", spans=(span,))
        ir = PromptIRV2(sections=(section,))

        c = Candidate(ir=ir).with_metric("tokens_before", 10).with_metric("tokens_after", 5)
        score = CandidateScorer.score(c)

        assert score.composite > 0.0
        assert score.cost_reduction == 0.5

        passed, reason = CandidateScorer.validate(c, quality_floor=0.85)
        assert passed is True

    def test_legacy_pipeline_still_works(self) -> None:
        """Old pipeline still functions (backward compat)."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        pipeline = build_optimizer_pipeline(cfg)

        request = Request(
            messages=[_req("Analyze this JSON: {'status':'ok'}")],
            model="gpt-4",
        )
        ctx = TransformContext()

        async def run():
            result = await pipeline.process(request, ctx)
            assert is_ok(result)
            assert result.unwrap() is not None
            return result

        asyncio.run(run())

    def test_end_to_end_v2_architecture(self) -> None:
        """Full integration: profiler -> planner -> pipeline_v2."""
        from lattice.transforms.content_profiler import ContentProfiler

        ctx = TransformContext()
        request = Request(
            messages=[_req('[{"status":"ok","name":"alpha"},{"status":"ok","name":"beta"}]')],
            model="gpt-4",
        )

        profiler = ContentProfiler()
        result1 = profiler.process(request, ctx)
        assert is_ok(result1)

        legacy = ctx.session_state.get("_lattice_schedule", {}).get("task_class")
        assert legacy is not None
        profile = profile_from_legacy(legacy)

        planner = UnifiedPlanner()
        plan = planner.plan(request, profile)

        pipeline_v2 = Pipeline()
        result2 = pipeline_v2.process(request, plan, ctx)

        assert is_ok(result2)
        output = result2.unwrap()
        assert output is not None
        text = "\n".join(m.content or "" for m in output.messages)
        assert len(text) > 0

    def test_production_v2_pipeline_full(self) -> None:
        """Use build_v2_pipeline() with use_v2_pipeline=True (proxy path)."""
        from lattice.pipeline.factory import build_v2_pipeline

        cfg = LatticeConfig(use_v2_pipeline=True)
        pipeline = build_v2_pipeline(cfg)

        # Registered transforms should include content_profiler + pipeline_v2
        names = [t.name for t in pipeline.transforms]
        assert "content_profiler" in names
        assert "pipeline_v2" in names
        assert "runtime_contract" in names

        request = Request(
            messages=[_req("Hello world")],
            model="gpt-4",
        )
        ctx = TransformContext()

        async def run():
            result = await pipeline.process(request, ctx)
            assert is_ok(result)
            output = result.unwrap()
            assert output is not None
            # ExecutionPlan should have been set by content_profiler
            assert "_lattice_execution_plan" in ctx.session_state or (
                "_lattice_schedule" in ctx.session_state
            )
            return output

        asyncio.run(run())
