"""End-to-end test for v2 proxy pipeline + reverse path."""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok
from lattice.pipeline.factory import build_default_pipeline
from lattice.proxy.bootstrap import build_proxy_runtime
from lattice.transport.types import Message, Request, Response


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestV2ProxyPipeline:
    """Proxy runtime with v2 pipeline — full process + reverse."""

    def test_v2_pipeline_runs_and_produces_execution_plan(self) -> None:
        cfg = LatticeConfig(provider_base_url="http://test")
        runtime = build_proxy_runtime(cfg)

        request = Request(
            messages=[_req("Hello world")],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = runtime.pipeline.compress(request, ctx)
        assert is_ok(result)
        compressed = result.unwrap()
        assert compressed is not None
        assert "content_profiler" in ctx.transforms_applied
        assert "_lattice_execution_plan" in ctx.session_state or (
            "_lattice_schedule" in ctx.session_state
        )

        response = Response(role="assistant", content="Hello!", model="gpt-4")
        reversed_resp = runtime.pipeline.reverse(response, ctx)
        assert reversed_resp is not None
        assert reversed_resp.content == "Hello!"

    def test_v2_pipeline_with_optimizer_transforms(self) -> None:
        """ExecutionPlan with optimizers triggers CandidateSearch beam."""
        from lattice.core.unified_planner import UnifiedPlanner, profile_from_legacy
        from lattice.ir.primitives import ExecutionPlan

        cfg = LatticeConfig()
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[_req('{"status":"ok","data":[1,2,3]}')],
            model="gpt-4",
        )
        ctx = TransformContext()

        # Run content_profiler first (produces ExecutionPlan)
        profiler = next(t for t in pipeline.transforms if t.name == "content_profiler")
        profiler.process(request, ctx)

        # Verify plan exists
        plan = ctx.session_state.get("_lattice_execution_plan")
        if plan is None:
            legacy = ctx.session_state.get("_lattice_schedule", {})
            profile = profile_from_legacy(legacy)
            if profile is not None:
                planner = UnifiedPlanner()
                plan = planner.plan(request, profile)

        assert plan is not None
        assert isinstance(plan, ExecutionPlan)
        assert plan.quality_floor >= 0.80

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        assert result.unwrap() is not None
        assert "content_profiler" in ctx.transforms_applied
