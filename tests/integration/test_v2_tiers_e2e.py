"""Test all 5 UnifiedPlanner tiers end-to-end."""

from __future__ import annotations

import asyncio

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok
from lattice.core.unified_planner import (
    SemanticProfile,
    TaskClass,
    UnifiedPlanner,
)
from lattice.proxy.bootstrap import build_proxy_runtime
from lattice.transport.types import Message, Request


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestV2TiersEndToEnd:
    """All 5 UnifiedPlanner tiers through the v2 pipeline."""

    def _run_pipeline(self, request: Request) -> tuple[Request, TransformContext]:
        """Run v2 pipeline synchronously."""
        cfg = LatticeConfig()
        runtime = build_proxy_runtime(cfg)
        ctx = TransformContext()

        async def _run() -> tuple[Request, TransformContext]:
            result = await runtime.pipeline.process(request, ctx)
            assert is_ok(result)
            return result.unwrap(), ctx

        return asyncio.run(_run())

    def test_fast_tier_runs_profiler_contract_only(self) -> None:
        """FAST tier: minimal latency, no beam search."""
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE,
            context_length=5,
            risk_total=0,
        )
        planner = UnifiedPlanner()
        plan = planner.plan(
            Request(messages=[_req("Hello")], model="gpt-4"),
            profile,
        )
        assert plan.latency_budget_ms == 30.0
        assert plan.quality_floor == 0.80
        assert "context_optimizer" not in plan.transforms

    def test_safe_tier_preserves_all(self) -> None:
        """SAFE tier: allows most safe transforms including context_optimizer."""
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE,
            context_length=3000,
            risk_total=55,
            is_conservative=False,
        )
        planner = UnifiedPlanner()
        plan = planner.plan(
            Request(messages=[_req("Error: something failed")], model="gpt-4"),
            profile,
        )
        assert plan.latency_budget_ms == 80.0  # SAFE budget
        # SAFE tier includes many safe transforms
        assert "runtime_contract" in plan.transforms
        assert "reference_optimizer" in plan.transforms
        assert "context_optimizer" in plan.transforms  # SAFE allows context_optimizer

    def test_reasoning_tier_debugging(self) -> None:
        """REASONING tier: high quality for debugging."""
        profile = SemanticProfile(
            task_class=TaskClass.DEBUGGING,
            context_length=10000,
            risk_total=30,
            is_conservative=True,
        )
        planner = UnifiedPlanner()
        plan = planner.plan(
            Request(messages=[_req("Traceback (most recent call last):")], model="gpt-4"),
            profile,
        )
        assert plan.quality_floor == 0.90
        assert plan.latency_budget_ms == 60.0

    def test_emergency_tier_minimal(self) -> None:
        """EMERGENCY tier: profiler + contract only."""
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE,
            context_length=500,
            risk_total=70,  # > 60 triggers EMERGENCY
        )
        planner = UnifiedPlanner()
        plan = planner.plan(
            Request(messages=[_req("Quick test")], model="gpt-4"),
            profile,
        )
        # EMERGENCY tier only allows content_profiler + runtime_contract
        assert "runtime_contract" in plan.transforms
        assert "context_optimizer" not in plan.transforms
        assert plan.latency_budget_ms == 10.0

    def test_full_pipeline_fast_tier(self) -> None:
        """FAST tier through full v2 proxy pipeline."""
        request = Request(messages=[_req("Short prompt")], model="gpt-4")
        compressed, ctx = self._run_pipeline(request)
        assert compressed is not None
        assert "pipeline_v2" in ctx.transforms_applied

    def test_full_pipeline_reasoning_tier(self) -> None:
        """REASONING tier through full v2 proxy pipeline."""
        request = Request(
            messages=[_req("Analyze why this TypeError occurs in line 42")],
            model="gpt-4",
        )
        compressed, ctx = self._run_pipeline(request)
        assert compressed is not None
        # v2 pipeline should have been applied
        assert "pipeline_v2" in ctx.transforms_applied
        # ExecutionPlan should exist
        assert "_lattice_execution_plan" in ctx.session_state or (
            "_lattice_schedule" in ctx.session_state
        )
