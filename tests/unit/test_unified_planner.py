"""Tests for UnifiedPlanner."""
from __future__ import annotations

from lattice.core.task_classifier import TaskClass
from lattice.core.transport import Message, Request
from lattice.core.unified_planner import SemanticProfile, Tier, UnifiedPlanner


class TestTierClassification:
    def test_fast_tier_simple_short(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE,
            context_length=200,
            risk_total=10,
        )
        assert planner.classify_tier(profile) == Tier.FAST

    def test_reasoning_tier(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(task_class=TaskClass.REASONING)
        assert planner.classify_tier(profile) == Tier.REASONING

    def test_debugging_tier(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(task_class=TaskClass.DEBUGGING)
        assert planner.classify_tier(profile) == Tier.REASONING

    def test_emergency_high_risk(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE, risk_total=70
        )
        assert planner.classify_tier(profile) == Tier.EMERGENCY

    def test_safe_tier_conservative(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.ANALYSIS,
            risk_total=20,
            is_conservative=True,
        )
        assert planner.classify_tier(profile) == Tier.SAFE

    def test_emergency_conservative_high_risk(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.ANALYSIS,
            risk_total=40,
            is_conservative=True,
        )
        assert planner.classify_tier(profile) == Tier.EMERGENCY


class TestPlanGeneration:
    def test_fast_plan_has_core_transforms(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(task_class=TaskClass.SIMPLE, context_length=200)
        request = Request(messages=[Message(role="user", content="Hi")])
        plan = planner.plan(request, profile)

        assert "content_profiler" in plan.transforms
        assert "runtime_contract" in plan.transforms
        assert "representation_optimizer" not in plan.transforms
        assert "tool_filter" in plan.transforms
        assert plan.quality_floor == 0.80
        assert plan.latency_budget_ms == 30.0

    def test_reasoning_plan_no_lossy(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(task_class=TaskClass.REASONING)
        request = Request(messages=[Message(role="user", content="Prove by induction")])
        plan = planner.plan(request, profile)

        assert "context_optimizer" not in plan.transforms
        assert "representation_optimizer" in plan.transforms
        assert plan.quality_floor == 0.92
        assert plan.latency_budget_ms == 60.0

    def test_debugging_plan_has_diagnostic(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(task_class=TaskClass.DEBUGGING)
        request = Request(messages=[Message(role="user", content="Traceback...")])
        plan = planner.plan(request, profile)

        assert "diagnostic_optimizer" in plan.transforms
        assert "context_optimizer" not in plan.transforms
        assert plan.quality_floor == 0.90

    def test_long_context_adds_context_optimizer(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.STRUCTURED,
            context_length=5000,
        )
        request = Request(messages=[Message(role="user", content="Large table")])
        plan = planner.plan(request, profile)

        assert "context_optimizer" in plan.transforms

    def test_streaming_reduces_budget(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE,
            is_streaming=True,
        )
        request = Request(messages=[Message(role="user", content="Hi")])
        plan = planner.plan(request, profile)

        assert plan.latency_budget_ms == 30.0  # min(30, 50) = 30.0

    def test_transform_order_is_stable(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(task_class=TaskClass.SIMPLE)
        plan = planner.plan(Request(messages=[]), profile)

        # content_profiler must always be first
        assert plan.transforms[0] == "content_profiler"
        # tool_filter is the last request-side transform

    def test_planner_no_duplicate_transforms(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.STRUCTURED,
            context_length=5000,
            has_tool_calls=True,
        )
        plan = planner.plan(Request(messages=[]), profile)

        assert len(plan.transforms) == len(set(plan.transforms))

    def test_emergency_plan_minimal(self) -> None:
        planner = UnifiedPlanner()
        profile = SemanticProfile(
            task_class=TaskClass.SIMPLE, risk_total=70
        )
        plan = planner.plan(Request(messages=[]), profile)

        assert plan.transforms == ("content_profiler", "runtime_contract")
        assert plan.latency_budget_ms == 10.0
