"""Tests for provider cache simulation and canonical runtime-state writes."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.planner.execution_plan import CachePlanEntry, ExecutionPlan
from lattice.planner.provider_strategy import (
    build_cache_plan_for_provider,
    simulate_provider_cache,
)
from lattice.planner.runtime_state import persist_execution_plan_state
from lattice.transport.types import Request


class TestProviderCacheSimulation:
    def test_simulation_reports_probability_for_openai(self) -> None:
        plan = build_cache_plan_for_provider("openai", segment_count=3, estimated_tokens=1000)
        sim = simulate_provider_cache(
            "openai",
            "gpt-4",
            estimated_tokens=1000,
            cache_plan=plan,
            prefix_manifest={"prefix_tokens": 200},
        )
        assert sim.cache_plan_entries == 3
        assert sim.hit_probability > 0.0
        assert sim.expected_cached_tokens > 0

    def test_simulation_without_plan_is_zero(self) -> None:
        sim = simulate_provider_cache(
            "generic",
            "gpt-4",
            estimated_tokens=1000,
            cache_plan=[],
            prefix_manifest=None,
        )
        assert sim.hit_probability == 0.0
        assert sim.expected_cached_tokens == 0

    def test_runtime_state_persistence_writes_all_views(self) -> None:
        request = Request()
        context = TransformContext(provider="openai", model="gpt-4")
        plan = ExecutionPlan(
            request_id="req-1",
            provider="openai",
            model="gpt-4",
            cache_plan=[
                CachePlanEntry(
                    segment_index=0,
                    provider_mode="auto_prefix",
                    expected_cached_tokens=256,
                    annotations={"stable": True},
                )
            ],
        )
        persist_execution_plan_state(
            request,
            context,
            plan,
            cache_plan=plan.cache_plan,
            cache_simulation={"hit_probability": 0.75, "expected_cached_tokens": 192},
        )

        assert "_lattice_execution_plan" in request.metadata
        assert "_lattice_cache_plan" in request.metadata
        assert "_lattice_cache_simulation" in request.metadata
        assert "_lattice_execution_plan" in context.session_state
        assert "_lattice_cache_plan" in context.session_state
        assert "_lattice_cache_simulation" in context.session_state
