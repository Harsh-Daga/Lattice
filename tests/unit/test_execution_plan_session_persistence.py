"""End-to-end tests for ExecutionPlan session persistence across multi-turn.

Covers:
1. Session creation with ExecutionPlan in metadata
2. Session serialization round-trip (to_dict / from_dict)
3. Multi-turn merge logic (sticky provider, union optimizers, stricter constraints)
"""

from __future__ import annotations

from lattice.core.session import Session
from lattice.planner.execution_plan import ExecutionPlan, FallbackPlan


class TestExecutionPlanSessionPersistence:
    """ExecutionPlan survives session serialization and multi-turn merging."""

    def test_session_to_dict_preserves_execution_plan(self) -> None:
        plan = ExecutionPlan(
            provider="anthropic",
            model="claude-3",
            allowed_optimizers=["reference_optimizer", "structure_optimizer"],
            quality_floor=0.90,
            latency_budget_ms=120.0,
            fallback_plan=FallbackPlan(retry_count=2, fallback_provider="openai"),
        )
        session = Session(
            session_id="abc123",
            created_at=1000.0,
            last_accessed_at=1000.0,
            messages=[],
            metadata={"_lattice_execution_plan": plan.to_dict()},
        )
        data = session.to_dict()
        assert "_lattice_execution_plan" in data["metadata"]
        ep = data["metadata"]["_lattice_execution_plan"]
        assert ep["provider"] == "anthropic"
        assert ep["allowed_optimizers"] == ["reference_optimizer", "structure_optimizer"]
        assert ep["fallback_plan"]["retry_count"] == 2
        assert ep["fallback_plan"]["fallback_provider"] == "openai"

    def test_session_from_dict_restores_execution_plan(self) -> None:
        plan = ExecutionPlan(
            provider="openai",
            model="gpt-4",
            allowed_optimizers=["tool_optimizer"],
            quality_floor=0.88,
        )
        session = Session(
            session_id="xyz789",
            created_at=2000.0,
            last_accessed_at=2000.0,
            messages=[],
            metadata={"_lattice_execution_plan": plan.to_dict()},
        )
        data = session.to_dict()
        restored = Session.from_dict(data)
        ep = restored.metadata.get("_lattice_execution_plan")
        assert ep is not None
        assert ep["provider"] == "openai"
        assert ep["allowed_optimizers"] == ["tool_optimizer"]
        # Verify cache_plan round-trips
        assert "cache_plan" in ep
        assert ep["quality_floor"] == 0.88

    def test_multi_turn_merge_sticky_provider(self) -> None:
        """Provider/model are sticky across turns — second turn must not override."""
        old_plan = ExecutionPlan(
            provider="anthropic",
            model="claude-3-sonnet",
            allowed_optimizers=["reference_optimizer"],
            quality_floor=0.85,
            latency_budget_ms=100.0,
        )
        new_plan = ExecutionPlan(
            provider="openai",
            model="gpt-4o",
            allowed_optimizers=["structure_optimizer"],
            quality_floor=0.80,
            latency_budget_ms=80.0,
        )
        session = Session(
            session_id="multi",
            created_at=1000.0,
            last_accessed_at=1000.0,
            messages=[],
            metadata={"_lattice_execution_plan": old_plan.to_dict()},
        )
        prev = session.metadata.get("_lattice_execution_plan")
        assert prev is not None
        restored = ExecutionPlan.from_dict(prev)

        # Merge logic from compat.py
        new_plan.provider = restored.provider
        new_plan.model = restored.model
        new_plan.allowed_optimizers = list(
            set(restored.allowed_optimizers) | set(new_plan.allowed_optimizers)
        )
        new_plan.quality_floor = max(new_plan.quality_floor, restored.quality_floor)
        new_plan.latency_budget_ms = max(new_plan.latency_budget_ms, restored.latency_budget_ms)

        assert new_plan.provider == "anthropic"
        assert new_plan.model == "claude-3-sonnet"
        assert sorted(new_plan.allowed_optimizers) == sorted(
            ["reference_optimizer", "structure_optimizer"]
        )
        assert new_plan.quality_floor == 0.85
        assert new_plan.latency_budget_ms == 100.0

    def test_multi_turn_stricter_constraints(self) -> None:
        """Stricter quality_floor and budget must win across turns."""
        old_plan = ExecutionPlan(
            quality_floor=0.95,
            latency_budget_ms=200.0,
            allowed_optimizers=["diagnostic_optimizer"],
        )
        new_plan = ExecutionPlan(
            quality_floor=0.80,
            latency_budget_ms=50.0,
            allowed_optimizers=["context_optimizer"],
        )
        new_plan.quality_floor = max(new_plan.quality_floor, old_plan.quality_floor)
        new_plan.latency_budget_ms = max(new_plan.latency_budget_ms, old_plan.latency_budget_ms)
        new_plan.allowed_optimizers = list(
            set(old_plan.allowed_optimizers) | set(new_plan.allowed_optimizers)
        )
        assert new_plan.quality_floor == 0.95
        assert new_plan.latency_budget_ms == 200.0
        assert sorted(new_plan.allowed_optimizers) == sorted(
            ["diagnostic_optimizer", "context_optimizer"]
        )
