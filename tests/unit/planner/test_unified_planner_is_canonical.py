"""UnifiedPlanner is the sole scheduling entry point after Phase 4."""

from __future__ import annotations


def test_unified_planner_imports() -> None:
    from lattice.planner import (
        ExecutionPlan,
        SemanticProfile,
        TaskClass,
        UnifiedPlanner,
        build_execution_plan,
        classify_task,
    )

    assert callable(UnifiedPlanner)
    assert callable(build_execution_plan)
    assert callable(classify_task)
    assert ExecutionPlan is not None
    assert SemanticProfile is not None
    assert TaskClass is not None


def test_scheduler_symbols_gone() -> None:
    import lattice.core

    assert not hasattr(lattice.core, "decide_schedule")
    assert not hasattr(lattice.core, "SchedulerDecision")
    assert not hasattr(lattice.core, "decide_optimizer_schedule")
    assert not hasattr(lattice.core, "OptimizerSchedule")
