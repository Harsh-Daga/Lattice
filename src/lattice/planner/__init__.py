"""Planning layer: request classification → ExecutionPlan.

Single source of truth for runtime decisions: which transforms to run, in what
order, with what quality floor and latency budget.
"""

from lattice.planner.execution_builder import build_execution_plan
from lattice.planner.execution_plan import (
    TIER_BUDGETS_MS,
    CachePlanEntry,
    ExecutionPlan,
    ExecutionTier,
    FallbackPlan,
    OptimizerDecision,
    RepresentationCandidate,
    RiskLevel,
    TransportPlanEntry,
    tier_budget_ms,
)
from lattice.planner.fallback_executor import (
    execute_with_fallback,
    execute_with_fallback_stream,
)
from lattice.planner.provider_strategy import (
    CacheSimulation,
    ProviderStrategy,
    build_cache_plan_for_provider,
    get_provider_strategy,
    preferred_optimizers_for_provider,
    simulate_provider_cache,
)
from lattice.planner.request_classifier import RequestClassifier
from lattice.planner.runtime_state import (
    coerce_execution_plan,
    get_canonical_request_value,
    get_canonical_state_value,
    get_ir_metadata,
    persist_execution_plan_state,
    persist_session_plan_state,
)
from lattice.planner.task_classifier import (
    ExecutionTier as TaskExecutionTier,
)
from lattice.planner.task_classifier import (
    TaskClass,
    TaskClassification,
    classify_task,
)
from lattice.planner.transport_planner import TransportPlan, build_transport_plan
from lattice.planner.unified_planner import (
    SemanticProfile,
    UnifiedPlanner,
    profile_from_legacy,
)
from lattice.planner.unified_planner import (
    Tier as PlanTier,
)

__all__ = [
    # plans
    "ExecutionPlan",
    "ExecutionTier",
    "RiskLevel",
    "OptimizerDecision",
    "RepresentationCandidate",
    "CachePlanEntry",
    "TransportPlanEntry",
    "FallbackPlan",
    "tier_budget_ms",
    "TIER_BUDGETS_MS",
    "TransportPlan",
    "build_transport_plan",
    # build
    "build_execution_plan",
    # planner
    "UnifiedPlanner",
    "SemanticProfile",
    "PlanTier",
    "profile_from_legacy",
    # classify
    "TaskClass",
    "TaskClassification",
    "classify_task",
    "TaskExecutionTier",
    "RequestClassifier",
    # runtime state bridges
    "coerce_execution_plan",
    "get_ir_metadata",
    "get_canonical_state_value",
    "get_canonical_request_value",
    "persist_execution_plan_state",
    "persist_session_plan_state",
    # provider strategy
    "ProviderStrategy",
    "CacheSimulation",
    "get_provider_strategy",
    "build_cache_plan_for_provider",
    "simulate_provider_cache",
    "preferred_optimizers_for_provider",
    # execution
    "execute_with_fallback",
    "execute_with_fallback_stream",
]
