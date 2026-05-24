"""LATTICE telemetry — metrics, downgrade taxonomy, cost estimation, agent stats, maintenance, sketches.

This is the single source of truth for everything observable about a running LATTICE proxy.

Use:
    from lattice.telemetry import (
        MetricsCollector, LatencyTracker,
        DowngradeTelemetry, DowngradeCategory, TransportOutcome,
        AgentStatsCollector, AgentMetrics,
        CostEstimator, CostEstimate, ModelPricing,
        MaintenanceCoordinator, MaintenanceResult,
        CountMinSketch, HyperLogLog,
    )
"""

from lattice.telemetry.agent_stats import (
    AgentMetrics,
    AgentStatsCollector,
    identify_agent,
)
from lattice.telemetry.cost_estimator import (
    CostEstimate,
    CostEstimator,
    ModelPricing,
    extract_cached_tokens,
    format_cost_usd,
    normalize_usage,
)
from lattice.telemetry.downgrade import (
    DowngradeCategory,
    DowngradeTelemetry,
    TransportOutcome,
)
from lattice.telemetry.maintenance import (
    MaintenanceCoordinator,
    MaintenanceResult,
)
from lattice.telemetry.metrics import (
    LatencyTracker,
    MetricsCollector,
)
from lattice.telemetry.streaming_sketches import (
    CountMinSketch,
    HyperLogLog,
)

__all__ = [
    "MetricsCollector",
    "LatencyTracker",
    "DowngradeCategory",
    "DowngradeTelemetry",
    "TransportOutcome",
    "AgentMetrics",
    "AgentStatsCollector",
    "identify_agent",
    "ModelPricing",
    "CostEstimate",
    "CostEstimator",
    "normalize_usage",
    "extract_cached_tokens",
    "format_cost_usd",
    "MaintenanceCoordinator",
    "MaintenanceResult",
    "CountMinSketch",
    "HyperLogLog",
]
