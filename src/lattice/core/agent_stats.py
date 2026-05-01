"""Agent stats — per-agent token usage, savings, and cost aggregation.

Tracks per-agent (Claude, Codex, Cursor, Copilot, OpenCode) metrics:
- Total requests, tokens processed, compression savings
- Estimated cost in USD
- Cache hit rate
- Provider distribution

Exposes:
- Prometheus metrics via existing MetricsCollector
- /stats endpoint via proxy server
- Weekly savings report via background task

Design decisions
----------------
1. Agent identification: via User-Agent header pattern matching or
   x-lattice-client-profile header.
2. Aggregation: in-memory with periodic flush to MetricsCollector.
3. Per-agent breakdown: available via /stats and Prometheus labels.
4. Cost: uses CostEstimator for per-request cost tracking.
5. Background reports: CronCreate for weekly summaries.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections import defaultdict
from typing import Any

import structlog

from lattice.core.cost_estimator import CostEstimator

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Agent identification
# ---------------------------------------------------------------------------

_AGENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "anthropic", "claude-code"),
    "codex": ("codex", "openai-codex"),
    "cursor": ("cursor", "cursor.sh", "cursor-ai"),
    "copilot": ("github-copilot", "copilot", "vscode-copilot"),
    "opencode": ("opencode", "open-code"),
    "other": (),
}


def identify_agent(user_agent: str | None, client_profile: str | None) -> str:
    """Identify the agent from User-Agent or client profile header.

    Returns the canonical agent name or "other".
    """
    text = (client_profile or "").lower() + " " + (user_agent or "").lower()
    for agent, patterns in _AGENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                return agent
    return "other"


# ---------------------------------------------------------------------------
# Per-agent stats aggregate
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class AgentMetrics:
    """Aggregated metrics for a single agent."""

    agent: str
    requests_total: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    cached_tokens_total: int = 0
    compressed_tokens_total: int = 0
    original_tokens_total: int = 0
    estimated_cost_usd: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    speculative_hits: int = 0
    speculative_misses: int = 0
    batch_dispatches: int = 0
    auto_continuation_turns: int = 0
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0

    @property
    def compression_savings_percent(self) -> float:
        if self.original_tokens_total > 0:
            return 1.0 - self.compressed_tokens_total / self.original_tokens_total
        return 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total > 0:
            return self.cache_hits / total
        return 0.0

    @property
    def avg_cost_per_request_usd(self) -> float:
        if self.requests_total > 0:
            return self.estimated_cost_usd / self.requests_total
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "requests_total": self.requests_total,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "cached_tokens_total": self.cached_tokens_total,
            "compression_savings_percent": round(self.compression_savings_percent, 4),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "avg_cost_per_request_usd": round(self.avg_cost_per_request_usd, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "speculative_hits": self.speculative_hits,
            "speculative_misses": self.speculative_misses,
            "batch_dispatches": self.batch_dispatches,
            "auto_continuation_turns": self.auto_continuation_turns,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
        }


class AgentStatsCollector:
    """Collects per-agent metrics across the proxy.

    Usage in gateway handler::

        collector = AgentStatsCollector(metrics=metrics, cost_estimator=cost_estimator)
        collector.record_request(
            agent="claude",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=1000,
            completion_tokens=500,
            compressed_tokens=800,
            original_tokens=1200,
            cached_tokens=200,
            cache_hit=True,
            cost_usd=0.015,
        )
    """

    def __init__(
        self,
        *,
        metrics: Any,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        self._metrics = metrics
        self._cost_estimator = cost_estimator or CostEstimator()
        self._agents: dict[str, AgentMetrics] = defaultdict(lambda: AgentMetrics(agent="unknown"))
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record_request(
        self,
        *,
        agent: str,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        compressed_tokens: int = 0,
        original_tokens: int = 0,
        cached_tokens: int = 0,
        cache_hit: bool = False,
        cost_usd: float = 0.0,
        speculative_hit: bool = False,
        batched: bool = False,
        auto_continuation_turns: int = 0,
    ) -> None:
        """Record a completed request for an agent."""
        async with self._lock:
            stats = self._agents[agent]
            stats.agent = agent
            stats.requests_total += 1
            stats.prompt_tokens_total += prompt_tokens
            stats.completion_tokens_total += completion_tokens
            stats.compressed_tokens_total += compressed_tokens
            stats.original_tokens_total += original_tokens
            stats.cached_tokens_total += cached_tokens
            stats.estimated_cost_usd += cost_usd
            if cache_hit:
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1
            if speculative_hit:
                stats.speculative_hits += 1
            else:
                stats.speculative_misses += 1
            if batched:
                stats.batch_dispatches += 1
            stats.auto_continuation_turns += auto_continuation_turns
            now = time.time()
            if stats.first_seen_at == 0:
                stats.first_seen_at = now
            stats.last_seen_at = now

        # Also export to Prometheus
        labels = {"agent": agent, "provider": provider, "model": model}
        self._metrics.increment("lattice_agent_requests_total", 1, labels)
        self._metrics.gauge("lattice_agent_prompt_tokens", float(prompt_tokens), labels)
        self._metrics.gauge("lattice_agent_completion_tokens", float(completion_tokens), labels)
        self._metrics.gauge("lattice_agent_cached_tokens", float(cached_tokens), labels)
        self._metrics.gauge("lattice_agent_cost_usd", cost_usd, labels)
        if cache_hit:
            self._metrics.increment("lattice_agent_cache_hits_total", 1, labels)
        if speculative_hit:
            self._metrics.increment("lattice_agent_speculative_hits_total", 1, labels)
        if batched:
            self._metrics.increment("lattice_agent_batch_dispatches_total", 1, labels)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_agent_stats(self, agent: str) -> AgentMetrics | None:
        """Get stats for a specific agent."""
        return self._agents.get(agent)

    def all_agents(self) -> list[str]:
        """List all tracked agent names."""
        return list(self._agents.keys())

    def global_summary(self) -> dict[str, Any]:
        """Aggregate across all agents."""
        total_requests = sum(a.requests_total for a in self._agents.values())
        total_prompt = sum(a.prompt_tokens_total for a in self._agents.values())
        total_completion = sum(a.completion_tokens_total for a in self._agents.values())
        total_cost = sum(a.estimated_cost_usd for a in self._agents.values())
        total_cache_hits = sum(a.cache_hits for a in self._agents.values())
        total_cache_misses = sum(a.cache_misses for a in self._agents.values())
        total_cached_tokens = sum(a.cached_tokens_total for a in self._agents.values())
        total_original = sum(a.original_tokens_total for a in self._agents.values())
        total_compressed = sum(a.compressed_tokens_total for a in self._agents.values())
        savings_pct = 1.0 - total_compressed / total_original if total_original > 0 else 0.0
        cache_rate = (
            total_cache_hits / (total_cache_hits + total_cache_misses)
            if (total_cache_hits + total_cache_misses) > 0
            else 0.0
        )
        return {
            "agents_tracked": len(self._agents),
            "total_requests": total_requests,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": total_cached_tokens,
            "compression_savings_percent": round(savings_pct, 4),
            "total_estimated_cost_usd": round(total_cost, 6),
            "avg_cost_per_request_usd": round(total_cost / total_requests, 6)
            if total_requests > 0
            else 0.0,
            "cache_hit_rate": round(cache_rate, 4),
            "per_agent": {name: stats.to_dict() for name, stats in self._agents.items()},
        }

    async def weekly_report(self) -> dict[str, Any]:
        """Generate a weekly savings report."""
        summary = self.global_summary()
        summary["report_type"] = "weekly"
        summary["generated_at"] = time.time()
        # Add ranking
        agents_by_savings = sorted(
            self._agents.items(),
            key=lambda x: x[1].compression_savings_percent,
            reverse=True,
        )
        summary["top_agents_by_savings"] = [
            {"agent": a, "savings_percent": round(s.compression_savings_percent, 4)}
            for a, s in agents_by_savings[:5]
        ]
        agents_by_cost = sorted(
            self._agents.items(),
            key=lambda x: x[1].estimated_cost_usd,
            reverse=True,
        )
        summary["top_agents_by_cost"] = [
            {"agent": a, "cost_usd": round(s.estimated_cost_usd, 6)} for a, s in agents_by_cost[:5]
        ]
        return summary


__all__ = [
    "AgentStatsCollector",
    "AgentMetrics",
    "identify_agent",
    "_AGENT_PATTERNS",
]
