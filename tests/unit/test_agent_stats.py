"""Tests for agent stats collector.


Covers:
- Agent identification from User-Agent and client profile
- Request recording with all metric fields
- Global summary aggregation
- Weekly report generation
- Cache hit rate calculation
- Compression savings calculation
- Cost aggregation
"""

from __future__ import annotations

import asyncio

import pytest

from lattice.core.agent_stats import AgentStatsCollector, identify_agent


class TestIdentifyAgent:
    def test_claude_from_user_agent(self) -> None:
        assert identify_agent("Claude-Code/1.0", None) == "claude"

    def test_claude_from_profile(self) -> None:
        assert identify_agent(None, "claude") == "claude"

    def test_codex_from_user_agent(self) -> None:
        assert identify_agent("OpenAI-Codex/1.0", None) == "codex"

    def test_cursor_from_user_agent(self) -> None:
        assert identify_agent("cursor.sh/0.1", None) == "cursor"

    def test_copilot_from_user_agent(self) -> None:
        assert identify_agent("GitHub-Copilot/1.0", None) == "copilot"

    def test_opencode_from_profile(self) -> None:
        assert identify_agent(None, "opencode") == "opencode"

    def test_other_when_no_match(self) -> None:
        assert identify_agent("Unknown-Agent/1.0", None) == "other"

    def test_profile_takes_precedence(self) -> None:
        # Even with a generic UA, profile identifies
        assert identify_agent("Mozilla/5.0", "cursor") == "cursor"


class TestAgentStatsCollector:
    @pytest.fixture
    async def collector(self):
        from lattice.core.metrics import MetricsCollector

        c = AgentStatsCollector(
            metrics=MetricsCollector(),
            cost_estimator=None,
        )
        return c

    async def test_record_single_request(self, collector) -> None:
        await collector.record_request(
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
            speculative_hit=False,
            batched=False,
            auto_continuation_turns=0,
        )
        stats = collector.get_agent_stats("claude")
        assert stats is not None
        assert stats.requests_total == 1
        assert stats.prompt_tokens_total == 1000
        assert stats.completion_tokens_total == 500
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
        assert stats.cached_tokens_total == 200
        assert stats.estimated_cost_usd == pytest.approx(0.015)
        assert stats.compression_savings_percent == pytest.approx(1.0 - 800 / 1200)

    async def test_record_multiple_agents(self, collector) -> None:
        for _ in range(3):
            await collector.record_request(
                agent="claude",
                provider="anthropic",
                model="claude-3-5-sonnet",
                prompt_tokens=1000,
                completion_tokens=500,
                compressed_tokens=800,
                original_tokens=1200,
                cache_hit=True,
                cost_usd=0.015,
            )
        for _ in range(2):
            await collector.record_request(
                agent="cursor",
                provider="openai",
                model="gpt-4o",
                prompt_tokens=2000,
                completion_tokens=1000,
                compressed_tokens=1500,
                original_tokens=2200,
                cache_hit=False,
                cost_usd=0.030,
            )

        claude = collector.get_agent_stats("claude")
        cursor = collector.get_agent_stats("cursor")
        assert claude.requests_total == 3
        assert cursor.requests_total == 2
        assert claude.estimated_cost_usd == pytest.approx(0.045)
        assert cursor.cache_misses == 2

    async def test_global_summary(self, collector) -> None:
        await collector.record_request(
            agent="claude",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=1000,
            completion_tokens=500,
            compressed_tokens=800,
            original_tokens=1200,
            cached_tokens=300,
            cache_hit=True,
            cost_usd=0.015,
        )
        await collector.record_request(
            agent="cursor",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=2000,
            completion_tokens=1000,
            compressed_tokens=1500,
            original_tokens=2200,
            cached_tokens=100,
            cache_hit=False,
            cost_usd=0.030,
        )

        summary = collector.global_summary()
        assert summary["total_requests"] == 2
        assert summary["total_prompt_tokens"] == 3000
        assert summary["total_completion_tokens"] == 1500
        assert summary["total_cached_tokens"] == 400
        assert summary["total_estimated_cost_usd"] == pytest.approx(0.045)
        assert summary["agents_tracked"] == 2
        assert "per_agent" in summary
        assert "claude" in summary["per_agent"]
        assert "cursor" in summary["per_agent"]

    async def test_cache_hit_rate(self, collector) -> None:
        for _ in range(7):
            await collector.record_request(
                agent="claude",
                provider="anthropic",
                model="claude-3-5-sonnet",
                prompt_tokens=100,
                completion_tokens=50,
                compressed_tokens=80,
                original_tokens=120,
                cache_hit=True,
                cost_usd=0.001,
            )
        for _ in range(3):
            await collector.record_request(
                agent="claude",
                provider="anthropic",
                model="claude-3-5-sonnet",
                prompt_tokens=100,
                completion_tokens=50,
                compressed_tokens=80,
                original_tokens=120,
                cache_hit=False,
                cost_usd=0.001,
            )

        stats = collector.get_agent_stats("claude")
        assert stats.cache_hits == 7
        assert stats.cache_misses == 3
        assert stats.cache_hit_rate == pytest.approx(0.7)

    async def test_avg_cost_per_request(self, collector) -> None:
        for _ in range(2):
            await collector.record_request(
                agent="claude",
                provider="anthropic",
                model="claude-3-5-sonnet",
                prompt_tokens=100,
                completion_tokens=50,
                compressed_tokens=80,
                original_tokens=120,
                cache_hit=False,
                cost_usd=0.010,
            )

        stats = collector.get_agent_stats("claude")
        assert stats.avg_cost_per_request_usd == pytest.approx(0.010)

    async def test_weekly_report(self, collector) -> None:
        await collector.record_request(
            agent="claude",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=1000,
            completion_tokens=500,
            compressed_tokens=800,
            original_tokens=1200,
            cache_hit=True,
            cost_usd=0.015,
        )
        await collector.record_request(
            agent="cursor",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=2000,
            completion_tokens=1000,
            compressed_tokens=1500,
            original_tokens=2200,
            cache_hit=False,
            cost_usd=0.030,
        )

        report = await collector.weekly_report()
        assert report["report_type"] == "weekly"
        assert report["total_requests"] == 2
        assert "top_agents_by_savings" in report
        assert "top_agents_by_cost" in report

    async def test_first_last_seen(self, collector) -> None:
        await collector.record_request(
            agent="claude",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
            cache_hit=False,
            cost_usd=0.001,
        )
        first_seen = collector.get_agent_stats("claude").first_seen_at
        await asyncio.sleep(0.01)
        await collector.record_request(
            agent="claude",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
            cache_hit=False,
            cost_usd=0.001,
        )
        last_seen = collector.get_agent_stats("claude").last_seen_at
        assert last_seen >= first_seen

    async def test_to_dict(self, collector) -> None:
        await collector.record_request(
            agent="claude",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=1000,
            completion_tokens=500,
            compressed_tokens=800,
            original_tokens=1200,
            cache_hit=True,
            cost_usd=0.015,
        )
        d = collector.get_agent_stats("claude").to_dict()
        assert d["agent"] == "claude"
        assert d["requests_total"] == 1
        assert "compression_savings_percent" in d
        assert "cache_hit_rate" in d

    async def test_prometheus_export(self, collector) -> None:
        await collector.record_request(
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
            batched=True,
            speculative_hit=True,
        )
        prom = collector._metrics.prometheus_output()
        assert "lattice_agent_requests_total" in prom
        assert 'agent="claude"' in prom
        assert "lattice_agent_cached_tokens" in prom
        assert "lattice_agent_cache_hits_total" in prom
        assert "lattice_agent_speculative_hits_total" in prom
        assert "lattice_agent_batch_dispatches_total" in prom
