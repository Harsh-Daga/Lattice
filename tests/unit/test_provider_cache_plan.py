"""Tests for CachePlanEntry serialization and adapter ingestion."""

from __future__ import annotations

from lattice.planner.execution_plan import CachePlanEntry
from lattice.planner.provider_strategy import build_cache_plan_for_provider
from lattice.providers.openai import OpenAIAdapter
from lattice.transport.types import Request


class TestProviderCachePlanIngestion:
    """CachePlan written into request metadata must be picked up by adapters."""

    def test_openai_reads_lattice_cache_plan(self) -> None:
        """OpenAIAdapter.serialize_request reads _lattice_cache_plan metadata."""
        adapter = OpenAIAdapter()
        req = Request(messages=[], model="openai/gpt-4")
        req.metadata["_lattice_cache_plan"] = [
            {
                "segment_index": 0,
                "provider_mode": "auto_prefix",
                "expected_cached_tokens": 512,
                "annotations": {"stable": True},
            }
        ]
        body = adapter.serialize_request(req)
        assert body["prompt_cache_key"] == "auto_prefix"

    def test_openai_prefers_explicit_prompt_cache_key(self) -> None:
        """Explicit prompt_cache_key in metadata wins over _lattice_cache_plan."""
        adapter = OpenAIAdapter()
        req = Request(messages=[], model="openai/gpt-4")
        req.metadata["prompt_cache_key"] = "my_custom_key"
        req.metadata["_lattice_cache_plan"] = [
            {"segment_index": 0, "provider_mode": "auto_prefix", "expected_cached_tokens": 100}
        ]
        body = adapter.serialize_request(req)
        assert body["prompt_cache_key"] == "my_custom_key"

    def test_openai_no_cache_plan_no_key(self) -> None:
        """Without cache plan or explicit key, no prompt_cache_key added."""
        adapter = OpenAIAdapter()
        req = Request(messages=[], model="openai/gpt-4")
        body = adapter.serialize_request(req)
        assert "prompt_cache_key" not in body

    def test_build_cache_plan_for_provider_openai(self) -> None:
        """Provider-specific cache plan generation for OpenAI."""
        plan = build_cache_plan_for_provider("openai", segment_count=3, estimated_tokens=1000)
        assert len(plan) == 3  # first 3 segments marked as stable prefix
        assert plan[0]["provider_mode"] == "auto_prefix"
        assert "expected_cached_tokens" in plan[0]

    def test_build_cache_plan_for_provider_anthropic(self) -> None:
        """Provider-specific cache plan for Anthropic — explicit breakpoint on first segment."""
        plan = build_cache_plan_for_provider("anthropic", segment_count=5, estimated_tokens=2000)
        assert len(plan) == 1
        assert plan[0]["provider_mode"] == "explicit_breakpoint"
        assert "expected_cached_tokens" in plan[0]
        assert plan[0]["annotations"].get("cache_control")
        assert plan[0]["annotations"]["cache_control"].get("type") == "ephemeral"

    def test_cache_plan_entry_to_dict(self) -> None:
        entry = CachePlanEntry(
            segment_index=0,
            provider_mode="auto_prefix",
            expected_cached_tokens=256,
            annotations={"stable": True},
        )
        d = entry.to_dict()
        assert d["segment_index"] == 0
        assert d["provider_mode"] == "auto_prefix"
        assert d["expected_cached_tokens"] == 256
        assert d["annotations"]["stable"] is True
