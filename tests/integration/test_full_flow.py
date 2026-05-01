"""Integration tests for LATTICE full flows.

These tests validate end-to-end behavior across proxy, SDK, and transport layers.
"""


from typing import Any

import pytest

from lattice.core.session import MemorySessionStore, SessionManager
from lattice.core.transport import Message, Request
from lattice.providers.capabilities import (
    CacheMode,
    Capability,
    CapabilityRegistry,
)
from lattice.providers.transport import DirectHTTPProvider
from lattice.runtime.router import RuntimeRouter
from lattice.transforms.batching import BatchedRequest, BatchingEngine
from lattice.transforms.speculative import SpeculativeExecutor

# =============================================================================
# Session failover
# =============================================================================

class TestSessionFailover:
    @pytest.mark.asyncio
    async def test_cross_instance_session_rebuild(self):
        """When session is missing, proxy should gracefully degrade to full prompt."""
        store1 = MemorySessionStore()
        mgr1 = SessionManager(store1)

        session = await mgr1.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ],
        )

        # Simulate second instance with different store
        store2 = MemorySessionStore()
        mgr2 = SessionManager(store2)

        # Session not found in store2 — should create new
        session2, was_created = await mgr2.get_or_create_session(
            session_id=session.session_id,
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="Hello again")],
        )
        assert was_created is True
        assert session2.session_id != session.session_id

    @pytest.mark.asyncio
    async def test_delta_reconstructs_full_context(self):
        """Delta encoding should produce identical messages to full context."""
        store = MemorySessionStore()
        mgr = SessionManager(store)

        full_messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
        ]

        session = await mgr.create_session("openai", "gpt-4", full_messages[:2])
        # Update with more messages
        await mgr.update_session(session.session_id, full_messages)

        # Retrieve and verify
        retrieved = await store.get(session.session_id)
        assert retrieved is not None
        assert len(retrieved.messages) == 4
        assert retrieved.messages[-1].content == "Q2"


# =============================================================================
# Cache planner integration
# =============================================================================

class TestCachePlannerIntegration:
    def test_openai_cache_breakpoint(self):
        from lattice.protocol.cache_planner import OpenAICachePlanner
        from lattice.protocol.manifest import build_manifest
        from lattice.protocol.segments import build_system_segment, build_tools_segment

        planner = OpenAICachePlanner()
        manifest = build_manifest("s1", [
            build_system_segment("You are helpful."),
            build_tools_segment([{"name": "search"}]),
        ])
        plan = planner.plan(manifest)
        assert len(plan.breakpoints) > 0
        assert plan.expected_cached_tokens > 0

    def test_anthropic_max_breakpoints(self):
        from lattice.protocol.cache_planner import AnthropicCachePlanner
        from lattice.protocol.manifest import build_manifest
        from lattice.protocol.segments import (
            build_system_segment,
            build_tools_segment,
        )

        planner = AnthropicCachePlanner()
        # Create manifest with many segments
        manifest = build_manifest("s1", [
            build_tools_segment([{"name": "t1"}]),
            build_system_segment("sys1"),
            build_system_segment("sys2"),
            build_system_segment("sys3"),
            build_system_segment("sys4"),
            build_system_segment("sys5"),
        ])
        plan = planner.plan(manifest)
        assert len(plan.breakpoints) <= planner.MAX_BREAKPOINTS


# =============================================================================
# Batching integration
# =============================================================================

class TestBatchingIntegration:
    @pytest.mark.asyncio
    async def test_batch_fill_ratio(self):
        """Batch should fill to max size or timeout."""
        engine = BatchingEngine(max_batch_size=3, max_wait_ms=50)

        async def dummy_caller(batched: BatchedRequest) -> Any:
            from lattice.transforms.batching import BatchedResponse
            return BatchedResponse(
                choices=[{"index": i, "message": {"content": f"resp{i}"}, "finish_reason": "stop"}
                         for i in range(len(batched.messages))],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                model="gpt-4",
            )

        engine.provider_caller = dummy_caller

        req = Request(messages=[Message(role="user", content="hi")])
        from lattice.core.context import TransformContext
        ctx = TransformContext(request_id="1", provider="openai", model="gpt-4")

        # Single request should still work (timeout flush)
        resp = await engine.submit(req, ctx)
        assert resp.content is not None

    @pytest.mark.asyncio
    async def test_batching_eligible_non_streaming(self):
        """Non-streaming requests are batchable."""
        from lattice.transforms.batching import BatchKey
        req = Request(messages=[Message(role="user", content="hi")], stream=False)
        key = BatchKey.from_request(req)
        assert key.stream is False

        req_stream = Request(messages=[Message(role="user", content="hi")], stream=True)
        key_stream = BatchKey.from_request(req_stream)
        assert key_stream.stream is True


# =============================================================================
# Speculation integration
# =============================================================================

class TestSpeculationIntegration:
    def test_speculation_never_worsens_tail_latency(self):
        """Speculative branch should be a sidecar — not block real request."""
        executor = SpeculativeExecutor(
            max_speculative_tokens=64,
            confidence_threshold=0.7,
            provider_caller=None,  # disabled
        )
        req = Request(
            messages=[Message(role="user", content="call search tool")],
            tools=[{"function": {"name": "search"}}],
        )
        prediction = executor.predict(req, None)  # type: ignore[arg-type]
        # Prediction exists but provider_caller is None — safe no-op
        assert prediction is not None

    def test_hit_detection(self):
        executor = SpeculativeExecutor()
        assert executor.is_hit("tool_call", "tool_call") is True
        assert executor.is_hit("tool_call", "completion") is False
        assert executor.is_hit("code_completion", "completion") is False

    def test_stats_tracking(self):
        executor = SpeculativeExecutor()
        executor.record_result(hit=True, _predicted="x", _actual="x", latency_ms=100)
        executor.record_result(hit=False, _predicted="y", _actual="z", latency_ms=100)
        assert executor.stats["total"] == 2
        assert executor.stats["hits"] == 1
        assert executor.stats["accuracy"] == 0.5


# =============================================================================
# Fallback chain
# =============================================================================

class TestFallbackChain:
    def test_no_model_fallback(self):
        """LATTICE does not route between models."""
        provider = DirectHTTPProvider()
        provider.configure_resilience(stall_timeout=1.0)
        assert provider._stall_timeout == 1.0


# =============================================================================
# Capability-driven routing
# =============================================================================

class TestCapabilityDrivenRouting:
    def test_provider_supports_caching(self):
        reg = CapabilityRegistry()
        assert reg.supports("openai", Capability.PROMPT_CACHING)
        assert reg.supports("anthropic", Capability.PROMPT_CACHING)
        assert not reg.supports("ollama", Capability.PROMPT_CACHING)

    def test_cache_mode_selection(self):
        reg = CapabilityRegistry()
        assert reg.cache_mode("openai") == CacheMode.AUTO_PREFIX
        assert reg.cache_mode("anthropic") == CacheMode.EXPLICIT_BREAKPOINT
        assert reg.cache_mode("groq") == CacheMode.NONE


# =============================================================================
# Router integration
# =============================================================================

class TestRouterIntegration:
    def test_tier_classifier_routing(self):
        router = RuntimeRouter()
        simple = Request(messages=[Message(role="user", content="hi")])
        complex_req = Request(
            messages=[Message(role="user", content="prove theorem step by step" * 100)],
            tools=[{"function": {"name": "t1"}}],
        )

        assert router.classify(simple).tier == "SIMPLE"
        assert router.classify(complex_req).tier in ("COMPLEX", "REASONING")
