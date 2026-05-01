"""End-to-end pipeline test for LATTICE Phase 0 transforms.

This test verifies:
1. Pipeline runs all transforms in priority order
2. Reference substitution is reversible
3. Tool output filtering reduces tokens
4. Metrics are collected correctly
5. Performance is within budget (<1ms total)
"""

import time
from typing import Any

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.transforms.prefix_opt import PrefixOptimizer
from lattice.transforms.rate_distortion import RateDistortionCompressor
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.tool_filter import ToolOutputFilter

# =============================================================================
# Pipeline fixture
# =============================================================================

@pytest.fixture
def pipeline() -> CompressorPipeline:
    """Build a pipeline with all Phase 0 transforms."""
    config = LatticeConfig()
    p = CompressorPipeline(config=config)
    p.register(PrefixOptimizer())
    p.register(ReferenceSubstitution())
    p.register(ToolOutputFilter())
    p.register(OutputCleanup())
    return p


# =============================================================================
# Reference Substitution
# =============================================================================

class TestReferenceSubstitution:
    """Comprehensive tests for the reversible reference substitution."""

    @pytest.mark.asyncio
    async def test_uuid_replacement(self, pipeline: CompressorPipeline) -> None:
        """UUIDs are replaced with short aliases."""
        request = Request(
            messages=[
                Message(
                    role="user",
                    content="Transaction 550e8400-e29b-41d4-a716-446655440000 failed",
                )
            ]
        )
        context = TransformContext()

        result = await pipeline.process(request, context)
        modified = unwrap(result)

        assert "ref_1" in modified.messages[0].content
        assert "550e8400" not in modified.messages[0].content
        assert context.transforms_applied == ["prefix_optimizer", "reference_sub", "tool_filter", "output_cleanup"]

    @pytest.mark.asyncio
    async def test_reversible_round_trip(self, pipeline: CompressorPipeline) -> None:
        """Reference substitution round-trips correctly."""
        original = Request(
            messages=[
                Message(
                    role="user",
                    content="UUID 550e8400-e29b-41d4-a716-446655440000 then 6ba7b810-9dad-11d1-80b4-00c04fd430c8 done",
                )
            ]
        )
        context = TransformContext()

        # Forward
        result = await pipeline.process(original, context)
        compressed = unwrap(result)
        compressed_text = compressed.messages[0].content

        # Reverse
        response = Response(content=compressed_text)
        reversed_response = await pipeline.reverse(response, context)

        assert reversed_response.content == original.messages[0].content

    @pytest.mark.asyncio
    async def test_duplicate_uuid_same_alias(self, pipeline: CompressorPipeline) -> None:
        """Same UUID gets same alias across messages."""
        request = Request(
            messages=[
                Message(role="user", content="First: 550e8400-e29b-41d4-a716-446655440000"),
                Message(role="user", content="Second: 550e8400-e29b-41d4-a716-446655440000"),
            ]
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        # Both messages should have the same alias
        assert "ref_1" in modified.messages[0].content
        assert "ref_1" in modified.messages[1].content
        assert "ref_2" not in modified.messages[0].content

    @pytest.mark.asyncio
    async def test_no_uuid_no_change(self, pipeline: CompressorPipeline) -> None:
        """Text without UUIDs is unchanged."""
        request = Request(
            messages=[Message(role="user", content="Hello, world.")]
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        assert modified.messages[0].content == "Hello, world."

    @pytest.mark.asyncio
    async def test_token_reduction(self, pipeline: CompressorPipeline) -> None:
        """Reference substitution saves tokens."""
        request = Request(
            messages=[
                Message(
                    role="user",
                    content=" " .join(["550e8400-e29b-41d4-a716-446655440000"] * 10),
                )
            ]
        )
        context = TransformContext()
        original_tokens = request.token_estimate

        result = await pipeline.process(request, context)
        modified = unwrap(result)
        modified_tokens = modified.token_estimate

        assert modified_tokens < original_tokens
        # Should be significant reduction (36 chars -> ~8 chars per UUID)
        assert modified_tokens <= original_tokens * 0.4


# =============================================================================
# Tool Output Filter
# =============================================================================

class TestToolOutputFilter:
    """Comprehensive tests for tool output filtering."""

    @pytest.mark.asyncio
    async def test_json_filtering(self, pipeline: CompressorPipeline) -> None:
        """JSON arrays have debug/metadata fields removed."""
        request = Request(
            messages=[
                Message(
                    role="tool",
                    content='''[
                        {
                            "id": "abc",
                            "name": "test",
                            "created_at": "2024-01-01T00:00:00Z",
                            "metadata": {"source": "db"},
                            "logs": ["processing"]
                        }
                    ]''',
                )
            ]
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)
        content = modified.messages[0].content

        assert '"id":' in content
        assert '"name":' in content
        assert '"created_at"' not in content
        assert '"metadata"' not in content
        assert '"logs"' not in content

    @pytest.mark.asyncio
    async def test_non_json_unchanged(self, pipeline: CompressorPipeline) -> None:
        """Non-JSON text is not modified."""
        request = Request(
            messages=[Message(role="user", content="The logs show processing.")]
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        assert modified.messages[0].content == "The logs show processing."

    @pytest.mark.asyncio
    async def test_token_reduction_tool_output(self, pipeline: CompressorPipeline) -> None:
        """Tool output filtering saves tokens."""
        items = [{"id": i, "name": f"item_{i}", "created_at": "2024-01-01", "metadata": {"x": i}} for i in range(100)]
        import json
        json_text = json.dumps(items, indent=2)
        request = Request(messages=[Message(role="tool", content=json_text)])
        context = TransformContext()
        original_tokens = request.token_estimate

        result = await pipeline.process(request, context)
        modified = unwrap(result)
        modified_tokens = modified.token_estimate

        assert modified_tokens < original_tokens
        # Should be ~50% reduction (removing created_at, metadata, indent)


# =============================================================================
# Prefix Optimizer
# =============================================================================

class TestPrefixOptimizer:
    """Comprehensive tests for prefix optimization."""

    @pytest.mark.asyncio
    async def test_system_prompt_in_prefix(self, pipeline: CompressorPipeline) -> None:
        """System message is included in prefix hash calculation."""
        request = Request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        assert "_prefix_hash" in modified.metadata
        assert "_prefix_tokens" in modified.metadata
        # Prefix = system prompt
        assert modified.metadata["_prefix_tokens"] > 0

    @pytest.mark.asyncio
    async def test_prefix_cache_miss_on_change(self, pipeline: CompressorPipeline) -> None:
        """Changing system prompt produces different hash."""
        req1 = Request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        ctx1 = TransformContext(session_id="sess_1")
        r1 = await pipeline.process(req1, ctx1)
        m1 = unwrap(r1)
        hash1 = m1.metadata["_prefix_hash"]

        req2 = Request(
            messages=[
                Message(role="system", content="You are a sarcastic assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        ctx2 = TransformContext(session_id="sess_1")
        r2 = await pipeline.process(req2, ctx2)
        m2 = unwrap(r2)
        hash2 = m2.metadata["_prefix_hash"]

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_prefix_cache_hit_same_session(self, pipeline: CompressorPipeline) -> None:
        """Same system prompt with pre-shared session state produces cache hit."""
        # First, compute the actual hash for this prefix
        req1 = Request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        from lattice.transforms.prefix_opt import PrefixOptimizer
        PrefixOptimizer()
        ctx1 = TransformContext(session_id="sess_2")
        r1 = await pipeline.process(req1, ctx1)
        m1 = unwrap(r1)
        actual_hash = m1.metadata["_prefix_hash"]

        # Now simulate a second request with shared session state
        req2 = Request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Goodbye"),
            ]
        )
        shared_session_state: dict[str, Any] = {
            "prefix_optimizer": {"prefix_hash": actual_hash}
        }
        ctx2 = TransformContext(
            session_id="sess_2",
            session_state=shared_session_state.copy(),
        )
        r2 = await pipeline.process(req2, ctx2)
        m2 = unwrap(r2)

        assert m2.metadata.get("_cache_hit") is True
        assert m2.metadata.get("_prefix_tokens") > 0


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformance:
    """Regression tests for performance budgets."""

    @pytest.mark.asyncio
    async def test_transform_latency_under_budget(self, pipeline: CompressorPipeline) -> None:
        """All transforms complete within 5ms budget."""
        request = Request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content=f"UUID: {'550e8400-e29b-41d4-a716-446655440000 ' * 100}"),
            ]
        )
        context = TransformContext()

        result = await pipeline.process(request, context)
        unwrap(result)

        # Check per-transform latency
        per_transform = context.metrics.get("transforms", {})
        for name, metrics in per_transform.items():
            latency_ms = metrics.get("latency_ms", 0)
            assert latency_ms < 5.0, f"{name} took {latency_ms:.3f}ms > 5ms budget"

        # Total should be under 10ms
        total_latency = context.elapsed_ms
        assert total_latency < 10.0, f"Total pipeline {total_latency:.3f}ms > 10ms budget"

    @pytest.mark.asyncio
    async def test_compression_ratio(self, pipeline: CompressorPipeline) -> None:
        """Pipeline achieves >=25% token reduction on synthetic workload."""
        import json

        # Synthetic: 100 tool outputs with debug fields and 50 UUIDs
        tool_outputs = []
        for i in range(50):
            tool_outputs.append({
                "id": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                "name": f"item_{i}",
                "created_at": "2024-01-01T00:00:00Z",
                "metadata": {"source": "db", "version": "1.0"},
                "logs": ["processing", "completed"],
            })

        request = Request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Here are the results:"),
                Message(role="tool", content=json.dumps(tool_outputs, indent=2)),
            ]
        )
        context = TransformContext()
        original_tokens = request.token_estimate

        result = await pipeline.process(request, context)
        modified = unwrap(result)
        modified_tokens = modified.token_estimate

        reduction = (original_tokens - modified_tokens) / original_tokens
        assert reduction >= 0.25, f"Reduction only {reduction:.1%}, expected >=25%"


# =============================================================================
# Pipeline Orchestration
# =============================================================================

class TestPipeline:
    """Tests for pipeline orchestration itself."""

    @pytest.mark.asyncio
    async def test_transform_priority_ordering(self) -> None:
        """Transforms apply in priority order (ascending)."""
        config = LatticeConfig()
        p = CompressorPipeline(config=config)
        p.register(PrefixOptimizer())   # priority 10
        p.register(ReferenceSubstitution())  # priority 20
        p.register(ToolOutputFilter())  # priority 30

        order = [t.priority for t in p.transforms]
        assert order == sorted(order)
        assert p.transforms[0].name == "prefix_optimizer"
        assert p.transforms[1].name == "reference_sub"
        assert p.transforms[2].name == "tool_filter"

    @pytest.mark.asyncio
    async def test_disable_transform_via_config(self) -> None:
        """Config can disable specific transforms."""
        config = LatticeConfig(transform_reference_sub=False)
        p = CompressorPipeline(config=config)
        p.register(PrefixOptimizer())
        p.register(ReferenceSubstitution())

        request = Request(
            messages=[Message(role="user", content="UUID: 550e8400-e29b-41d4-a716-446655440000")]
        )
        context = TransformContext()
        result = await p.process(request, context)
        modified = unwrap(result)

        # Reference substitution should NOT have been applied
        assert "550e8400" in modified.messages[0].content
        assert "reference_sub" not in context.transforms_applied

    @pytest.mark.asyncio
    async def test_graceful_degradation(self) -> None:
        """Transform failure with graceful_degradation=True continues processing."""
        class BrokenTransform:
            """A transform that always fails."""
            name = "broken"
            priority = 15
            enabled = True
            def process(self, _request, _context):
                raise AssertionError("Kaboom")

        config = LatticeConfig(graceful_degradation=True)
        p = CompressorPipeline(config=config)
        p.register(BrokenTransform())  # type: ignore[arg-type]
        p.register(ReferenceSubstitution())

        request = Request(
            messages=[Message(role="user", content="UUID: 550e8400-e29b-41d4-a716-446655440000")]
        )
        context = TransformContext()
        result = await p.process(request, context)
        unwrap(result)

        # Despite broken transform, reference sub still ran
        assert "broken" not in context.transforms_applied
        assert "reference_sub" in context.transforms_applied

    @pytest.mark.asyncio
    async def test_metrics_collected(self, pipeline: CompressorPipeline) -> None:
        """Metrics are populated after pipeline runs."""
        request = Request(messages=[Message(role="user", content="Hello")])
        context = TransformContext()
        await pipeline.process(request, context)

        assert context.metrics["tokens_in"] >= 0
        assert context.metrics["tokens_out"] >= 0
        assert context.metrics["transform_latency_ms"] >= 0
        assert bool(context.transforms_applied)

    @pytest.mark.asyncio
    async def test_runtime_budget_skips_remaining_transforms(self) -> None:
        """Runtime contract latency budget stops optional transform spend."""
        class SlowReferenceSub(ReferenceSubstitution):
            priority = 10

            def process(self, request, context):
                time.sleep(0.003)
                return super().process(request, context)

        class LaterRateDistortion(RateDistortionCompressor):
            priority = 20

        p = CompressorPipeline(config=LatticeConfig())
        p.register(SlowReferenceSub())
        p.register(LaterRateDistortion(distortion_budget=0.03, max_input_tokens=1))

        request = Request(
            messages=[
                Message(
                    role="user",
                    content=(
                        "UUID: 550e8400-e29b-41d4-a716-446655440000. "
                        "This intro sentence is low value. "
                        "What is the final numeric answer for the report? "
                        "Another low priority sentence for context."
                    ),
                )
            ],
            metadata={
                "_lattice_runtime_contract": {
                    "max_transform_latency_ms": 0.001,
                }
            },
        )
        context = TransformContext()
        result = await p.process(request, context)
        unwrap(result)

        assert "reference_sub" in context.transforms_applied
        assert "rate_distortion" not in context.transforms_applied
        assert "rate_distortion" in context.metrics["runtime_budget_skipped"]
        pipeline_metrics = context.metrics["transforms"]["pipeline"]
        assert pipeline_metrics["runtime_budget_exhausted"] is True
        budget = request.metadata.get("_lattice_runtime_budget", {})
        assert budget == {}
        modified = unwrap(result)
        budget = modified.metadata["_lattice_runtime_budget"]
        assert budget["exhausted"] is True
        assert budget["skipped_count"] == 1
        assert budget["skipped_transforms"] == ["rate_distortion"]
        assert budget["actual_transform_ms"] > 0
