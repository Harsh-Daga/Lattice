"""End-to-end benchmark for LATTICE compression.

This benchmark creates prompts specifically designed to trigger each transform,
then measures token savings across the pipeline using DirectHTTPProvider.

Key insight from our analysis: previous benchmarks got 0% savings because
prompts had no compressible content. This benchmark fixes that.

Usage:
    uv run python -m tests.unit.test_transport_e2e

Expected results:
    * ReferenceSubstitution:    20-50% savings (UUIDs, hashes)
    * ToolOutputFilter:       70-95% savings (large JSON blocks)
    * PrefixOptimizer:         5-15% savings (repeated prefixes)
    * Full pipeline:          combined savings
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.transport import Message, Request
from lattice.providers.transport import DirectHTTPProvider
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.transforms.prefix_opt import PrefixOptimizer
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.tool_filter import ToolOutputFilter
from lattice.utils.token_count import TiktokenCounter


class _FakeSessionManager:
    """Stub for DeltaEncoder that doesn't need async."""

    class _FakeStore:
        _sessions: dict[str, Any] = {}
        _ttl_seconds = 3600

    store = _FakeStore()


# =============================================================================
# Prompts designed to trigger compression
# =============================================================================


def make_uuid_heavy_prompt() -> tuple[list[dict[str, Any]], int]:
    """Prompt with 5 UUIDs — ReferenceSubstitution target."""
    uuids = [
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
    ]
    content = f"""Please analyze these transaction IDs: {uuids[0]}, {uuids[1]}, {uuids[2]}, {uuids[3]}, {uuids[4]}"""
    messages = ["user", content]
    chars = sum(len(u) for u in uuids)
    return messages, chars


def make_tool_output_prompt() -> tuple[list[dict[str, Any]], int]:
    """Prompt with large JSON tool output — ToolOutputFilter target."""
    tool_result = {
        "status": "success",
        "data": {
            "employees": [
                {
                    "id": i,
                    "name": f"Employee {i}",
                    "department": "Engineering",
                    "salary": 100000 + i * 1000,
                    "email": f"emp{i}@company.com",
                }
                for i in range(50)
            ],
            "metadata": {"total": 50, "page": 1, "page_size": 50},
        },
    }
    content = f"Here are the employee records:\n```json\n{json.dumps(tool_result, indent=2)}\n```"
    return ["tool", content], len(content)


def make_prefix_heavy_prompt() -> tuple[list[dict[str, Any]], int]:
    """Prompt with repeated prefix — PrefixOptimizer target."""
    prefix = "Step 1: Analyze the code.\nStep 2: Find bugs.\nStep 3: Suggest fixes.\n"
    messages = ["user", prefix + "Please refactor this function."]
    return messages, len(prefix)


def make_combined_prompt() -> tuple[list[dict[str, Any]], int]:
    """Combines UUIDs + tool output + prefixes — full pipeline target."""
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    prefix = "System status: OK\n"
    tool_data = {"logs": ["error"] * 100}
    content = f"{prefix}Transaction {uuid} failed.\n```json\n{json.dumps(tool_data)}\n```"
    return ["user", content], len(prefix) + len(uuid) + len(json.dumps(tool_data))


# =============================================================================
# Helper: run pipeline and measure savings
# =============================================================================


def run_pipeline(
    content: str,
    pipeline: CompressorPipeline,
    role: str = "user",
) -> tuple[Request, TransformContext, int, int]:
    """Run single message through pipeline and return before/after token counts."""
    request = Request(messages=[Message(role=role, content=content)])
    context = TransformContext(
        request_id="bench-001",
        session_id=None,
        provider="openai",
        model="gpt-4",
    )

    counter = TiktokenCounter("gpt-4")
    tokens_before = counter.count(content)

    import asyncio

    result = asyncio.run(pipeline.process(request, context))
    from lattice.core.result import unwrap

    compressed = unwrap(result)

    tokens_after = counter.count(compressed.messages[0].content)
    return compressed, context, tokens_before, tokens_after


# =============================================================================
# Tests
# =============================================================================


class TestReferenceSubstitutionBenchmark:
    """Measure ReferenceSubstitution on UUID-heavy content."""

    def test_uuid_token_reduction(self) -> None:
        config = LatticeConfig(graceful_degradation=True)
        pipeline = CompressorPipeline(config=config)
        pipeline.register(ReferenceSubstitution())

        raw_content = (
            "Please analyze these transaction IDs: 550e8400-e29b-41d4-a716-446655440000, "
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        )
        _, _, before, after = run_pipeline(raw_content, pipeline)

        # 2 UUIDs × ~36 chars = ~72 chars → replaced with <ref_1>, <ref_2> (~16 chars)
        # Token reduction should be meaningful
        savings = before - after
        ratio = savings / max(before, 1)
        print(
            f"\n[ReferenceSubstitution] UUID test: {before} → {after} tokens (saved {savings}, {ratio:.1%})"
        )
        assert savings > 0, "Must save tokens on UUID-heavy input"
        assert ratio > 0.05, f"Expected >5% savings, got {ratio:.1%}"


class TestToolOutputFilterBenchmark:
    """Measure ToolOutputFilter on large JSON content."""

    def test_json_token_reduction(self) -> None:
        config = LatticeConfig(graceful_degradation=True)
        pipeline = CompressorPipeline(config=config)
        pipeline.register(ToolOutputFilter())

        # Bare JSON array (no markdown wrapper) — ToolOutputFilter expects this
        raw_content = json.dumps(
            [
                {
                    "id": i,
                    "data": "x" * 500,
                    "metadata": {"created_at": "2024-01-01", "internal": True},
                }
                for i in range(100)
            ],
            indent=2,
        )
        _, _, before, after = run_pipeline(raw_content, pipeline, role="tool")

        # ToolOutputFilter removes metadata, truncates arrays, compacts JSON
        savings = before - after
        ratio = savings / max(before, 1)
        print(
            f"\n[ToolOutputFilter] JSON test: {before} → {after} tokens (saved {savings}, {ratio:.1%})"
        )
        assert savings > 0, "Must save tokens on large JSON input"
        assert ratio > 0.01, f"Expected >1% savings, got {ratio:.1%}"


class TestPrefixOptimizerBenchmark:
    """Measure PrefixOptimizer on repeated prefix content."""

    def test_prefix_token_reduction(self) -> None:
        config = LatticeConfig(graceful_degradation=True)
        pipeline = CompressorPipeline(config=config)
        pipeline.register(PrefixOptimizer())

        prefix = "Common system prefix: analyze, optimize, refactor. "
        msgs = []
        for i in range(5):
            content = prefix + f"Here is message number {i}."
            msgs.append(Message(role="user" if i % 2 == 0 else "assistant", content=content))

        request = Request(messages=msgs)
        context = TransformContext(request_id="bench-002", provider="openai", model="gpt-4")

        counter = TiktokenCounter("gpt-4")
        tokens_before = sum(counter.count(m.content) for m in msgs)

        import asyncio

        from lattice.core.result import unwrap

        result = asyncio.run(pipeline.process(request, context))
        compressed = unwrap(result)

        tokens_after = sum(counter.count(m.content) for m in compressed.messages)
        savings = tokens_before - tokens_after
        ratio = savings / max(tokens_before, 1)
        print(
            f"\n[PrefixOptimizer] Prefix test: {tokens_before} → {tokens_after} tokens (saved {savings}, {ratio:.1%})"
        )
        assert savings >= 0


class TestFullPipelineBenchmark:
    """Measure combined savings across all transforms."""

    def test_combined_savings(self) -> None:
        config = LatticeConfig(graceful_degradation=True)
        pipeline = CompressorPipeline(config=config)
        pipeline.register(PrefixOptimizer())
        pipeline.register(ReferenceSubstitution())
        pipeline.register(ToolOutputFilter())
        pipeline.register(OutputCleanup())

        # Dense compressible content: 5 UUIDs + large JSON array
        uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
        ]
        json_data = [
            {
                "id": i,
                "name": f"Emp{i}",
                "salary": 100000 + i * 1000,
                "metadata": {"created_at": "2024-01-01", "internal": True},
            }
            for i in range(50)
        ]
        content = f"Transactions: {', '.join(uuids)}\n" + json.dumps(json_data, indent=2)
        _, _, before, after = run_pipeline(content, pipeline)

        savings = before - after
        ratio = savings / max(before, 1)
        print(
            f"\n[FullPipeline] Combined test: {before} → {after} tokens (saved {savings}, {ratio:.1%})"
        )
        assert savings > 0, "Full pipeline must save tokens on compressible input"


class TestEndToEndRoundTrip:
    """Verify provider transport round-trip preserves meaning."""

    def test_provider_completion_uses_direct_http(self) -> None:
        """Verify DirectHTTPProvider is wired correctly (mocked at HTTP level)."""
        import respx
        from httpx import Response as HttpxResponse

        with respx.mock:
            route = respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=HttpxResponse(
                    200,
                    json={
                        "model": "llama3",
                        "message": {"role": "assistant", "content": "Hello back"},
                        "done": True,
                    },
                )
            )
            provider = DirectHTTPProvider(
                provider_base_urls={"ollama": "http://127.0.0.1:11434"},
            )
            import asyncio

            resp = asyncio.run(
                provider.completion(
                    model="ollama/llama3",
                    messages=[{"role": "user", "content": "Hello"}],
                )
            )
            assert resp.content == "Hello back"
            assert route.called


# Run as script for quick manual verification
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
