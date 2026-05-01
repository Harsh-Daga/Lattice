"""Cross-surface consistency tests — Phase 9 requirement.

Verifies that Proxy, SDK, MCP, and direct pipeline surfaces produce
the same compression behavior for identical inputs.

Cross-cutting rule: Treat proxy, SDK, MCP, and integrations as different
surfaces over one runtime core.
"""

from __future__ import annotations

from typing import Any

import pytest

from lattice.client import LatticeClient
from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline_factory import build_default_pipeline
from lattice.core.result import unwrap
from lattice.core.serialization import message_from_dict, message_to_dict
from lattice.core.transport import Request
from lattice.integrations.mcp import LatticeMCPTools

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "The transaction ID is 550e8400-e29b-41d4-a716-446655440000"},
        {"role": "assistant", "content": "I see the transaction."},
        {"role": "user", "content": "What about 6ba7b810-9dad-11d1-80b4-00c04fd430c8?"},
    ]


@pytest.fixture
def config() -> LatticeConfig:
    return LatticeConfig(compression_mode="safe", graceful_degradation=True)


# =============================================================================
# Surface implementations
# =============================================================================


async def _run_direct_pipeline(
    messages: list[dict[str, Any]], config: LatticeConfig
) -> dict[str, Any]:
    """Simulate the proxy surface: direct pipeline.process()."""
    pipeline = build_default_pipeline(config)
    request = Request(messages=[message_from_dict(m) for m in messages], model="gpt-4")
    ctx = TransformContext(provider="openai", model="gpt-4")
    result = await pipeline.process(request, ctx)
    compressed = unwrap(result)
    return {
        "messages": [message_to_dict(m) for m in compressed.messages],
        "tokens_before": request.token_estimate,
        "tokens_after": compressed.token_estimate,
        "transforms": ctx.transforms_applied,
        "profile": compressed.metadata.get("_lattice_profile"),
    }


async def _run_sdk_client(messages: list[dict[str, Any]], config: LatticeConfig) -> dict[str, Any]:
    """Simulate the SDK surface: LatticeClient.compress()."""
    client = LatticeClient(config=config)
    # compress() is sync and calls asyncio.run internally;
    # use to_thread to avoid 'cannot be called from a running event loop'
    import asyncio

    result = await asyncio.to_thread(client.compress, messages=messages, model="openai/gpt-4")
    return {
        "messages": result.compressed_messages,
        "tokens_before": result.original_tokens,
        "tokens_after": result.compressed_tokens,
        "transforms": result.transforms_applied,
        "profile": result.runtime.get("profile"),
    }


async def _run_mcp_tool(messages: list[dict[str, Any]], config: LatticeConfig) -> dict[str, Any]:
    """Simulate the MCP surface: LatticeMCPTools.lattice_compress()."""
    tools = LatticeMCPTools()
    tools.config = config
    tools.pipeline = build_default_pipeline(config)
    import asyncio

    result = await asyncio.to_thread(
        tools.lattice_compress, messages=messages, model="openai/gpt-4"
    )
    return {
        "messages": result["compressed_messages"],
        "tokens_before": result["tokens_before"],
        "tokens_after": result["tokens_after"],
        "transforms": result["transforms_applied"],
        "profile": result.get("content_profile"),
    }


# =============================================================================
# Consistency tests
# =============================================================================


class TestCrossSurfaceCompression:
    """All surfaces produce the same compressed output for the same input."""

    @pytest.mark.asyncio
    async def test_proxy_vs_sdk_same_output(
        self, sample_messages: list[dict[str, Any]], config: LatticeConfig
    ) -> None:
        """Direct pipeline and SDK client produce identical compressed messages."""
        proxy_result = await _run_direct_pipeline(sample_messages, config)
        sdk_result = await _run_sdk_client(sample_messages, config)

        # Message content should be identical
        proxy_contents = [m.get("content", "") for m in proxy_result["messages"]]
        sdk_contents = [m.get("content", "") for m in sdk_result["messages"]]
        assert proxy_contents == sdk_contents

        # Token counts should match
        assert proxy_result["tokens_before"] == sdk_result["tokens_before"]
        assert proxy_result["tokens_after"] == sdk_result["tokens_after"]

    @pytest.mark.asyncio
    async def test_proxy_vs_mcp_same_output(
        self, sample_messages: list[dict[str, Any]], config: LatticeConfig
    ) -> None:
        """Direct pipeline and MCP tool produce identical compressed messages."""
        proxy_result = await _run_direct_pipeline(sample_messages, config)
        mcp_result = await _run_mcp_tool(sample_messages, config)

        proxy_contents = [m.get("content", "") for m in proxy_result["messages"]]
        mcp_contents = [m.get("content", "") for m in mcp_result["messages"]]
        assert proxy_contents == mcp_contents

        assert proxy_result["tokens_before"] == mcp_result["tokens_before"]
        assert proxy_result["tokens_after"] == mcp_result["tokens_after"]

    @pytest.mark.asyncio
    async def test_sdk_vs_mcp_same_output(
        self, sample_messages: list[dict[str, Any]], config: LatticeConfig
    ) -> None:
        """SDK client and MCP tool produce identical compressed messages."""
        sdk_result = await _run_sdk_client(sample_messages, config)
        mcp_result = await _run_mcp_tool(sample_messages, config)

        sdk_contents = [m.get("content", "") for m in sdk_result["messages"]]
        mcp_contents = [m.get("content", "") for m in mcp_result["messages"]]
        assert sdk_contents == mcp_contents

        assert sdk_result["tokens_before"] == mcp_result["tokens_before"]
        assert sdk_result["tokens_after"] == mcp_result["tokens_after"]


class TestCrossSurfaceJsonSafety:
    """Structured content is safe on all surfaces."""

    @pytest.mark.asyncio
    async def test_tool_json_preserved_all_surfaces(self, config: LatticeConfig) -> None:
        """Tool output JSON is never corrupted by any surface."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "content": '{"id": "123", "status": "ok", "result": [1, 2, 3]}'},
        ]

        proxy_result = await _run_direct_pipeline(messages, config)
        sdk_result = await _run_sdk_client(messages, config)
        mcp_result = await _run_mcp_tool(messages, config)

        for result in (proxy_result, sdk_result, mcp_result):
            tool_msg = result["messages"][-1]
            parsed = __import__("json").loads(tool_msg["content"])
            assert parsed["id"] == "123"
            assert parsed["status"] == "ok"
            assert parsed["result"] == [1, 2, 3]


class TestCrossSurfaceModeConsistency:
    """Compression mode behaves the same across all surfaces."""

    @pytest.mark.asyncio
    async def test_safe_mode_produces_same_reduction(
        self, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Safe mode yields identical reduction ratios across surfaces."""
        config = LatticeConfig(compression_mode="safe")

        proxy_result = await _run_direct_pipeline(sample_messages, config)
        sdk_result = await _run_sdk_client(sample_messages, config)
        mcp_result = await _run_mcp_tool(sample_messages, config)

        proxy_ratio = proxy_result["tokens_after"] / max(proxy_result["tokens_before"], 1)
        sdk_ratio = sdk_result["tokens_after"] / max(sdk_result["tokens_before"], 1)
        mcp_ratio = mcp_result["tokens_after"] / max(mcp_result["tokens_before"], 1)

        # Ratios should be identical (or very close due to rounding)
        assert proxy_ratio == pytest.approx(sdk_ratio, abs=0.01)
        assert proxy_ratio == pytest.approx(mcp_ratio, abs=0.01)

    @pytest.mark.asyncio
    async def test_aggressive_mode_produces_same_reduction(
        self, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Aggressive mode yields identical reduction ratios across surfaces."""
        config = LatticeConfig(compression_mode="aggressive")

        proxy_result = await _run_direct_pipeline(sample_messages, config)
        sdk_result = await _run_sdk_client(sample_messages, config)
        mcp_result = await _run_mcp_tool(sample_messages, config)

        proxy_ratio = proxy_result["tokens_after"] / max(proxy_result["tokens_before"], 1)
        sdk_ratio = sdk_result["tokens_after"] / max(sdk_result["tokens_before"], 1)
        mcp_ratio = mcp_result["tokens_after"] / max(mcp_result["tokens_before"], 1)

        assert proxy_ratio == pytest.approx(sdk_ratio, abs=0.01)
        assert proxy_ratio == pytest.approx(mcp_ratio, abs=0.01)


class TestCrossSurfaceIdempotence:
    """Running compression twice on the same input yields the same output."""

    @pytest.mark.asyncio
    async def test_sdk_idempotent(
        self, sample_messages: list[dict[str, Any]], config: LatticeConfig
    ) -> None:
        """LatticeClient.compress is idempotent."""
        client = LatticeClient(config=config)
        import asyncio

        result1 = await asyncio.to_thread(
            client.compress, messages=sample_messages, model="openai/gpt-4"
        )
        result2 = await asyncio.to_thread(
            client.compress, messages=result1.compressed_messages, model="openai/gpt-4"
        )

        contents1 = [m.get("content", "") for m in result1.compressed_messages]
        contents2 = [m.get("content", "") for m in result2.compressed_messages]
        assert contents1 == contents2

    @pytest.mark.asyncio
    async def test_mcp_idempotent(
        self, sample_messages: list[dict[str, Any]], config: LatticeConfig
    ) -> None:
        """LatticeMCPTools.lattice_compress is idempotent."""
        tools = LatticeMCPTools()
        tools.config = config
        tools.pipeline = build_default_pipeline(config)

        import asyncio

        result1 = await asyncio.to_thread(
            tools.lattice_compress, messages=sample_messages, model="openai/gpt-4"
        )
        result2 = await asyncio.to_thread(
            tools.lattice_compress, messages=result1["compressed_messages"], model="openai/gpt-4"
        )

        contents1 = [m.get("content", "") for m in result1["compressed_messages"]]
        contents2 = [m.get("content", "") for m in result2["compressed_messages"]]
        assert contents1 == contents2
