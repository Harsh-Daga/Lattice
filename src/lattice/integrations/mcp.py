"""MCP (Model Context Protocol) tools for LATTICE.

Exposes LATTICE functionality as MCP tool definitions that can be
registered with any MCP server. Enables agents to:
- Compress context on demand (all 11 transforms)
- Start/stop sessions with delta encoding
- Get compression metrics and provider health
- Count tokens

Dependencies:
    pip install lattice-transport[mcp]
    # or: pip install mcp>=1.0.0

Usage (server-side):
    from lattice.integrations.mcp import lattice_tools
    # Register with your MCP server
    server.add_tool(lattice_tools["lattice_compress"])
"""

from __future__ import annotations

import uuid
from typing import Any

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.pipeline_factory import build_default_pipeline, pipeline_summary
from lattice.core.result import is_err, unwrap, unwrap_err
from lattice.core.serialization import message_from_dict, message_to_dict
from lattice.core.session import MemorySessionStore, Session
from lattice.core.transport import Request

_MCP_AVAILABLE = False
try:
    import mcp.types  # noqa: F401

    _MCP_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Tool definitions
# =============================================================================


class LatticeMCPTools:
    """MCP tool definitions for LATTICE.

    Each method is a callable that MCP servers can register.
    Schema follows the MCP protocol.
    """

    def __init__(self) -> None:
        self.config = LatticeConfig.auto()
        self.store = MemorySessionStore()
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> CompressorPipeline:
        """Build the full compression pipeline."""
        return build_default_pipeline(self.config)

    # ------------------------------------------------------------------
    # lattice_compress
    # ------------------------------------------------------------------

    def lattice_compress(
        self, messages: list[dict[str, Any]], model: str = "openai/gpt-4"
    ) -> dict[str, Any]:
        """Compress a list of messages using the full LATTICE pipeline.

        Args:
            messages: List of OpenAI-compatible message dicts.
            model: Model identifier (determines tokenizer).

        Returns:
            Dict with compressed_messages, tokens_before, tokens_after,
            compression_ratio, transforms_applied, and content_profile.
        """
        from lattice.providers.transport import _resolve_provider_name

        request = Request(
            messages=[message_from_dict(m) for m in messages],
            model=model,
        )
        context = TransformContext(
            provider=_resolve_provider_name(model),
            model=model,
        )

        async def _run() -> dict[str, Any]:
            result = await self.pipeline.process(request, context)
            if is_err(result):
                err = unwrap_err(result)
                return {"error": err.message, "compressed_messages": messages}
            compressed = unwrap(result)
            compressed_messages = [message_to_dict(m) for m in compressed.messages]
            return {
                "compressed_messages": compressed_messages,
                "tokens_before": request.token_estimate,
                "tokens_after": compressed.token_estimate,
                "compression_ratio": round(
                    (request.token_estimate - compressed.token_estimate)
                    / max(request.token_estimate, 1),
                    4,
                ),
                "transforms_applied": context.transforms_applied,
                "content_profile": context.session_state.get("content_profile"),
                "runtime": compressed.metadata.get("_lattice_runtime", {}),
                "runtime_budget": compressed.metadata.get("_lattice_runtime_budget", {}),
            }

        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        # In an async context — schedule in a thread pool to avoid deadlock
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result()

    # ------------------------------------------------------------------
    # lattice_session_start
    # ------------------------------------------------------------------

    def lattice_session_start(self, provider: str = "", model: str = "") -> dict[str, str]:
        """Start a new LATTICE compression session.

        Args:
            provider: Provider name (openai, anthropic, groq, etc.).
            model: Model identifier.

        Returns:
            Dict with session_id.
        """
        if not provider:
            raise ValueError("provider is required")
        import time as time_mod

        sid = f"sess-{uuid.uuid4().hex[:12]}"
        session = Session(
            session_id=sid,
            created_at=time_mod.time(),
            last_accessed_at=time_mod.time(),
            provider=provider,
            model=model,
            messages=[],
        )

        async def _persist() -> None:
            await self.store.set(session)

        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_persist())
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _persist())
                future.result()

        return {"session_id": sid}

    # ------------------------------------------------------------------
    # lattice_stats
    # ------------------------------------------------------------------

    def lattice_stats(self) -> dict[str, Any]:
        """Get current LATTICE compression statistics.

        Returns:
            Dict with version, available_transforms, and session_count.
        """
        from lattice._version import __version__

        return {
            "version": __version__,
            "available_transforms": [t.name for t in self.pipeline.transforms],
            "pipeline": pipeline_summary(self.pipeline),
            "session_count": self.store.session_count,
        }

    # ------------------------------------------------------------------
    # lattice_count_tokens
    # ------------------------------------------------------------------

    def lattice_count_tokens(self, text: str, model: str = "gpt-4") -> dict[str, Any]:
        """Count tokens in a text string.

        Args:
            text: Text to tokenize.
            model: Model name for tokenizer selection.

        Returns:
            Dict with token_count.
        """
        from lattice.utils.token_count import TiktokenCounter

        try:
            counter = TiktokenCounter(model)
            count = counter.count(text)
        except Exception:
            count = len(text) // 4
        return {"token_count": count, "model": model}


# =============================================================================
# MCP Tool schema definitions (for MCP servers)
# =============================================================================


def _make_tool_schema() -> dict[str, Any]:
    """Return MCP tool schema definitions."""
    return {
        "lattice_compress": {
            "description": "Compress a list of messages using LATTICE transforms. "
            "Replaces long identifiers with short aliases, filters tool output noise, "
            "deduplicates messages, and cleans conversational fluff.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "List of OpenAI-compatible messages to compress",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant", "tool"],
                                },
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "model": {"type": "string", "default": "gpt-4"},
                },
                "required": ["messages"],
            },
        },
        "lattice_session_start": {
            "description": "Start a new LATTICE compression session for multi-turn delta encoding.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "default": "openai"},
                    "model": {"type": "string", "default": ""},
                },
            },
        },
        "lattice_stats": {
            "description": "Get LATTICE compression statistics and transform status.",
            "inputSchema": {"type": "object"},
        },
        "lattice_count_tokens": {
            "description": "Count tokens in a text string using tiktoken.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "model": {"type": "string", "default": "gpt-4"},
                },
                "required": ["text"],
            },
        },
    }


# Public exports
lattice_tools = LatticeMCPTools()
tool_schema = _make_tool_schema()

__all__ = ["LatticeMCPTools", "lattice_tools", "tool_schema"]
