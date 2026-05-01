"""Provider-aware cache planners for LATTICE.

Optimizes prompt segment ordering and annotations to maximize provider-side
cache hit rates.

OpenAI caching:
- Exact-prefix based, automatic for prompts >1024 tokens on supported models.
- Exposes cached_tokens in usage.
- Best practice: stable prefix placement, consistent content, static tools first.

Anthropic caching:
- Exact-prefix based, requires explicit cache_control breakpoints.
- Order: tools → system → messages.
- cache_control: {type: "ephemeral"} marks a breakpoint.

Reference:
- https://platform.openai.com/docs/guides/prompt-caching
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lattice.protocol.manifest import Manifest
from lattice.protocol.segments import Segment, SegmentType
from lattice.providers.capabilities import (
    CacheMode,
    CacheSemantics,
    get_capability_registry,
)

# =============================================================================
# CachePlanner protocol
# =============================================================================

@dataclasses.dataclass(slots=True)
class CachePlan:
    """Result of cache planning for a manifest.

    Attributes:
        segments: Reordered/annotated segments for the provider.
        expected_cached_tokens: Estimated tokens that will hit cache.
        breakpoints: Indices where cache breaks occur.
        annotations: Provider-specific metadata per segment.
    """

    segments: list[Segment]
    expected_cached_tokens: int = 0
    breakpoints: list[int] = dataclasses.field(default_factory=list)
    annotations: dict[str, Any] = dataclasses.field(default_factory=dict)


class CachePlanner:
    """Base class for provider-specific cache planners."""

    def plan(self, manifest: Manifest) -> CachePlan:
        """Generate a cache plan for the given manifest.

        Args:
            manifest: The conversation manifest.

        Returns:
            CachePlan with optimized segment ordering and annotations.
        """
        raise NotImplementedError


# =============================================================================
# OpenAI cache planner
# =============================================================================

class OpenAICachePlanner(CachePlanner):
    """Optimizes manifest for OpenAI prompt caching.

    OpenAI's caching is automatic for long prompts on supported models.
    The key is keeping the prefix STABLE and EXACT across turns.

    Strategy:
    1. Order segments: tools → system → docs → artifacts → messages.
    2. Ensure system/tools segments never change between turns.
    3. Only append to messages segment (never mutate earlier parts).
    """

    def __init__(
        self,
        provider: str = "openai",
        cache: CacheSemantics | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache or get_capability_registry().cache_semantics(provider)

    def plan(self, manifest: Manifest) -> CachePlan:
        segments = list(manifest.segments)

        # OpenAI rewards stable exact prefixes
        # The default canonical order is already optimal
        # We just need to ensure no mutations to prefix segments

        # Calculate estimated cached tokens = prefix segments
        cached_tokens = sum(
            s.token_estimate
            for s in segments
            if s.type in (SegmentType.TOOLS, SegmentType.SYSTEM, SegmentType.DOCS)
        )

        # Find breakpoint index (first non-stable segment)
        breakpoints: list[int] = []
        for i, seg in enumerate(segments):
            if seg.type == SegmentType.MESSAGES:
                breakpoints.append(i)
                break

        return CachePlan(
            segments=segments,
            expected_cached_tokens=cached_tokens,
            breakpoints=breakpoints or [len(segments)],
            annotations={
                "provider": self.provider,
                "auto_caching": True,
                "cache": self.cache.to_dict(),
            },
        )


# =============================================================================
# Anthropic cache planner
# =============================================================================

class AnthropicCachePlanner(CachePlanner):
    """Optimizes manifest for Anthropic prompt caching.

    Anthropic requires explicit cache_control breakpoints.
    Maximum 4 cache breakpoints per request.
    Order matters: tools → system → messages.

    Strategy:
    1. Place cache_control on the LAST item of each stable prefix block.
    2. Maximum 4 breakpoints: end of tools, end of system, end of docs,
       end of most recent message block.
    3. Never place breakpoints on the final message block (would waste a
       breakpoint since it can't be reused until the next turn).
    """

    MAX_BREAKPOINTS = 4

    def __init__(
        self,
        provider: str = "anthropic",
        cache: CacheSemantics | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache or get_capability_registry().cache_semantics(provider)
        self.max_breakpoints = self.cache.max_breakpoints or self.MAX_BREAKPOINTS

    def plan(self, manifest: Manifest) -> CachePlan:
        segments = list(manifest.segments)
        annotated: list[Segment] = []
        breakpoints: list[int] = []
        breakpoint_count = 0

        for i, seg in enumerate(segments):
            meta = dict(seg.metadata)

            # Place cache_control at the end of stable prefix blocks
            if (
                seg.type in (SegmentType.TOOLS, SegmentType.SYSTEM, SegmentType.DOCS)
                and breakpoint_count < self.max_breakpoints
            ):
                meta["cache_control"] = {"type": "ephemeral"}
                if self.cache.default_ttl_seconds == 3600:
                    meta["cache_control"]["ttl"] = "1h"
                breakpoints.append(i)
                breakpoint_count += 1

            annotated.append(
                Segment(
                    type=seg.type,
                    version=seg.version,
                    hash=seg.hash,
                    parts=seg.parts,
                    metadata=meta,
                    created_at=seg.created_at,
                )
            )

        # Calculate expected cached tokens
        cached_tokens = (
            sum(
                seg.token_estimate
                for seg in annotated[: breakpoints[-1] + 1]
            )
            if breakpoints
            else 0
        )

        return CachePlan(
            segments=annotated,
            expected_cached_tokens=cached_tokens,
            breakpoints=breakpoints,
            annotations={
                "provider": self.provider,
                "cache_breakpoints": len(breakpoints),
                "max_breakpoints": self.max_breakpoints,
                "cache": self.cache.to_dict(),
            },
        )


# =============================================================================
# Explicit context cache planner
# =============================================================================

class ContextCachePlanner(CachePlanner):
    """Planner for providers with explicit reusable context-cache resources.

    Gemini and Vertex expose cached content/context resources rather than
    Anthropic-style inline cache_control breakpoints. The planner marks stable
    prefix segments as eligible for external cache resource creation while
    keeping request order unchanged.
    """

    def __init__(
        self,
        provider: str = "gemini",
        cache: CacheSemantics | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache or get_capability_registry().cache_semantics(provider)

    def plan(self, manifest: Manifest) -> CachePlan:
        annotated: list[Segment] = []
        cacheable_indices: list[int] = []

        for i, seg in enumerate(manifest.segments):
            meta = dict(seg.metadata)
            if seg.type in (SegmentType.TOOLS, SegmentType.SYSTEM, SegmentType.DOCS, SegmentType.ARTIFACTS):
                meta["cache_resource"] = {
                    "eligible": True,
                    "mode": self.cache.mode.value,
                    "ttl_seconds": self.cache.default_ttl_seconds,
                }
                cacheable_indices.append(i)
            annotated.append(
                Segment(
                    type=seg.type,
                    version=seg.version,
                    hash=seg.hash,
                    parts=seg.parts,
                    metadata=meta,
                    created_at=seg.created_at,
                )
            )

        cached_tokens = sum(annotated[i].token_estimate for i in cacheable_indices)
        return CachePlan(
            segments=annotated,
            expected_cached_tokens=cached_tokens,
            breakpoints=cacheable_indices,
            annotations={
                "provider": self.provider,
                "explicit_context_cache": self.cache.supports_explicit_resource,
                "cache": self.cache.to_dict(),
            },
        )


# =============================================================================
# Generic / fallback planner
# =============================================================================

class GenericCachePlanner(CachePlanner):
    """Fallback planner that preserves canonical order without annotations."""

    def __init__(self, provider: str = "generic") -> None:
        self.provider = provider

    def plan(self, manifest: Manifest) -> CachePlan:
        return CachePlan(
            segments=list(manifest.segments),
            expected_cached_tokens=0,
            breakpoints=[],
            annotations={"provider": self.provider, "auto_caching": False},
        )


# =============================================================================
# Planner registry
# =============================================================================

def get_cache_planner(provider: str) -> CachePlanner:
    """Get the appropriate cache planner for a provider.

    Args:
        provider: Provider name (openai, anthropic, ollama, etc.).

    Returns:
        CachePlanner instance.
    """
    provider_key = provider.lower()
    registry = get_capability_registry()
    cache = registry.cache_semantics(provider_key)

    if cache.mode == CacheMode.AUTO_PREFIX:
        return OpenAICachePlanner(provider=provider_key, cache=cache)
    if cache.mode == CacheMode.EXPLICIT_BREAKPOINT:
        return AnthropicCachePlanner(provider=provider_key, cache=cache)
    if cache.mode == CacheMode.EXPLICIT_CONTEXT:
        return ContextCachePlanner(provider=provider_key, cache=cache)
    return GenericCachePlanner(provider=provider_key if registry.get(provider_key) else "generic")
