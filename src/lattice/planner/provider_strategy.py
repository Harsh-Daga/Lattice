"""ProviderStrategy — provider-aware optimization decisions.

Different providers need different optimization plans. This module maps
provider capabilities to specific representation and transport strategies.

Phase 10 from the architecture refactor.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderStrategy:
    """Optimization strategy tailored to a specific provider."""

    provider: str
    preferred_optimizers: tuple[str, ...]
    cache_mode: str
    prefix_stable: bool  # OpenAI-style stable prefix caching
    supports_breakpoints: bool  # Anthropic-style explicit breakpoints
    supports_cached_content: bool  # Gemini/Vertex cachedContent
    max_context_tokens: int
    notes: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class CacheSimulation:
    """Estimated provider cache behavior for a request."""

    provider: str
    model: str
    cache_mode: str
    hit_probability: float
    expected_cached_tokens: int
    stable_prefix_tokens: int
    cache_plan_entries: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "cache_mode": self.cache_mode,
            "hit_probability": self.hit_probability,
            "expected_cached_tokens": self.expected_cached_tokens,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "cache_plan_entries": self.cache_plan_entries,
            "reason": self.reason,
        }


# Provider-specific strategies
_PROVIDER_STRATEGIES: dict[str, ProviderStrategy] = {
    "openai": ProviderStrategy(
        provider="openai",
        preferred_optimizers=(
            "reference_optimizer",
            "structure_optimizer",
            "tool_optimizer",
            "path_prefix",
        ),
        cache_mode="auto_prefix",
        prefix_stable=True,
        supports_breakpoints=False,
        supports_cached_content=False,
        max_context_tokens=128_000,
        notes="Stable prefix + prompt cache key + compact JSON",
    ),
    "anthropic": ProviderStrategy(
        provider="anthropic",
        preferred_optimizers=(
            "reference_optimizer",
            "structure_optimizer",
            "tool_optimizer",
            "path_prefix",
        ),
        cache_mode="explicit_breakpoint",
        prefix_stable=False,
        supports_breakpoints=True,
        supports_cached_content=False,
        max_context_tokens=200_000,
        notes="Cache breakpoints + system/tool separation",
    ),
    "ollama": ProviderStrategy(
        provider="ollama",
        preferred_optimizers=(
            "reference_optimizer",
            "structure_optimizer",
            "context_optimizer",
            "diagnostic_optimizer",
        ),
        cache_mode="none",
        prefix_stable=False,
        supports_breakpoints=False,
        supports_cached_content=False,
        max_context_tokens=128_000,
        notes="No provider cache assumption; prioritize token reduction",
    ),
    "gemini": ProviderStrategy(
        provider="gemini",
        preferred_optimizers=(
            "reference_optimizer",
            "structure_optimizer",
            "tool_optimizer",
        ),
        cache_mode="explicit_context",
        prefix_stable=False,
        supports_breakpoints=False,
        supports_cached_content=True,
        max_context_tokens=1_000_000,
        notes="cachedContent references for long context",
    ),
    "vertex": ProviderStrategy(
        provider="vertex",
        preferred_optimizers=(
            "reference_optimizer",
            "structure_optimizer",
            "tool_optimizer",
        ),
        cache_mode="explicit_context",
        prefix_stable=False,
        supports_breakpoints=False,
        supports_cached_content=True,
        max_context_tokens=1_000_000,
        notes="Same as Gemini via Vertex AI",
    ),
}


def get_provider_strategy(provider: str) -> ProviderStrategy:
    """Return the strategy for a provider, defaulting to generic."""
    return _PROVIDER_STRATEGIES.get(
        provider.lower(),
        ProviderStrategy(
            provider=provider.lower(),
            preferred_optimizers=(
                "reference_optimizer",
                "structure_optimizer",
                "tool_optimizer",
            ),
            cache_mode="none",
            prefix_stable=False,
            supports_breakpoints=False,
            supports_cached_content=False,
            max_context_tokens=128_000,
            notes="Generic provider — conservative optimization",
        ),
    )


def build_cache_plan_for_provider(
    provider: str,
    segment_count: int,
    estimated_tokens: int,
) -> list[dict[str, Any]]:
    """Build a provider-specific cache plan.

    Phase 10 — cache_arbitrage moves from transform to plan.
    """
    strategy = get_provider_strategy(provider)
    plan: list[dict[str, Any]] = []

    if strategy.cache_mode == "auto_prefix" and strategy.prefix_stable:
        # OpenAI: first 3 segments are the stable prefix
        for i in range(min(3, segment_count)):
            plan.append(
                {
                    "segment_index": i,
                    "provider_mode": "auto_prefix",
                    "expected_cached_tokens": estimated_tokens // max(segment_count, 1),
                    "annotations": {"stable": True},
                }
            )

    elif strategy.cache_mode == "explicit_breakpoint" and strategy.supports_breakpoints:
        # Anthropic: breakpoint after system + tools
        plan.append(
            {
                "segment_index": 0,
                "provider_mode": "explicit_breakpoint",
                "expected_cached_tokens": estimated_tokens // 2,
                "annotations": {"cache_control": {"type": "ephemeral"}},
            }
        )

    elif strategy.cache_mode == "explicit_context" and strategy.supports_cached_content:
        # Gemini: entire context as cached content reference
        plan.append(
            {
                "segment_index": 0,
                "provider_mode": "explicit_context",
                "expected_cached_tokens": estimated_tokens,
                "annotations": {"cachedContent": True},
            }
        )

    return plan


def simulate_provider_cache(
    provider: str,
    model: str,
    *,
    estimated_tokens: int,
    cache_plan: list[dict[str, Any]] | None = None,
    prefix_manifest: Any | None = None,
) -> CacheSimulation:
    """Estimate cache hit probability from provider strategy + plan shape."""
    strategy = get_provider_strategy(provider)
    plan_entries = cache_plan or []
    stable_prefix_tokens = 0

    if prefix_manifest is not None:
        if isinstance(prefix_manifest, dict):
            stable_prefix_tokens = int(prefix_manifest.get("prefix_tokens", 0))
        else:
            stable_prefix_tokens = int(getattr(prefix_manifest, "prefix_tokens", 0))

    if not plan_entries:
        return CacheSimulation(
            provider=provider,
            model=model,
            cache_mode=strategy.cache_mode,
            hit_probability=0.0,
            expected_cached_tokens=0,
            stable_prefix_tokens=stable_prefix_tokens,
            cache_plan_entries=0,
            reason="no_cache_plan",
        )

    cached_tokens = 0
    provider_modes = {_entry_value(entry, "provider_mode", "") for entry in plan_entries}
    for entry in plan_entries:
        cached_tokens += int(_entry_value(entry, "expected_cached_tokens", 0))

    # Deterministic heuristic: provider-native cache modes plus stable prefixes
    # improve the hit probability, while mixed provider modes keep it lower.
    base = {
        "auto_prefix": 0.78,
        "explicit_breakpoint": 0.66,
        "explicit_context": 0.83,
        "none": 0.0,
    }.get(strategy.cache_mode, 0.5)
    if "auto_prefix" in provider_modes:
        base += 0.05
    if "explicit_breakpoint" in provider_modes:
        base += 0.04
    if "explicit_context" in provider_modes:
        base += 0.06
    prefix_bonus = min(0.12, stable_prefix_tokens / max(estimated_tokens, 1) * 0.25)
    hit_probability = min(0.99, max(0.0, base + prefix_bonus))
    expected_cached = int(round(cached_tokens * hit_probability))

    return CacheSimulation(
        provider=provider,
        model=model,
        cache_mode=strategy.cache_mode,
        hit_probability=round(hit_probability, 4),
        expected_cached_tokens=expected_cached,
        stable_prefix_tokens=stable_prefix_tokens,
        cache_plan_entries=len(plan_entries),
        reason="provider_strategy_simulation",
    )


def _entry_value(entry: Any, key: str, default: Any) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def preferred_optimizers_for_provider(provider: str) -> tuple[str, ...]:
    """Return the optimizer preference order for a provider."""
    return get_provider_strategy(provider).preferred_optimizers


__all__ = [
    "ProviderStrategy",
    "CacheSimulation",
    "get_provider_strategy",
    "build_cache_plan_for_provider",
    "simulate_provider_cache",
    "preferred_optimizers_for_provider",
]
