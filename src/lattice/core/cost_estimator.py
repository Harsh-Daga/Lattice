"""Real-time cost estimator for LLM requests.

Tracks per-provider / per-model pricing and computes the estimated and
actual cost of each request. Exposes cost metadata via response headers
and aggregates for dashboards.

Pricing tables are kept current as of 2026-04. They are overridable via
config for private deployments or custom pricing tiers.

Supported providers
-------------------
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini, o1, o3-mini, etc.)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash)
- Cohere (Command R, Command R+)
- Mistral (Large, Medium, Small)
- Ollama (local — zero cost)
- DeepSeek (deepseek-chat, deepseek-coder)
- OpenRouter (pass-through pricing)

Design decisions
----------------
1. Pricing is stored as **USD per 1M tokens** (industry standard).
2. Cache-hit discounts are applied automatically when provider usage
   reports ``cached_tokens``.
3. Tool-call input tokens are counted as prompt tokens.
4. Output tokens include reasoning tokens (for o1/o3 models).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Pricing tables — USD per 1_000_000 tokens
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ModelPricing:
    """Per-model pricing structure."""

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None
    # Optional: reasoning / thinking tokens (e.g. o1, o3-mini)
    reasoning_per_1m: float | None = None


# OpenAI pricing (as of 2026-04)
_OPENAI_PRICING: dict[str, ModelPricing] = {
    "gpt-4": ModelPricing(input_per_1m=30.00, output_per_1m=60.00),
    "gpt-4-turbo": ModelPricing(input_per_1m=10.00, output_per_1m=30.00),
    "gpt-4o": ModelPricing(
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=1.25,
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
    ),
    "o1": ModelPricing(
        input_per_1m=15.00,
        output_per_1m=60.00,
        reasoning_per_1m=60.00,
    ),
    "o1-mini": ModelPricing(
        input_per_1m=3.00,
        output_per_1m=12.00,
        reasoning_per_1m=12.00,
    ),
    "o3-mini": ModelPricing(
        input_per_1m=1.10,
        output_per_1m=4.40,
        reasoning_per_1m=4.40,
    ),
    "gpt-3.5-turbo": ModelPricing(input_per_1m=0.50, output_per_1m=1.50),
}

# Anthropic pricing (as of 2026-04)
_ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=0.30,
    ),
    "claude-3-opus-20240229": ModelPricing(
        input_per_1m=15.00,
        output_per_1m=75.00,
        cached_input_per_1m=1.50,
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_per_1m=0.25,
        output_per_1m=1.25,
        cached_input_per_1m=0.03,
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        input_per_1m=1.00,
        output_per_1m=5.00,
        cached_input_per_1m=0.08,
    ),
}

# Google Gemini pricing (as of 2026-04)
_GOOGLE_PRICING: dict[str, ModelPricing] = {
    "gemini-1.5-pro": ModelPricing(input_per_1m=3.50, output_per_1m=10.50),
    "gemini-1.5-flash": ModelPricing(input_per_1m=0.35, output_per_1m=1.05),
    "gemini-1.5-pro-002": ModelPricing(input_per_1m=3.50, output_per_1m=10.50),
    "gemini-1.5-flash-002": ModelPricing(input_per_1m=0.35, output_per_1m=1.05),
}

# Cohere pricing
_COHERE_PRICING: dict[str, ModelPricing] = {
    "command-r": ModelPricing(input_per_1m=0.50, output_per_1m=1.50),
    "command-r-plus": ModelPricing(input_per_1m=3.00, output_per_1m=15.00),
}

# Mistral pricing
_MISTRAL_PRICING: dict[str, ModelPricing] = {
    "mistral-large-latest": ModelPricing(input_per_1m=2.00, output_per_1m=6.00),
    "mistral-medium": ModelPricing(input_per_1m=0.70, output_per_1m=2.70),
    "mistral-small": ModelPricing(input_per_1m=0.20, output_per_1m=0.60),
}

# DeepSeek pricing
_DEEPSEEK_PRICING: dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(input_per_1m=0.14, output_per_1m=0.28),
    "deepseek-coder": ModelPricing(input_per_1m=0.14, output_per_1m=0.28),
    "deepseek-reasoner": ModelPricing(
        input_per_1m=0.55,
        output_per_1m=2.19,
        reasoning_per_1m=2.19,
    ),
}

# Ollama (local) — zero cost
_OLLAMA_PRICING: dict[str, ModelPricing] = {
    "ollama": ModelPricing(input_per_1m=0.0, output_per_1m=0.0),
}

# AWS Bedrock exposes upstream provider model IDs (for example
# ``anthropic.claude...`` and ``cohere.command-r...``).  Reuse canonical
# upstream pricing after normalizing the Bedrock model ID.
_BEDROCK_PRICING: dict[str, ModelPricing] = {
    **_ANTHROPIC_PRICING,
    **_COHERE_PRICING,
    **_MISTRAL_PRICING,
}


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_TABLES: dict[str, dict[str, ModelPricing]] = {
    "openai": _OPENAI_PRICING,
    "azure": _OPENAI_PRICING,
    "anthropic": _ANTHROPIC_PRICING,
    "bedrock": _BEDROCK_PRICING,
    "google": _GOOGLE_PRICING,
    "gemini": _GOOGLE_PRICING,
    "vertex": _GOOGLE_PRICING,
    "cohere": _COHERE_PRICING,
    "mistral": _MISTRAL_PRICING,
    "deepseek": _DEEPSEEK_PRICING,
    "ollama": _OLLAMA_PRICING,
}


def _normalize_model_name(model: str) -> str:
    """Strip provider prefixes and version suffixes for lookup."""
    model = model.strip()

    # Strip common prefixes
    for prefix in (
        "openai/",
        "azure/",
        "anthropic/",
        "bedrock/",
        "google/",
        "gemini/",
        "vertex/",
        "mistral/",
        "cohere/",
        "deepseek/",
        "ollama/",
    ):
        if model.startswith(prefix):
            model = model[len(prefix) :]

    # Bedrock model IDs encode the upstream provider in a dotted prefix, e.g.
    # ``anthropic.claude-3-5-sonnet-20241022-v2:0``.
    for prefix in ("anthropic.", "amazon.", "cohere.", "mistral.", "meta."):
        if model.startswith(prefix):
            model = model[len(prefix) :]
            break

    # Bedrock appends deployment revisions that should not affect pricing.
    model = re.sub(r"-v\d+(?::\d+)?$", "", model)
    return model


def _first_int(data: dict[str, Any], *keys: str) -> int:
    """Return the first integer-valued key from a usage-like dict."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, int):
            return value
    return 0


def extract_cached_tokens(usage: Any) -> int:
    """Extract provider cache-read tokens from raw or normalized usage."""
    if not isinstance(usage, dict):
        return 0

    cached = _first_int(
        usage,
        "cached_tokens",
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "CacheReadInputTokens",
        "cachedContentTokenCount",
        "cached_content_token_count",
    )
    if cached:
        return cached

    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _first_int(details, "cached_tokens")
        if cached:
            return cached

    for container_key in ("usageMetadata", "usage_metadata"):
        container = usage.get(container_key)
        if isinstance(container, dict):
            cached = extract_cached_tokens(container)
            if cached:
                return cached
    return 0


def normalize_usage(usage: Any) -> dict[str, int]:
    """Normalize OpenAI, Anthropic, Bedrock, and Gemini usage fields."""
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    usage_root = usage
    for container_key in ("usageMetadata", "usage_metadata"):
        container = usage.get(container_key)
        if isinstance(container, dict):
            usage_root = container
            break

    prompt_tokens = _first_int(
        usage_root,
        "prompt_tokens",
        "input_tokens",
        "inputTokens",
        "InputTokens",
        "promptTokenCount",
        "prompt_token_count",
    )
    completion_tokens = _first_int(
        usage_root,
        "completion_tokens",
        "output_tokens",
        "outputTokens",
        "OutputTokens",
        "candidatesTokenCount",
        "candidates_token_count",
    )
    reasoning_tokens = _first_int(usage_root, "reasoning_tokens")
    completion_details = usage_root.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = reasoning_tokens or _first_int(
            completion_details,
            "reasoning_tokens",
        )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": extract_cached_tokens(usage),
        "reasoning_tokens": reasoning_tokens,
    }


_extract_cached_tokens = extract_cached_tokens
_normalize_usage = normalize_usage


def _lookup_pricing(provider: str, model: str) -> ModelPricing | None:
    """Look up pricing for a provider/model pair.

    Falls back to prefix matching if exact model not found.
    """
    table = _PROVIDER_TABLES.get(provider.lower())
    if table is None:
        return None

    model = _normalize_model_name(model)

    # Exact match
    if model in table:
        return table[model]

    # Prefix match (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
    for key in sorted(table.keys(), key=len, reverse=True):
        if model.startswith(key):
            return table[key]

    return None


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CostEstimate:
    """Result of a cost estimation."""

    prompt_cost_usd: float
    completion_cost_usd: float
    cached_savings_usd: float
    total_cost_usd: float
    currency: str = "USD"
    model: str = ""
    provider: str = ""
    pricing_source: str = "builtin"


class CostEstimator:
    """Real-time cost tracker for LLM requests.

    Usage::

        estimator = CostEstimator()
        estimate = estimator.estimate_request(provider="openai", model="gpt-4o",
                                               prompt_tokens=1000, completion_tokens=500)
        print(f"Estimated ${estimate.total_cost_usd:.6f}")
    """

    def __init__(
        self,
        custom_pricing: dict[str, dict[str, ModelPricing]] | None = None,
    ) -> None:
        self._custom = custom_pricing or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_request(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> CostEstimate:
        """Estimate cost for a request BEFORE calling the provider.

        Args:
            provider: Provider identifier (openai, anthropic, etc.).
            model: Model name.
            prompt_tokens: Estimated input tokens.
            completion_tokens: Estimated output tokens.
            cached_tokens: Tokens expected to hit provider-side cache.
            reasoning_tokens: Estimated reasoning tokens (o1, o3-mini, etc.).

        Returns:
            CostEstimate with prompt, completion, savings, and total.
        """
        pricing = self._resolve_pricing(provider, model)
        if pricing is None:
            return CostEstimate(
                prompt_cost_usd=0.0,
                completion_cost_usd=0.0,
                cached_savings_usd=0.0,
                total_cost_usd=0.0,
                model=model,
                provider=provider,
                pricing_source="unknown",
            )

        prompt_tokens = max(0, prompt_tokens)
        completion_tokens = max(0, completion_tokens)
        reasoning_tokens = max(0, reasoning_tokens)
        cached_tokens = min(max(0, cached_tokens), prompt_tokens)

        if cached_tokens > 0 and pricing.cached_input_per_1m is not None:
            uncached_prompt = prompt_tokens - cached_tokens
            prompt_cost = (uncached_prompt / 1_000_000) * pricing.input_per_1m
            cached_cost = (cached_tokens / 1_000_000) * pricing.cached_input_per_1m
            # Savings = what it would have cost at full input rate minus cached rate
            full_cached_cost = (cached_tokens / 1_000_000) * pricing.input_per_1m
            cached_savings = full_cached_cost - cached_cost
        else:
            prompt_cost = (prompt_tokens / 1_000_000) * pricing.input_per_1m
            cached_cost = 0.0
            cached_savings = 0.0

        completion_cost = (completion_tokens / 1_000_000) * pricing.output_per_1m

        reasoning_cost = 0.0
        if reasoning_tokens > 0 and pricing.reasoning_per_1m is not None:
            reasoning_cost = (reasoning_tokens / 1_000_000) * pricing.reasoning_per_1m

        total = prompt_cost + cached_cost + completion_cost + reasoning_cost

        return CostEstimate(
            prompt_cost_usd=prompt_cost + cached_cost,
            completion_cost_usd=completion_cost + reasoning_cost,
            cached_savings_usd=cached_savings,
            total_cost_usd=total,
            model=model,
            provider=provider,
            pricing_source="builtin",
        )

    def compute_actual(
        self,
        *,
        provider: str,
        model: str,
        usage: dict[str, Any],
    ) -> CostEstimate:
        """Compute actual cost from provider usage dict.

        Args:
            provider: Provider identifier.
            model: Model name.
            usage: Usage dict with keys like ``prompt_tokens``,
                ``completion_tokens``, ``cached_tokens``, etc.

        Returns:
            CostEstimate reflecting actual consumption.
        """
        normalized = normalize_usage(usage)
        return self.estimate_request(
            provider=provider,
            model=model,
            prompt_tokens=normalized["prompt_tokens"],
            completion_tokens=normalized["completion_tokens"],
            cached_tokens=normalized["cached_tokens"],
            reasoning_tokens=normalized["reasoning_tokens"],
        )

    def get_pricing_info(self, provider: str, model: str) -> dict[str, Any] | None:
        """Return raw pricing info for a provider/model."""
        pricing = self._resolve_pricing(provider, model)
        if pricing is None:
            return None
        return {
            "input_per_1m": pricing.input_per_1m,
            "output_per_1m": pricing.output_per_1m,
            "cached_input_per_1m": pricing.cached_input_per_1m,
            "reasoning_per_1m": pricing.reasoning_per_1m,
        }

    def add_custom_pricing(
        self,
        provider: str,
        model: str,
        input_per_1m: float,
        output_per_1m: float,
        cached_input_per_1m: float | None = None,
        reasoning_per_1m: float | None = None,
    ) -> None:
        """Register custom pricing for a provider/model."""
        self._custom.setdefault(provider.lower(), {})
        self._custom[provider.lower()][model] = ModelPricing(
            input_per_1m=input_per_1m,
            output_per_1m=output_per_1m,
            cached_input_per_1m=cached_input_per_1m,
            reasoning_per_1m=reasoning_per_1m,
        )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(estimates: list[CostEstimate]) -> CostEstimate:
        """Aggregate multiple estimates into a single summary."""
        total_prompt = sum(e.prompt_cost_usd for e in estimates)
        total_completion = sum(e.completion_cost_usd for e in estimates)
        total_savings = sum(e.cached_savings_usd for e in estimates)
        return CostEstimate(
            prompt_cost_usd=total_prompt,
            completion_cost_usd=total_completion,
            cached_savings_usd=total_savings,
            total_cost_usd=total_prompt + total_completion,
            model="aggregate",
            provider="aggregate",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_pricing(self, provider: str, model: str) -> ModelPricing | None:
        # Custom pricing takes precedence
        custom_table = self._custom.get(provider.lower())
        if custom_table:
            normalized = _normalize_model_name(model)
            if normalized in custom_table:
                return custom_table[normalized]
            for key in sorted(custom_table.keys(), key=len, reverse=True):
                if normalized.startswith(key):
                    return custom_table[key]
        return _lookup_pricing(provider, model)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def format_cost_usd(value: float) -> str:
    """Format a USD value for display.

    Uses µ (micro) for very small values, ¢ (cent) for moderate, $ for large.
    """
    if value < 0.000_001:
        return f"${value:.2e}"
    if value < 0.01:
        micro = value * 1_000_000
        return f"{micro:.1f}µ"
    if value < 1.0:
        cents = value * 100
        return f"{cents:.2f}¢"
    return f"${value:.4f}"


__all__ = [
    "CostEstimator",
    "CostEstimate",
    "ModelPricing",
    "extract_cached_tokens",
    "format_cost_usd",
    "normalize_usage",
]
