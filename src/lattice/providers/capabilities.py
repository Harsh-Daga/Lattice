"""Provider capabilities database for LATTICE.

Tracks per-provider feature support so LATTICE can adapt behavior:
- Prompt caching support and semantics
- Streaming support
- Tool calling support
- Multimodal support
- Reasoning/thinking support
- HTTP/2 and HTTP/3 support
- Max context length
- Supported models

This replaces hardcoded assumptions with capability-detected behavior.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

# =============================================================================
# Capability flags
# =============================================================================


class Capability(enum.IntFlag):
    """Feature capabilities that a provider may support."""

    CHAT_COMPLETIONS = 1 << 0
    STREAMING = 1 << 1
    TOOL_CALLS = 1 << 2
    MULTIMODAL = 1 << 3
    REASONING = 1 << 4
    PROMPT_CACHING = 1 << 5
    BATCHING = 1 << 6
    STRUCTURED_OUTPUT = 1 << 7
    RESPONSES_API = 1 << 8
    HTTP2 = 1 << 9
    HTTP3 = 1 << 10
    WEBSOCKET = 1 << 11


# =============================================================================
# Cache semantics
# =============================================================================


class CacheMode(enum.Enum):
    """How a provider implements prompt caching."""

    NONE = "none"
    AUTO_PREFIX = "auto_prefix"  # OpenAI: automatic exact-prefix
    EXPLICIT_BREAKPOINT = "explicit"  # Anthropic: cache_control breakpoints
    ADAPTER_MANAGED = "adapter"  # Provider adapter handles it
    EXPLICIT_CONTEXT = "explicit_context"  # Gemini/Vertex: cachedContent resource


@dataclasses.dataclass(frozen=True, slots=True)
class CacheSemantics:
    """Provider prompt/cache behavior that LATTICE can optimize against."""

    mode: CacheMode = CacheMode.NONE
    min_cache_tokens: int = 0
    hit_increment_tokens: int = 0
    max_breakpoints: int = 0
    default_ttl_seconds: int | None = None
    supported_ttl_seconds: tuple[int, ...] = ()
    telemetry_fields: tuple[str, ...] = ()
    cache_key_param: str | None = None
    retention_param: str | None = None
    supports_explicit_resource: bool = False
    notes: str = ""

    def supports_ttl(self, ttl_seconds: int) -> bool:
        return ttl_seconds in self.supported_ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "min_cache_tokens": self.min_cache_tokens,
            "hit_increment_tokens": self.hit_increment_tokens,
            "max_breakpoints": self.max_breakpoints,
            "default_ttl_seconds": self.default_ttl_seconds,
            "supported_ttl_seconds": list(self.supported_ttl_seconds),
            "telemetry_fields": list(self.telemetry_fields),
            "cache_key_param": self.cache_key_param,
            "retention_param": self.retention_param,
            "supports_explicit_resource": self.supports_explicit_resource,
            "notes": self.notes,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class RateLimitSemantics:
    """Header conventions and accounting assumptions for provider limits."""

    retry_after_header: str = "retry-after"
    request_limit_header: str | None = None
    request_remaining_header: str | None = None
    request_reset_header: str | None = None
    token_limit_header: str | None = None
    token_remaining_header: str | None = None
    token_reset_header: str | None = None
    cache_hits_count_against_token_limits: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# ProviderCapability
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderCapability:
    """Complete capability snapshot for a provider."""

    provider: str
    capabilities: Capability
    cache_mode: CacheMode
    max_context_tokens: int
    max_output_tokens: int
    supported_models: tuple[str, ...]
    default_base_url: str
    cache: CacheSemantics = dataclasses.field(default_factory=CacheSemantics)
    rate_limits: RateLimitSemantics = dataclasses.field(default_factory=RateLimitSemantics)
    notes: str = ""

    def supports(self, cap: Capability) -> bool:
        return bool(self.capabilities & cap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "capabilities": [c.name for c in Capability if self.supports(c)],
            "cache_mode": self.cache_mode.value,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "supported_models": list(self.supported_models),
            "default_base_url": self.default_base_url,
            "cache": self.cache.to_dict(),
            "rate_limits": self.rate_limits.to_dict(),
            "notes": self.notes,
        }


# =============================================================================
# Built-in capability database
# =============================================================================

_BUILTIN_CAPABILITIES: dict[str, ProviderCapability] = {
    "openai": ProviderCapability(
        provider="openai",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.MULTIMODAL
            | Capability.REASONING
            | Capability.PROMPT_CACHING
            | Capability.STRUCTURED_OUTPUT
            | Capability.RESPONSES_API
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.AUTO_PREFIX,
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        supported_models=(
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5",
            "gpt-5-codex",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
        ),
        default_base_url="https://api.openai.com",
        cache=CacheSemantics(
            mode=CacheMode.AUTO_PREFIX,
            min_cache_tokens=1024,
            hit_increment_tokens=128,
            default_ttl_seconds=600,
            supported_ttl_seconds=(300, 600, 3600, 86400),
            telemetry_fields=("usage.prompt_tokens_details.cached_tokens",),
            cache_key_param="prompt_cache_key",
            retention_param="prompt_cache_retention",
            notes="Automatic exact-prefix cache. In-memory retention is typically 5-10 minutes idle, up to 1 hour; selected models support 24h retention.",
        ),
        rate_limits=RateLimitSemantics(
            request_limit_header="x-ratelimit-limit-requests",
            request_remaining_header="x-ratelimit-remaining-requests",
            request_reset_header="x-ratelimit-reset-requests",
            token_limit_header="x-ratelimit-limit-tokens",
            token_remaining_header="x-ratelimit-remaining-tokens",
            token_reset_header="x-ratelimit-reset-tokens",
        ),
        notes="Prompt caching automatic for prompts >=1024 tokens on recent models.",
    ),
    "anthropic": ProviderCapability(
        provider="anthropic",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.MULTIMODAL
            | Capability.REASONING
            | Capability.PROMPT_CACHING
            | Capability.STRUCTURED_OUTPUT
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.EXPLICIT_BREAKPOINT,
        max_context_tokens=200_000,
        max_output_tokens=8192,
        supported_models=(
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ),
        default_base_url="https://api.anthropic.com",
        cache=CacheSemantics(
            mode=CacheMode.EXPLICIT_BREAKPOINT,
            min_cache_tokens=1024,
            max_breakpoints=4,
            default_ttl_seconds=300,
            supported_ttl_seconds=(300, 3600),
            telemetry_fields=(
                "usage.cache_read_input_tokens",
                "usage.cache_creation_input_tokens",
                "usage.cache_creation.ephemeral_5m_input_tokens",
                "usage.cache_creation.ephemeral_1h_input_tokens",
            ),
            notes="Explicit cache_control breakpoints with 5m default and optional 1h TTL on supported models.",
        ),
        rate_limits=RateLimitSemantics(
            request_limit_header="anthropic-ratelimit-requests-limit",
            request_remaining_header="anthropic-ratelimit-requests-remaining",
            request_reset_header="anthropic-ratelimit-requests-reset",
            token_limit_header="anthropic-ratelimit-tokens-limit",
            token_remaining_header="anthropic-ratelimit-tokens-remaining",
            token_reset_header="anthropic-ratelimit-tokens-reset",
            cache_hits_count_against_token_limits=False,
        ),
        notes="Max 4 cache_control breakpoints per request. Order: tools -> system -> messages.",
    ),
    "ollama": ProviderCapability(
        provider="ollama",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.REASONING
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        supported_models=("llama3", "llama3.1", "mistral", "qwen2", "phi3"),
        default_base_url="http://127.0.0.1:11434",
        rate_limits=RateLimitSemantics(notes="Local runtime; limits depend on host hardware."),
        notes="Native /api/chat endpoint. Thinking traces supported via think tags.",
    ),
    "groq": ProviderCapability(
        provider="groq",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.STRUCTURED_OUTPUT
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.NONE,
        max_context_tokens=8_192,
        max_output_tokens=4_096,
        supported_models=(
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b",
            "gemma-7b-it",
        ),
        default_base_url="https://api.groq.com/openai",
        rate_limits=RateLimitSemantics(
            request_limit_header="x-ratelimit-limit-requests",
            request_remaining_header="x-ratelimit-remaining-requests",
            request_reset_header="x-ratelimit-reset-requests",
            token_limit_header="x-ratelimit-limit-tokens",
            token_remaining_header="x-ratelimit-remaining-tokens",
            token_reset_header="x-ratelimit-reset-tokens",
        ),
        notes="OpenAI-compatible endpoint. Very fast TTFT.",
    ),
    "azure": ProviderCapability(
        provider="azure",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.MULTIMODAL
            | Capability.REASONING
            | Capability.PROMPT_CACHING
            | Capability.STRUCTURED_OUTPUT
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.AUTO_PREFIX,
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        supported_models=("gpt-4o", "gpt-4", "gpt-35-turbo"),
        default_base_url="",
        cache=CacheSemantics(
            mode=CacheMode.AUTO_PREFIX,
            min_cache_tokens=1024,
            hit_increment_tokens=128,
            telemetry_fields=("usage.prompt_tokens_details.cached_tokens",),
            notes="Azure OpenAI follows OpenAI-compatible cached-token usage where supported by the deployment.",
        ),
        rate_limits=RateLimitSemantics(
            request_limit_header="x-ratelimit-limit-requests",
            request_remaining_header="x-ratelimit-remaining-requests",
            request_reset_header="x-ratelimit-reset-requests",
            token_limit_header="x-ratelimit-limit-tokens",
            token_remaining_header="x-ratelimit-remaining-tokens",
            token_reset_header="x-ratelimit-reset-tokens",
        ),
        notes="Azure OpenAI. Base URL and model features are deployment-specific.",
    ),
    "bedrock": ProviderCapability(
        provider="bedrock",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.MULTIMODAL
            | Capability.REASONING
            | Capability.PROMPT_CACHING
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.EXPLICIT_BREAKPOINT,
        max_context_tokens=200_000,
        max_output_tokens=4096,
        supported_models=(
            "claude-3-5-sonnet",
            "claude-3-opus",
            "claude-3-sonnet",
            "llama-3-70b",
            "mistral-large",
        ),
        default_base_url="",
        cache=CacheSemantics(
            mode=CacheMode.EXPLICIT_BREAKPOINT,
            min_cache_tokens=1024,
            max_breakpoints=4,
            default_ttl_seconds=300,
            supported_ttl_seconds=(300, 3600),
            telemetry_fields=(
                "usage.CacheReadInputTokens",
                "usage.CacheWriteInputTokens",
                "usage.CacheDetails",
            ),
            notes="Bedrock prompt caching uses cachePoint/cache_control checkpoints; 5m default and 1h on supported models.",
        ),
        rate_limits=RateLimitSemantics(cache_hits_count_against_token_limits=False),
        notes="AWS Bedrock. Uses AWS SigV4 auth.",
    ),
    "gemini": ProviderCapability(
        provider="gemini",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.MULTIMODAL
            | Capability.PROMPT_CACHING
            | Capability.STRUCTURED_OUTPUT
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.EXPLICIT_CONTEXT,
        max_context_tokens=1_000_000,
        max_output_tokens=8192,
        supported_models=(
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ),
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        cache=CacheSemantics(
            mode=CacheMode.EXPLICIT_CONTEXT,
            min_cache_tokens=1024,
            default_ttl_seconds=3600,
            telemetry_fields=("usage_metadata.cachedContentTokenCount",),
            supports_explicit_resource=True,
            notes="Gemini supports implicit caching and explicit cachedContent resources; minimum tokens vary by model.",
        ),
        notes="OpenAI-compatible endpoint plus native cachedContents API. Multimodal native.",
    ),
    "vertex": ProviderCapability(
        provider="vertex",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.MULTIMODAL
            | Capability.PROMPT_CACHING
            | Capability.STRUCTURED_OUTPUT
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.EXPLICIT_CONTEXT,
        max_context_tokens=1_000_000,
        max_output_tokens=8192,
        supported_models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"),
        default_base_url="",
        cache=CacheSemantics(
            mode=CacheMode.EXPLICIT_CONTEXT,
            min_cache_tokens=2048,
            default_ttl_seconds=3600,
            telemetry_fields=("usageMetadata.cachedContentTokenCount",),
            supports_explicit_resource=True,
            notes="Vertex AI has implicit caching and explicit context cache resources; default explicit cache TTL is 60 minutes.",
        ),
        notes="Google Vertex AI Gemini. Base URL is project/location-specific.",
    ),
    "ollama-cloud": ProviderCapability(
        provider="ollama-cloud",
        capabilities=(
            Capability.CHAT_COMPLETIONS
            | Capability.STREAMING
            | Capability.TOOL_CALLS
            | Capability.REASONING
            | Capability.HTTP2
        ),
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        supported_models=("gpt-oss", "llama", "qwen", "mistral"),
        default_base_url="https://ollama.com",
        notes="Ollama cloud/native-compatible endpoint; model limits vary.",
    ),
    "together": ProviderCapability(
        provider="together",
        capabilities=Capability.CHAT_COMPLETIONS
        | Capability.STREAMING
        | Capability.TOOL_CALLS
        | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=8192,
        supported_models=("meta-llama", "mistral", "qwen"),
        default_base_url="https://api.together.xyz/v1",
        notes="OpenAI-compatible provider; cache semantics not exposed consistently.",
    ),
    "deepseek": ProviderCapability(
        provider="deepseek",
        capabilities=Capability.CHAT_COMPLETIONS
        | Capability.STREAMING
        | Capability.TOOL_CALLS
        | Capability.REASONING
        | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=8192,
        supported_models=("deepseek-chat", "deepseek-reasoner"),
        default_base_url="https://api.deepseek.com",
        notes="OpenAI-compatible provider with reasoning models.",
    ),
    "perplexity": ProviderCapability(
        provider="perplexity",
        capabilities=Capability.CHAT_COMPLETIONS | Capability.STREAMING | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=8192,
        supported_models=("sonar", "sonar-pro", "sonar-reasoning"),
        default_base_url="https://api.perplexity.ai",
        notes="OpenAI-compatible search/research provider; tool semantics differ.",
    ),
    "mistral": ProviderCapability(
        provider="mistral",
        capabilities=Capability.CHAT_COMPLETIONS
        | Capability.STREAMING
        | Capability.TOOL_CALLS
        | Capability.STRUCTURED_OUTPUT
        | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=8192,
        supported_models=("mistral-large", "mistral-small", "codestral"),
        default_base_url="https://api.mistral.ai/v1",
        notes="OpenAI-compatible chat surface for supported models.",
    ),
    "fireworks": ProviderCapability(
        provider="fireworks",
        capabilities=Capability.CHAT_COMPLETIONS
        | Capability.STREAMING
        | Capability.TOOL_CALLS
        | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=8192,
        supported_models=("accounts/fireworks/models/*",),
        default_base_url="https://api.fireworks.ai/inference/v1",
        notes="OpenAI-compatible hosted open-model provider.",
    ),
    "openrouter": ProviderCapability(
        provider="openrouter",
        capabilities=Capability.CHAT_COMPLETIONS
        | Capability.STREAMING
        | Capability.TOOL_CALLS
        | Capability.MULTIMODAL
        | Capability.HTTP2,
        cache_mode=CacheMode.ADAPTER_MANAGED,
        max_context_tokens=1_000_000,
        max_output_tokens=16_384,
        supported_models=("openrouter/*",),
        default_base_url="https://openrouter.ai/api/v1",
        cache=CacheSemantics(
            mode=CacheMode.ADAPTER_MANAGED,
            telemetry_fields=("usage.prompt_tokens_details.cached_tokens",),
            notes="Caching depends on the selected upstream provider and OpenRouter routing.",
        ),
        notes="Aggregator; concrete capabilities depend on selected upstream model/provider.",
    ),
    "cohere": ProviderCapability(
        provider="cohere",
        capabilities=Capability.CHAT_COMPLETIONS
        | Capability.STREAMING
        | Capability.TOOL_CALLS
        | Capability.STRUCTURED_OUTPUT
        | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=128_000,
        max_output_tokens=8192,
        supported_models=("command-r", "command-r-plus"),
        default_base_url="https://api.cohere.com/compatibility/v1",
        notes="OpenAI-compatible compatibility endpoint.",
    ),
    "ai21": ProviderCapability(
        provider="ai21",
        capabilities=Capability.CHAT_COMPLETIONS | Capability.STREAMING | Capability.HTTP2,
        cache_mode=CacheMode.NONE,
        max_context_tokens=256_000,
        max_output_tokens=8192,
        supported_models=("jamba",),
        default_base_url="https://api.ai21.com/studio/v1",
        notes="OpenAI-compatible chat support depends on model family.",
    ),
}


# =============================================================================
# Capability registry
# =============================================================================


class CapabilityRegistry:
    """Queryable database of provider capabilities."""

    def __init__(self, capabilities: dict[str, ProviderCapability] | None = None) -> None:
        self._caps = dict(capabilities) if capabilities else dict(_BUILTIN_CAPABILITIES)

    def get(self, provider: str) -> ProviderCapability | None:
        return self._caps.get(provider.lower())

    def supports(self, provider: str, cap: Capability) -> bool:
        pc = self._caps.get(provider.lower())
        return pc.supports(cap) if pc else False

    def cache_mode(self, provider: str) -> CacheMode:
        pc = self._caps.get(provider.lower())
        return pc.cache_mode if pc else CacheMode.NONE

    def cache_semantics(self, provider: str) -> CacheSemantics:
        pc = self._caps.get(provider.lower())
        return pc.cache if pc else CacheSemantics()

    def rate_limit_semantics(self, provider: str) -> RateLimitSemantics:
        pc = self._caps.get(provider.lower())
        return pc.rate_limits if pc else RateLimitSemantics()

    def supports_cache_ttl(self, provider: str, ttl_seconds: int) -> bool:
        return self.cache_semantics(provider).supports_ttl(ttl_seconds)

    def max_context(self, provider: str) -> int:
        pc = self._caps.get(provider.lower())
        return pc.max_context_tokens if pc else 128_000

    def list_providers(self) -> list[str]:
        return list(self._caps.keys())

    def providers_with(self, cap: Capability) -> list[str]:
        return [p for p, pc in self._caps.items() if pc.supports(cap)]

    def register(self, capability: ProviderCapability) -> None:
        self._caps[capability.provider.lower()] = capability

    def to_dict(self) -> dict[str, Any]:
        return {p: c.to_dict() for p, c in self._caps.items()}


# =============================================================================
# Global singleton (lazy)
# =============================================================================

_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry()
    return _registry


__all__ = [
    "Capability",
    "CacheMode",
    "CacheSemantics",
    "RateLimitSemantics",
    "ProviderCapability",
    "CapabilityRegistry",
    "get_capability_registry",
]
