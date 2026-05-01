"""Tests for provider capabilities database."""

from lattice.providers.capabilities import (
    CacheMode,
    CacheSemantics,
    Capability,
    CapabilityRegistry,
    ProviderCapability,
    RateLimitSemantics,
    get_capability_registry,
)


class TestCapabilityFlags:
    def test_combination(self):
        caps = Capability.CHAT_COMPLETIONS | Capability.STREAMING
        assert caps & Capability.CHAT_COMPLETIONS
        assert caps & Capability.STREAMING
        assert not (caps & Capability.TOOL_CALLS)

    def test_all_providers_have_chat(self):
        reg = CapabilityRegistry()
        for provider in reg.list_providers():
            assert reg.supports(provider, Capability.CHAT_COMPLETIONS)


class TestCacheMode:
    def test_values(self):
        assert CacheMode.AUTO_PREFIX.value == "auto_prefix"
        assert CacheMode.EXPLICIT_BREAKPOINT.value == "explicit"
        assert CacheMode.EXPLICIT_CONTEXT.value == "explicit_context"
        assert CacheMode.NONE.value == "none"


class TestCapabilityRegistry:
    def test_get_openai(self):
        reg = CapabilityRegistry()
        cap = reg.get("openai")
        assert cap is not None
        assert cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.AUTO_PREFIX

    def test_get_anthropic(self):
        reg = CapabilityRegistry()
        cap = reg.get("anthropic")
        assert cap is not None
        assert cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.EXPLICIT_BREAKPOINT

    def test_get_ollama(self):
        reg = CapabilityRegistry()
        cap = reg.get("ollama")
        assert cap is not None
        assert not cap.supports(Capability.PROMPT_CACHING)

    def test_bedrock_prompt_caching(self):
        reg = CapabilityRegistry()
        cap = reg.get("bedrock")
        assert cap is not None
        assert cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.EXPLICIT_BREAKPOINT
        assert cap.cache.max_breakpoints == 4

    def test_gemini_context_cache(self):
        reg = CapabilityRegistry()
        cap = reg.get("gemini")
        assert cap is not None
        assert cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.EXPLICIT_CONTEXT
        assert cap.cache.supports_explicit_resource is True

    def test_get_unknown(self):
        reg = CapabilityRegistry()
        assert reg.get("unknown_provider") is None

    def test_supports(self):
        reg = CapabilityRegistry()
        assert reg.supports("openai", Capability.MULTIMODAL)
        assert not reg.supports("groq", Capability.MULTIMODAL)
        assert not reg.supports("unknown", Capability.STREAMING)

    def test_cache_mode(self):
        reg = CapabilityRegistry()
        assert reg.cache_mode("openai") == CacheMode.AUTO_PREFIX
        assert reg.cache_mode("anthropic") == CacheMode.EXPLICIT_BREAKPOINT
        assert reg.cache_mode("ollama") == CacheMode.NONE
        assert reg.cache_mode("unknown") == CacheMode.NONE

    def test_cache_semantics(self):
        reg = CapabilityRegistry()
        openai_cache = reg.cache_semantics("openai")
        assert openai_cache.min_cache_tokens == 1024
        assert openai_cache.hit_increment_tokens == 128
        assert openai_cache.cache_key_param == "prompt_cache_key"

        anthropic_cache = reg.cache_semantics("anthropic")
        assert anthropic_cache.max_breakpoints == 4
        assert anthropic_cache.supports_ttl(300)
        assert anthropic_cache.supports_ttl(3600)

    def test_rate_limit_semantics(self):
        reg = CapabilityRegistry()
        openai_limits = reg.rate_limit_semantics("openai")
        assert openai_limits.token_remaining_header == "x-ratelimit-remaining-tokens"
        assert reg.rate_limit_semantics("unknown").retry_after_header == "retry-after"

    def test_supports_cache_ttl(self):
        reg = CapabilityRegistry()
        assert reg.supports_cache_ttl("anthropic", 3600)
        assert not reg.supports_cache_ttl("anthropic", 7200)

    def test_max_context(self):
        reg = CapabilityRegistry()
        assert reg.max_context("openai") == 128_000
        assert reg.max_context("anthropic") == 200_000
        assert reg.max_context("gemini") == 1_000_000
        assert reg.max_context("unknown") == 128_000

    def test_providers_with(self):
        reg = CapabilityRegistry()
        multimodal = reg.providers_with(Capability.MULTIMODAL)
        assert "openai" in multimodal
        assert "anthropic" in multimodal
        assert "groq" not in multimodal

    def test_register_custom(self):
        reg = CapabilityRegistry()
        custom = ProviderCapability(
            provider="custom",
            capabilities=Capability.CHAT_COMPLETIONS,
            cache_mode=CacheMode.NONE,
            max_context_tokens=4096,
            max_output_tokens=512,
            supported_models=("custom-model",),
            default_base_url="https://custom.example.com",
            cache=CacheSemantics(),
            rate_limits=RateLimitSemantics(),
        )
        reg.register(custom)
        assert reg.get("custom") is not None
        assert reg.max_context("custom") == 4096

    def test_to_dict(self):
        reg = CapabilityRegistry()
        d = reg.to_dict()
        assert "openai" in d
        assert "capabilities" in d["openai"]
        assert isinstance(d["openai"]["capabilities"], list)
        assert "cache" in d["openai"]
        assert "rate_limits" in d["openai"]

    def test_singleton(self):
        r1 = get_capability_registry()
        r2 = get_capability_registry()
        assert r1 is r2


class TestProviderCapability:
    def test_to_dict(self):
        cap = ProviderCapability(
            provider="test",
            capabilities=Capability.CHAT_COMPLETIONS | Capability.STREAMING,
            cache_mode=CacheMode.NONE,
            max_context_tokens=1000,
            max_output_tokens=100,
            supported_models=("m1",),
            default_base_url="http://test",
        )
        d = cap.to_dict()
        assert d["provider"] == "test"
        assert "CHAT_COMPLETIONS" in d["capabilities"]
        assert "STREAMING" in d["capabilities"]
        assert "TOOL_CALLS" not in d["capabilities"]
