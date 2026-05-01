"""Phase 2: Strict Provider Contract Layer tests.

Verifies exact provider and model routing, no fallback guessing,
no hidden defaults.
"""

from __future__ import annotations

import pytest

from lattice.core.errors import ProviderError
from lattice.gateway.compat import ProviderDetectionError, detect_provider
from lattice.providers.capabilities import (
    Capability,
    CacheMode,
    get_capability_registry,
)
from lattice.providers.transport import (
    DirectHTTPProvider,
    _resolve_provider_name,
)


# =============================================================================
# Exact provider resolution
# =============================================================================

class TestExactProviderResolution:
    def test_detect_from_model_prefix(self) -> None:
        assert detect_provider("openai/gpt-4") == "openai"
        assert detect_provider("anthropic/claude-3") == "anthropic"
        assert detect_provider("ollama/llama3") == "ollama"

    def test_detect_from_explicit_hint(self) -> None:
        assert detect_provider("gpt-4", provider_hint="openai") == "openai"
        assert detect_provider("claude-3", provider_hint="anthropic") == "anthropic"

    def test_bare_model_rejected(self) -> None:
        with pytest.raises(ProviderDetectionError):
            detect_provider("gpt-4")
        with pytest.raises(ProviderDetectionError):
            detect_provider("claude-3-sonnet")

    def test_unknown_prefix_rejected(self) -> None:
        with pytest.raises(ProviderDetectionError):
            detect_provider("unknown/model")

    def test_resolve_provider_name_explicit(self) -> None:
        assert _resolve_provider_name("openai/gpt-4", "anthropic") == "anthropic"
        assert _resolve_provider_name("openai/gpt-4", "openai") == "openai"

    def test_resolve_provider_name_from_model(self) -> None:
        assert _resolve_provider_name("groq/llama-3b") == "groq"
        assert _resolve_provider_name("ollama/llama3") == "ollama"

    def test_resolve_provider_name_bare_model_fails(self) -> None:
        with pytest.raises(ProviderError):
            _resolve_provider_name("gpt-4")


# =============================================================================
# Missing provider config / base URL
# =============================================================================

class TestMissingProviderConfig:
    def test_missing_base_url_raises(self) -> None:
        provider = DirectHTTPProvider()
        with pytest.raises(ProviderError) as exc_info:
            provider._resolve_base_url("nonexistent-provider")
        assert "No base URL" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_missing_api_key_raises(self) -> None:
        provider = DirectHTTPProvider()
        with pytest.raises(ProviderError) as exc_info:
            provider._resolve_api_key("openai")
        assert "No API key" in str(exc_info.value)
        assert exc_info.value.status_code == 401

    def test_ollama_no_auth_allowed(self) -> None:
        provider = DirectHTTPProvider()
        key = provider._resolve_api_key("ollama")
        assert key == ""

    def test_explicit_api_base_overrides_default(self) -> None:
        provider = DirectHTTPProvider()
        base = provider._resolve_base_url("openai", api_base="https://custom.example.com")
        assert base == "https://custom.example.com"

    def test_per_provider_base_url_override(self) -> None:
        provider = DirectHTTPProvider(
            provider_base_urls={"openai": "https://proxy.example.com"}
        )
        base = provider._resolve_base_url("openai")
        assert base == "https://proxy.example.com"


# =============================================================================
# ollama vs ollama-cloud distinction
# =============================================================================

class TestOllamaDistinction:
    def test_ollama_local_base_url(self) -> None:
        provider = DirectHTTPProvider()
        base = provider._resolve_base_url("ollama")
        assert base == "http://127.0.0.1:11434"

    def test_ollama_cloud_base_url(self) -> None:
        provider = DirectHTTPProvider()
        base = provider._resolve_base_url("ollama-cloud")
        assert base == "https://ollama.com"

    def test_ollama_cloud_different_from_ollama(self) -> None:
        provider = DirectHTTPProvider()
        local_base = provider._resolve_base_url("ollama")
        cloud_base = provider._resolve_base_url("ollama-cloud")
        assert local_base != cloud_base

    def test_ollama_prefix_resolution(self) -> None:
        assert detect_provider("ollama/llama3") == "ollama"
        assert detect_provider("ollama-cloud/llama3") == "ollama-cloud"


# =============================================================================
# Capability registry coverage
# =============================================================================

class TestCapabilityRegistry:
    def test_openai_capabilities(self) -> None:
        registry = get_capability_registry()
        cap = registry.get("openai")
        assert cap is not None
        assert cap.supports(Capability.CHAT_COMPLETIONS)
        assert cap.supports(Capability.STREAMING)
        assert cap.supports(Capability.TOOL_CALLS)
        assert cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.AUTO_PREFIX

    def test_anthropic_capabilities(self) -> None:
        registry = get_capability_registry()
        cap = registry.get("anthropic")
        assert cap is not None
        assert cap.supports(Capability.CHAT_COMPLETIONS)
        assert cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.EXPLICIT_BREAKPOINT
        assert cap.cache.max_breakpoints == 4

    def test_ollama_no_caching(self) -> None:
        registry = get_capability_registry()
        cap = registry.get("ollama")
        assert cap is not None
        assert not cap.supports(Capability.PROMPT_CACHING)
        assert cap.cache_mode == CacheMode.NONE

    def test_unknown_provider_returns_none(self) -> None:
        registry = get_capability_registry()
        assert registry.get("nonexistent") is None

    def test_providers_with_capability(self) -> None:
        registry = get_capability_registry()
        providers = registry.providers_with(Capability.PROMPT_CACHING)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" not in providers

    def test_max_context_tokens(self) -> None:
        registry = get_capability_registry()
        assert registry.max_context("openai") == 128_000
        assert registry.max_context("anthropic") == 200_000
        assert registry.max_context("nonexistent") == 128_000  # default


# =============================================================================
# Adapter purity
# =============================================================================

class TestAdapterPurity:
    def test_adapters_do_not_mutate_input(self) -> None:
        from lattice.providers.openai import OpenAIAdapter
        from lattice.core.transport import Message, Request

        adapter = OpenAIAdapter()
        original = Request(
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        serialized = adapter.serialize_request(original)
        # The adapter should return a new dict without mutating the request
        assert isinstance(serialized, dict)
        assert original.model == "gpt-4"

    def test_adapter_supports_check(self) -> None:
        from lattice.providers.openai import OpenAIAdapter
        from lattice.providers.anthropic import AnthropicAdapter

        openai = OpenAIAdapter()
        anthropic = AnthropicAdapter()

        assert openai.supports("openai/gpt-4")
        assert not openai.supports("anthropic/claude-3")

        assert anthropic.supports("anthropic/claude-3")
        assert not anthropic.supports("openai/gpt-4")
