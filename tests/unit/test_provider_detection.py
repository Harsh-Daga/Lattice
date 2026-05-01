"""Tests for proxy provider detection and routing.

Covers:
- _detect_provider with explicit provider hint and model prefix
- bare model rejection
"""

from __future__ import annotations

import pytest

from lattice.proxy.server import ProviderDetectionError, _detect_provider


class TestDetectProvider:
    def test_provider_hint_override(self) -> None:
        assert _detect_provider("gpt-4", provider_hint="ollama") == "ollama"
        assert _detect_provider("claude-3", provider_hint="openai") == "openai"
        assert _detect_provider("glm-5.1", provider_hint="anthropic") == "anthropic"

    def test_provider_hint_unknown_raises(self) -> None:
        with pytest.raises(ProviderDetectionError, match="Unknown provider hint"):
            _detect_provider("gpt-4", provider_hint="unknown-provider")

    def test_provider_prefix_in_model(self) -> None:
        assert _detect_provider("ollama/llama3") == "ollama"
        assert _detect_provider("anthropic/claude-3") == "anthropic"
        assert _detect_provider("openai/gpt-4") == "openai"
        assert _detect_provider("azure/gpt-4o") == "azure"
        assert _detect_provider("bedrock/claude-3") == "bedrock"
        assert _detect_provider("groq/llama-3.1-70b") == "groq"
        assert _detect_provider("together/llama-3") == "together"
        assert _detect_provider("deepseek/deepseek-chat") == "deepseek"
        assert _detect_provider("perplexity/sonar") == "perplexity"
        assert _detect_provider("mistral/mistral-small") == "mistral"
        assert _detect_provider("fireworks/llama-3.1-70b") == "fireworks"
        assert _detect_provider("openrouter/anthropic/claude-3") == "openrouter"
        assert _detect_provider("cohere/command-r") == "cohere"
        assert _detect_provider("ai21/jamba-1.5-mini") == "ai21"

    def test_provider_prefix_unknown_raises(self) -> None:
        with pytest.raises(ProviderDetectionError, match="Unknown provider prefix"):
            _detect_provider("unknown/llama3")

    def test_bare_model_raises(self) -> None:
        """Bare model names must raise — provider must be explicit."""
        with pytest.raises(ProviderDetectionError, match="Provider not specified"):
            _detect_provider("gpt-4o")
        with pytest.raises(ProviderDetectionError, match="Provider not specified"):
            _detect_provider("claude-3-sonnet")
        with pytest.raises(ProviderDetectionError, match="Provider not specified"):
            _detect_provider("llama3.1")
        with pytest.raises(ProviderDetectionError, match="Provider not specified"):
            _detect_provider("some-random-model")

    def test_priority_order(self) -> None:
        # hint beats prefix
        assert _detect_provider("ollama/llama3", provider_hint="openai") == "openai"
        # prefix is used when no hint
        assert _detect_provider("ollama/gpt-4o") == "ollama"

    def test_ollama_cloud_provider(self) -> None:
        assert _detect_provider("ollama-cloud/glm-5.1") == "ollama-cloud"
