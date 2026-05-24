"""ProviderRegistry registers all 17 adapters."""

from __future__ import annotations

from lattice.providers import ProviderRegistry


def test_all_17_providers_registered() -> None:
    reg = ProviderRegistry()
    expected = {
        "openai",
        "anthropic",
        "azure",
        "bedrock",
        "gemini",
        "vertex",
        "groq",
        "deepseek",
        "mistral",
        "cohere",
        "perplexity",
        "fireworks",
        "together",
        "openrouter",
        "ai21",
        "ollama",
        "ollama-cloud",
    }
    found = {a.name for a in reg.adapters}
    assert expected.issubset(found), f"Missing: {expected - found}"
