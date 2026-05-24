"""Model-string → adapter routing."""

from __future__ import annotations

from lattice.core.errors import ProviderError
from lattice.providers.adapters import (
    AI21Adapter,
    AnthropicAdapter,
    AzureAdapter,
    BedrockAdapter,
    CohereAdapter,
    DeepSeekAdapter,
    FireworksAdapter,
    GeminiAdapter,
    GroqAdapter,
    MistralAdapter,
    OllamaAdapter,
    OllamaCloudAdapter,
    OpenAIAdapter,
    OpenRouterAdapter,
    PerplexityAdapter,
    ProviderAdapter,
    TogetherAdapter,
    VertexAdapter,
)

_PROVIDER_ALIASES: dict[str, str] = {}


class ProviderRegistry:
    """Maps model strings to the correct ``ProviderAdapter``."""

    def __init__(self, adapters: list[ProviderAdapter] | None = None) -> None:
        self._adapters = list(adapters) if adapters else []
        if not self._adapters:
            self._adapters = [
                GroqAdapter(),
                TogetherAdapter(),
                DeepSeekAdapter(),
                PerplexityAdapter(),
                MistralAdapter(),
                FireworksAdapter(),
                OpenRouterAdapter(),
                CohereAdapter(),
                AI21Adapter(),
                GeminiAdapter(),
                VertexAdapter(),
                OllamaCloudAdapter(),
                OllamaAdapter(),
                AnthropicAdapter(),
                AzureAdapter(),
                BedrockAdapter(),
                OpenAIAdapter(),
            ]

    @property
    def adapters(self) -> list[ProviderAdapter]:
        return list(self._adapters)

    def resolve(self, model: str) -> ProviderAdapter:
        for adapter in self._adapters:
            if adapter.supports(model):
                return adapter
        raise ProviderError(
            provider="unknown",
            status_code=400,
            message=f"No provider adapter for model '{model}'",
        )

    def get_adapter(self, name: str) -> ProviderAdapter:
        resolved = _PROVIDER_ALIASES.get(name, name)
        for adapter in self._adapters:
            if adapter.name == resolved:
                return adapter
        raise ProviderError(
            provider=name,
            status_code=400,
            message=f"No adapter registered for provider '{name}'",
        )

    def list_adapters(self) -> list[str]:
        return [a.name for a in self._adapters]

    def iter_adapters(self):
        yield from self._adapters


def _resolve_provider_name(
    model: str,
    provider_name: str | None = None,
    registry: ProviderRegistry | None = None,
) -> str:
    """Resolve provider from explicit hint or model prefix."""
    if provider_name:
        return provider_name.lower()
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        reg = registry or ProviderRegistry()
        if prefix in reg.list_adapters():
            return prefix
    raise ProviderError(
        provider="unknown",
        status_code=400,
        message=(
            f"Provider not specified. Use either: "
            f"1) provider_name parameter, or "
            f"2) model prefix like 'groq/llama-3b' (got model='{model}')"
        ),
    )


def should_retry(status_code: int, retry_on: tuple[int, ...]) -> bool:
    return status_code in retry_on
