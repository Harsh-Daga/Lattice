"""Provider adapters — one per provider family.

17 providers across 8 files (some adapters live in the same module — e.g. Gemini+Vertex).
"""

from lattice.providers.adapters.anthropic import AnthropicAdapter
from lattice.providers.adapters.azure import AzureAdapter
from lattice.providers.adapters.base import (
    ProviderAdapter,
    _format_sse_event,
    _pop_system,
    _remap_tool_choice,
    _remap_tools,
    _strip_provider_prefix,
)
from lattice.providers.adapters.bedrock import BedrockAdapter
from lattice.providers.adapters.gemini import GeminiAdapter, VertexAdapter
from lattice.providers.adapters.ollama import OllamaAdapter, OllamaCloudAdapter
from lattice.providers.adapters.openai import OpenAIAdapter
from lattice.providers.adapters.openai_compatible import (
    AI21Adapter,
    CohereAdapter,
    DeepSeekAdapter,
    FireworksAdapter,
    GroqAdapter,
    MistralAdapter,
    OpenAICompatibleAdapter,
    OpenRouterAdapter,
    PerplexityAdapter,
    TogetherAdapter,
)

__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "GroqAdapter",
    "TogetherAdapter",
    "DeepSeekAdapter",
    "PerplexityAdapter",
    "MistralAdapter",
    "FireworksAdapter",
    "OpenRouterAdapter",
    "CohereAdapter",
    "AI21Adapter",
    "AnthropicAdapter",
    "AzureAdapter",
    "BedrockAdapter",
    "GeminiAdapter",
    "VertexAdapter",
    "OllamaAdapter",
    "OllamaCloudAdapter",
    "_pop_system",
    "_remap_tool_choice",
    "_remap_tools",
    "_strip_provider_prefix",
    "_format_sse_event",
]
