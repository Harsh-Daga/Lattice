"""LATTICE provider layer.

Public API for users:
    from lattice.providers import DirectHTTPProvider, ProviderRegistry
    from lattice.providers import (
        OpenAIAdapter, AnthropicAdapter, ..., AI21Adapter,
    )
    from lattice.providers import (
        Capability, CacheMode, CacheSemantics, ProviderCapability,
    )
"""

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
    OpenAICompatibleAdapter,
    OpenRouterAdapter,
    PerplexityAdapter,
    ProviderAdapter,
    TogetherAdapter,
    VertexAdapter,
)
from lattice.providers.capabilities import (
    CacheMode,
    CacheSemantics,
    Capability,
    ProviderCapability,
    RateLimitSemantics,
)
from lattice.providers.credentials import CredentialResolver, ProviderCredentials
from lattice.providers.mcp_to_anthropic import convert_mcp_to_anthropic, is_mcp_tool
from lattice.providers.schema_filter import sanitize_json_schema, sanitize_tool_definitions
from lattice.providers.stream_state import AnthropicStreamState, StreamResult
from lattice.providers.tool_sanitizer import (
    AnthropicToolSanitizer,
    BedrockToolSanitizer,
    ToolSanitizer,
    restore_tool_call_ids,
    sanitize_tool_ids,
)
from lattice.transport import (
    ConnectionPoolManager,
    DirectHTTPProvider,
    ProviderRegistry,
    RateLimitTracker,
    StreamStallDetector,
    _resolve_provider_name,
)

__all__ = [
    "DirectHTTPProvider",
    "ProviderRegistry",
    "ConnectionPoolManager",
    "RateLimitTracker",
    "StreamStallDetector",
    "_resolve_provider_name",
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
    "Capability",
    "CacheMode",
    "CacheSemantics",
    "RateLimitSemantics",
    "ProviderCapability",
    "AnthropicStreamState",
    "StreamResult",
    "ToolSanitizer",
    "AnthropicToolSanitizer",
    "BedrockToolSanitizer",
    "sanitize_tool_ids",
    "restore_tool_call_ids",
    "sanitize_json_schema",
    "sanitize_tool_definitions",
    "convert_mcp_to_anthropic",
    "is_mcp_tool",
    "ProviderCredentials",
    "CredentialResolver",
]
