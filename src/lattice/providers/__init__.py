"""Provider adapters and direct HTTP transport for LATTICE.

Exports
-------
* ``ProviderAdapter`` — base protocol
* ``ProviderRegistry`` — model-string → adapter resolver
* ``DirectHTTPProvider`` — HTTP/2 connection pool + adapter dispatch
* ``ConnectionPoolManager`` — per-provider ``httpx.AsyncClient`` pooling
* ``OpenAIAdapter`` / ``AnthropicAdapter`` / ``OllamaAdapter`` / ``AzureAdapter`` / ``BedrockAdapter``
"""

from lattice.providers.anthropic import AnthropicAdapter
from lattice.providers.azure import AzureAdapter
from lattice.providers.base import ProviderAdapter
from lattice.providers.bedrock import BedrockAdapter
from lattice.providers.ollama import OllamaAdapter
from lattice.providers.openai import OpenAIAdapter
from lattice.providers.openai_compatible import AI21Adapter
from lattice.providers.transport import (
    ConnectionPoolManager,
    DirectHTTPProvider,
    ProviderRegistry,
)

__all__ = [
    "AI21Adapter",
    "AnthropicAdapter",
    "AzureAdapter",
    "BedrockAdapter",
    "ConnectionPoolManager",
    "DirectHTTPProvider",
    "OllamaAdapter",
    "OpenAIAdapter",
    "ProviderAdapter",
    "ProviderRegistry",
]
