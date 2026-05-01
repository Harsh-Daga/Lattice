"""OpenAI-compatible provider adapters.

Most modern inference providers (Groq, Together, DeepSeek, Perplexity,
Mistral, Fireworks) speak the OpenAI chat-completions API with minor
variations in base URL and auth header format.

This module provides:
- ``OpenAICompatibleAdapter`` — generic base for any OpenAI-compatible provider
- Concrete adapters for popular providers with correct defaults

Routing Philosophy
------------------
* Models are portable — ``llama-3.1-70b`` can be on Groq, Together, or Fireworks.
* The caller decides the provider via ``provider_name`` or the ``provider/model`` prefix.
* ``map_model_name`` only strips the prefix and applies provider-native aliases.
"""

from __future__ import annotations

from typing import Any

from lattice.core.transport import Request, Response
from lattice.providers.base import _strip_provider_prefix
from lattice.providers.openai import OpenAIAdapter

# =============================================================================
# Generic base
# =============================================================================


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Generic adapter for any provider that speaks OpenAI's API.

    Override ``name``, ``_PREFIXES``, and ``_DEFAULT_BASE_URL`` for each
    concrete provider.
    """

    name = "openai-compatible"
    _PREFIXES: set[str] = set()
    _DEFAULT_BASE_URL: str = ""

    def supports(self, model: str) -> bool:
        if "/" not in model:
            return False
        prefix = model.split("/", 1)[0].lower()
        return prefix in self._PREFIXES

    def chat_endpoint(self, _model: str, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/v1/chat/completions"

    def map_model_name(self, model: str) -> str:
        """Strip the provider prefix if present.

        The caller is responsible for providing the exact model ID the
        provider expects.  We do not maintain provider-native aliases here.
        """
        return _strip_provider_prefix(model, self._PREFIXES)

    def retry_config(self) -> dict[str, Any]:
        return {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_on": (429, 502, 503, 504),
        }

    def extra_headers(self, _request: Request) -> dict[str, str]:
        return {}


# =============================================================================
# Concrete adapters
# =============================================================================


class GroqAdapter(OpenAICompatibleAdapter):
    """Groq — ultra-fast inference (OpenAI-compatible).

    Docs: https://console.groq.com/docs/openai
    """

    name = "groq"
    _PREFIXES = {"groq"}
    _DEFAULT_BASE_URL = "https://api.groq.com/openai"

    def retry_config(self) -> dict[str, Any]:
        # Groq is very fast but rate-limits aggressively
        return {
            "max_retries": 5,
            "backoff_factor": 0.5,
            "retry_on": (429, 502, 503, 504),
        }


class TogetherAdapter(OpenAICompatibleAdapter):
    """Together AI — inference API for open-source models.

    Docs: https://docs.together.ai/docs/openai-api-compatibility
    """

    name = "together"
    _PREFIXES = {"together"}
    _DEFAULT_BASE_URL = "https://api.together.xyz"


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """DeepSeek — Chinese provider, very cost-effective.

    Docs: https://platform.deepseek.com/api-docs
    """

    name = "deepseek"
    _PREFIXES = {"deepseek"}
    _DEFAULT_BASE_URL = "https://api.deepseek.com"

    def serialize_request(self, request: Request) -> dict[str, Any]:
        body = super().serialize_request(request)
        # DeepSeek supports reasoning_effort via metadata
        reasoning_effort = request.metadata.get("reasoning_effort")
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        return body

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """DeepSeek includes ``reasoning_content`` in its responses."""
        resp = super().deserialize_response(data)
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        reasoning = msg.get("reasoning_content")
        if reasoning:
            resp.metadata["reasoning"] = reasoning
        return resp


class PerplexityAdapter(OpenAICompatibleAdapter):
    """Perplexity — search-augmented LLM API.

    Docs: https://docs.perplexity.ai/guides/getting-started
    """

    name = "perplexity"
    _PREFIXES = {"perplexity"}
    _DEFAULT_BASE_URL = "https://api.perplexity.ai"

    def serialize_request(self, request: Request) -> dict[str, Any]:
        body = super().serialize_request(request)
        # Perplexity supports search_recency_filter and return_images
        search_filter = request.metadata.get("search_recency_filter")
        if search_filter:
            body["search_recency_filter"] = search_filter
        return_images = request.metadata.get("return_images")
        if return_images is not None:
            body["return_images"] = return_images
        return body


class MistralAdapter(OpenAICompatibleAdapter):
    """Mistral AI — European LLM provider.

    Docs: https://docs.mistral.ai/api/
    """

    name = "mistral"
    _PREFIXES = {"mistral"}
    _DEFAULT_BASE_URL = "https://api.mistral.ai"


class FireworksAdapter(OpenAICompatibleAdapter):
    """Fireworks AI — fast inference for open-source models.

    Docs: https://docs.fireworks.ai/getting-started/openai-compatibility
    """

    name = "fireworks"
    _PREFIXES = {"fireworks"}
    _DEFAULT_BASE_URL = "https://api.fireworks.ai/inference"


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter — unified API for 100+ providers.

    Docs: https://openrouter.ai/docs
    """

    name = "openrouter"
    _PREFIXES = {"openrouter"}
    _DEFAULT_BASE_URL = "https://openrouter.ai/api"

    def serialize_request(self, request: Request) -> dict[str, Any]:
        body = super().serialize_request(request)
        # OpenRouter supports extra headers for routing and cost control
        if request.metadata.get("openrouter_provider"):
            body["provider"] = {"order": [request.metadata["openrouter_provider"]]}
        if request.metadata.get("openrouter_max_price"):
            body["max_price"] = request.metadata["openrouter_max_price"]
        return body

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        h = super().auth_headers(api_key)
        # OpenRouter recommends these headers for tracking
        h["HTTP-Referer"] = "https://lattice.dev"
        h["X-Title"] = "LATTICE"
        return h

    def extra_headers(self, request: Request) -> dict[str, str]:
        h: dict[str, str] = {}
        if request.metadata.get("openrouter_provider"):
            h["X-OpenRouter-Provider"] = request.metadata["openrouter_provider"]
        return h


class CohereAdapter(OpenAICompatibleAdapter):
    """Cohere — Command models (OpenAI-compatible beta).

    Docs: https://docs.cohere.com/docs/openai-compatibility
    """

    name = "cohere"
    _PREFIXES = {"cohere"}
    _DEFAULT_BASE_URL = "https://api.cohere.com/compatibility/v1"


class AI21Adapter(OpenAICompatibleAdapter):
    """AI21 Labs — Jurassic / Jamba models (OpenAI-compatible).

    Docs: https://docs.ai21.com/reference/jamba-instruct-api
    """

    name = "ai21"
    _PREFIXES = {"ai21"}
    _DEFAULT_BASE_URL = "https://api.ai21.com/studio/v1"

    def chat_endpoint(self, _model: str, base_url: str) -> str:
        # AI21 uses /chat/completions under the compatibility endpoint
        return f"{base_url.rstrip('/')}/chat/completions"
