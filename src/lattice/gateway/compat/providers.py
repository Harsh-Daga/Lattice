from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

Handler = Callable[..., Awaitable[Any]]

_PROVIDER_FALLBACK_BASE_URLS: dict[str, str] = {}

_WELL_KNOWN_PROVIDER_URLS: dict[str, str] = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com/openai",
    "together": "https://api.together.xyz",
    "deepseek": "https://api.deepseek.com",
    "perplexity": "https://api.perplexity.com",
    "mistral": "https://api.mistral.ai",
    "fireworks": "https://api.fireworks.ai",
    "openrouter": "https://openrouter.ai/api",
    "cohere": "https://api.cohere.com",
    "ai21": "https://api.ai21.com",
}


def _resolve_provider_upstream_url(
    provider_name: str,
    path: str,
    provider: Any,
    *,
    query_params: str = "",
) -> str:
    """Resolve upstream URL with multi-tier fallback for passthrough endpoints.

    Resolution order:
    1. ``provider_base_urls`` (config / env)
    2. ``_PROVIDER_FALLBACK_BASE_URLS`` (runtime overrides)
    3. ``default_api_base`` (global default from config)
    4. ``_WELL_KNOWN_PROVIDER_URLS`` (hardcoded well-known endpoints)
    5. Raise ValueError
    """
    base_url = provider.provider_base_urls.get(provider_name)
    if not base_url:
        base_url = _PROVIDER_FALLBACK_BASE_URLS.get(provider_name)
    if not base_url:
        base_url = getattr(provider, "default_api_base", None) or ""
    if not base_url:
        base_url = _WELL_KNOWN_PROVIDER_URLS.get(provider_name, "")
    if not base_url:
        raise ValueError(
            f"No base URL configured for provider '{provider_name}'. "
            f"Set provider_base_urls['{provider_name}'] or "
            f"LATTICE_PROVIDER_BASE_URLS env var."
        )
    upstream_url = f"{base_url.rstrip('/')}{path}"
    if query_params:
        upstream_url = f"{upstream_url}?{query_params}"
    return upstream_url


def _resolve_passthrough_provider(
    headers: dict[str, str],
    *,
    path: str = "",
    body: dict[str, Any] | None = None,
) -> str:
    """Simple header-based provider detection for metadata/lightweight endpoints.

    ``model_metadata_provider`` pattern: detect Anthropic
    auth signals first, then Gemini, then explicit header, else OpenAI.
    """
    from lattice.gateway.routing import RequestSignals

    signals = RequestSignals(
        method="GET",
        path=path,
        headers=headers,
        body=body or {},
        model="",
    )

    # 1. Explicit header wins (operator override)
    explicit = signals.headers.get("x-lattice-provider")
    if explicit:
        return explicit.strip().lower()

    # 2. Gemini API key — exclusive Google signal
    if signals.headers.get("x-goog-api-key"):
        return "gemini"

    # 3. Anthropic auth signals — strongest match first
    auth = signals.headers.get("authorization", "")
    if auth.startswith("Bearer sk-ant-"):
        return "anthropic"
    if signals.headers.get("anthropic-version"):
        return "anthropic"
    if signals.headers.get("anthropic-beta"):
        return "anthropic"

    # 4. x-api-key alone is shared across many providers — only match
    #    when combined with anthropic-version or anthropic-beta above.
    #    Standing alone it's too ambiguous.

    # 5. Default — OpenAI (the most common /v1/models consumer)
    return "openai"


def _prepare_codex_upstream_headers(
    headers: dict[str, str],
) -> dict[str, str]:
    """Prepare upstream headers for Codex (ChatGPT session) requests.

    Injects ``ChatGPT-Account-ID`` from the JWT claim for upstream routing.
    This is a no-op when the auth is not a Codex JWT.
    """
    from lattice.integrations.codex.auth import _is_codex_jwt, _resolve_codex_routing_headers

    auth = headers.get("authorization", "")
    if not _is_codex_jwt(auth):
        return headers

    resolved = _resolve_codex_routing_headers(
        auth,
        headers.get("openai-beta", ""),
        headers.get("chatgpt-account-id", ""),
    )
    return {**headers, **resolved}
