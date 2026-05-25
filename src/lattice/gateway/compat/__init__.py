"""HTTP compatibility — public surface preserved at lattice.gateway.compat."""

from __future__ import annotations

from lattice.gateway.compat.anthropic_handler import (
    AnthropicCompatDeps,
    make_anthropic_handler,
)
from lattice.gateway.compat.anthropic_messages import (
    compress_anthropic_body,
    deserialize_anthropic_request,
    deserialize_anthropic_response,
    extract_anthropic_text_blocks,
    replace_anthropic_text_blocks,
    serialize_anthropic_response,
)
from lattice.gateway.compat.anthropic_passthrough import anthropic_passthrough
from lattice.gateway.compat.handler import HTTPCompatHandler
from lattice.gateway.compat.headers import (
    _extract_cached_tokens,
    _runtime_header_values,
    _usage_total_tokens,
    build_routing_headers,
)
from lattice.gateway.compat.openai_chat import (
    chat_completions_websocket_passthrough,
    make_chat_completion_handler,
)
from lattice.gateway.compat.openai_chat_deps import ChatCompatDeps
from lattice.gateway.compat.operational import (
    OperationalRouteDeps,
    build_proxy_stats_payload,
    register_operational_routes,
)
from lattice.gateway.compat.providers import (
    _PROVIDER_FALLBACK_BASE_URLS,
    _WELL_KNOWN_PROVIDER_URLS,
    _prepare_codex_upstream_headers,
    _resolve_passthrough_provider,
    _resolve_provider_upstream_url,
)
from lattice.gateway.compat.responses_body import (
    compress_responses_body,
    extract_responses_text_blocks,
    replace_responses_text_blocks,
)
from lattice.gateway.compat.responses_handler import (
    ResponsesCompatDeps,
    make_models_handler,
    make_responses_handler,
    models_passthrough,
)
from lattice.gateway.compat.responses_passthrough import (
    responses_passthrough,
    responses_websocket_passthrough,
)
from lattice.gateway.compat.translation import (
    deserialize_openai_request,
    detect_new_messages,
    is_local_origin,
    serialize_messages,
    serialize_openai_response,
)

Handler = __import__(
    "collections.abc", fromlist=["Callable"]
).Callable[..., __import__("typing").Awaitable[__import__("typing").Any]]

__all__ = [
    "AnthropicCompatDeps",
    "ChatCompatDeps",
    "Handler",
    "HTTPCompatHandler",
    "OperationalRouteDeps",
    "ResponsesCompatDeps",
    "_PROVIDER_FALLBACK_BASE_URLS",
    "_WELL_KNOWN_PROVIDER_URLS",
    "_extract_cached_tokens",
    "_prepare_codex_upstream_headers",
    "_resolve_passthrough_provider",
    "_resolve_provider_upstream_url",
    "_runtime_header_values",
    "_usage_total_tokens",
    "anthropic_passthrough",
    "build_proxy_stats_payload",
    "build_routing_headers",
    "chat_completions_websocket_passthrough",
    "compress_anthropic_body",
    "compress_responses_body",
    "deserialize_anthropic_request",
    "deserialize_anthropic_response",
    "deserialize_openai_request",
    "detect_new_messages",
    "extract_anthropic_text_blocks",
    "extract_responses_text_blocks",
    "is_local_origin",
    "make_anthropic_handler",
    "make_chat_completion_handler",
    "make_models_handler",
    "make_responses_handler",
    "models_passthrough",
    "register_operational_routes",
    "replace_anthropic_text_blocks",
    "replace_responses_text_blocks",
    "responses_passthrough",
    "responses_websocket_passthrough",
    "serialize_anthropic_response",
    "serialize_messages",
    "serialize_openai_response",
]
