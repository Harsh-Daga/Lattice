from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import status
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.gateway.compat.providers import (
    _WELL_KNOWN_PROVIDER_URLS,
    _resolve_passthrough_provider,
    _resolve_provider_upstream_url,
)
from lattice.transport.serialization import message_from_dict, message_to_dict
from lattice.transport.types import Request

Handler = Callable[..., Awaitable[Any]]


@dataclasses.dataclass(slots=True)
class ResponsesCompatDeps:
    """Dependencies for OpenAI Responses passthrough handlers."""

    responses_passthrough: Callable[..., Awaitable[Any]]
    provider: Any
    pipeline: Any = None
    config: Any = None
    logger: Any = None


async def models_passthrough(
    fastapi_request: Any,
    provider: Any,
    *,
    session_id: str | None = None,
) -> Any:
    """Passthrough /v1/models with simple header-based provider detection.

    The /v1/models endpoint (GET, empty body) provides zero signals for
    ProviderRouter's adapter-scoring system.  We use a simple header-based
    heuristic instead, ``model_metadata_provider`` pattern:
    Anthropic auth → api.anthropic.com, Gemini API key → generativelanguage,
    explicit header → named provider, else OpenAI.

    This is a pure passthrough — no compression, no pipeline, no transformation.
    """
    import logging

    _log = logging.getLogger("lattice.gateway.models")

    headers: dict[str, str] = {}
    for k, v in fastapi_request.headers.items():
        kl = k.lower()
        if kl in (
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "upgrade",
            "te",
            "trailer",
            "proxy-authenticate",
            "proxy-authorization",
        ):
            continue
        headers[k] = v

    provider_name = _resolve_passthrough_provider(headers, path="/v1/models")

    try:
        upstream_url = _resolve_provider_upstream_url(
            provider_name,
            "/v1/models",
            provider,
            query_params=str(fastapi_request.query_params),
        )
    except ValueError as exc:
        return JSONResponse(
            {"error": "provider_not_configured", "message": str(exc)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    base_url = provider.provider_base_urls.get(provider_name) or _WELL_KNOWN_PROVIDER_URLS.get(
        provider_name, ""
    )

    _log.debug("models_passthrough upstream=%s provider=%s", upstream_url, provider_name)

    client = provider.pool.get_client(provider_name, base_url)
    try:
        http_resp = await client.get(upstream_url, headers=headers)
    except httpx.TimeoutException:
        _log.warning("models_passthrough_timeout upstream=%s", upstream_url)
        return JSONResponse(
            {"error": "upstream_timeout"},
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        _log.warning("models_passthrough_error error=%s", exc)
        return JSONResponse(
            {"error": "upstream_error", "message": str(exc)},
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    response_headers = dict(http_resp.headers)
    response_headers.pop("content-encoding", None)
    response_headers.pop("content-length", None)
    for _hop in ("transfer-encoding", "connection", "keep-alive"):
        response_headers.pop(_hop, None)

    return StarletteResponse(
        content=http_resp.content,
        status_code=http_resp.status_code,
        headers=response_headers,
    )


def make_models_handler(deps: ResponsesCompatDeps) -> Handler:
    """Create /v1/models passthrough handler."""

    async def _handle_models(
        request: Any,
        x_lattice_session_id: str | None = None,
    ) -> Any:
        return await models_passthrough(
            request,
            deps.provider,
            session_id=x_lattice_session_id or "",
        )

    return _handle_models


def make_responses_handler(deps: ResponsesCompatDeps) -> Handler:
    """Create /v1/responses* passthrough handler with optional pipeline compression."""

    async def _handle_responses(
        method: str,
        request: Any,
        response_id: str | None = None,
        x_lattice_session_id: str | None = None,
        x_lattice_disable_transforms: str | None = None,
    ) -> Any:
        path = "/v1/responses" if response_id is None else f"/v1/responses/{response_id}"
        raw_body = await request.body() if method == "POST" else b""

        # GET/DELETE or no body or transforms disabled — pure passthrough
        if method != "POST" or not raw_body or x_lattice_disable_transforms or not deps.pipeline:
            return await deps.responses_passthrough(
                method,
                path,
                raw_body,
                request,
                deps.provider,
                session_id=x_lattice_session_id or "",
            )

        body_json: dict[str, Any] = {}
        try:
            body_json = json.loads(raw_body)
        except json.JSONDecodeError:
            return await deps.responses_passthrough(
                method,
                path,
                raw_body,
                request,
                deps.provider,
                session_id=x_lattice_session_id or "",
            )

        msgs = []
        for m in body_json.get("messages", body_json.get("input", [])):
            if isinstance(m, dict):
                msgs.append(message_from_dict(m))
        internal_request = Request(
            messages=msgs,
            model=body_json.get("model", ""),
        )
        ctx = TransformContext(
            request_id=str(time.time()),
            session_id=x_lattice_session_id or "",
            provider="openai",
            model=internal_request.model,
        )

        result = deps.pipeline.compress(internal_request, ctx)
        if is_err(result):
            if deps.config and getattr(deps.config, "graceful_degradation", False):
                if deps.logger:
                    deps.logger.warning("responses_pipeline_degraded", error=str(result))
                compressed = internal_request
            else:
                return JSONResponse(
                    {"error": "pipeline_failed", "message": "Transform error"},
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )
        else:
            compressed = unwrap(result)

        compressed_body = dict(body_json)
        if "messages" in compressed_body:
            compressed_body["messages"] = [message_to_dict(m) for m in compressed.messages]
        if "input" in compressed_body:
            compressed_body["input"] = [message_to_dict(m) for m in compressed.messages]
        passthrough_body = json.dumps(compressed_body).encode("utf-8")

        return await deps.responses_passthrough(
            method,
            path,
            passthrough_body,
            request,
            deps.provider,
            session_id=x_lattice_session_id or "",
        )

    return _handle_responses
