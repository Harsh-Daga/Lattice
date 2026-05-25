from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import status
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response as StarletteResponse

from lattice.proxy.middleware import attach_routing_headers
from lattice.telemetry.downgrade import TransportOutcome

Handler = Callable[..., Awaitable[Any]]

from lattice.gateway.compat.headers import build_routing_headers
from lattice.gateway.compat.providers import (
    _WELL_KNOWN_PROVIDER_URLS,
    _resolve_provider_upstream_url,
)


async def anthropic_passthrough(
    method: str,
    path: str,
    body: bytes,
    fastapi_request: Any,
    provider: Any,
    *,
    compressed_tokens: int = 0,
    original_tokens: int = 0,
    logger: Any,
    session_id: str | None = None,
) -> Any:
    """Forward Anthropic Messages API requests to the upstream provider.

    Provider resolution is delegated to :class:`ProviderRouter`.  There are
    no hardcoded defaults — if no adapter matches, the request fails with a
    descriptive 400.
    """
    # ------------------------------------------------------------------
    # 1. Collect headers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Parse body
    # ------------------------------------------------------------------
    body_json: dict[str, Any] | None = None
    if body:
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError as exc:
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_json",
                        "message": f"Request body is not valid JSON: {exc}",
                    },
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    # ------------------------------------------------------------------
    # 3. Provider detection — zero defaults
    # ------------------------------------------------------------------
    from lattice.gateway.routing import (
        ProviderAmbiguityError,
        ProviderNotDetectedError,
        ProviderRouter,
        RequestSignals,
    )

    router = ProviderRouter(provider.registry)
    signals = RequestSignals.from_request(
        method=method,
        path=path,
        headers=headers,
        body=body_json or {},
        model=body_json.get("model", "") if body_json else "",
    )

    try:
        result = router.resolve(signals)
    except (ProviderNotDetectedError, ProviderAmbiguityError) as exc:
        logger.warning("anthropic_provider_detection_failed", error=str(exc))
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "provider_detection_failed",
                    "message": str(exc),
                },
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    provider_name = result.provider

    # ------------------------------------------------------------------
    # 4. Resolve upstream URL with multi-tier fallback
    # ------------------------------------------------------------------
    try:
        upstream_url = _resolve_provider_upstream_url(
            provider_name,
            path,
            provider,
            query_params=str(fastapi_request.query_params),
        )
    except ValueError as exc:
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "provider_not_configured",
                    "message": str(exc),
                },
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    base_url = provider.provider_base_urls.get(provider_name) or _WELL_KNOWN_PROVIDER_URLS.get(
        provider_name, ""
    )

    # ------------------------------------------------------------------
    # 5. Derive model_used for routing headers
    # ------------------------------------------------------------------
    # 5. Derive model_used from actual request
    # ------------------------------------------------------------------
    model_used = body_json.get("model", "") if body_json else ""
    if "/" in model_used:
        model_used = model_used.split("/", 1)[1]
    if not model_used:
        model_used = "unknown"

    # ------------------------------------------------------------------
    # 6. Streaming detection
    # ------------------------------------------------------------------
    is_streaming = False
    if body_json:
        is_streaming = body_json.get("stream", False)

    logger.info(
        "anthropic_passthrough_start",
        url=upstream_url,
        provider=provider_name,
        confidence=result.confidence.name,
        reason=result.reason,
        is_streaming=is_streaming,
        has_auth=bool(headers.get("authorization")),
        body_bytes=len(body),
    )

    client = provider.pool.get_client(provider_name, base_url)
    http_version = provider.pool.get_http_version(provider_name, base_url)

    # ------------------------------------------------------------------
    # 7. Streaming path
    # ------------------------------------------------------------------
    if is_streaming:

        async def _stream_relay() -> Any:
            try:
                async with client.stream(
                    method, upstream_url, content=body, headers=headers
                ) as resp:
                    if not resp.is_success:
                        error_body = await resp.aread()
                        logger.error(
                            "anthropic_passthrough_stream_error",
                            status_code=resp.status_code,
                            error_body=error_body.decode("utf-8", errors="replace")[:500],
                        )
                        error_payload = {
                            "type": "error",
                            "error": {
                                "type": "upstream_error",
                                "message": f"HTTP {resp.status_code}",
                            },
                        }
                        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                        return
                    async for chunk in resp.aiter_text():
                        yield chunk
            except httpx.TimeoutException as exc:
                logger.error(
                    "anthropic_passthrough_stream_timeout",
                    error=str(exc),
                )
                error_payload = {
                    "type": "error",
                    "error": {"type": "timeout_error", "message": str(exc)},
                }
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            except httpx.HTTPError as exc:
                logger.error(
                    "anthropic_passthrough_stream_http_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                error_payload = {
                    "type": "error",
                    "error": {"type": "upstream_error", "message": str(exc)},
                }
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

        transport_outcome = TransportOutcome(
            http_version=http_version,
        )
        attach_routing_headers(
            fastapi_request,
            None,
            build_routing_headers(
                model_used=model_used,
                compressed_tokens=compressed_tokens,
                original_tokens=original_tokens,
                session_id=session_id or "",
                transport_outcome=transport_outcome,
            ),
        )
        return StreamingResponse(
            _stream_relay(),
            media_type="text/event-stream",
        )

    # ------------------------------------------------------------------
    # 8. Non-streaming path
    # ------------------------------------------------------------------
    try:
        http_resp = await client.request(method, upstream_url, content=body, headers=headers)
    except httpx.TimeoutException as exc:
        logger.error("anthropic_passthrough_timeout", error=str(exc))
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "timeout_error",
                    "message": str(exc),
                },
            },
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.error(
            "anthropic_passthrough_http_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "upstream_error",
                    "message": str(exc),
                },
            },
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    response_headers = {
        k: v
        for k, v in http_resp.headers.items()
        if k.lower()
        in (
            "content-type",
            "x-request-id",
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-tokens-remaining",
        )
    }
    transport_outcome = TransportOutcome(
        http_version=http_version,
    )
    attach_routing_headers(
        fastapi_request,
        None,
        build_routing_headers(
            model_used=model_used,
            compressed_tokens=compressed_tokens,
            original_tokens=original_tokens,
            session_id=session_id or "",
            transport_outcome=transport_outcome,
        ),
    )
    return StarletteResponse(
        content=http_resp.content,
        status_code=http_resp.status_code,
        headers=response_headers,
    )


