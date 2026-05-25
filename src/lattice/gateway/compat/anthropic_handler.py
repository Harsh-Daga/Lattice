from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from fastapi import status
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response as StarletteResponse

from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.gateway.compat.anthropic_messages import (
    deserialize_anthropic_request,
    deserialize_anthropic_response,
    serialize_anthropic_response,
)
from lattice.gateway.compat.headers import build_routing_headers
from lattice.gateway.compat.providers import _WELL_KNOWN_PROVIDER_URLS
from lattice.proxy.middleware import attach_routing_headers
from lattice.telemetry.downgrade import TransportOutcome
from lattice.transport.serialization import message_to_dict

Handler = Callable[..., Awaitable[Any]]


@dataclasses.dataclass(slots=True)
class AnthropicCompatDeps:
    """Dependencies for Anthropic messages passthrough handler."""

    anthropic_passthrough: Callable[..., Awaitable[Any]]
    provider: Any
    pipeline: Any = None
    config: Any = None
    session_manager: Any = None
    logger: Any = None


def make_anthropic_handler(deps: AnthropicCompatDeps) -> Handler:
    """Create Anthropic messages passthrough handler."""

    async def _handle_anthropic_message(
        fastapi_request: Any,
        x_lattice_session_id: str | None = None,
        x_lattice_disable_transforms: str | None = None,
    ) -> Any:
        raw_body = await fastapi_request.body()

        if not deps.pipeline or x_lattice_disable_transforms:
            return await deps.anthropic_passthrough(
                "POST",
                "/v1/messages",
                raw_body,
                fastapi_request,
                deps.provider,
                session_id=x_lattice_session_id or "",
            )

        body_json: dict[str, Any] = {}
        try:
            body_json = json.loads(raw_body)
        except json.JSONDecodeError:
            return await deps.anthropic_passthrough(
                "POST",
                "/v1/messages",
                raw_body,
                fastapi_request,
                deps.provider,
                session_id=x_lattice_session_id or "",
            )

        request = deserialize_anthropic_request(body_json)
        ctx = TransformContext(
            request_id=str(time.time()),
            session_id=x_lattice_session_id or "",
            provider="anthropic",
            model=request.model or body_json.get("model", ""),
        )
        ctx.session_state["client_profile"] = "default"

        result = deps.pipeline.compress(request, ctx)
        if is_err(result):
            if deps.config and getattr(deps.config, "graceful_degradation", False):
                if deps.logger:
                    deps.logger.warning("anthropic_pipeline_degraded", error=str(result))
                compressed_request = request
            else:
                return JSONResponse(
                    {
                        "type": "error",
                        "error": {
                            "type": "pipeline_failed",
                            "message": "Transform error — set graceful_degradation=true to continue",
                        },
                    },
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )
        else:
            compressed_request = unwrap(result)

        compressed_tokens = compressed_request.token_estimate
        original_tokens = request.token_estimate

        compressed_body = dict(body_json)
        compressed_messages = [message_to_dict(m) for m in compressed_request.messages]
        compressed_body["messages"] = compressed_messages
        passthrough_body = json.dumps(compressed_body).encode("utf-8")

        streaming = compressed_request.stream
        model_used = compressed_request.model or body_json.get("model", "")

        is_streaming = streaming
        client = deps.provider.pool.get_client(
            "anthropic",
            deps.provider.provider_base_urls.get("anthropic")
            or _WELL_KNOWN_PROVIDER_URLS.get("anthropic", ""),
        )
        upstream_url = (
            deps.provider.provider_base_urls.get("anthropic")
            or _WELL_KNOWN_PROVIDER_URLS.get("anthropic", "")
        ).rstrip("/") + "/v1/messages"
        http_version = deps.provider.pool.get_http_version(
            "anthropic",
            deps.provider.provider_base_urls.get("anthropic")
            or _WELL_KNOWN_PROVIDER_URLS.get("anthropic", ""),
        )

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

        if is_streaming:

            async def _stream_relay() -> AsyncIterator[str]:
                try:
                    async with client.stream(
                        "POST", upstream_url, content=passthrough_body, headers=headers
                    ) as resp:
                        if not resp.is_success:
                            error_body = await resp.aread()
                            if deps.logger:
                                deps.logger.error(
                                    "anthropic_handler_stream_error",
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
                    if deps.logger:
                        deps.logger.error("anthropic_handler_stream_timeout", error=str(exc))
                    error_payload = {
                        "type": "error",
                        "error": {"type": "timeout_error", "message": str(exc)},
                    }
                    yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                except httpx.HTTPError as exc:
                    if deps.logger:
                        deps.logger.error(
                            "anthropic_handler_stream_http_error",
                            error=str(exc),
                            error_type=type(exc).__name__,
                        )
                    error_payload = {
                        "type": "error",
                        "error": {"type": "upstream_error", "message": str(exc)},
                    }
                    yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

            transport_outcome = TransportOutcome(http_version=http_version)
            attach_routing_headers(
                fastapi_request,
                ctx,
                build_routing_headers(
                    model_used=model_used,
                    compressed_tokens=compressed_tokens,
                    original_tokens=original_tokens,
                    session_id=x_lattice_session_id or "",
                    transport_outcome=transport_outcome,
                ),
            )
            return StreamingResponse(
                _stream_relay(),
                media_type="text/event-stream",
            )
        else:
            http_resp = await client.request(
                "POST", upstream_url, content=passthrough_body, headers=headers
            )
            try:
                resp_json = json.loads(http_resp.content)
            except json.JSONDecodeError:
                resp_json = {}
            if resp_json and http_resp.is_success and deps.pipeline:
                internal_response = deserialize_anthropic_response(resp_json)
                internal_response = deps.pipeline.reverse(internal_response, ctx)
                resp_json = serialize_anthropic_response(internal_response, compressed_request)
                response_body = json.dumps(resp_json).encode("utf-8")
            else:
                response_body = http_resp.content

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
            transport_outcome = TransportOutcome(http_version=http_version)
            attach_routing_headers(
                fastapi_request,
                ctx,
                build_routing_headers(
                    model_used=model_used,
                    compressed_tokens=compressed_tokens,
                    original_tokens=original_tokens,
                    session_id=x_lattice_session_id or "",
                    transport_outcome=transport_outcome,
                ),
            )
            return StarletteResponse(
                content=response_body,
                status_code=http_resp.status_code,
                headers=response_headers,
            )

    return _handle_anthropic_message
