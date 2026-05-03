"""Route registration helpers for proxy app factory."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from fastapi import FastAPI, Header, WebSocket, status
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

from lattice.gateway.compat import (
    AnthropicCompatDeps,
    ChatCompatDeps,
    HTTPCompatHandler,
    ResponsesCompatDeps,
    make_anthropic_handler,
    make_chat_completion_handler,
    make_models_handler,
    make_responses_handler,
)
from lattice.gateway.server import ClientConnectionInfo, LLMTPGateway


def register_native_lattice_routes(app: FastAPI, gateway: LLMTPGateway) -> None:
    """Register native LATTICE session and gateway endpoints."""

    @app.post("/lattice/session/start")
    async def lattice_session_start(body: dict[str, Any]) -> dict[str, Any]:
        return await gateway.handle_session_start(body)

    @app.post("/lattice/session/append")
    async def lattice_session_append(body: dict[str, Any]) -> Any:
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse(
                {"error": "missing_session_id"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        result = await gateway.handle_session_append(body)
        if result is None:
            return JSONResponse(
                {"error": "session_not_found", "session_id": session_id},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return result

    @app.get("/lattice/session/{session_id}")
    async def lattice_session_get(session_id: str) -> Any:
        session = await gateway.handle_session_get(session_id)
        if session is None:
            return JSONResponse(
                {"error": "session_not_found"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return session

    @app.post("/lattice/session/invalidate")
    async def lattice_session_invalidate(body: dict[str, Any]) -> Any:
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse(
                {"error": "missing_session_id"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        result = await gateway.handle_session_invalidate(body)
        if result is None:
            return JSONResponse(
                {"error": "session_not_found", "session_id": session_id},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return result

    @app.post("/lattice/gateway")
    async def lattice_gateway(fastapi_request: FastAPIRequest) -> Any:
        raw_body = await fastapi_request.body()
        client = fastapi_request.client
        client_info = {
            "client_id": "",
            "remote_addr": client.host if client else "",
            "user_agent": fastapi_request.headers.get("user-agent", ""),
        }
        output, response_meta = await gateway.handle_request(
            raw_body,
            dict(fastapi_request.headers.items()),
            client_info=ClientConnectionInfo(**client_info),
        )
        response_headers: dict[str, str] = {}
        if "x-lattice-framing" in response_meta:
            response_headers["x-lattice-framing"] = response_meta["x-lattice-framing"]
        if raw_body[:4] == b"LATT":
            return StarletteResponse(
                content=output, media_type="application/octet-stream", headers=response_headers
            )
        try:
            return JSONResponse(json.loads(output.decode("utf-8")), headers=response_headers)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return StarletteResponse(
                content=output, media_type="application/json", headers=response_headers
            )


@dataclasses.dataclass(slots=True)
class ProviderCompatRouteDeps:
    """Dependencies for provider compatibility route registration."""

    config: Any
    pipeline: Any
    provider: Any
    session_manager: Any
    batching_engine: Any
    speculative_executor: Any
    semantic_cache: Any
    cost_estimator: Any
    auto_continuation: Any
    agent_stats: Any
    metrics: Any
    logger: Any
    deserialize_openai_request: Any
    serialize_messages: Any
    serialize_openai_response: Any
    build_routing_headers: Any
    detect_new_messages: Any
    get_cache_planner: Any
    message_cls: Any
    provider_timeout_error: Any
    provider_error: Any
    sse_done: str
    anthropic_passthrough: Any
    responses_passthrough: Any
    responses_websocket_passthrough: Any
    chat_completions_websocket_passthrough: Any = None
    maintenance: Any = None


def register_provider_compat_routes(
    app: FastAPI,
    compat: HTTPCompatHandler,
    deps: ProviderCompatRouteDeps,
) -> None:
    """Register OpenAI/Anthropic compatibility routes and handlers."""
    compat.chat_completion_handler = make_chat_completion_handler(
        ChatCompatDeps(
            config=deps.config,
            pipeline=deps.pipeline,
            provider=deps.provider,
            session_manager=deps.session_manager,
            batching_engine=deps.batching_engine,
            speculative_executor=deps.speculative_executor,
            semantic_cache=deps.semantic_cache,
            cost_estimator=deps.cost_estimator,
            auto_continuation=deps.auto_continuation,
            agent_stats=deps.agent_stats,
            metrics=deps.metrics,
            logger=deps.logger,
            deserialize_openai_request=deps.deserialize_openai_request,
            serialize_messages=deps.serialize_messages,
            serialize_openai_response=deps.serialize_openai_response,
            build_routing_headers=deps.build_routing_headers,
            detect_new_messages=deps.detect_new_messages,
            get_cache_planner=deps.get_cache_planner,
            message_cls=deps.message_cls,
            provider_timeout_error=deps.provider_timeout_error,
            provider_error=deps.provider_error,
            sse_done=deps.sse_done,
            maintenance=deps.maintenance,
        )
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: dict[str, Any],
        x_lattice_session_id: str | None = Header(default=None),
        x_lattice_disable_transforms: str | None = Header(default=None),
        x_lattice_client_profile: str | None = Header(default=None),
        x_lattice_provider: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        return await compat.handle_chat_completion(
            body,
            x_lattice_session_id=x_lattice_session_id,
            x_lattice_disable_transforms=x_lattice_disable_transforms,
            x_lattice_client_profile=x_lattice_client_profile,
            x_lattice_provider=x_lattice_provider,
            authorization=authorization,
            x_api_key=x_api_key,
        )

    compat.anthropic_handler = make_anthropic_handler(
        AnthropicCompatDeps(
            anthropic_passthrough=deps.anthropic_passthrough,
            provider=deps.provider,
        )
    )

    @app.post("/v1/messages")
    async def messages_post(
        fastapi_request: FastAPIRequest,
        x_lattice_session_id: str | None = Header(default=None),
    ) -> Any:
        return await compat.handle_anthropic_message(
            fastapi_request, x_lattice_session_id=x_lattice_session_id
        )

    responses_deps = ResponsesCompatDeps(
        responses_passthrough=deps.responses_passthrough,
        provider=deps.provider,
    )
    compat.models_handler = make_models_handler(responses_deps)
    compat.responses_handler = make_responses_handler(responses_deps)

    @app.get("/v1/models")
    async def models_get(request: FastAPIRequest) -> Any:
        return await compat.handle_models(request)

    @app.post("/v1/responses")
    async def responses_post(request: FastAPIRequest) -> Any:
        return await compat.handle_responses_api("POST", request)

    @app.get("/v1/responses/{response_id:path}")
    async def responses_get(request: FastAPIRequest, response_id: str) -> Any:
        return await compat.handle_responses_api("GET", request, response_id)

    @app.delete("/v1/responses/{response_id:path}")
    async def responses_delete(request: FastAPIRequest, response_id: str) -> Any:
        return await compat.handle_responses_api("DELETE", request, response_id)

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket) -> None:
        logger = getattr(deps, "logger", None)
        if logger is not None:
            try:
                logger.info(
                    "responses_websocket_route_entered",
                    path=str(getattr(getattr(websocket, "url", None), "path", "")),
                    headers={k: v for k, v in websocket.headers.items()},
                )
            except Exception:
                pass
        await deps.responses_websocket_passthrough(websocket)

    @app.websocket("/v1/responses/")
    async def responses_websocket_slash(websocket: WebSocket) -> None:
        await responses_websocket(websocket)

    @app.websocket("/v1/chat/completions")
    async def chat_completions_websocket_route(websocket: WebSocket) -> None:
        """WebSocket handler for chat completions (Codex CLI, custom clients).

        Accepts a WS upgrade, reads the first text frame as the JSON body,
        processes it through the Lattice pipeline, proxies to the upstream
        provider via SSE, and pipes events back as WS text frames.
        """
        logger = getattr(deps, "logger", None)
        if logger is not None:
            try:
                logger.info(
                    "chat_completions_ws_route_entered",
                    headers={k: v for k, v in websocket.headers.items()},
                )
            except Exception:
                pass
        await deps.chat_completions_websocket_passthrough(websocket)

    responses_aliases = (
        "/v1/codex/responses",
        "/v1/codex/responses/",
        "/backend-api/responses",
        "/backend-api/responses/",
        "/backend-api/codex/responses",
        "/backend-api/codex/responses/",
    )
    for base_path in responses_aliases:
        app.add_api_route(base_path, responses_post, methods=["POST"])
        app.add_api_route(f"{base_path}/{{response_id:path}}", responses_get, methods=["GET"])
        app.add_api_route(
            f"{base_path}/{{response_id:path}}",
            responses_delete,
            methods=["DELETE"],
        )
        app.add_api_websocket_route(base_path, responses_websocket)
