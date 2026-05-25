"""WebSocket chat completions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.gateway.compat.translation import deserialize_openai_request
from lattice.transport.serialization import message_to_dict

Handler = Callable[..., Awaitable[Any]]

__all__ = ["chat_completions_websocket_passthrough"]

_ws_lib: Any | None = None


def _ensure_ws_lib() -> None:
    global _ws_lib
    if _ws_lib is None:
        import websockets as _impl

        _ws_lib = _impl


async def chat_completions_websocket_passthrough(
    websocket: Any, *, logger: Any = None, pipeline: Any = None, provider: Any = None
) -> None:
    """Relay chat completions WS traffic through Lattice pipeline.

    Uses injected pipeline and provider from the runtime (not rebuilding)."""
    _ensure_ws_lib()
    await websocket.accept()

    import json as _json


    try:
        json_body = await websocket.receive_text()
        body = _json.loads(json_body)
    except (_json.JSONDecodeError, RuntimeError, KeyError):
        await websocket.close(code=1007, reason="invalid_json")
        return

    model = body.get("model", "")
    if not model:
        if logger:
            logger.warning("ws_chat_completions_no_model")
        await websocket.send_text(_json.dumps({"error": "model field is required"}))
        await websocket.close(code=1007, reason="missing model")
        return

    request = deserialize_openai_request(body)

    if pipeline is None:
        # Fallback: build a default pipeline (for standalone testing)
        from lattice.core.config import LatticeConfig
        from lattice.pipeline.factory import build_default_pipeline

        config = LatticeConfig.auto()
        pipeline = build_default_pipeline(config)

    provider_name = model.split("/")[0] if "/" in model else "openai"
    ctx = TransformContext(request_id=f"ws-chat-{model}", provider=provider_name, model=model)
    result = pipeline.compress(request, ctx)

    if is_err(result):
        await websocket.send_text(_json.dumps({"error": "pipeline_failed"}))
        await websocket.close(code=1011)
        return

    compressed = unwrap(result)
    compressed_messages = [message_to_dict(m) for m in compressed.messages]
    provider_name = provider

    from lattice.providers.credentials import CredentialResolver
    from lattice.providers.transport import DirectHTTPProvider, ProviderRegistry

    registry = ProviderRegistry()
    credentials = CredentialResolver()
    resolved = credentials.resolve(provider_name)
    provider_obj = DirectHTTPProvider(
        registry=registry, default_api_key=resolved.api_key, credentials=credentials
    )

    try:
        async for chunk in provider_obj.completion_stream(
            model=model,
            messages=compressed_messages,
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            provider_name=provider_name,
        ):
            if chunk is not None:
                await websocket.send_text(_json.dumps(chunk))
    except Exception as exc:
        if logger:
            logger.warning("ws_chat_completions_stream_error", error=str(exc))
    finally:
        await websocket.close()


