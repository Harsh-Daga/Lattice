"""Codex WebSocket passthrough handler.

Robust relay between Codex client and OpenAI Responses upstream.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from typing import Any

import websockets as _ws_lib

from lattice.integrations.codex.auth import (
    _is_codex_jwt,
    _resolve_codex_routing_headers,
)

_WS_FIRST_FRAME_TIMEOUT = 60.0
_WS_CONNECT_RETRY_MAX = 3
_WS_CONNECT_TIMEOUT = 30.0
_WS_FALLBACK_HTTP_TIMEOUT = 120.0


async def responses_websocket_passthrough(websocket: Any, *, logger: Any) -> None:
    """Relay WebSocket traffic between client and OpenAI Responses upstream.

    Supports both standard OpenAI Responses and Codex-specific routing:
    * Standard → ``wss://api.openai.com/v1/responses``
    * Codex     → ``wss://chatgpt.com/backend-api/codex/responses``
      (detected via JWT ``https://api.openai.com/auth.chatgpt_account_id`` claim)

    Robust implementation:
    * Subprotocol forwarding (client → upstream)
    * First-frame timeout (60s) so zombie clients don't hold slots
    * OpenAI-Beta header injection (required by OpenAI)
    * Env OPENAI_API_KEY fallback when client doesn't send auth
    * WebSocket → HTTP/SSE fallback when upstream WS fails (prevents
      Codex from exhausting its WS retry budget)
    * Configurable retry with jitter for upstream connect
    * Graceful bidirectional relay with deterministic cleanup
    """
    request_id = f"ws-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # 1. Extract client headers & subprotocol
    # ------------------------------------------------------------------
    ws_headers = dict(websocket.headers)

    client_subprotocols: list[str] = []
    raw_protocol = ws_headers.get("sec-websocket-protocol", "")
    if raw_protocol:
        client_subprotocols = [p.strip() for p in raw_protocol.split(",") if p.strip()]

    # Accept client connection with the requested subprotocol
    if client_subprotocols:
        await websocket.accept(subprotocol=client_subprotocols[0])
    else:
        await websocket.accept()

    # ------------------------------------------------------------------
    # 2. Build upstream headers (strip hop-by-hop, add routing)
    # ------------------------------------------------------------------
    _skip_headers = frozenset(
        {
            "host",
            "connection",
            "upgrade",
            "sec-websocket-key",
            "sec-websocket-version",
            "sec-websocket-extensions",
            "sec-websocket-accept",
            "sec-websocket-protocol",
            "content-length",
            "transfer-encoding",
        }
    )
    # Also strip any headers that _resolve_codex_routing_headers will re-add
    # with canonical casing, to avoid duplicate headers with mixed case.
    _headers_to_dedup = frozenset({
        "chatgpt-account-id",
        "openai-beta",
        "authorization",
    })
    upstream_headers: dict[str, str] = {}
    for k, v in ws_headers.items():
        kl = k.lower()
        if kl not in _skip_headers and kl not in _headers_to_dedup:
            upstream_headers[k] = v

    # Resolve Codex JWT vs standard API key routing
    auth = ws_headers.get("authorization", "")
    openai_beta = ws_headers.get("openai-beta", "")
    chatgpt_account_id = ws_headers.get("chatgpt-account-id", "")

    is_chatgpt_auth = False
    if auth:
        resolved = _resolve_codex_routing_headers(auth, openai_beta, chatgpt_account_id)
        upstream_headers.update(resolved)
        is_chatgpt_auth = _is_codex_jwt(auth)

    if is_chatgpt_auth:
        upstream_url = "wss://chatgpt.com/backend-api/codex/responses"
        logger.info("codex_websocket_routing", upstream=upstream_url, request_id=request_id)
    else:
        # Standard OpenAI API
        base = os.environ.get("LATTICE_OPENAI_BASE_URL", "https://api.openai.com")
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        upstream_url = f"{ws_base.rstrip('/')}/v1/responses"
        logger.info("openai_websocket_routing", upstream=upstream_url, request_id=request_id)

    # Fallback auth from env if client didn't send Authorization
    _lower_headers = {k.lower(): v for k, v in upstream_headers.items()}
    if "authorization" not in _lower_headers:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            upstream_headers["Authorization"] = f"Bearer {api_key}"
            logger.debug("ws_auth_fallback_env", request_id=request_id)
        else:
            logger.warning(
                "ws_no_auth",
                request_id=request_id,
                message="No Authorization header from client and OPENAI_API_KEY not set",
            )

    # Inject required OpenAI-Beta header if missing — OpenAI returns 500 without it
    if "openai-beta" not in _lower_headers:
        upstream_headers["OpenAI-Beta"] = "responses_websockets=2026-02-06"

    logger.debug(
        "ws_upstream_headers",
        request_id=request_id,
        header_keys=[k for k in upstream_headers if k.lower() != "authorization"],
        subprotocols=client_subprotocols,
    )

    # ------------------------------------------------------------------
    # 3. Receive first client frame with timeout
    # ------------------------------------------------------------------
    first_msg_raw: str | None = None
    try:
        first_msg_raw = await asyncio.wait_for(
            websocket.receive_text(),
            timeout=_WS_FIRST_FRAME_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.info(
            "ws_first_frame_timeout",
            request_id=request_id,
            timeout=_WS_FIRST_FRAME_TIMEOUT,
        )
        with contextlib.suppress(Exception):
            await websocket.close(code=1001, reason="first-frame timeout")
        return
    except Exception as exc:
        logger.warning("ws_first_frame_error", request_id=request_id, error=str(exc))
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="first-frame error")
        return

    # ------------------------------------------------------------------
    # 4. Connect upstream with retry + jitter
    # ------------------------------------------------------------------
    upstream_ws = None
    last_err: Exception | None = None

    for attempt in range(_WS_CONNECT_RETRY_MAX):
        try:
            use_ssl: bool | None = True if upstream_url.startswith("wss://") else None
            upstream_ws = await _ws_lib.connect(
                upstream_url,
                additional_headers=upstream_headers,
                subprotocols=(
                    [_ws_lib.Subprotocol(p) for p in client_subprotocols]
                    if client_subprotocols and hasattr(_ws_lib, "Subprotocol")
                    else client_subprotocols or None
                ),
                ssl=use_ssl,
                open_timeout=_WS_CONNECT_TIMEOUT,
                close_timeout=10,
                ping_interval=20,
                ping_timeout=20,
            )
            break
        except Exception as exc:
            last_err = exc
            logger.warning(
                "ws_upstream_connect_attempt_failed",
                request_id=request_id,
                attempt=attempt + 1,
                max_attempts=_WS_CONNECT_RETRY_MAX,
                error=str(exc),
            )
            if attempt < _WS_CONNECT_RETRY_MAX - 1:
                delay = min(2 ** attempt, 8) + (time.time() % 1)
                await asyncio.sleep(delay)

    if upstream_ws is None:
        logger.error(
            "ws_upstream_connect_failed_all_attempts",
            request_id=request_id,
            error=str(last_err),
        )
        # WS → HTTP/SSE fallback
        fallback_http_url = upstream_url.replace("wss://", "https://").replace("ws://", "http://")
        await _ws_http_fallback(
            websocket,
            first_msg_raw,
            fallback_http_url,
            upstream_headers,
            is_chatgpt_auth,
            logger,
            request_id,
        )
        return

    try:
        await upstream_ws.send(first_msg_raw)
        logger.debug("ws_first_frame_sent_upstream", request_id=request_id)
    except Exception as exc:
        logger.warning("ws_first_frame_send_failed", request_id=request_id, error=str(exc))
        with contextlib.suppress(Exception):
            await upstream_ws.close()
        fallback_http_url = upstream_url.replace("wss://", "https://").replace("ws://", "http://")
        await _ws_http_fallback(
            websocket,
            first_msg_raw,
            fallback_http_url,
            upstream_headers,
            is_chatgpt_auth,
            logger,
            request_id,
        )
        return

    # ------------------------------------------------------------------
    # 5. Bidirectional relay — FIRST_COMPLETED so one side exiting immediately
    #    cancels the other (prevents hung connections when upstream closes).
    # ------------------------------------------------------------------
    async def _client_to_upstream() -> None:
        try:
            while True:
                msg = await websocket.receive_text()
                await upstream_ws.send(msg)
        except asyncio.CancelledError:
            raise
        except _ws_lib.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass

    async def _upstream_to_client() -> None:
        try:
            while True:
                msg = await upstream_ws.recv()
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_text(msg)
        except asyncio.CancelledError:
            raise
        except _ws_lib.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass

    c2u_task = asyncio.create_task(_client_to_upstream(), name="c2u")
    u2c_task = asyncio.create_task(_upstream_to_client(), name="u2c")
    try:
        done, pending = await asyncio.wait(
            {c2u_task, u2c_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        # Re-raise CancelledError so callers can propagate it
        for t in done:
            exc = t.exception()
            if isinstance(exc, asyncio.CancelledError):
                raise exc
    finally:
        for t in {c2u_task, u2c_task}:
            if not t.done():
                t.cancel()
        with contextlib.suppress(Exception):
            await upstream_ws.close()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _ws_http_fallback(
    websocket: Any,
    first_msg_raw: str,
    upstream_url: str,
    upstream_headers: dict[str, str],
    is_chatgpt_auth: bool,
    logger: Any,
    request_id: str,
) -> None:
    """Fall back to HTTP POST streaming when upstream WebSocket fails.

    Converts the WS ``response.create`` message to an HTTP POST to
    ``/v1/responses`` (or chatgpt equivalent), reads SSE events, and relays
    each ``data:`` line as a WS text message to the client.
    """
    import json as _json

    import httpx

    body: dict[str, Any]
    try:
        parsed = _json.loads(first_msg_raw)
    except (_json.JSONDecodeError, TypeError):
        parsed = {}

    # Normalize envelope: {"type":"response.create","response":{...}} → inner dict
    if isinstance(parsed, dict) and isinstance(parsed.get("response"), dict):
        body = dict(parsed["response"])
    elif isinstance(parsed, dict):
        body = dict(parsed)
        if body.get("type") == "response.create":
            body.pop("type", None)
    else:
        body = {}

    if body.get("type") in {"response.create", "response"}:
        body.pop("type", None)

    body["stream"] = True

    http_headers = dict(upstream_headers)
    http_headers["content-type"] = "application/json"
    http_headers.pop("openai-beta", None)

    logger.debug("ws_http_fallback_start", request_id=request_id, url=upstream_url)

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                upstream_url,
                headers=http_headers,
                json=body,
                timeout=_WS_FALLBACK_HTTP_TIMEOUT,
            ) as resp:
                if resp.status_code != 200:
                    error_body = b""
                    async for chunk in resp.aiter_bytes():
                        error_body += chunk
                        if len(error_body) > 2000:
                            break
                    logger.error(
                        "ws_http_fallback_error_status",
                        request_id=request_id,
                        status=resp.status_code,
                        body=error_body.decode("utf-8", errors="replace")[:500],
                    )
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "server_error",
                            "message": f"Upstream returned {resp.status_code}",
                        },
                    }
                    await websocket.send_text(_json.dumps(error_event))
                    return

                # Relay SSE data: lines as WS text messages
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                continue
                            try:
                                await websocket.send_text(data)
                            except Exception:
                                return

                # Flush remainder
                for line in buffer.strip().splitlines():
                    line = line.strip()
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        with contextlib.suppress(Exception):
                            await websocket.send_text(line[6:])
    except Exception as exc:
        logger.error("ws_http_fallback_failed", request_id=request_id, error=str(exc))
        error_event = {
            "type": "error",
            "error": {
                "type": "server_error",
                "message": f"Fallback failed: {exc!s}"[:200],
            },
        }
        with contextlib.suppress(Exception):
            await websocket.send_text(_json.dumps(error_event))
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
