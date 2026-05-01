"""Proxy client SDK for LATTICE.

Provides an HTTP client that talks to a running LATTICE proxy server.
This is the recommended mode for production use when you want:
- Centralized compression on the proxy
- Session persistence across clients
- Batching and speculative execution
- Metrics and observability

Usage:
    >>> from lattice.sdk import LatticeProxyClient
    >>> client = LatticeProxyClient(base_url="http://localhost:8787")
    >>> response = await client.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello"}],
    ... )
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx
import structlog

from lattice.protocol.dictionary_codec import DictionaryCodec
from lattice.protocol.framing import BinaryFramer, FrameFlags, FrameType

logger = structlog.get_logger()


# =============================================================================
# LatticeProxyClient
# =============================================================================


class LatticeProxyClient:
    """HTTP client for a LATTICE proxy server.

    Mirrors the OpenAI SDK interface while adding LATTICE-specific
    headers for session management, provider routing, and transform control.

    Attributes:
        base_url: Proxy server base URL (e.g. ``http://localhost:8787``).
        api_key: Optional API key for proxy authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8787",
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._framer = BinaryFramer()
        self._dictionary_codecs: dict[str, DictionaryCodec] = {}
        self._log = logger.bind(module="lattice_proxy_client")

        self.chat = _ProxyChatCompletionNamespace(self)
        self.sessions = _ProxySessionNamespace(self)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build request headers."""
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    async def health(self) -> dict[str, Any]:
        """Check proxy health."""
        resp = await self._client.get(f"{self.base_url}/healthz")
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    async def stats(self) -> dict[str, Any]:
        """Get proxy compression statistics."""
        resp = await self._client.get(f"{self.base_url}/stats")
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())


# =============================================================================
# Proxy chat completion namespace
# =============================================================================


class _ProxyChatCompletionNamespace:
    """Proxy-backed chat.completions namespace."""

    def __init__(self, client: LatticeProxyClient) -> None:
        self._client = client

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
        session_id: str | None = None,
        provider: str | None = None,
        disable_transforms: bool = False,
        client_profile: str | None = None,
        native_wire: bool = False,
        dictionary_wire: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request through the LATTICE proxy.

        Args:
            model: Model identifier.
            messages: OpenAI-compatible message list.
            stream: Whether to stream the response.
            session_id: Optional LATTICE session ID.
            provider: Explicit provider hint (e.g. ``"groq"``).
            disable_transforms: If True, skip LATTICE compression.
            client_profile: Client profile for adaptive transforms.
            **kwargs: Additional OpenAI-compatible params.

        Returns:
            OpenAI-compatible response dict.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stop is not None:
            body["stop"] = stop
        body.update(kwargs)

        headers: dict[str, str] = {}
        if session_id:
            headers["x-lattice-session-id"] = session_id
        if provider:
            headers["x-lattice-provider"] = provider
        if disable_transforms:
            headers["x-lattice-disable-transforms"] = "true"
        if client_profile:
            headers["x-lattice-client-profile"] = client_profile

        if native_wire:
            return await self._client.sessions._gateway_request(
                body,
                session_id=session_id,
                headers=headers,
                dictionary_wire=dictionary_wire,
            )

        resp = await self._client._client.post(
            f"{self._client.base_url}/v1/chat/completions",
            json=body,
            headers=self._client._headers(headers),
        )
        resp.raise_for_status()
        result = cast(dict[str, Any], resp.json())
        result["transport"] = {
            "framing": "json",
            "delta": resp.headers.get("x-lattice-delta", "bypassed"),
            "http_version": resp.headers.get("x-lattice-http-version", ""),
        }
        return result

    async def create_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        session_id: str | None = None,
        provider: str | None = None,
        disable_transforms: bool = False,
        client_profile: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion through the LATTICE proxy.

        Yields OpenAI-compatible delta chunks.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stop is not None:
            body["stop"] = stop
        body.update(kwargs)

        headers: dict[str, str] = {}
        if session_id:
            headers["x-lattice-session-id"] = session_id
        if provider:
            headers["x-lattice-provider"] = provider
        if disable_transforms:
            headers["x-lattice-disable-transforms"] = "true"
        if client_profile:
            headers["x-lattice-client-profile"] = client_profile

        async with self._client._client.stream(
            "POST",
            f"{self._client.base_url}/v1/chat/completions",
            json=body,
            headers=self._client._headers(headers),
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            async for raw_line in resp.aiter_text():
                buffer += raw_line
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: ") :]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        data: dict[str, Any] = json.loads(payload)
                    except Exception:
                        continue
                    yield data
            if buffer.strip():
                line = buffer.strip()
                if line.startswith("data: "):
                    payload = line[len("data: ") :]
                    if payload.strip() != "[DONE]":
                        with contextlib.suppress(Exception):
                            yield json.loads(payload)


# =============================================================================
# Proxy session namespace
# =============================================================================


class _ProxySessionNamespace:
    """Proxy-backed session management namespace."""

    def __init__(self, client: LatticeProxyClient) -> None:
        self._client = client

    async def start(self, provider: str, model: str) -> dict[str, str]:
        """Start a new session on the proxy."""
        if not provider:
            raise ValueError("provider is required")
        if not model:
            raise ValueError("model is required")
        resp = await self._client._client.post(
            f"{self._client.base_url}/lattice/session/start",
            json={"provider": provider, "model": model},
            headers=self._client._headers(),
        )
        resp.raise_for_status()
        return cast(dict[str, str], resp.json())

    async def get(self, session_id: str) -> dict[str, Any]:
        """Get session by ID."""
        resp = await self._client._client.get(
            f"{self._client.base_url}/lattice/session/{session_id}",
            headers=self._client._headers(),
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    async def _gateway_request(
        self,
        body: dict[str, Any],
        *,
        session_id: str | None = None,
        headers: dict[str, str] | None = None,
        dictionary_wire: bool = False,
    ) -> dict[str, Any]:
        """Send a native binary request through ``/lattice/gateway``."""
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        wire_headers = {"Content-Type": "application/octet-stream"}
        if headers:
            wire_headers.update(headers)
        if session_id:
            wire_headers["x-lattice-session-id"] = session_id

        flags = FrameFlags.NONE
        codec: DictionaryCodec | None = None
        codec_key = session_id or "__default__"
        if dictionary_wire:
            codec = self._client._dictionary_codecs.get(codec_key)
            if codec is None:
                codec = DictionaryCodec(session_id=session_id)
                self._client._dictionary_codecs[codec_key] = codec
            payload = codec.compress(payload)
            flags |= FrameFlags.DICT_COMPRESSED

        frame = self._client._framer.encode_request(payload, flags=flags)[0]
        resp = await self._client._client.post(
            f"{self._client.base_url}/lattice/gateway",
            content=frame.to_bytes(),
            headers=self._client._headers(wire_headers),
        )
        resp.raise_for_status()
        out_frame = self._client._framer.decode_frame(resp.content)
        if out_frame.frame_type == FrameType.ERROR:
            raise ValueError("native gateway returned error frame")
        payload_bytes = out_frame.payload
        if out_frame.flags & FrameFlags.DICT_COMPRESSED:
            if codec is None:
                codec = self._client._dictionary_codecs.get(codec_key)
                if codec is None:
                    codec = DictionaryCodec(session_id=session_id)
                    self._client._dictionary_codecs[codec_key] = codec
            payload_bytes = codec.decompress(payload_bytes)
        with contextlib.suppress(Exception):
            result = cast(dict[str, Any], json.loads(payload_bytes.decode("utf-8")))
            result.setdefault("transport", {"framing": "binary", "delta": "bypassed"})
            return result
        return {
            "raw": payload_bytes.decode("utf-8", errors="replace"),
            "transport": {"framing": "binary", "delta": "bypassed"},
        }

    async def append(self, session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Append messages to a session."""
        resp = await self._client._client.post(
            f"{self._client.base_url}/lattice/session/append",
            json={"session_id": session_id, "messages": messages},
            headers=self._client._headers(),
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())
