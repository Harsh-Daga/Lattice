"""Persistent tunneling sidecar for LATTICE agent integration.

When an agent (Claude, Cursor, Codex, etc.) is laced with LATTICE,
instead of configuring the agent to talk directly to the proxy HTTP endpoint,
`lattice lace` starts a local sidecar that:

1. Opens a persistent WebSocket tunnel to the LATTICE proxy.
2. Exposes a local Unix domain socket (or TCP port) for the agent.
3. Auto-reconnects with replay window on disconnections.
4. Forwards all traffic bidirectionally.

Architecture
------------
```
┌─────────┐     ┌──────────┐     WebSocket      ┌─────────┐
│  Agent  │────▶│ Sidecar  │◄─────tunnel─────▶│  Proxy  │
│ (env)   │     │ (local)  │                   │ (cloud) │
└─────────┘     └──────────┘                   └─────────┘
       localhost:8788    wss://proxy/lattice/stream
```

Design decisions
----------------
1. **Local socket**: Unix domain socket when available (faster, more secure),
   falls back to TCP localhost:8788.
2. **WebSocket protocol**: Native binary framing over WebSocket for efficiency.
   Falls back to JSON text frames.
3. **Auto-reconnect**: Exponential backoff with jitter. Replay window
   buffers last N seconds of traffic for seamless reconnect.
4. **Bidirectional**: Both request (agent→proxy) and response
   (proxy→agent) streams are tunneled.
5. **Session affinity**: Session ID is sticky to the tunnel so
   reconnections resume the same session.
6. **Graceful shutdown**: SIGTERM drains pending requests before exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import signal
import socket
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from lattice.core.config import LatticeConfig
from lattice.protocol.framing import BinaryFramer, Frame, FrameType

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReplayBuffer:
    """Circular buffer of recent frames for replay on reconnect."""

    max_duration_seconds: float = 30.0
    _frames: deque[tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=10000))

    def append(self, frame_bytes: bytes) -> None:
        """Append a frame with timestamp."""
        self._frames.append((time.time(), frame_bytes))
        # Prune old frames
        cutoff = time.time() - self.max_duration_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def get_replay(self) -> list[bytes]:
        """Get all frames for replay."""
        return [f for _, f in self._frames]

    def clear(self) -> None:
        self._frames.clear()


# ---------------------------------------------------------------------------
# TunnelState
# ---------------------------------------------------------------------------

class TunnelState:
    """State machine for the tunnel connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    SHUTDOWN = "shutdown"

# ---------------------------------------------------------------------------
# LocalSocketServer
# ---------------------------------------------------------------------------

class LocalSocketServer:
    """Accepts agent connections on a local Unix domain socket or TCP port."""

    def __init__(
        self,
        *,
        unix_path: pathlib.Path | None = None,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 8788,
    ) -> None:
        self.unix_path = unix_path
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self._server: asyncio.Server | None = None
        self._agent_reader: asyncio.StreamReader | None = None
        self._agent_writer: asyncio.StreamWriter | None = None
        self._on_frame: callable | None = None

    async def start(self) -> None:
        """Start listening for agent connections."""
        if self.unix_path:
            # Remove stale socket
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.unix_path)
            self._server = await asyncio.start_unix_server(
                self._handle_agent,
                path=str(self.unix_path),
            )
            logger.info("local_socket_listening", path=str(self.unix_path), type="unix")
        else:
            self._server = await asyncio.start_server(
                self._handle_agent,
                host=self.tcp_host,
                port=self.tcp_port,
            )
            logger.info(
                "local_socket_listening",
                host=self.tcp_host,
                port=self.tcp_port,
                type="tcp",
            )

    async def stop(self) -> None:
        """Stop the local socket server."""
        if self._agent_writer and not self._agent_writer.is_closing():
            self._agent_writer.close()
            with contextlib.suppress(Exception):
                await self._agent_writer.wait_closed()
            self._agent_writer = None
            self._agent_reader = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.unix_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.unix_path)

    async def send_to_agent(self, data: bytes) -> None:
        """Send data to the connected agent."""
        if self._agent_writer and not self._agent_writer.is_closing():
            self._agent_writer.write(data)
            await self._agent_writer.drain()

    def is_agent_connected(self) -> bool:
        return self._agent_writer is not None and not self._agent_writer.is_closing()

    async def _handle_agent(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new agent connection."""
        # Only one agent at a time
        if self.is_agent_connected():
            with contextlib.suppress(Exception):
                writer.write_eof()
            writer.close()
            writer.transport.abort()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return

        self._agent_reader = reader
        self._agent_writer = writer
        peer = writer.get_extra_info("socket")
        logger.info("agent_connected", peer=str(peer))

        try:
            while True:
                # Read length-prefixed frame from agent
                length_bytes = await reader.readexactly(4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, "big")
                frame = await reader.readexactly(length)
                if not frame:
                    break
                if self._on_frame:
                    await self._on_frame(frame)
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            logger.warning("agent_connection_error", error=str(exc))
        finally:
            logger.info("agent_disconnected")
            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            self._agent_reader = None
            self._agent_writer = None

    def set_frame_handler(self, handler: callable) -> None:
        self._on_frame = handler

# ---------------------------------------------------------------------------
# HTTPProxyServer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _HTTPRequest:
    """Parsed HTTP/1.1 request."""

    method: str
    path: str
    version: str
    headers: dict[str, str]
    body: bytes


class HTTPProxyServer:
    """Local HTTP proxy that accepts agent connections and forwards to the LATTICE proxy.

    Supports HTTP/1.1 keep-alive, request-level retry with exponential backoff,
    and streaming response forwarding (SSE).  This is the production interface
    that agents (Claude, Codex, Cursor, etc.) actually talk to.
    """

    def __init__(
        self,
        *,
        proxy_url: str,
        host: str = "127.0.0.1",
        port: int = 8788,
        max_retries: int = 3,
        base_retry_interval: float = 0.5,
    ) -> None:
        self.proxy_url = proxy_url.rstrip("/")
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.base_retry_interval = base_retry_interval
        self._server: asyncio.Server | None = None
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start the HTTP proxy server."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            follow_redirects=False,
        )
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self.host,
            port=self.port,
        )
        logger.info(
            "http_proxy_listening",
            host=self.host,
            port=self.port,
            proxy=self.proxy_url,
        )

    async def stop(self) -> None:
        """Stop the HTTP proxy server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a persistent HTTP/1.1 connection."""
        peer = writer.get_extra_info("peername")
        logger.info("agent_http_connected", peer=str(peer))
        try:
            while True:
                request = await self._read_request(reader)
                if request is None:
                    break
                should_close = await self._forward_request(request, writer)
                if should_close:
                    break
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            logger.warning("agent_http_error", error=str(exc), peer=str(peer), exc_info=True)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            logger.info("agent_http_disconnected", peer=str(peer))

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> _HTTPRequest | None:
        """Read and parse a single HTTP/1.1 request."""
        # Read headers
        header_lines: list[bytes] = []
        while True:
            line = await reader.readline()
            if not line:
                return None
            header_lines.append(line)
            if line == b"\r\n":
                break

        if not header_lines:
            return None

        # Parse first line
        first_line = header_lines[0].decode("utf-8", errors="replace").strip()
        parts = first_line.split(" ", 2)
        if len(parts) != 3:
            return None
        method, path, version = parts

        # Parse headers
        headers: dict[str, str] = {}
        content_length = 0
        for line in header_lines[1:]:
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            if ":" in line_str:
                k, v = line_str.split(":", 1)
                headers[k.strip().lower()] = v.strip()
                if k.strip().lower() == "content-length":
                    try:
                        content_length = int(v.strip())
                    except ValueError:
                        content_length = 0

        # Read body
        body = await reader.readexactly(content_length) if content_length > 0 else b""

        return _HTTPRequest(method=method, path=path, version=version, headers=headers, body=body)

    async def _forward_request(
        self, request: _HTTPRequest, writer: asyncio.StreamWriter
    ) -> bool:
        """Forward a request to the proxy and write the response.

        Returns True if the connection should be closed after this request.
        """
        url = f"{self.proxy_url}{request.path}"
        fwd_headers = {k: v for k, v in request.headers.items() if k not in ("host", "content-length")}
        fwd_headers["host"] = self.proxy_url.replace("http://", "").replace("https://", "").split("/")[0]

        is_streaming = self._is_streaming_request(request)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if is_streaming:
                    await self._forward_streaming(request, url, fwd_headers, writer)
                else:
                    await self._forward_buffered(request, url, fwd_headers, writer)
                break
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                wait = min(self.base_retry_interval * (2 ** attempt), 30.0)
                logger.info(
                    "proxy_retry",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    wait=wait,
                    url=url,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                logger.warning("proxy_forward_error", error=str(exc), url=url)
                await self._send_error(writer, 502, b"Bad Gateway")
                break
        else:
            # All retries exhausted
            logger.error(
                "proxy_retries_exhausted",
                error=str(last_error),
                url=url,
            )
            await self._send_error(writer, 504, b"Gateway Timeout")

        # Determine if connection should close
        conn = request.headers.get("connection", "").lower()
        if conn == "close" or request.version == "HTTP/1.0":
            return True
        return False

    def _is_streaming_request(self, request: _HTTPRequest) -> bool:
        """Detect if this is a streaming request (SSE)."""
        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept:
            return True
        # Check JSON body for stream: true
        if request.body:
            try:
                body_json = json.loads(request.body.decode("utf-8", errors="ignore"))
                if body_json.get("stream") is True:
                    return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        return False

    async def _forward_buffered(
        self,
        request: _HTTPRequest,
        url: str,
        headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Forward a non-streaming request with buffered response."""
        assert self._client is not None
        response = await self._client.request(
            request.method,
            url,
            headers=headers,
            content=request.body or None,
        )
        body = await response.aread()

        status_line = f"HTTP/1.1 {response.status_code} {response.reason_phrase}\r\n"
        writer.write(status_line.encode())
        for k, v in response.headers.items():
            writer.write(f"{k}: {v}\r\n".encode())
        writer.write(f"Content-Length: {len(body)}\r\n\r\n".encode())
        writer.write(body)
        await writer.drain()

    async def _forward_streaming(
        self,
        request: _HTTPRequest,
        url: str,
        headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Forward a streaming request, streaming the response back."""
        assert self._client is not None
        async with self._client.stream(
            request.method,
            url,
            headers=headers,
            content=request.body or None,
        ) as response:
            status_line = f"HTTP/1.1 {response.status_code} {response.reason_phrase}\r\n"
            writer.write(status_line.encode())
            for k, v in response.headers.items():
                # Remove Content-Length if present — we will use chunked
                if k.lower() == "content-length":
                    continue
                writer.write(f"{k}: {v}\r\n".encode())
            writer.write(b"Transfer-Encoding: chunked\r\n\r\n")
            await writer.drain()

            async for chunk in response.aiter_raw():
                if chunk:
                    writer.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()

    async def _send_error(
        self, writer: asyncio.StreamWriter, code: int, message: bytes
    ) -> None:
        """Send an HTTP error response."""
        body = message
        writer.write(f"HTTP/1.1 {code} Error\r\n".encode())
        writer.write(f"Content-Length: {len(body)}\r\n".encode())
        writer.write(b"Connection: close\r\n\r\n")
        writer.write(body)
        await writer.drain()


# ---------------------------------------------------------------------------
# WebSocketTunnel
# ---------------------------------------------------------------------------

class WebSocketTunnel:
    """WebSocket tunnel to the LATTICE proxy."""

    def __init__(
        self,
        *,
        proxy_url: str,
        session_id: str,
        reconnect_interval: float = 1.0,
        max_reconnect_interval: float = 30.0,
        framer: BinaryFramer | None = None,
    ) -> None:
        self.proxy_url = proxy_url.replace("http://", "ws://").replace("https://", "wss://")
        self.session_id = session_id
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.framer = framer or BinaryFramer()
        self._ws_reader: asyncio.StreamReader | None = None
        self._ws_writer: asyncio.StreamWriter | None = None
        self._on_frame: callable | None = None
        self._state = TunnelState.DISCONNECTED

    @property
    def state(self) -> str:
        return self._state

    async def connect(self) -> bool:
        """Connect to the proxy WebSocket endpoint."""
        self._state = TunnelState.CONNECTING
        try:
            url = f"{self.proxy_url}/lattice/stream?session_id={self.session_id}"
            # Using plain asyncio.open_connection for WebSocket handshake
            # In production, use websockets library for proper handshake
            host, port = self._parse_host_port(self.proxy_url)
            self._ws_reader, self._ws_writer = await asyncio.open_connection(
                host, port, ssl=(self.proxy_url.startswith("wss"))
            )
            # Send WebSocket upgrade request
            path = f"/lattice/stream?session_id={self.session_id}"
            upgrade = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                f"Sec-WebSocket-Version: 13\r\n"
                f"\r\n"
            )
            self._ws_writer.write(upgrade.encode())
            await self._ws_writer.drain()

            # Read response (simplified — assumes success)
            response = await self._ws_reader.readline()
            if b"101" not in response:
                raise ConnectionError(f"WebSocket handshake failed: {response}")
            # Drain remaining headers
            while True:
                line = await self._ws_reader.readline()
                if line == b"\r\n":
                    break

            self._state = TunnelState.CONNECTED
            logger.info("websocket_connected", url=url)
            return True
        except Exception as exc:
            self._state = TunnelState.DISCONNECTED
            logger.warning("websocket_connect_failed", error=str(exc))
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        if self._ws_writer:
            self._ws_writer.close()
            with contextlib.suppress(Exception):
                await self._ws_writer.wait_closed()
            self._ws_writer = None
            self._ws_reader = None
        self._state = TunnelState.DISCONNECTED

    async def send(self, data: bytes) -> None:
        """Send data through the tunnel."""
        if self._state == TunnelState.CONNECTED and self._ws_writer:
            # Send as WebSocket binary frame (simplified)
            header = b"\x82"  # FIN=1, opcode=binary
            length = len(data)
            if length < 126:
                header += bytes([length])
            else:
                header += b"\x7e" + length.to_bytes(2, "big")
            self._ws_writer.write(header + data)
            await self._ws_writer.drain()

    async def receive_loop(self) -> None:
        """Receive frames from the tunnel indefinitely."""
        try:
            while self._state == TunnelState.CONNECTED:
                # Simplified WebSocket frame parsing
                # Read first byte (FIN + opcode)
                first = await self._ws_reader.readexactly(1)
                if not first:
                    break
                masked = False
                length_byte = await self._ws_reader.readexactly(1)
                length = length_byte[0] & 0x7F
                if length == 126:
                    length = int.from_bytes(await self._ws_reader.readexactly(2), "big")
                elif length == 127:
                    length = int.from_bytes(await self._ws_reader.readexactly(8), "big")
                if masked:
                    mask = await self._ws_reader.readexactly(4)
                payload = await self._ws_reader.readexactly(length)
                if masked:
                    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                if self._on_frame:
                    await self._on_frame(payload)
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            logger.warning("websocket_receive_error", error=str(exc))
        finally:
            self._state = TunnelState.DISCONNECTED

    def set_frame_handler(self, handler: callable) -> None:
        self._on_frame = handler

    @staticmethod
    def _parse_host_port(url: str) -> tuple[str, int]:
        """Parse host and port from ws://host:port/path."""
        is_wss = url.startswith("wss://")
        url = url.replace("wss://", "").replace("ws://", "")
        if "/" in url:
            url = url.split("/", 1)[0]
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            return host, int(port_str)
        return url, 443 if is_wss else 80

# ---------------------------------------------------------------------------
# TunnelSidecar
# ---------------------------------------------------------------------------

class TunnelSidecar:
    """Main sidecar that exposes a local HTTP proxy for agents.

    Architecture (current):
        Agent → HTTPProxyServer (localhost:8788) → Proxy (HTTP/2)

    The WebSocket tunnel component is preserved for future activation
    once the proxy-side ``/lattice/stream`` endpoint is implemented.
    """

    _UNSET = object()

    def __init__(
        self,
        *,
        config: LatticeConfig,
        session_id: str | None = None,
        unix_socket_path: pathlib.Path | None | Any = _UNSET,
        tcp_port: int = 8788,
    ) -> None:
        self.config = config
        self.session_id = session_id or f"tun_{uuid.uuid4().hex[:16]}"
        if unix_socket_path is self._UNSET:
            self.unix_socket_path = pathlib.Path(
                tempfile.gettempdir()
            ) / f"lattice-{self.session_id}.sock"
        else:
            self.unix_socket_path = unix_socket_path
        self.tcp_port = tcp_port
        self._proxy = HTTPProxyServer(
            proxy_url=config.proxy_url(),
            host="127.0.0.1",
            port=self.tcp_port,
        )
        self._replay = ReplayBuffer()
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []

    async def run(self) -> None:
        """Run the sidecar until shutdown."""
        logger.info(
            "sidecar_starting",
            session_id=self.session_id,
            tcp_port=self.tcp_port,
            proxy=self.config.proxy_url(),
        )

        await self._proxy.start()

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        logger.info("sidecar_shutting_down")
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._proxy.stop()
        logger.info("sidecar_stopped")

    def shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_event.set()

    @property
    def connect_url(self) -> str:
        """URL for the agent to connect to."""
        return f"http://127.0.0.1:{self.tcp_port}"


# ---------------------------------------------------------------------------
# SidecarThread — synchronous wrapper for background use
# ---------------------------------------------------------------------------

import threading


class SidecarThread:
    """Run a ``TunnelSidecar`` in a background thread with a dedicated event loop.

    Usage::

        sidecar = TunnelSidecar(config=config, tcp_port=8788)
        runner = SidecarThread(sidecar)
        runner.start()
        # ... agent runs ...
        runner.stop(timeout=5.0)
    """

    def __init__(self, sidecar: TunnelSidecar) -> None:
        self.sidecar = sidecar
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the sidecar in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self.sidecar.run())
            finally:
                self._loop.close()
                self._loop = None

        self._thread = threading.Thread(target=_run, daemon=True, name="lattice-sidecar")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown and wait for the thread to finish."""
        self.sidecar.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive() and self._loop is not None:
                # Forcefully cancel pending tasks
                try:
                    self._loop.call_soon_threadsafe(
                        lambda: [t.cancel() for t in asyncio.all_tasks(self._loop)]
                    )
                except RuntimeError:
                    pass
                self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = [
    "TunnelSidecar",
    "WebSocketTunnel",
    "LocalSocketServer",
    "ReplayBuffer",
    "TunnelState",
    "SidecarThread",
]
