"""Tests for persistent tunneling sidecar.

Covers:
- ReplayBuffer time-based pruning and circular eviction
- TunnelState constants
- LocalSocketServer start/stop, frame handler, agent connection rejection
- HTTPProxyServer start/stop, GET/POST forwarding, streaming, retry
- WebSocketTunnel host/port parsing, connect/disconnect, frame send/receive
- TunnelSidecar initialization, shutdown, connect_url, run loop
- Graceful shutdown with pending tasks cleanup
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

from lattice.core.tunnel_sidecar import (
    HTTPProxyServer,
    LocalSocketServer,
    ReplayBuffer,
    TunnelSidecar,
    TunnelState,
    WebSocketTunnel,
)

# =============================================================================
# ReplayBuffer
# =============================================================================


class TestReplayBuffer:
    def test_append_and_get(self) -> None:
        buf = ReplayBuffer(max_duration_seconds=60.0)
        buf.append(b"frame1")
        buf.append(b"frame2")
        assert buf.get_replay() == [b"frame1", b"frame2"]

    def test_clear(self) -> None:
        buf = ReplayBuffer()
        buf.append(b"x")
        buf.clear()
        assert buf.get_replay() == []

    def test_time_pruning(self) -> None:
        buf = ReplayBuffer(max_duration_seconds=0.05)
        buf.append(b"old")
        time.sleep(0.06)
        buf.append(b"new")
        replay = buf.get_replay()
        assert b"old" not in replay
        assert b"new" in replay

    def test_maxlen_eviction(self) -> None:
        buf = ReplayBuffer(max_duration_seconds=3600.0)
        # deque maxlen=10000
        for i in range(10005):
            buf.append(bytes([i % 256]))
        assert len(buf.get_replay()) == 10000

    def test_empty_replay(self) -> None:
        buf = ReplayBuffer()
        assert buf.get_replay() == []


# =============================================================================
# TunnelState
# =============================================================================


class TestTunnelState:
    def test_states(self) -> None:
        assert TunnelState.DISCONNECTED == "disconnected"
        assert TunnelState.CONNECTING == "connecting"
        assert TunnelState.CONNECTED == "connected"
        assert TunnelState.RECONNECTING == "reconnecting"
        assert TunnelState.SHUTDOWN == "shutdown"


# =============================================================================
# LocalSocketServer
# =============================================================================


class TestLocalSocketServer:
    async def test_start_stop_tcp(self) -> None:
        srv = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)
        await srv.start()
        assert srv._server is not None
        await srv.stop()

    async def test_frame_handler_called(self) -> None:
        received: list[bytes] = []

        async def _handler(frame: bytes) -> None:
            received.append(frame)

        srv = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)
        srv.set_frame_handler(_handler)
        await srv.start()

        port = srv._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        # Send length-prefixed frame
        frame = b"hello world"
        writer.write(len(frame).to_bytes(4, "big") + frame)
        await writer.drain()

        # Give handler a moment
        await asyncio.sleep(0.05)

        writer.close()
        await writer.wait_closed()
        await srv.stop()

        assert received == [b"hello world"]

    async def test_send_to_agent(self) -> None:
        srv = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)
        await srv.start()
        port = srv._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await asyncio.sleep(0.05)  # let _handle_agent run

        await srv.send_to_agent(b"from server")
        data = await reader.readexactly(11)
        assert data == b"from server"

        writer.close()
        await writer.wait_closed()
        await srv.stop()

    async def test_only_one_agent(self) -> None:
        srv = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)
        await srv.start()
        port = srv._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        r1, w1 = await asyncio.open_connection("127.0.0.1", port)
        await asyncio.sleep(0.02)
        assert srv.is_agent_connected()

        # Second connection should be closed immediately by the server
        r2, w2 = await asyncio.open_connection("127.0.0.1", port)
        # Server closes the writer; reading should return EOF quickly
        data = await r2.read()
        assert data == b""

        w1.close()
        await w1.wait_closed()
        w2.close()
        await w2.wait_closed()
        await srv.stop()

    async def test_multiple_frames(self) -> None:
        received: list[bytes] = []

        async def _handler(frame: bytes) -> None:
            received.append(frame)

        srv = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)
        srv.set_frame_handler(_handler)
        await srv.start()
        port = srv._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        for i in range(5):
            frame = f"frame-{i}".encode()
            writer.write(len(frame).to_bytes(4, "big") + frame)
        await writer.drain()
        await asyncio.sleep(0.05)

        writer.close()
        await writer.wait_closed()
        await srv.stop()

        assert len(received) == 5
        assert received[0] == b"frame-0"
        assert received[4] == b"frame-4"

    async def test_agent_disconnect_cleanup(self) -> None:
        srv = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)
        await srv.start()
        port = srv._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        _, writer = await asyncio.open_connection("127.0.0.1", port)
        await asyncio.sleep(0.02)
        assert srv.is_agent_connected()

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        assert not srv.is_agent_connected()

        await srv.stop()


# =============================================================================
# HTTPProxyServer
# =============================================================================


class TestHTTPProxyServer:
    async def test_start_stop(self) -> None:
        proxy = HTTPProxyServer(proxy_url="http://127.0.0.1:1", port=0)
        await proxy.start()
        assert proxy._server is not None
        assert proxy._client is not None
        await proxy.stop()
        assert proxy._server is None
        assert proxy._client is None

    async def test_forward_get_request(self) -> None:
        """Proxy should forward a GET and return the upstream response."""
        upstream_body = b'{"models":["gpt-4"]}'

        async def _upstream_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while True:
                line = await reader.readline()
                if line == b"\r\n":
                    break
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
            writer.write(f"Content-Length: {len(upstream_body)}\r\n\r\n".encode())
            writer.write(upstream_body)
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(_upstream_handler, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]

        proxy = HTTPProxyServer(proxy_url=f"http://127.0.0.1:{upstream_port}", port=0)
        await proxy.start()
        proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await writer.drain()

        response = await reader.read()
        assert b"200 OK" in response
        assert upstream_body in response

        writer.close()
        await writer.wait_closed()
        await proxy.stop()
        upstream.close()
        await upstream.wait_closed()

    async def test_forward_post_request(self) -> None:
        """Proxy should forward a POST with body."""
        received_body: bytes = b""

        async def _upstream_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal received_body
            headers = b""
            content_length = 0
            while True:
                line = await reader.readline()
                headers += line
                if line == b"\r\n":
                    break
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
            received_body = await reader.readexactly(content_length)
            writer.write(b"HTTP/1.1 201 Created\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(_upstream_handler, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]

        proxy = HTTPProxyServer(proxy_url=f"http://127.0.0.1:{upstream_port}", port=0)
        await proxy.start()
        proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        body = b'{"messages":[{"role":"user","content":"hi"}]}'
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: close\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()

        response = await reader.read()
        assert b"201 Created" in response
        assert received_body == body

        writer.close()
        await writer.wait_closed()
        await proxy.stop()
        upstream.close()
        await upstream.wait_closed()

    async def test_forward_streaming_request(self) -> None:
        """Proxy should stream SSE responses back using chunked encoding."""
        upstream_body = b'data: {"chunk": 1}\n\ndata: {"chunk": 2}\n\n'

        async def _upstream_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while True:
                line = await reader.readline()
                if line == b"\r\n":
                    break
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n")
            writer.write(b"Transfer-Encoding: chunked\r\n\r\n")
            for chunk in [b'data: {"chunk": 1}\n\n', b'data: {"chunk": 2}\n\n']:
                writer.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(_upstream_handler, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]

        proxy = HTTPProxyServer(proxy_url=f"http://127.0.0.1:{upstream_port}", port=0)
        await proxy.start()
        proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        body = b'{"stream": true}'
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: close\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()

        response = await reader.read()
        assert b"200 OK" in response
        assert b"Transfer-Encoding: chunked" in response
        assert b'data: {"chunk": 1}' in response
        assert b'data: {"chunk": 2}' in response

        writer.close()
        await writer.wait_closed()
        await proxy.stop()
        upstream.close()
        await upstream.wait_closed()

    async def test_retry_on_failure(self) -> None:
        """Proxy should retry on connection failure."""
        proxy = HTTPProxyServer(
            proxy_url="http://127.0.0.1:1",
            port=0,
            max_retries=2,
            base_retry_interval=0.01,
        )
        await proxy.start()
        proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await writer.drain()

        response = await reader.read()
        assert b"504" in response

        writer.close()
        await writer.wait_closed()
        await proxy.stop()

    async def test_is_streaming_request(self) -> None:
        from lattice.core.tunnel_sidecar import _HTTPRequest

        proxy = HTTPProxyServer(proxy_url="http://127.0.0.1:1")
        req = _HTTPRequest(
            method="POST",
            path="/v1/chat/completions",
            version="HTTP/1.1",
            headers={"content-type": "application/json"},
            body=json.dumps({"stream": True}).encode(),
        )
        assert proxy._is_streaming_request(req) is True

        req2 = _HTTPRequest(
            method="POST",
            path="/v1/chat/completions",
            version="HTTP/1.1",
            headers={"content-type": "application/json"},
            body=json.dumps({"stream": False}).encode(),
        )
        assert proxy._is_streaming_request(req2) is False

        req3 = _HTTPRequest(
            method="GET",
            path="/v1/models",
            version="HTTP/1.1",
            headers={"accept": "text/event-stream"},
            body=b"",
        )
        assert proxy._is_streaming_request(req3) is True


# =============================================================================
# WebSocketTunnel
# =============================================================================


class TestWebSocketTunnel:
    def test_parse_host_port_wss_default(self) -> None:
        assert WebSocketTunnel._parse_host_port("wss://proxy.example.com/path") == (
            "proxy.example.com",
            443,
        )

    def test_parse_host_port_ws_default(self) -> None:
        assert WebSocketTunnel._parse_host_port("ws://proxy.example.com/path") == (
            "proxy.example.com",
            80,
        )

    def test_parse_host_port_custom_port(self) -> None:
        assert WebSocketTunnel._parse_host_port("wss://proxy.example.com:8443/lattice") == (
            "proxy.example.com",
            8443,
        )

    def test_parse_host_port_no_path(self) -> None:
        assert WebSocketTunnel._parse_host_port("ws://localhost:8788") == ("localhost", 8788)

    def test_url_normalization(self) -> None:
        t = WebSocketTunnel(proxy_url="https://proxy.example.com", session_id="s1")
        assert t.proxy_url == "wss://proxy.example.com"

        t2 = WebSocketTunnel(proxy_url="http://proxy.example.com", session_id="s2")
        assert t2.proxy_url == "ws://proxy.example.com"

    def test_state_initial(self) -> None:
        t = WebSocketTunnel(proxy_url="ws://localhost:8788", session_id="sid")
        assert t.state == TunnelState.DISCONNECTED

    async def test_connect_to_nonexistent_host(self) -> None:
        # Use a port that is extremely unlikely to be open
        t = WebSocketTunnel(proxy_url="ws://127.0.0.1:1", session_id="sid")
        result = await t.connect()
        assert result is False
        assert t.state == TunnelState.DISCONNECTED

    async def test_disconnect_when_not_connected(self) -> None:
        t = WebSocketTunnel(proxy_url="ws://localhost:8788", session_id="sid")
        # Should not raise
        await t.disconnect()
        assert t.state == TunnelState.DISCONNECTED

    async def test_connect_to_minimal_ws_server(self) -> None:
        """Start a minimal server that accepts the WS upgrade, verify connect succeeds."""
        server_received = b""

        async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal server_received
            # Read upgrade request
            while True:
                line = await reader.readline()
                server_received += line
                if line == b"\r\n":
                    break
            # Send 101 response
            writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
            await writer.drain()
            # Echo back any WebSocket frames
            try:
                while True:
                    first = await reader.readexactly(1)
                    length_byte = await reader.readexactly(1)
                    length = length_byte[0] & 0x7F
                    if length == 126:
                        length = int.from_bytes(await reader.readexactly(2), "big")
                    payload = await reader.readexactly(length)
                    # Echo back
                    header = b"\x82"
                    if length < 126:
                        header += bytes([length])
                    else:
                        header += b"\x7e" + length.to_bytes(2, "big")
                    writer.write(header + payload)
                    await writer.drain()
            except asyncio.IncompleteReadError:
                pass
            writer.close()

        srv = await asyncio.start_server(_handle_client, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]

        t = WebSocketTunnel(proxy_url=f"ws://127.0.0.1:{port}", session_id="sid")
        result = await t.connect()
        assert result is True
        assert t.state == TunnelState.CONNECTED

        # Start receive loop in background
        received: list[bytes] = []
        t.set_frame_handler(lambda f: received.append(f))
        recv_task = asyncio.create_task(t.receive_loop())

        await t.send(b"ping")
        await asyncio.sleep(0.05)

        await t.disconnect()
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task
        srv.close()
        await srv.wait_closed()

        assert b"ping" in received

    async def test_send_when_not_connected(self) -> None:
        t = WebSocketTunnel(proxy_url="ws://localhost:8788", session_id="sid")
        # Should not raise
        await t.send(b"data")


# =============================================================================
# TunnelSidecar
# =============================================================================


class DummyConfig:
    """Minimal config for sidecar tests."""

    def __init__(self) -> None:
        self._proxy_url = "http://127.0.0.1:9999"

    def proxy_url(self) -> str:
        return self._proxy_url


class TestTunnelSidecar:
    def test_init(self) -> None:
        config = DummyConfig()
        sidecar = TunnelSidecar(config=config, session_id="test_sid")
        assert sidecar.session_id == "test_sid"
        assert sidecar._proxy is not None
        assert sidecar._replay is not None
        assert not sidecar._shutdown_event.is_set()

    def test_init_auto_session_id(self) -> None:
        config = DummyConfig()
        sidecar = TunnelSidecar(config=config)
        assert sidecar.session_id.startswith("tun_")
        assert len(sidecar.session_id) == 20  # "tun_" + 16 hex chars

    def test_connect_url(self) -> None:
        config = DummyConfig()
        sidecar = TunnelSidecar(config=config, tcp_port=9999)
        assert sidecar.connect_url == "http://127.0.0.1:9999"

    async def test_shutdown_sets_event(self) -> None:
        config = DummyConfig()
        sidecar = TunnelSidecar(config=config)
        sidecar.shutdown()
        assert sidecar._shutdown_event.is_set()

    async def test_run_starts_proxy_server(self) -> None:
        config = DummyConfig()
        sidecar = TunnelSidecar(config=config, tcp_port=0)

        run_task = asyncio.create_task(sidecar.run())
        await asyncio.sleep(0.3)  # let it start proxy server
        assert sidecar._proxy._server is not None

        sidecar.shutdown()
        await run_task

    async def test_run_graceful_cleanup(self) -> None:
        config = DummyConfig()
        sidecar = TunnelSidecar(config=config, tcp_port=0)

        run_task = asyncio.create_task(sidecar.run())
        await asyncio.sleep(0.3)
        sidecar.shutdown()
        await run_task

        # After shutdown, server should be stopped
        assert sidecar._proxy._server is None
