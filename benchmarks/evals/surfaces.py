"""Deterministic evals for protocol, transport, capabilities, and agent integrations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmarks.evals.report import EvalSectionReport
from lattice.core.config import LatticeConfig
from lattice.integrations.claude.install import (
    apply_provider_scope as apply_claude_scope,
)
from lattice.integrations.claude.install import (
    build_install_env as build_claude_env,
)
from lattice.integrations.claude.install import (
    revert_provider_scope as revert_claude_scope,
)
from lattice.integrations.codex.install import (
    apply_provider_scope as apply_codex_scope,
)
from lattice.integrations.codex.install import (
    build_install_env as build_codex_env,
)
from lattice.integrations.codex.install import (
    revert_provider_scope as revert_codex_scope,
)
from lattice.integrations.copilot.install import (
    apply_provider_scope as apply_copilot_scope,
)
from lattice.integrations.copilot.install import (
    build_install_env as build_copilot_env,
)
from lattice.integrations.copilot.install import (
    revert_provider_scope as revert_copilot_scope,
)
from lattice.integrations.cursor.install import (
    apply_provider_scope as apply_cursor_scope,
)
from lattice.integrations.cursor.install import (
    build_install_env as build_cursor_env,
)
from lattice.integrations.opencode.install import (
    apply_provider_scope as apply_opencode_scope,
)
from lattice.integrations.opencode.install import (
    build_install_env as build_opencode_env,
)
from lattice.integrations.opencode.install import (
    revert_provider_scope as revert_opencode_scope,
)
from lattice.protocol.cache_planner import get_cache_planner
from lattice.protocol.dictionary_codec import DictionaryCodec
from lattice.protocol.framing import BinaryFramer, FrameFlags, FrameType
from lattice.protocol.manifest import manifest_from_messages, manifest_summary
from lattice.protocol.multiplex import MultiStreamMux, ReliabilityMode, StreamType
from lattice.protocol.reliability import SelectiveReliability
from lattice.protocol.resume import ReplayWindow, StreamChunk, StreamManager
from lattice.core.tunnel_sidecar import (
    HTTPProxyServer,
    LocalSocketServer,
    ReplayBuffer,
    TunnelSidecar,
    WebSocketTunnel,
)
from lattice.providers.transport import ConnectionPoolManager
from lattice.providers.capabilities import CacheMode, Capability, get_capability_registry


@contextmanager
def _temp_home() -> Any:
    with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "home", return_value=Path(tmp)):
        yield Path(tmp)


_SENSITIVE_ENV_KEY_PARTS = ("key", "secret", "token", "password", "credential", "session")


def _redact_sensitive_mapping(value: Any, key: str | None = None) -> Any:
    """Redact likely secrets from nested mapping values before reporting."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child_name = str(child_key).lower()
            if any(part in child_name for part in _SENSITIVE_ENV_KEY_PARTS):
                redacted[child_key] = "<redacted>"
            else:
                redacted[child_key] = _redact_sensitive_mapping(child_value, child_name)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_mapping(item, key) for item in value]
    return value


async def run_protocol_eval() -> EvalSectionReport:
    """Run deterministic protocol and wire-layer evals."""
    messages = [
        {"role": "system", "content": "You are LATTICE's protocol evaluator."},
        {"role": "user", "content": "Explain the manifest, wire, and cache plan."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
            },
        }
    ]
    manifest = manifest_from_messages(
        "eval-protocol",
        messages,
        tools=tools,
        model="openai/gpt-4o-mini",
        provider="openai",
    )
    manifest_info = manifest_summary(manifest)

    cache_plans: dict[str, dict[str, Any]] = {}
    for provider in ("openai", "anthropic", "gemini", "vertex", "bedrock"):
        plan = get_cache_planner(provider).plan(manifest)
        cache_plans[provider] = {
            "cached_tokens": plan.expected_cached_tokens,
            "breakpoints": len(plan.breakpoints),
            "annotation_keys": sorted(plan.annotations.keys()),
        }

    payload = json.dumps(
        {
            "manifest": manifest_info,
            "messages": messages,
            "tools": tools,
        },
        sort_keys=True,
    ).encode("utf-8")

    codec = DictionaryCodec(session_id="eval-protocol")
    compressed = codec.compress(payload)
    restored = codec.decompress(compressed)
    snapshot = codec.to_snapshot()
    replayed = DictionaryCodec.from_snapshot(snapshot).decompress(compressed)

    framer = BinaryFramer(max_frame_payload=32)
    frames = framer.encode_request(compressed, flags=FrameFlags.DICT_COMPRESSED)
    decoded = framer.decode_frame(frames[0].to_bytes())
    boundary_frame = framer.encode_frame_with_boundary(
        b"payload",
        boundary_type="reasoning",
        reliability="high",
        priority=7,
    )

    mux = MultiStreamMux(max_streams=4)
    primary = mux.create_stream(StreamType.PRIMARY, priority=0, reliability=ReliabilityMode.RELIABLE)
    tool = mux.create_stream(StreamType.TOOL, priority=1, reliability=ReliabilityMode.PARTIAL)
    migrated = mux.migrate_stream(tool.stream_id)
    mux.close_stream(tool.stream_id)
    reliability = SelectiveReliability(max_retries=3)
    critical = reliability.should_retransmit(boundary_frame, 0)
    low = reliability.should_retransmit(framer.encode_frame_with_boundary(b"x", "sentence", "low"), 0)

    stream_manager = StreamManager(window_capacity=3, token_ttl_seconds=30)
    stream_id = stream_manager.create_stream()
    stream_manager.append_chunk(stream_id, 0, "data: first")
    stream_manager.append_chunk(stream_id, 1, "data: second")
    resume_token = stream_manager.create_resume_token(stream_id, 2)
    validated = stream_manager.validate_resume_token(resume_token)
    replay = stream_manager.replay(stream_id, 1)
    window = ReplayWindow(2)
    window.append(StreamChunk(sequence=0, data="a"))
    window.append(StreamChunk(sequence=1, data="b"))

    checks = {
        "manifest_segments": manifest_info["segment_count"] >= 2,
        "cache_plans": all(v["breakpoints"] >= 1 for v in cache_plans.values()),
        "dictionary_roundtrip": restored == payload and replayed == payload,
        "frame_roundtrip": decoded.frame_type == FrameType.REQUEST and bool(decoded.flags & FrameFlags.DICT_COMPRESSED),
        "chunking": len(frames) >= 1,
        "boundary_flags": bool(boundary_frame.flags & FrameFlags.BOUNDARY_REASONING),
        "mux_primary_active": primary.stream_id == 0 and mux.active_count == 1,
        "mux_migration": migrated is True,
        "reliability": critical is True and low is False,
        "resume_valid": validated is not None and len(replay) >= 1,
        "replay_window": len(window.replay_from(1)) == 1,
    }
    return EvalSectionReport(
        name="protocol_eval",
        kind="local",
        summary={"passed": sum(1 for ok in checks.values() if ok), "total": len(checks)},
        details={"checks": checks, "cache_plans": cache_plans, "manifest": manifest_info},
    )


async def run_transport_eval() -> EvalSectionReport:
    """Run deterministic transport and networking evals."""
    pool = ConnectionPoolManager(http2=True)
    client_a = pool.get_client("openai", "https://api.openai.com")
    client_b = pool.get_client("openai", "https://api.openai.com")
    http_version = pool.get_http_version("openai", "https://api.openai.com")

    framer = BinaryFramer(max_frame_payload=32)
    request_frames = framer.encode_request(b"payload", flags=FrameFlags.DICT_COMPRESSED)
    decoded = framer.decode_frame(request_frames[0].to_bytes())
    accepted, reason = framer.decode_negotiation_outcome(
        framer.encode_negotiation_outcome(True, fallback_reason="")
    )

    stream_manager = StreamManager(window_capacity=3, token_ttl_seconds=30)
    stream_id = stream_manager.create_stream()
    stream_manager.append_chunk(stream_id, 0, "data: first")
    stream_manager.append_chunk(stream_id, 1, "data: second")
    resume_token = stream_manager.create_resume_token(stream_id, 2)
    resume_meta = stream_manager.get_resume_metadata(stream_id)
    token_valid = stream_manager.validate_resume_token(resume_token) is not None

    replay = ReplayBuffer(max_duration_seconds=0.05)
    replay.append(b"old")
    await asyncio.sleep(0.06)
    replay.append(b"new")
    replay_ok = b"old" not in replay.get_replay() and b"new" in replay.get_replay()

    socket_received: list[bytes] = []
    socket_echo = b""
    socket_server = LocalSocketServer(tcp_host="127.0.0.1", tcp_port=0)

    async def _socket_handler(frame: bytes) -> None:
        socket_received.append(frame)

    socket_server.set_frame_handler(_socket_handler)
    await socket_server.start()
    socket_port = socket_server._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    sock_reader, sock_writer = await asyncio.open_connection("127.0.0.1", socket_port)
    frame = b"transport"
    sock_writer.write(len(frame).to_bytes(4, "big") + frame)
    await sock_writer.drain()
    await asyncio.sleep(0.05)
    await socket_server.send_to_agent(b"wire")
    socket_echo = await sock_reader.readexactly(4)
    sock_writer.close()
    await sock_writer.wait_closed()
    await socket_server.stop()

    backend_requests: list[dict[str, Any]] = []

    async def _backend(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_line = await reader.readline()
        if not request_line:
            writer.close()
            return
        method, path, version = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
        headers: dict[str, str] = {}
        content_length = 0
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
            key, value = line.decode("utf-8", errors="replace").split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            headers[key] = value
            if key == "content-length":
                content_length = int(value)
        body = await reader.readexactly(content_length) if content_length else b""
        backend_requests.append(
            {
                "method": method,
                "path": path,
                "version": version,
                "headers": headers,
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        if "stream" in path or b'"stream": true' in body:
            chunk_one = b"data: stream-one\n\n"
            chunk_two = b"data: stream-two\n\n"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
                + f"{len(chunk_one):X}\r\n".encode("utf-8")
                + chunk_one
                + b"\r\n"
                + f"{len(chunk_two):X}\r\n".encode("utf-8")
                + chunk_two
                + b"\r\n0\r\n\r\n"
            )
        else:
            payload = json.dumps({"ok": True, "path": path, "host": headers.get("host", "")}).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
                + payload
            )
        await writer.drain()
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    backend_server = await asyncio.start_server(_backend, host="127.0.0.1", port=0)
    backend_port = backend_server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    proxy = HTTPProxyServer(proxy_url=f"http://127.0.0.1:{backend_port}", host="127.0.0.1", port=0)
    await proxy.start()
    proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def _read_http_response(reader: asyncio.StreamReader) -> dict[str, Any]:
        status = await reader.readline()
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
            key, value = line.decode("utf-8", errors="replace").split(":", 1)
            headers[key.strip().lower()] = value.strip()
        body = b""
        if headers.get("transfer-encoding", "").lower() == "chunked":
            while True:
                size_line = await reader.readline()
                if not size_line:
                    break
                size = int(size_line.strip(), 16)
                if size == 0:
                    await reader.readline()
                    break
                body += await reader.readexactly(size)
                await reader.readexactly(2)
        else:
            length = int(headers.get("content-length", "0") or "0")
            if length:
                body = await reader.readexactly(length)
        return {
            "status": status.decode("utf-8", errors="replace").strip(),
            "headers": headers,
            "body": body.decode("utf-8", errors="replace"),
        }

    proxy_reader, proxy_writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    request_body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    proxy_writer.write(
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        + f"Host: 127.0.0.1:{proxy_port}\r\n".encode("utf-8")
        + b"Connection: keep-alive\r\n"
        + f"Content-Length: {len(request_body)}\r\n\r\n".encode("utf-8")
        + request_body
    )
    await proxy_writer.drain()
    proxied = await _read_http_response(proxy_reader)

    stream_body = json.dumps({"stream": True}).encode("utf-8")
    proxy_writer.write(
        b"POST /v1/chat/completions/stream HTTP/1.1\r\n"
        + f"Host: 127.0.0.1:{proxy_port}\r\n".encode("utf-8")
        + b"Connection: close\r\n"
        + b"Accept: text/event-stream\r\n"
        + f"Content-Length: {len(stream_body)}\r\n\r\n".encode("utf-8")
        + stream_body
    )
    await proxy_writer.drain()
    streamed = await _read_http_response(proxy_reader)
    proxy_writer.close()
    with contextlib.suppress(Exception):
        await proxy_writer.wait_closed()

    await proxy.stop()
    backend_server.close()
    await backend_server.wait_closed()

    ws_events: list[bytes] = []

    async def _ws_backend(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await reader.readuntil(b"\r\n\r\n")
        if b"Upgrade: websocket" not in request and b"upgrade: websocket" not in request.lower():
            writer.close()
            return
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n\r\n"
        )
        await writer.drain()
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=1.0)
            if data:
                ws_events.append(data)
        except Exception:
            pass
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    ws_server = await asyncio.start_server(_ws_backend, host="127.0.0.1", port=0)
    ws_port = ws_server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    tunnel = WebSocketTunnel(proxy_url=f"ws://127.0.0.1:{ws_port}", session_id="transport-eval")
    ws_connected = await tunnel.connect()
    await tunnel.send(b"hello-wire")
    await asyncio.sleep(0.05)
    await tunnel.disconnect()
    ws_server.close()
    await ws_server.wait_closed()

    sidecar = TunnelSidecar(
        config=LatticeConfig(provider_base_url="http://127.0.0.1:11434"),
        tcp_port=8799,
    )

    checks = {
        "connection_pool_reuse": client_a is client_b and http_version in {"http/2", "http/1.1"},
        "framing_roundtrip": decoded.frame_type == FrameType.REQUEST and bool(decoded.flags & FrameFlags.DICT_COMPRESSED),
        "negotiation_roundtrip": accepted is True and reason == "",
        "resume_window": token_valid and resume_meta["resumed"] is True and resume_meta["replay_chunks"] == 2,
        "replay_buffer": replay_ok,
        "local_socket": socket_received == [frame] and socket_echo == b"wire",
        "proxy_roundtrip": proxied["status"].startswith("HTTP/1.1 200") and backend_requests and backend_requests[0]["headers"].get("host", "").endswith(f":{backend_port}"),
        "proxy_streaming": "data: stream-one" in streamed["body"] and "data: stream-two" in streamed["body"],
        "websocket_tunnel": ws_connected is True and tunnel.state == "disconnected" and bool(ws_events),
        "sidecar_connect_url": sidecar.connect_url == "http://127.0.0.1:8799",
    }
    return EvalSectionReport(
        name="transport_eval",
        kind="local",
        summary={"passed": sum(1 for ok in checks.values() if ok), "total": len(checks)},
        details={
            "checks": checks,
            "http_version": http_version,
            "proxy": {
                "request_count": len(backend_requests),
                "stream_response": streamed["body"],
                "response": proxied["body"],
            },
            "websocket": {
                "connected": ws_connected,
                "frames": len(ws_events),
            },
        },
    )


async def run_integration_eval() -> EvalSectionReport:
    """Run deterministic integration evals for agent setup surfaces."""
    port = 8787
    env_builders = {
        "claude": build_claude_env,
        "codex": build_codex_env,
        "cursor": build_cursor_env,
        "opencode": build_opencode_env,
        "copilot": build_copilot_env,
    }
    envs: dict[str, dict[str, str]] = {
        name: builder(port=port, backend="proxy") for name, builder in env_builders.items()
    }

    with _temp_home() as home:
        mutations = {
            "claude": apply_claude_scope(port),
            "codex": apply_codex_scope(port),
            "cursor": apply_cursor_scope(port),
            "opencode": apply_opencode_scope(port),
            "copilot": apply_copilot_scope(port),
        }

        # Validate written config files and then revert them.
        patch_paths = {
            "claude": home / ".claude" / "settings.json",
            "codex": home / ".codex" / "config.toml",
            "opencode": home / ".config" / "opencode" / "opencode.json",
            "copilot": home / ".copilot" / "config.json",
        }

        wrote = {
            "claude": patch_paths["claude"].exists(),
            "codex": patch_paths["codex"].exists(),
            "opencode": patch_paths["opencode"].exists(),
            "copilot": patch_paths["copilot"].exists(),
            "cursor": mutations["cursor"]["kind"] == "instructions",
        }
        for name, revert in (
            ("claude", revert_claude_scope),
            ("codex", revert_codex_scope),
            ("opencode", revert_opencode_scope),
            ("copilot", revert_copilot_scope),
        ):
            revert(mutations[name])

        reverted = {
            "claude": not patch_paths["claude"].exists() or '"ANTHROPIC_BASE_URL"' not in patch_paths["claude"].read_text(),
            "codex": patch_paths["codex"].exists() and "# --- LATTICE persistent provider ---" not in patch_paths["codex"].read_text(),
            "opencode": patch_paths["opencode"].exists() and "x-lattice-provider" not in patch_paths["opencode"].read_text(),
            "copilot": patch_paths["copilot"].exists() and "lattice_init" not in patch_paths["copilot"].read_text(),
        }

    checks = {
        "claude_env": envs["claude"].get("ANTHROPIC_BASE_URL") == f"http://127.0.0.1:{port}",
        "codex_env": envs["codex"].get("OPENAI_BASE_URL") == f"http://127.0.0.1:{port}/v1",
        "cursor_env": envs["cursor"].get("OPENAI_BASE_URL", "").endswith(f":{port}/v1"),
        "opencode_env": envs["opencode"].get("OPENAI_BASE_URL") == f"http://127.0.0.1:{port}/v1",
        "copilot_env": "COPILOT_PROVIDER_BASE_URL" in envs["copilot"],
        "writes": all(wrote.values()),
        "reverts": all(reverted.values()),
    }
    return EvalSectionReport(
        name="integration_eval",
        kind="local",
        summary={"passed": sum(1 for ok in checks.values() if ok), "total": len(checks)},
        details={
            "checks": checks,
            "envs": _redact_sensitive_mapping(envs),
            "wrote": wrote,
            "reverted": reverted,
        },
    )


async def run_capability_eval() -> EvalSectionReport:
    """Run a local provider capability matrix eval."""
    registry = get_capability_registry()
    providers = [
        "openai",
        "anthropic",
        "gemini",
        "vertex",
        "bedrock",
        "ollama",
        "azure",
        "groq",
        "deepseek",
        "mistral",
        "cohere",
        "openrouter",
        "fireworks",
        "together",
        "perplexity",
        "ai21",
    ]
    rows: list[dict[str, Any]] = []
    cache_modes: dict[str, int] = {}
    for provider in providers:
        cap = registry.get(provider)
        if cap is None:
            continue
        cache_modes[cap.cache_mode.value] = cache_modes.get(cap.cache_mode.value, 0) + 1
        rows.append(
            {
                "provider": provider,
                "cache_mode": cap.cache_mode.value,
                "supports_prompt_caching": cap.supports(Capability.PROMPT_CACHING),
                "supports_tool_calls": cap.supports(Capability.TOOL_CALLS),
                "supports_streaming": cap.supports(Capability.STREAMING),
                "supports_reasoning": cap.supports(Capability.REASONING),
                "supports_structured_output": cap.supports(Capability.STRUCTURED_OUTPUT),
                "supports_http2": cap.supports(Capability.HTTP2),
                "default_base_url": cap.default_base_url,
            }
        )

    checks = {
        "matrix_size": len(rows) >= 10,
        "openai_cache": any(row["provider"] == "openai" and row["cache_mode"] == CacheMode.AUTO_PREFIX.value for row in rows),
        "anthropic_cache": any(row["provider"] == "anthropic" and row["cache_mode"] == CacheMode.EXPLICIT_BREAKPOINT.value for row in rows),
        "explicit_context": any(row["cache_mode"] == CacheMode.EXPLICIT_CONTEXT.value for row in rows),
        "cache_modes_present": len(cache_modes) >= 3,
    }
    return EvalSectionReport(
        name="capability_eval",
        kind="local",
        summary={"passed": sum(1 for ok in checks.values() if ok), "total": len(checks)},
        details={"checks": checks, "cache_modes": cache_modes, "providers": rows},
    )
