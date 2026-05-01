"""LLMTP gateway server and native session handlers."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.result import is_err, unwrap
from lattice.core.serialization import message_to_dict, request_from_dict, response_to_dict
from lattice.core.session import SessionManager
from lattice.core.transport import Response
from lattice.protocol.cache_planner import get_cache_planner
from lattice.protocol.dictionary_codec import DictionaryCodec
from lattice.protocol.framing import BinaryFramer, FrameFlags, FrameType, MessageAssembler
from lattice.protocol.manifest import manifest_summary
from lattice.protocol.resume import StreamManager
from lattice.providers.transport import DirectHTTPProvider


@dataclasses.dataclass(slots=True)
class ClientConnectionInfo:
    """Connection metadata for native gateway requests."""

    client_id: str = ""
    remote_addr: str = ""
    user_agent: str = ""


class LLMTPGateway:
    """Entry point for LATTICE-native requests and session operations."""

    def __init__(
        self,
        session_manager: SessionManager,
        pipeline: CompressorPipeline,
        provider: DirectHTTPProvider,
        framer: BinaryFramer,
        stream_manager: StreamManager,
        *,
        store: Any | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.pipeline = pipeline
        self.provider = provider
        self.framer = framer
        self.stream_manager = stream_manager
        self.store = store or session_manager.store

    async def handle_request(
        self,
        raw_data: bytes,
        headers: dict[str, str],
        client_info: ClientConnectionInfo,
    ) -> tuple[bytes, dict[str, str]]:
        """Process JSON or binary gateway requests and return same-format response.

        Returns a tuple of (response_bytes, response_metadata). Response metadata
        includes headers like ``x-lattice-framing`` for observability.
        """
        if raw_data[:4] == b"LATT":
            return await self._handle_binary(raw_data, headers, client_info)
        return await self._handle_json(raw_data, headers, client_info)

    async def _handle_binary(
        self,
        raw_data: bytes,
        headers: dict[str, str],
        _client: ClientConnectionInfo,
    ) -> tuple[bytes, dict[str, str]]:
        frame = self.framer.decode_frame(raw_data)
        assembler = MessageAssembler()
        assembler.append(frame)
        if assembler.frame_type not in (FrameType.REQUEST, FrameType.SESSION_START):
            error = self.framer.encode_error(400, f"unsupported frame type: {assembler.frame_type}")
            return error.to_bytes(), {"x-lattice-framing": "native"}

        dict_wire = bool(frame.flags & FrameFlags.DICT_COMPRESSED)
        session_id_hint = headers.get("x-lattice-session-id", "")
        codec: DictionaryCodec | None = None
        payload = assembler.payload
        if dict_wire:
            codec = await self._dictionary_codec_for_session(session_id_hint)
            try:
                payload = codec.decompress(payload)
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                error = self.framer.encode_error(422, f"dict_decode_failed: {exc}")
                return error.to_bytes(), {"x-lattice-framing": "native"}

        body = json.loads(payload.decode("utf-8"))
        request = request_from_dict(body)
        try:
            provider_name = self._resolve_provider_name(body, headers, request.model)
        except ValueError as exc:
            error = self.framer.encode_error(400, str(exc))
            return error.to_bytes(), {"x-lattice-framing": "native"}
        ctx = TransformContext(
            session_id=body.get("session_id"),
            provider=provider_name,
            model=request.model,
        )
        result = await self.pipeline.process(request, ctx)
        if is_err(result):
            error = self.framer.encode_error(422, "pipeline_failed")
            return error.to_bytes(), {"x-lattice-framing": "native"}
        compressed_request = unwrap(result)
        messages = [message_to_dict(m) for m in compressed_request.messages]
        resp = await self.provider.completion(
            model=compressed_request.model,
            messages=messages,
            temperature=compressed_request.temperature,
            max_tokens=compressed_request.max_tokens,
            top_p=compressed_request.top_p,
            tools=compressed_request.tools,
            tool_choice=compressed_request.tool_choice,
            stop=compressed_request.stop,
            stream=False,
            provider_name=provider_name,
            extra_headers=compressed_request.extra_headers,
            extra_body=compressed_request.extra_body,
            **compressed_request.metadata,
        )
        restored = await self._reverse_response(resp, ctx)
        payload = json.dumps(response_to_dict(restored, compressed_request.model)).encode("utf-8")
        response_flags = FrameFlags.NONE
        if dict_wire and codec is not None:
            payload = codec.compress(payload)
            response_flags |= FrameFlags.DICT_COMPRESSED
            if session_id_hint:
                await self._persist_dictionary_codec(session_id_hint, codec)
        out_frame = self.framer.encode_response(payload, flags=response_flags)[0]
        return out_frame.to_bytes(), {"x-lattice-framing": "native"}

    async def _handle_json(
        self,
        raw_data: bytes,
        headers: dict[str, str],
        _client: ClientConnectionInfo,
    ) -> tuple[bytes, dict[str, str]]:
        body = json.loads(raw_data.decode("utf-8"))
        request = request_from_dict(body)
        try:
            provider_name = self._resolve_provider_name(body, headers, request.model)
        except ValueError as exc:
            return (
                json.dumps({"error": "provider_detection_failed", "message": str(exc)}).encode("utf-8"),
                {"x-lattice-framing": "json"},
            )
        ctx = TransformContext(
            session_id=body.get("session_id"),
            provider=provider_name,
            model=request.model,
        )
        result = await self.pipeline.process(request, ctx)
        if is_err(result):
            return (
                json.dumps({"error": "pipeline_failed"}).encode("utf-8"),
                {"x-lattice-framing": "json"},
            )
        compressed_request = unwrap(result)
        messages = [message_to_dict(m) for m in compressed_request.messages]
        resp = await self.provider.completion(
            model=compressed_request.model,
            messages=messages,
            temperature=compressed_request.temperature,
            max_tokens=compressed_request.max_tokens,
            top_p=compressed_request.top_p,
            tools=compressed_request.tools,
            tool_choice=compressed_request.tool_choice,
            stop=compressed_request.stop,
            stream=False,
            provider_name=provider_name,
            extra_headers=compressed_request.extra_headers,
            extra_body=compressed_request.extra_body,
            **compressed_request.metadata,
        )
        restored: Response = await self._reverse_response(resp, ctx)
        return (
            json.dumps(response_to_dict(restored, compressed_request.model)).encode("utf-8"),
            {"x-lattice-framing": "json"},
        )

    async def _reverse_response(self, response: Response, context: TransformContext) -> Response:
        """Handle sync or async reverse pipeline implementations."""
        reversed_result = self.pipeline.reverse(response, context)
        if inspect.isawaitable(reversed_result):
            return await reversed_result
        return reversed_result

    def _resolve_provider_name(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        model: str,
    ) -> str:
        """Resolve provider for gateway requests.

        Provider must be explicit via body/header or model prefix.
        Bare models are rejected to prevent cross-provider fallback.
        """
        for key in ("provider_name", "provider"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

        header_map = {k.lower(): v for k, v in headers.items()}
        header_hint = header_map.get("x-lattice-provider")
        if header_hint and header_hint.strip():
            return header_hint.strip().lower()

        if "/" in model:
            return model.split("/", 1)[0].strip().lower()

        raise ValueError(
            f"Provider not specified. Use either a provider field/header or model prefix "
            f"like 'ollama-cloud/kimi-k2.6:cloud' (got model='{model}')"
        )

    def _summarize_cache_plan(self, provider_name: str, manifest: Any) -> dict[str, Any] | None:
        if manifest is None:
            return None
        try:
            plan = get_cache_planner(provider_name).plan(manifest)
        except (ImportError, ValueError, KeyError) as exc:
            return {
                "provider": provider_name,
                "planner_failure": f"{type(exc).__name__}: {exc}",
            }
        return {
            "provider": provider_name,
            "expected_cached_tokens": plan.expected_cached_tokens,
            "breakpoints": plan.breakpoints,
            "annotations": plan.annotations,
        }

    async def handle_session_start(self, body: dict[str, Any]) -> dict[str, Any]:
        from lattice.core.serialization import message_from_dict

        messages = [message_from_dict(msg) for msg in body.get("messages", [])]
        provider_name = self._resolve_provider_name(body, {}, str(body.get("model", "")))
        session, _ = await self.session_manager.get_or_create_session(
            session_id=None,
            provider=provider_name,
            model=body.get("model", ""),
            messages=messages,
            tools=body.get("tools"),
        )
        return {
            "session_id": session.session_id,
            "anchor_version": session.manifest.anchor_version if session.manifest else 0,
            "anchor_hash": session.manifest.anchor_hash if session.manifest else "",
            "manifest": manifest_summary(session.manifest) if session.manifest else {},
            "cache_plan": self._summarize_cache_plan(provider_name, session.manifest),
        }

    async def handle_session_append(self, body: dict[str, Any]) -> dict[str, Any] | None:
        from lattice.core.serialization import message_from_dict
        from lattice.protocol.manifest import build_manifest, manifest_from_messages

        session_id = body.get("session_id")
        if not session_id:
            return None
        new_messages = [message_from_dict(msg) for msg in body.get("messages", [])]
        current = await self.session_manager.store.get(session_id)
        if current is None:
            return None
        current_version = current.manifest.anchor_version if current.manifest else current.version
        base_manifest = manifest_from_messages(
            session_id=current.session_id,
            messages=[{"role": m.role, "content": m.content} for m in new_messages],
            tools=current.tool_schemas or None,
            model=current.model,
            provider=current.provider,
        )
        new_manifest = build_manifest(
            session_id=current.session_id,
            segments=base_manifest.segments,
            anchor_version=current_version + 1,
            metadata=base_manifest.metadata,
            manifest_id=base_manifest.manifest_id,
        )
        session = await self.session_manager.update_session(session_id, new_messages, manifest=new_manifest)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "anchor_version": new_manifest.anchor_version,
            "anchor_hash": new_manifest.anchor_hash,
            "message_count": session.message_count,
            "manifest": manifest_summary(new_manifest),
            "cache_plan": self._summarize_cache_plan(current.provider, new_manifest),
        }

    async def handle_session_get(self, session_id: str) -> dict[str, Any] | None:
        session = await self.store.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "provider": session.provider,
            "model": session.model,
            "message_count": session.message_count,
            "anchor_version": session.manifest.anchor_version if session.manifest else 0,
            "anchor_hash": session.manifest.anchor_hash if session.manifest else "",
            "manifest": manifest_summary(session.manifest) if session.manifest else {},
            "cache_plan": self._summarize_cache_plan(session.provider, session.manifest),
            "created_at": session.created_at,
            "last_accessed_at": session.last_accessed_at,
        }

    async def handle_session_invalidate(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Invalidate segments in a session manifest.

        Args:
            body: Must contain ``session_id`` and optionally
                ``invalidate_hashes`` (list of segment hashes to remove)
                or ``replace_messages`` (new message parts).

        Returns:
            Updated session info, or None if session not found.
        """
        from lattice.protocol.manifest import apply_delta

        session_id = body.get("session_id")
        if not session_id:
            return None
        session = await self.store.get(session_id)
        if session is None or session.manifest is None:
            return None

        invalidate_hashes = body.get("invalidate_hashes")
        replace_messages = body.get("replace_messages")

        new_manifest = apply_delta(
            session.manifest,
            invalidate_hashes=invalidate_hashes,
            replace_messages=replace_messages,
        )
        session.manifest = new_manifest
        session.bump_version()
        await self.store.set(session)
        return {
            "session_id": session.session_id,
            "anchor_version": new_manifest.anchor_version,
            "anchor_hash": new_manifest.anchor_hash,
            "manifest": manifest_summary(new_manifest),
            "cache_plan": self._summarize_cache_plan(session.provider, new_manifest),
        }

    async def _dictionary_codec_for_session(self, session_id: str) -> DictionaryCodec:
        """Load the dictionary codec snapshot for a session, if any."""
        if not session_id:
            return DictionaryCodec()
        session = await self.store.get(session_id)
        if session is None:
            return DictionaryCodec(session_id=session_id)
        snapshot = session.metadata.get("dict_wire_state")
        if isinstance(snapshot, dict):
            with contextlib.suppress(Exception):
                return DictionaryCodec.from_snapshot(snapshot)
        return DictionaryCodec(session_id=session_id)

    async def _persist_dictionary_codec(self, session_id: str, codec: DictionaryCodec) -> None:
        """Persist dictionary codec state back into the session metadata."""
        if not session_id:
            return
        session = await self.store.get(session_id)
        if session is None:
            return
        session.metadata["dict_wire_state"] = codec.to_snapshot()
        await self.store.set(session)
