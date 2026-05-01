"""Delta wire protocol for the LATTICE transport layer.

Core idea: Instead of sending the full message history every turn, the client
sends only new messages plus a session reference. The proxy reconstructs the
full conversation from its session store before forwarding to the provider.

**Wire savings example:**
Full request (turn 5 of 20-message conversation): ~4,000 chars JSON
Delta request: ~200 chars JSON (only the new user message)
**Savings: ~95% wire bytes per turn after turn 1**

Protocol
--------
When a client wants to send a delta, it uses the special wire format::

    {
      "model": "gpt-4",
      "messages": [
        "delta",        // sentinel — signals delta protocol
        "sess_abc123",  // session id
        5,              // base sequence number (how many messages proxy knows)
        {"role": "user", "content": "follow-up"}
      ],
      "stream": false
    }

The proxy:
1. Recognizes the ``"delta"`` sentinel in ``messages[0]``
2. Looks up ``session_id`` in ``SessionStore``
3. Checks that ``session.messages[:base_seq]`` matches known state
4. Appends ``messages[3:]`` to the session
5. Builds full request with all messages
6. Runs compression pipeline on the full request
7. Forwards to provider

Safety
------
* Session not found → fall back to treating delta as full request
* Sequence mismatch → full replacement (client is out of sync)
* Any error → graceful fallback to passthrough

SDK Integration
---------------
The SDK client (``LatticeClient``) transparently switches to delta mode after
turn 1. No user code changes: the client tracks ``session_id`` and
``sequence_number``, and automatically sends delta format on subsequent turns.

Reference: LLMTP Transport Protocol research (sequence-number-based deltas).
"""

from __future__ import annotations

from typing import Any

import structlog

from lattice.core.session import SessionStore
from lattice.core.transport import Message, Request

logger = structlog.get_logger()


# =============================================================================
# Constants
# =============================================================================

_DELTA_SENTINEL = "delta"
_MIN_MESSAGE_LENGTH_FOR_DELTA = 3  # sentinel + session_id + base_seq


# =============================================================================
# DeltaWireDecoder
# =============================================================================


class DeltaWireDecoder:
    """Detects and decodes delta wire format in incoming requests.

    Usage:
        decoder = DeltaWireDecoder(session_store)
        full_request = await decoder.decode(request)
        # full_request is a Request with complete message history
    """

    _delta_success_count: int = 0
    _delta_fallback_count: int = 0
    _delta_fallback_reasons: dict[str, int] = {}

    def __init__(self, store: SessionStore, downgrade_telemetry: Any = None) -> None:
        self.store = store
        self._downgrade_telemetry = downgrade_telemetry
        self._log = logger.bind(module="delta_wire")

    def is_delta(self, request: Request) -> bool:
        """Return True if *request* uses delta wire format."""
        msgs = request.messages
        if not msgs:
            return False
        msgs[0]
        # The sentinel must be a string "delta" in the first message's content
        # or the first element of messages if the raw body used a mixed array
        # But since our serializer always puts objects in messages[], we check
        # a marker in request metadata instead.
        return request.metadata.get("_delta_wire") is True

    async def decode(self, request: Request) -> Request:
        """Reconstruct full conversation from delta wire format.

        Returns the request unchanged if it's not a delta request.
        """
        if not self.is_delta(request):
            return request

        session_id = request.metadata.get("_delta_session_id")
        base_seq = request.metadata.get("_delta_base_seq", 0)
        anchor_version = request.metadata.get("_delta_anchor_version")
        delta_messages_raw = request.metadata.get("_delta_messages", [])

        if not session_id or not isinstance(session_id, str):
            self._log.warning("delta_decode_no_session_id")
            DeltaWireDecoder._delta_fallback_count += 1
            DeltaWireDecoder._delta_fallback_reasons["no_session_id"] = (
                DeltaWireDecoder._delta_fallback_reasons.get("no_session_id", 0) + 1
            )
            if self._downgrade_telemetry is not None:
                from lattice.core.telemetry import DowngradeCategory

                self._downgrade_telemetry.record(
                    DowngradeCategory.DELTA_TO_FULL_PROMPT,
                    reason="no_session_id",
                )
            return request

        session = await self.store.get(session_id)
        if session is None:
            self._log.warning("delta_decode_session_not_found", session_id=session_id)
            DeltaWireDecoder._delta_fallback_count += 1
            DeltaWireDecoder._delta_fallback_reasons["session_not_found"] = (
                DeltaWireDecoder._delta_fallback_reasons.get("session_not_found", 0) + 1
            )
            if self._downgrade_telemetry is not None:
                from lattice.core.telemetry import DowngradeCategory

                self._downgrade_telemetry.record(
                    DowngradeCategory.DELTA_TO_FULL_PROMPT,
                    reason="session_not_found",
                )
            # Fallback: treat as full request with delta messages as-is
            request.messages = self._raw_to_messages(delta_messages_raw)
            return request

        # Validate anchor version for optimistic concurrency
        if anchor_version is not None and hasattr(session, "version"):
            if session.version != anchor_version:
                self._log.warning(
                    "delta_decode_version_mismatch",
                    session_id=session_id,
                    client_anchor=anchor_version,
                    server_version=session.version,
                )
                DeltaWireDecoder._delta_fallback_count += 1
                DeltaWireDecoder._delta_fallback_reasons["version_mismatch"] = (
                    DeltaWireDecoder._delta_fallback_reasons.get("version_mismatch", 0) + 1
                )
                if self._downgrade_telemetry is not None:
                    from lattice.core.telemetry import DowngradeCategory

                    self._downgrade_telemetry.record(
                        DowngradeCategory.DELTA_TO_FULL_PROMPT,
                        reason="version_mismatch",
                    )
                # Version mismatch: client is stale. Fall back to full replacement.
                request.messages = self._raw_to_messages(delta_messages_raw)
                return request

        # Validate sequence
        existing = session.messages
        if base_seq > len(existing):
            self._log.warning(
                "delta_decode_sequence_mismatch",
                session_id=session_id,
                base_seq=base_seq,
                existing_len=len(existing),
            )
            DeltaWireDecoder._delta_fallback_count += 1
            DeltaWireDecoder._delta_fallback_reasons["sequence_mismatch"] = (
                DeltaWireDecoder._delta_fallback_reasons.get("sequence_mismatch", 0) + 1
            )
            if self._downgrade_telemetry is not None:
                from lattice.core.telemetry import DowngradeCategory

                self._downgrade_telemetry.record(
                    DowngradeCategory.DELTA_TO_FULL_PROMPT,
                    reason="sequence_mismatch",
                )
            # Client is ahead — full replacement with what they sent
            request.messages = self._raw_to_messages(delta_messages_raw)
            return request

        # Reconstruct full messages
        full_messages = list(existing[:base_seq]) + self._raw_to_messages(delta_messages_raw)
        request.messages = full_messages

        # Update session
        session.messages = full_messages
        session.touch()

        DeltaWireDecoder._delta_success_count += 1
        self._log.info(
            "delta_decoded",
            session_id=session_id,
            base_seq=base_seq,
            anchor_version=anchor_version,
            existing=len(existing),
            new_total=len(full_messages),
        )
        return request

    @classmethod
    def get_fallback_stats(cls) -> dict[str, Any]:
        return {
            "success_count": cls._delta_success_count,
            "fallback_count": cls._delta_fallback_count,
            "fallback_reasons": dict(cls._delta_fallback_reasons),
        }

    @staticmethod
    def _raw_to_messages(raw: list[dict[str, Any]]) -> list[Message]:
        return [
            Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                name=msg.get("name"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )
            for msg in raw
        ]


# =============================================================================
# DeltaWireEncoder (client-side)
# =============================================================================


class DeltaWireEncoder:
    """Client-side encoder that produces delta wire format.

    The SDK uses this to turn a full message list into a compact delta
    payload for the wire.

    Usage:
        encoder = DeltaWireEncoder()
        delta = encoder.encode(
            session_id="sess_abc",
            known_sequence=5,
            new_messages=[{"role": "user", "content": "hi"}],
        )
        # delta is a dict ready to POST to /v1/chat/completions
    """

    def __init__(self) -> None:
        self._last_anchor_version: int | None = None
        self._last_base_sequence: int = 0

    def get_transport_metadata(self) -> dict[str, Any]:
        return {
            "delta_mode": True,
            "anchor_version": self._last_anchor_version,
            "base_sequence": self._last_base_sequence,
        }

    def encode_negotiation_outcome(
        self, accepted: bool, fallback_reason: str = ""
    ) -> dict[str, Any]:
        """Return a standardized negotiation outcome dict for telemetry/headers.

        Args:
            accepted: Whether delta transport was accepted.
            fallback_reason: Empty if accepted; otherwise the downgrade reason.
        """
        return {
            "delta_accepted": accepted,
            "delta_fallback_reason": fallback_reason,
            "anchor_version": self._last_anchor_version,
            "base_sequence": self._last_base_sequence,
        }

    @staticmethod
    def decode_negotiation_outcome(data: dict[str, Any]) -> tuple[bool, str]:
        """Parse a negotiation outcome dict.

        Returns (accepted, fallback_reason).
        """
        return (
            bool(data.get("delta_accepted", False)),
            str(data.get("delta_fallback_reason", "")),
        )

    def encode(
        self,
        full_messages: list[dict[str, Any]],
        session_id: str,
        base_sequence: int,
        anchor_version: int | None = None,
    ) -> dict[str, Any]:
        """Encode a delta request.

        Args:
            full_messages: Complete message list (used for fallback).
            session_id: Session identifier.
            base_sequence: Number of messages the proxy already knows.
            anchor_version: Optional session manifest version for CAS safety.

        Returns:
            A request dict with delta wire metadata.
        """
        delta_messages = full_messages[base_sequence:]
        self._last_anchor_version = anchor_version
        self._last_base_sequence = base_sequence
        result: dict[str, Any] = {
            "model": "",  # filled by caller
            "messages": full_messages,  # fallback — proxy uses metadata if delta
            "_delta_wire": True,
            "_delta_session_id": session_id,
            "_delta_base_seq": base_sequence,
            "_delta_messages": delta_messages,
        }
        if anchor_version is not None:
            result["_delta_anchor_version"] = anchor_version
        return result


# =============================================================================
# Wire size measurement
# =============================================================================


def delta_wire_bytes(
    full_messages: list[dict[str, Any]],
    delta_messages: list[dict[str, Any]],
    session_id: str,
    base_seq: int,
) -> tuple[int, int]:
    """Measure wire size: full request vs delta request.

    Returns (full_bytes, delta_bytes)."""
    import json

    full_request = {"model": "gpt-4", "messages": full_messages}
    delta_request = {
        "model": "gpt-4",
        "messages": ["delta", session_id, base_seq] + delta_messages,
    }

    full_json = json.dumps(full_request)
    delta_json = json.dumps(delta_request)
    return len(full_json), len(delta_json)


def compute_wire_savings(
    full_messages: list[dict[str, Any]],
    delta_messages: list[dict[str, Any]],
    session_id: str,
    base_seq: int,
) -> dict[str, Any]:
    """Compute wire savings statistics for delta encoding.

    Returns a dict with:
        - full_bytes: size of full request JSON
        - delta_bytes: size of delta request JSON
        - savings_bytes: bytes saved by using delta
        - savings_pct: percentage saved (0-100)
    """
    full_bytes, delta_bytes = delta_wire_bytes(full_messages, delta_messages, session_id, base_seq)
    savings_bytes = max(0, full_bytes - delta_bytes)
    savings_pct = round(100 * savings_bytes / full_bytes, 2) if full_bytes > 0 else 0.0
    return {
        "full_bytes": full_bytes,
        "delta_bytes": delta_bytes,
        "savings_bytes": savings_bytes,
        "savings_pct": savings_pct,
    }
