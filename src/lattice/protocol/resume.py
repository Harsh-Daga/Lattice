"""Resumable streaming for LATTICE.

Implements stream IDs and a replay window so clients can reconnect and
continue from the last acknowledged event after disconnects.

Key concepts:
- **Stream**: A single response stream with a unique ID.
- **Replay window**: Circular buffer of recent SSE chunks.
- **Acknowledgements**: Clients send acks for received chunks.
- **Resume token**: Short-lived token for reconnecting to a stream.

Design
------
Each streaming response gets a `stream_id` and chunks are numbered
sequentially. The proxy keeps a replay window (default 1000 chunks).
If a client disconnects and reconnects with a `resume_token`, the proxy
replays missed chunks from the window.

The resume token is signed with a server secret to prevent tampering.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# StreamChunk
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class StreamChunk:
    """A single chunk in a resumable stream.

    Attributes:
        sequence: Monotonic sequence number within the stream.
        data: The SSE payload (already serialized).
        timestamp: Unix timestamp when the chunk was produced.
    """

    sequence: int
    data: str
    timestamp: float = dataclasses.field(default_factory=time.time)


# =============================================================================
# ReplayWindow
# =============================================================================


class ReplayWindow:
    """Circular buffer of recent stream chunks for replay.

    Thread-safe for single-producer single-consumer within one event loop.
    For multi-producer, wrap access in a lock.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = max(capacity, 1)
        self._chunks: list[StreamChunk | None] = [None] * self._capacity
        self._head: int = 0  # Next write position
        self._count: int = 0  # Total chunks written
        self._log = logger.bind(module="replay_window")

    def append(self, chunk: StreamChunk) -> None:
        """Add a chunk to the window."""
        self._chunks[self._head] = chunk
        self._head = (self._head + 1) % self._capacity
        self._count += 1

    def replay_from(self, sequence: int) -> list[StreamChunk]:
        """Return all chunks with sequence >= the given sequence.

        Returns empty list if the requested sequence is too old
        (fallen out of the window).
        """
        if self._count == 0:
            return []

        # Find the oldest available sequence
        oldest_seq = max(0, self._count - self._capacity)
        if sequence < oldest_seq:
            self._log.warning(
                "replay_sequence_too_old",
                requested=sequence,
                oldest_available=oldest_seq,
            )
            return []

        result: list[StreamChunk] = []
        # Iterate through window in order
        start_idx = (self._head - min(self._count, self._capacity)) % self._capacity
        for i in range(min(self._count, self._capacity)):
            idx = (start_idx + i) % self._capacity
            chunk = self._chunks[idx]
            if chunk is not None and chunk.sequence >= sequence:
                result.append(chunk)
        return result

    @property
    def count(self) -> int:
        return min(self._count, self._capacity)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "capacity": self._capacity,
            "total_written": self._count,
            "available": min(self._count, self._capacity),
        }


# =============================================================================
# StreamState
# =============================================================================


@dataclasses.dataclass(slots=True)
class StreamState:
    """Mutable state for an active or recently-closed stream."""

    stream_id: str
    created_at: float
    last_activity_at: float
    window: ReplayWindow
    client_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    is_closed: bool = False
    final_sequence: int = 0
    sequence: int = 0
    was_resumed: bool = False

    def touch(self) -> None:
        self.last_activity_at = time.time()


# =============================================================================
# ResumeTokenManager
# =============================================================================


class ResumeTokenManager:
    """Creates and validates cryptographically signed resume tokens.

    Tokens are short-lived (default 5 minutes) and bound to a specific
    stream_id + sequence number.
    """

    def __init__(self, secret: str | None = None, ttl_seconds: int = 300) -> None:
        self._secret = (secret or secrets.token_hex(32)).encode("utf-8")
        self._ttl_seconds = ttl_seconds
        self._log = logger.bind(module="resume_token_manager")

    def create_token(self, stream_id: str, sequence: int) -> str:
        """Create a signed resume token.

        Args:
            stream_id: The stream to resume.
            sequence: The next sequence number the client expects.

        Returns:
            Base64-encoded token string.
        """
        payload = {
            "stream_id": stream_id,
            "sequence": sequence,
            "expires": time.time() + self._ttl_seconds,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = hmac.new(self._secret, payload_bytes, hashlib.sha256).hexdigest()[:16]
        token_bytes = payload_bytes + b":" + signature.encode("utf-8")
        return base64.urlsafe_b64encode(token_bytes).decode("utf-8").rstrip("=")

    def validate_token(self, token: str) -> tuple[str, int] | None:
        """Validate a resume token and return (stream_id, sequence).

        Returns None if the token is invalid, expired, or tampered.
        """
        try:
            # Pad token for base64 decoding
            padding = 4 - (len(token) % 4)
            if padding != 4:
                token += "=" * padding
            decoded = base64.urlsafe_b64decode(token.encode("utf-8"))
            payload_bytes, sig = decoded.rsplit(b":", 1)
            payload = json.loads(payload_bytes)

            # Check expiry
            if payload["expires"] < time.time():
                self._log.debug("resume_token_expired")
                return None

            # Verify signature
            expected_sig = hmac.new(self._secret, payload_bytes, hashlib.sha256).hexdigest()[:16]
            if not hmac.compare_digest(expected_sig.encode("utf-8"), sig):
                self._log.warning("resume_token_signature_invalid")
                return None

            return payload["stream_id"], payload["sequence"]
        except Exception:
            self._log.debug("resume_token_decode_failed")
            return None


# =============================================================================
# StreamManager
# =============================================================================


class StreamManager:
    """Manages active streams and their replay windows.

    Usage:
        manager = StreamManager()
        stream_id = manager.create_stream()
        manager.append_chunk(stream_id, sequence=0, data="data: {...}\n\n")
        # Client disconnects, reconnects with resume token
        token = manager.create_resume_token(stream_id, sequence=5)
        stream_id, seq = manager.validate_resume_token(token)
        chunks = manager.replay(stream_id, seq)
    """

    def __init__(
        self,
        window_capacity: int = 1000,
        token_ttl_seconds: int = 300,
        max_streams: int = 10000,
    ) -> None:
        self._streams: dict[str, StreamState] = {}
        self._token_manager = ResumeTokenManager(ttl_seconds=token_ttl_seconds)
        self._window_capacity = window_capacity
        self._max_streams = max_streams
        self._log = logger.bind(module="stream_manager")

    def create_stream(self) -> str:
        """Create a new stream and return its ID."""
        stream_id = f"stream-{secrets.token_urlsafe(16)}"
        now = time.time()
        self._streams[stream_id] = StreamState(
            stream_id=stream_id,
            created_at=now,
            last_activity_at=now,
            window=ReplayWindow(capacity=self._window_capacity),
        )
        self._log.debug("stream_created", stream_id=stream_id)
        # Evict oldest if over limit
        if len(self._streams) > self._max_streams:
            oldest = min(self._streams.values(), key=lambda s: s.last_activity_at)
            del self._streams[oldest.stream_id]
            self._log.info("stream_evicted", stream_id=oldest.stream_id)
        return stream_id

    def append_chunk(self, stream_id: str, sequence: int, data: str) -> bool:
        """Append a chunk to a stream's replay window.

        Returns False if the stream does not exist or is closed.
        """
        state = self._streams.get(stream_id)
        if state is None or state.is_closed:
            return False
        state.window.append(StreamChunk(sequence=sequence, data=data))
        state.sequence = max(state.sequence, sequence)
        state.touch()
        return True

    def close_stream(self, stream_id: str, final_sequence: int) -> None:
        """Mark a stream as closed."""
        state = self._streams.get(stream_id)
        if state:
            state.is_closed = True
            state.final_sequence = final_sequence
            self._log.debug("stream_closed", stream_id=stream_id, final_sequence=final_sequence)

    def migrate_connection(self, stream_id: str, new_client_info: dict[str, Any]) -> bool:
        """Re-bind a stream to a new client connection."""
        state = self._streams.get(stream_id)
        if state is None:
            return False
        state.client_info = dict(new_client_info)
        state.touch()
        self._log.info("connection_migrated", stream_id=stream_id)
        return True

    def replay(self, stream_id: str, sequence: int) -> list[StreamChunk]:
        """Replay chunks from a stream starting at the given sequence."""
        state = self._streams.get(stream_id)
        if state is None:
            return []
        state.was_resumed = True
        return state.window.replay_from(sequence)

    def create_resume_token(self, stream_id: str, sequence: int) -> str:
        """Create a resume token for a stream."""
        return self._token_manager.create_token(stream_id, sequence)

    def validate_resume_token(self, token: str) -> tuple[str, int] | None:
        """Validate a resume token and return (stream_id, sequence)."""
        return self._token_manager.validate_token(token)

    def get_stats(self, stream_id: str) -> dict[str, Any] | None:
        """Return stats for a stream."""
        state = self._streams.get(stream_id)
        if state is None:
            return None
        return {
            "stream_id": stream_id,
            "created_at": state.created_at,
            "last_activity_at": state.last_activity_at,
            "is_closed": state.is_closed,
            "client_info": state.client_info,
            "window": state.window.stats,
        }

    def get_resume_metadata(self, stream_id: str) -> dict[str, Any]:
        state = self._streams.get(stream_id)
        if state is None:
            return {"resumed": False, "reason": "stream_not_found"}
        return {
            "resumed": True,
            "was_resumed": state.was_resumed,
            "replay_chunks": state.window.count,
            "stream_sequence": state.sequence,
        }

    @property
    def stream_count(self) -> int:
        return len(self._streams)
