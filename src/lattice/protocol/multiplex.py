"""Multi-stream multiplexer for LATTICE protocol.

Provides QUIC-like stream multiplexing for LLM sessions, with per-stream
typing, priority, reliability, and lifecycle management.
"""

from __future__ import annotations

import dataclasses
import enum
import time

from lattice.protocol.resume import ReplayWindow


class StreamType(enum.Enum):
    """Categories of multiplexed streams."""

    PRIMARY = "primary"
    SPECULATIVE = "speculative"
    REASONING = "reasoning"
    TOOL = "tool"
    DRAFT = "draft"


class StreamState(enum.Enum):
    """Lifecycle states for a stream."""

    IDLE = "idle"
    ACTIVE = "active"
    CLOSED = "closed"
    MIGRATED = "migrated"


class ReliabilityMode(enum.Enum):
    """Delivery reliability guarantees."""

    RELIABLE = "reliable"
    PARTIAL = "partial"
    BEST_EFFORT = "best_effort"


@dataclasses.dataclass(slots=True)
class Stream:
    """A single multiplexed stream within a session.

    Attributes:
        stream_id: Monotonically increasing integer ID.
        stream_type: Semantic category of the stream.
        priority: Lower values are scheduled first.
        reliability: Delivery guarantee level.
        state: Current lifecycle state.
        window: Replay window for resumability.
        created_at: Unix timestamp when the stream was created.
    """

    stream_id: int
    stream_type: StreamType
    priority: int = 0
    reliability: ReliabilityMode = ReliabilityMode.RELIABLE
    state: StreamState = StreamState.IDLE
    window: ReplayWindow = dataclasses.field(default_factory=lambda: ReplayWindow(1000))
    created_at: float = dataclasses.field(default_factory=time.time)


class MultiStreamMux:
    """QUIC-like stream multiplexer for LLM sessions.

    Manages multiple independent streams within one session, each with its own
    type, priority, reliability, and flow control.
    """

    def __init__(self, max_streams: int = 32) -> None:
        self._streams: dict[int, Stream] = {}
        self._next_id: int = 0
        self.max_streams = max_streams

    def create_stream(
        self,
        stream_type: StreamType,
        priority: int = 0,
        reliability: ReliabilityMode = ReliabilityMode.RELIABLE,
    ) -> Stream:
        """Create a new stream within the session.

        Args:
            stream_type: Semantic category for the stream.
            priority: Scheduling priority (lower = higher priority).
            reliability: Delivery guarantee level.

        Returns:
            The newly created Stream.

        Raises:
            RuntimeError: If the maximum number of streams has been reached.
        """
        if len(self._streams) >= self.max_streams:
            raise RuntimeError("max_streams reached")
        sid = self._next_id
        self._next_id += 1
        stream = Stream(
            stream_id=sid,
            stream_type=stream_type,
            priority=priority,
            reliability=reliability,
            state=StreamState.ACTIVE,
        )
        self._streams[sid] = stream
        return stream

    def close_stream(self, sid: int) -> None:
        """Close a stream by ID."""
        stream = self._streams.get(sid)
        if stream:
            stream.state = StreamState.CLOSED

    def get_priority_order(self) -> list[int]:
        """Return stream IDs in priority order for scheduling.

        Only includes streams whose state is ACTIVE.
        """
        active = [s for s in self._streams.values() if s.state == StreamState.ACTIVE]
        active.sort(key=lambda s: s.priority)
        return [s.stream_id for s in active]

    @property
    def active_count(self) -> int:
        """Number of currently active streams."""
        return sum(1 for s in self._streams.values() if s.state == StreamState.ACTIVE)

    @property
    def stream_count(self) -> int:
        """Total number of streams (including closed / migrated)."""
        return len(self._streams)

    def get_stream(self, sid: int) -> Stream | None:
        """Retrieve a stream by ID, or None if not found."""
        return self._streams.get(sid)

    def migrate_stream(self, sid: int) -> bool:
        """Migrate a stream to a new connection.

        Sets the stream state to MIGRATED and returns True.  Returns False
        if the stream does not exist or is already closed.
        """
        stream = self._streams.get(sid)
        if stream is None:
            return False
        if stream.state == StreamState.CLOSED:
            return False
        stream.state = StreamState.MIGRATED
        return True
