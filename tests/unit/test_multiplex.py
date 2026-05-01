"""Tests for lattice.protocol.multiplex.

Covers:
- Stream / enum definitions
- MultiStreamMux CRUD and scheduling order
- Max-stream enforcement
- Stream migration semantics
"""

from __future__ import annotations

import pytest

from lattice.protocol.multiplex import (
    MultiStreamMux,
    ReliabilityMode,
    Stream,
    StreamState,
    StreamType,
)


class TestEnumsAndDataclass:
    def test_stream_type_has_expected_members(self) -> None:
        assert StreamType.PRIMARY.value == "primary"
        assert StreamType.SPECULATIVE.value == "speculative"
        assert StreamType.REASONING.value == "reasoning"
        assert StreamType.TOOL.value == "tool"
        assert StreamType.DRAFT.value == "draft"

    def test_stream_state_has_expected_members(self) -> None:
        assert StreamState.IDLE.value == "idle"
        assert StreamState.ACTIVE.value == "active"
        assert StreamState.CLOSED.value == "closed"
        assert StreamState.MIGRATED.value == "migrated"

    def test_reliability_mode_has_expected_members(self) -> None:
        assert ReliabilityMode.RELIABLE.value == "reliable"
        assert ReliabilityMode.PARTIAL.value == "partial"
        assert ReliabilityMode.BEST_EFFORT.value == "best_effort"

    def test_stream_defaults(self) -> None:
        s = Stream(stream_id=0, stream_type=StreamType.PRIMARY)
        assert s.priority == 0
        assert s.reliability == ReliabilityMode.RELIABLE
        assert s.state == StreamState.IDLE
        assert s.window is not None
        assert s.created_at > 0


class TestMultiStreamMux:
    def test_create_stream_basic(self) -> None:
        mux = MultiStreamMux()
        s = mux.create_stream(StreamType.PRIMARY)
        assert isinstance(s, Stream)
        assert s.stream_id == 0
        assert s.stream_type == StreamType.PRIMARY
        assert s.state == StreamState.ACTIVE
        assert mux.stream_count == 1

    def test_create_stream_with_options(self) -> None:
        mux = MultiStreamMux()
        s = mux.create_stream(StreamType.SPECULATIVE, priority=5, reliability=ReliabilityMode.BEST_EFFORT)
        assert s.priority == 5
        assert s.reliability == ReliabilityMode.BEST_EFFORT

    def test_multiple_streams_increment_ids(self) -> None:
        mux = MultiStreamMux()
        s0 = mux.create_stream(StreamType.PRIMARY)
        s1 = mux.create_stream(StreamType.TOOL)
        assert s0.stream_id == 0
        assert s1.stream_id == 1
        assert mux.stream_count == 2

    def test_close_stream(self) -> None:
        mux = MultiStreamMux()
        s = mux.create_stream(StreamType.PRIMARY)
        mux.close_stream(s.stream_id)
        assert s.state == StreamState.CLOSED
        assert mux.active_count == 0

    def test_get_priority_order(self) -> None:
        mux = MultiStreamMux()
        s0 = mux.create_stream(StreamType.PRIMARY, priority=2)
        s1 = mux.create_stream(StreamType.SPECULATIVE, priority=1)
        s2 = mux.create_stream(StreamType.TOOL, priority=3)
        order = mux.get_priority_order()
        assert order == [s1.stream_id, s0.stream_id, s2.stream_id]

    def test_priority_order_ignores_closed_and_migrated(self) -> None:
        mux = MultiStreamMux()
        s0 = mux.create_stream(StreamType.PRIMARY, priority=1)
        s1 = mux.create_stream(StreamType.SPECULATIVE, priority=2)
        s2 = mux.create_stream(StreamType.TOOL, priority=3)
        mux.close_stream(s1.stream_id)
        mux.migrate_stream(s2.stream_id)
        order = mux.get_priority_order()
        assert order == [s0.stream_id]

    def test_active_count(self) -> None:
        mux = MultiStreamMux()
        mux.create_stream(StreamType.PRIMARY)
        mux.create_stream(StreamType.TOOL)
        assert mux.active_count == 2
        mux.close_stream(0)
        assert mux.active_count == 1

    def test_stream_count_includes_closed(self) -> None:
        mux = MultiStreamMux()
        mux.create_stream(StreamType.PRIMARY)
        mux.create_stream(StreamType.TOOL)
        mux.close_stream(0)
        assert mux.stream_count == 2

    def test_get_stream(self) -> None:
        mux = MultiStreamMux()
        s = mux.create_stream(StreamType.PRIMARY)
        fetched = mux.get_stream(s.stream_id)
        assert fetched is s
        assert mux.get_stream(999) is None

    def test_get_stream_returns_none_after_nonexistent(self) -> None:
        mux = MultiStreamMux()
        assert mux.get_stream(0) is None

    def test_migrate_stream_success(self) -> None:
        mux = MultiStreamMux()
        s = mux.create_stream(StreamType.PRIMARY)
        ok = mux.migrate_stream(s.stream_id)
        assert ok is True
        assert s.state == StreamState.MIGRATED

    def test_migrate_stream_missing_returns_false(self) -> None:
        mux = MultiStreamMux()
        assert mux.migrate_stream(0) is False

    def test_migrate_stream_closed_returns_false(self) -> None:
        mux = MultiStreamMux()
        s = mux.create_stream(StreamType.PRIMARY)
        mux.close_stream(s.stream_id)
        assert mux.migrate_stream(s.stream_id) is False

    def test_max_streams_raises(self) -> None:
        mux = MultiStreamMux(max_streams=2)
        mux.create_stream(StreamType.PRIMARY)
        mux.create_stream(StreamType.TOOL)
        with pytest.raises(RuntimeError, match="max_streams reached"):
            mux.create_stream(StreamType.REASONING)

    def test_close_nonexistent_noop(self) -> None:
        mux = MultiStreamMux()
        mux.close_stream(999)  # should not raise
        assert mux.active_count == 0
