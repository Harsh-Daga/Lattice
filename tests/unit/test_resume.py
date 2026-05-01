"""Tests for LATTICE resumable streaming protocol.

Tests cover:
- ReplayWindow circular buffer
- ResumeTokenManager signing and validation
- StreamManager lifecycle
- Token expiry and tamper detection
"""

from __future__ import annotations

from lattice.protocol.resume import (
    ReplayWindow,
    ResumeTokenManager,
    StreamChunk,
    StreamManager,
)

# =============================================================================
# ReplayWindow
# =============================================================================

class TestReplayWindow:
    def test_append_and_replay(self) -> None:
        window = ReplayWindow(capacity=10)
        window.append(StreamChunk(sequence=0, data="a"))
        window.append(StreamChunk(sequence=1, data="b"))
        chunks = window.replay_from(0)
        assert len(chunks) == 2
        assert chunks[0].data == "a"
        assert chunks[1].data == "b"

    def test_replay_from_middle(self) -> None:
        window = ReplayWindow(capacity=10)
        for i in range(5):
            window.append(StreamChunk(sequence=i, data=str(i)))
        chunks = window.replay_from(3)
        assert len(chunks) == 2
        assert chunks[0].data == "3"
        assert chunks[1].data == "4"

    def test_replay_too_old(self) -> None:
        window = ReplayWindow(capacity=3)
        for i in range(10):
            window.append(StreamChunk(sequence=i, data=str(i)))
        # Sequence 0 has fallen out of the 3-slot window
        chunks = window.replay_from(0)
        assert chunks == []

    def test_replay_empty(self) -> None:
        window = ReplayWindow(capacity=10)
        assert window.replay_from(0) == []

    def test_circular_wrap(self) -> None:
        window = ReplayWindow(capacity=3)
        for i in range(5):
            window.append(StreamChunk(sequence=i, data=str(i)))
        # Window should contain 2, 3, 4
        chunks = window.replay_from(2)
        assert len(chunks) == 3
        assert chunks[0].data == "2"

    def test_stats(self) -> None:
        window = ReplayWindow(capacity=10)
        for i in range(5):
            window.append(StreamChunk(sequence=i, data=str(i)))
        stats = window.stats
        assert stats["capacity"] == 10
        assert stats["total_written"] == 5
        assert stats["available"] == 5


# =============================================================================
# ResumeTokenManager
# =============================================================================

class TestResumeTokenManager:
    def test_create_and_validate(self) -> None:
        mgr = ResumeTokenManager(secret="test_secret")
        token = mgr.create_token("stream_1", sequence=5)
        result = mgr.validate_token(token)
        assert result is not None
        assert result[0] == "stream_1"
        assert result[1] == 5

    def test_invalid_token(self) -> None:
        mgr = ResumeTokenManager(secret="test_secret")
        assert mgr.validate_token("not_a_valid_token") is None

    def test_tampered_token(self) -> None:
        mgr = ResumeTokenManager(secret="test_secret")
        token = mgr.create_token("stream_1", sequence=5)
        # Tamper with the token
        tampered = token[:-4] + "xxxx"
        assert mgr.validate_token(tampered) is None

    def test_expired_token(self) -> None:
        mgr = ResumeTokenManager(secret="test_secret", ttl_seconds=-1)
        token = mgr.create_token("stream_1", sequence=5)
        assert mgr.validate_token(token) is None

    def test_different_secrets(self) -> None:
        mgr1 = ResumeTokenManager(secret="secret_a")
        mgr2 = ResumeTokenManager(secret="secret_b")
        token = mgr1.create_token("stream_1", sequence=5)
        assert mgr2.validate_token(token) is None


# =============================================================================
# StreamManager
# =============================================================================

class TestStreamManager:
    def test_create_stream(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        assert sid.startswith("stream-")
        assert mgr.stream_count == 1

    def test_append_and_replay(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        mgr.append_chunk(sid, 0, "data: hello\n\n")
        mgr.append_chunk(sid, 1, "data: world\n\n")
        chunks = mgr.replay(sid, 0)
        assert len(chunks) == 2

    def test_append_to_nonexistent_stream(self) -> None:
        mgr = StreamManager()
        assert mgr.append_chunk("nonexistent", 0, "data") is False

    def test_replay_nonexistent_stream(self) -> None:
        mgr = StreamManager()
        assert mgr.replay("nonexistent", 0) == []

    def test_close_stream(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        mgr.append_chunk(sid, 0, "data")
        mgr.close_stream(sid, final_sequence=0)
        # Can still replay closed stream
        chunks = mgr.replay(sid, 0)
        assert len(chunks) == 1
        # But can't append new chunks
        assert mgr.append_chunk(sid, 1, "more") is False

    def test_resume_token_flow(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        for i in range(5):
            mgr.append_chunk(sid, i, f"data: {i}\n\n")

        token = mgr.create_resume_token(sid, sequence=3)
        result = mgr.validate_resume_token(token)
        assert result is not None
        resumed_sid, seq = result
        assert resumed_sid == sid
        assert seq == 3

        chunks = mgr.replay(resumed_sid, seq)
        assert len(chunks) == 2  # sequences 3 and 4

    def test_eviction_when_over_max(self) -> None:
        mgr = StreamManager(max_streams=2)
        s1 = mgr.create_stream()
        mgr.create_stream()
        mgr.create_stream()
        assert mgr.stream_count == 2
        # s1 should have been evicted (oldest)
        assert mgr.replay(s1, 0) == []

    def test_get_stats(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        mgr.append_chunk(sid, 0, "data")
        stats = mgr.get_stats(sid)
        assert stats is not None
        assert stats["stream_id"] == sid
        assert stats["is_closed"] is False
        assert stats["window"]["total_written"] == 1

    def test_get_stats_missing(self) -> None:
        mgr = StreamManager()
        assert mgr.get_stats("nonexistent") is None

    def test_connection_migration(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        ok = mgr.migrate_connection(sid, {"ip": "10.0.0.1", "network": "wifi"})
        assert ok is True
        stats = mgr.get_stats(sid)
        assert stats is not None
        assert stats["client_info"]["network"] == "wifi"

    def test_migration_replay(self) -> None:
        mgr = StreamManager()
        sid = mgr.create_stream()
        mgr.append_chunk(sid, 0, "data: a\n\n")
        mgr.migrate_connection(sid, {"ip": "10.0.0.2"})
        mgr.append_chunk(sid, 1, "data: b\n\n")
        chunks = mgr.replay(sid, 0)
        assert [c.sequence for c in chunks] == [0, 1]
