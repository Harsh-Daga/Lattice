"""Tests for StreamStallDetector."""

from __future__ import annotations

import pytest

from lattice.providers.stall_detector import StreamStallDetector


def test_true_stall_detection() -> None:
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-1")
    # No chunks recorded, silence > openai tolerance (30000)
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=35000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-1",
        )
        is True
    )


def test_false_positive_slow_stream() -> None:
    detector = StreamStallDetector()
    detector.start_stream("groq", "stream-2")
    # Record slow but steady chunks
    for _ in range(5):
        detector.record_chunk("groq", "chunk", 5000.0, tokens=10, stream_id="stream-2")
    # Still within grace period (groq tolerance=15000, half=7500)
    assert (
        detector.is_stalled(
            "groq",
            since_last_chunk_ms=5000.0,
            fallback_timeout_ms=30000.0,
            stream_id="stream-2",
        )
        is False
    )


def test_velocity_drop_detection() -> None:
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-3")
    # Fast chunks establish high velocity
    for _ in range(5):
        detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-3")
    # Sudden silence: velocity drops to 0
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=35000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-3",
        )
        is True
    )


def test_per_provider_tolerance() -> None:
    detector = StreamStallDetector()
    detector.start_stream("anthropic", "stream-a")
    detector.start_stream("groq", "stream-g")

    # anthropic tolerance=45000, groq=15000
    assert (
        detector.is_stalled(
            "anthropic",
            since_last_chunk_ms=40000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-a",
        )
        is False
    )

    assert (
        detector.is_stalled(
            "groq",
            since_last_chunk_ms=40000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-g",
        )
        is True
    )


def test_state_reset_between_streams() -> None:
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-1")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-1")
    detector.end_stream("stream-1")

    detector.start_stream("openai", "stream-2")
    # New stream should not inherit old state
    stats = detector.get_stream_stats("stream-2")
    assert stats["chunk_count"] == 0
    assert stats["total_tokens"] == 0


def test_thinking_phase_grace_period() -> None:
    detector = StreamStallDetector()
    detector.start_stream("anthropic", "stream-t")
    # Record some chunks then switch to thinking
    detector.record_chunk("anthropic", "chunk", 200.0, tokens=10, stream_id="stream-t")
    detector.record_chunk("anthropic", "thinking", 200.0, tokens=0, stream_id="stream-t")
    # In thinking phase, still within grace period for moderate silence
    # anthropic tolerance=45000, half=22500
    assert (
        detector.is_stalled(
            "anthropic",
            since_last_chunk_ms=20000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-t",
        )
        is False
    )


def test_stream_stats_emitted() -> None:
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-s")
    detector.record_chunk("openai", "first_chunk", 500.0, tokens=10, stream_id="stream-s")
    detector.record_chunk("openai", "chunk", 200.0, tokens=20, stream_id="stream-s")

    stats = detector.get_stream_stats("stream-s")
    assert stats["provider"] == "openai"
    assert stats["chunk_count"] == 2
    assert stats["total_tokens"] == 30
    assert stats["phase"] == "chunk"
    assert "rolling_inter_chunk_ms" in stats
    assert "token_velocity" in stats


def test_tacc_uses_stall_signal() -> None:
    from lattice.transport.congestion import TACCController

    tacc = TACCController(enabled=True)
    tacc.record_stall_state("openai", True)
    stats = tacc.stats("openai")
    assert stats.get("last_stall_detected") is True

    # After recording False it should clear
    tacc.record_stall_state("openai", False)
    stats = tacc.stats("openai")
    assert stats.get("last_stall_detected") is False


def test_fallback_timeout_absolute_ceiling() -> None:
    detector = StreamStallDetector()
    detector.start_stream("groq", "stream-f")
    # Even a healthy stream is stalled if it exceeds fallback_timeout
    for _ in range(5):
        detector.record_chunk("groq", "chunk", 100.0, tokens=50, stream_id="stream-f")
    assert (
        detector.is_stalled(
            "groq",
            since_last_chunk_ms=25000.0,
            fallback_timeout_ms=20000.0,
            stream_id="stream-f",
        )
        is True
    )


def test_grace_period_prevents_false_positive() -> None:
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-g")
    for _ in range(5):
        detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-g")
    # Silence is less than half tolerance (30000/2=15000)
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=10000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-g",
        )
        is False
    )


def test_concurrent_streams_same_provider_isolated() -> None:
    """Two concurrent streams from the same provider do not collide."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-a")
    detector.start_stream("openai", "stream-b")

    # Stream A gets regular chunks
    for _ in range(5):
        detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-a")

    # Stream B gets no chunks
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=35000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-b",
        )
        is True
    )

    # Stream A is still healthy
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=5000.0,
            fallback_timeout_ms=60000.0,
            stream_id="stream-a",
        )
        is False
    )


def test_concurrent_stream_cleanup_is_per_stream() -> None:
    """Ending one stream does not affect the other."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-a")
    detector.start_stream("openai", "stream-b")

    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-a")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-b")

    detector.end_stream("stream-a")

    # Stream B should still be trackable
    stats = detector.get_stream_stats("stream-b")
    assert stats["chunk_count"] == 1
    assert stats["total_tokens"] == 50


def test_interleaved_chunks_maintain_correct_state() -> None:
    """Interleaved chunks from concurrent streams update the right state."""
    detector = StreamStallDetector()
    detector.start_stream("groq", "stream-1")
    detector.start_stream("groq", "stream-2")

    # Interleave chunks
    detector.record_chunk("groq", "chunk", 100.0, tokens=10, stream_id="stream-1")
    detector.record_chunk("groq", "chunk", 200.0, tokens=20, stream_id="stream-2")
    detector.record_chunk("groq", "chunk", 150.0, tokens=15, stream_id="stream-1")
    detector.record_chunk("groq", "chunk", 250.0, tokens=25, stream_id="stream-2")

    stats1 = detector.get_stream_stats("stream-1")
    stats2 = detector.get_stream_stats("stream-2")

    assert stats1["chunk_count"] == 2
    assert stats1["total_tokens"] == 25
    assert stats2["chunk_count"] == 2
    assert stats2["total_tokens"] == 45


def test_legacy_provider_fallback_still_works() -> None:
    """Callers that do not pass stream_id fall back to provider index."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-legacy")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50)

    stats = detector.get_stream_stats("stream-legacy")
    assert stats["chunk_count"] == 1
    assert stats["total_tokens"] == 50


def test_get_stream_stats_unknown_returns_empty() -> None:
    """get_stream_stats should return empty dict for unknown streams."""
    detector = StreamStallDetector()
    assert detector.get_stream_stats("nonexistent") == {}
    assert detector.get_stream_stats("") == {}


def test_is_stalled_unknown_stream_is_conservative() -> None:
    """is_stalled without a valid stream_id should be conservative."""
    detector = StreamStallDetector()
    # Short silence — not stalled even for unknown stream
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=10000.0,
            fallback_timeout_ms=60000.0,
            stream_id="unknown",
        )
        is False
    )
    # Long silence — stalled because we have no state to prove health
    assert (
        detector.is_stalled(
            "openai",
            since_last_chunk_ms=35000.0,
            fallback_timeout_ms=60000.0,
            stream_id="unknown",
        )
        is True
    )


def test_cleanup_stale_streams_removes_old() -> None:
    """cleanup_stale_streams should remove abandoned streams."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-old")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-old")

    # Not stale yet
    assert detector.cleanup_stale_streams(max_age_ms=300000.0) == 0
    assert detector.get_stream_stats("stream-old") != {}

    # Force stale by manipulating last_chunk_at directly
    with detector._lock:
        detector._streams["stream-old"].last_chunk_at -= 400000.0

    assert detector.cleanup_stale_streams(max_age_ms=300000.0) == 1
    assert detector.get_stream_stats("stream-old") == {}


def test_end_stream_without_id_does_nothing() -> None:
    """end_stream with no stream_id should not remove arbitrary streams."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-x")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-x")

    detector.end_stream("")
    # Stream should still exist
    assert detector.get_stream_stats("stream-x") != {}


def test_unknown_stream_id_silently_ignored() -> None:
    """record_chunk for an unknown stream_id should not auto-create state."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-known")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-known")

    # Unknown stream should be silently ignored
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-unknown")
    assert detector.get_stream_stats("stream-unknown") == {}

    # Known stream should not be affected
    stats = detector.get_stream_stats("stream-known")
    assert stats["chunk_count"] == 1


def test_cleanup_stale_streams_does_not_affect_active() -> None:
    """cleanup_stale_streams should only remove abandoned streams."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-active")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-active")

    detector.start_stream("openai", "stream-old")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-old")

    # Force stream-old to be stale
    with detector._lock:
        detector._streams["stream-old"].last_chunk_at -= 400000.0

    removed = detector.cleanup_stale_streams(max_age_ms=300000.0)
    assert removed == 1
    assert detector.get_stream_stats("stream-old") == {}
    assert detector.get_stream_stats("stream-active") != {}


def test_ignored_chunk_counter_increments() -> None:
    """Chunks for unknown stream IDs should increment the ignored counter."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-known")

    assert detector.get_ignored_chunk_count() == 0
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-unknown")
    assert detector.get_ignored_chunk_count() == 1
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-unknown")
    assert detector.get_ignored_chunk_count() == 2


def test_strict_mode_raises_on_unknown_stream() -> None:
    """Strict mode should raise when a chunk arrives for an unknown stream."""
    detector = StreamStallDetector(strict_mode=True)
    detector.start_stream("openai", "stream-known")

    with pytest.raises(RuntimeError, match="unknown stream_id"):
        detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-unknown")


def test_per_provider_ignored_chunk_tracking() -> None:
    """Ignored chunks should be tracked per-provider in stats."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-o")
    detector.start_stream("anthropic", "stream-a")

    detector.record_chunk("openai", "chunk", 100.0, tokens=10, stream_id="stream-unknown-1")
    detector.record_chunk("openai", "chunk", 100.0, tokens=10, stream_id="stream-unknown-2")
    detector.record_chunk("anthropic", "chunk", 100.0, tokens=10, stream_id="stream-unknown-3")

    stats = detector.get_ignored_chunk_stats()
    assert stats["total"] == 3
    assert stats["by_provider"]["openai"] == 2
    assert stats["by_provider"]["anthropic"] == 1


def test_ignored_chunk_stats_returns_zero_when_clean() -> None:
    """No ignored chunks when all stream IDs are valid."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-known")
    detector.record_chunk("openai", "chunk", 100.0, tokens=10, stream_id="stream-known")

    stats = detector.get_ignored_chunk_stats()
    assert stats["total"] == 0
    assert stats["by_provider"] == {}


def test_repeated_unknown_chunks_increment_counter() -> None:
    """Repeated unknown chunks should keep incrementing the counter."""
    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-known")

    for _ in range(5):
        detector.record_chunk("openai", "chunk", 100.0, tokens=10, stream_id="stream-bad")

    stats = detector.get_ignored_chunk_stats()
    assert stats["total"] == 5
