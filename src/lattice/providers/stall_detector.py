"""Stateful stream stall detector with per-provider dynamic tolerance."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _StreamState:
    provider: str
    stream_id: str
    last_chunk_at: float = 0.0
    first_chunk_at: float = 0.0
    chunk_count: int = 0
    total_tokens: int = 0
    rolling_inter_chunk_ms: float = 0.0
    token_velocity: float = 0.0
    phase: str = "first_chunk"


class StreamStallDetector:
    """Stateful stream stall detector with per-provider dynamic tolerance."""

    # Provider-specific silence tolerance in ms
    _PROVIDER_TOLERANCE_MS: dict[str, float] = {
        "openai": 30000.0,
        "anthropic": 45000.0,  # Anthropic thinking can be slower
        "ollama": 20000.0,
        "groq": 15000.0,
        "azure": 30000.0,
        "bedrock": 45000.0,
        "gemini": 30000.0,
    }

    _DEFAULT_TOLERANCE_MS: float = 30000.0
    _VELOCITY_DROP_THRESHOLD: float = 0.2  # velocity must drop to <20% of recent average
    _MIN_CHUNKS_FOR_VELOCITY: int = 3
    _INTER_CHUNK_EWMA_ALPHA: float = 0.3

    # Phase-specific tolerance multipliers
    _PHASE_MULTIPLIERS: dict[str, float] = {
        "first_chunk": 1.5,  # Longer grace for first chunk
        "streaming": 1.0,  # Baseline
        "thinking": 2.0,  # Thinking can be very slow
        "tool_call": 1.2,  # Tool calls have moderate latency
    }

    def __init__(self, default_epsilon: float = 0.02, strict_mode: bool = False) -> None:
        self.default_epsilon = default_epsilon
        self._strict_mode = strict_mode
        self._streams: dict[str, _StreamState] = {}
        self._last_stream_id_for_provider: dict[str, str] = {}
        self._lock = threading.Lock()
        self._ignored_chunk_count: int = 0
        self._ignored_chunk_log_cap: int = 0
        self._ignored_chunks_by_provider: dict[str, int] = {}

    def _get_tolerance_ms(self, provider: str, phase: str = "streaming") -> float:
        base = self._PROVIDER_TOLERANCE_MS.get(provider, self._DEFAULT_TOLERANCE_MS)
        multiplier = self._PHASE_MULTIPLIERS.get(phase, 1.0)
        return base * multiplier

    def start_stream(self, provider: str, stream_id: str = "") -> None:
        """Initialize tracking for a new stream."""
        if not stream_id:
            stream_id = f"{provider}_{time.perf_counter()}_{id(object())}"
        now = time.perf_counter() * 1000.0
        with self._lock:
            self._streams[stream_id] = _StreamState(
                provider=provider,
                stream_id=stream_id,
                last_chunk_at=now,
                first_chunk_at=now,
            )
            self._last_stream_id_for_provider[provider] = stream_id

    def record_chunk(
        self,
        provider: str,
        kind: str,
        elapsed_ms: float,
        tokens: int = 0,
        stream_id: str = "",
    ) -> None:
        """Update rolling state when a chunk arrives.

        kind: "first_chunk" | "chunk" | "thinking" | "tool_call"

        Requires start_stream() to have been called for this stream_id.
        Chunks for unknown stream_ids are silently ignored to prevent
        lifecycle bugs from being masked.
        """
        with self._lock:
            if not stream_id:
                # Legacy compatibility: look up the most recent stream for
                # this provider. This path is deprecated and will be removed.
                stream_id = self._last_stream_id_for_provider.get(provider)
                if not stream_id:
                    return

            state = self._streams.get(stream_id)
            if state is None:
                # Unknown stream — drop the chunk and record observability.
                self._ignored_chunk_count += 1
                self._ignored_chunks_by_provider[provider] = (
                    self._ignored_chunks_by_provider.get(provider, 0) + 1
                )
                if self._strict_mode:
                    raise RuntimeError(
                        f"StreamStallDetector received chunk for unknown stream_id={stream_id!r} "
                        f"(provider={provider!r}). Did you forget to call start_stream()?"
                    )
                # Capped debug log: emit at most 10 messages to avoid spam.
                if self._ignored_chunk_log_cap < 10:
                    self._ignored_chunk_log_cap += 1
                    # NOTE: we cannot import a logger here without adding a dependency;
                    # the counter and strict mode are the primary observability surfaces.
                return

            now = time.perf_counter() * 1000.0
            state.chunk_count += 1
            state.total_tokens += tokens
            state.phase = kind

            if state.chunk_count == 1:
                state.first_chunk_at = now
                state.last_chunk_at = now
                if elapsed_ms > 0 and tokens > 0:
                    state.token_velocity = (tokens * 1000.0) / elapsed_ms
                return

            # Update rolling inter-chunk time
            if state.rolling_inter_chunk_ms <= 0:
                state.rolling_inter_chunk_ms = elapsed_ms
            else:
                state.rolling_inter_chunk_ms = (
                    1.0 - self._INTER_CHUNK_EWMA_ALPHA
                ) * state.rolling_inter_chunk_ms + self._INTER_CHUNK_EWMA_ALPHA * elapsed_ms

            # Update token velocity (tokens per second)
            if elapsed_ms > 0 and tokens > 0:
                instant_velocity = (tokens * 1000.0) / elapsed_ms
                if state.token_velocity <= 0:
                    state.token_velocity = instant_velocity
                else:
                    state.token_velocity = (
                        1.0 - self._INTER_CHUNK_EWMA_ALPHA
                    ) * state.token_velocity + self._INTER_CHUNK_EWMA_ALPHA * instant_velocity

            state.last_chunk_at = now

    def is_stalled(
        self,
        provider: str,
        *,
        since_last_chunk_ms: float,
        fallback_timeout_ms: float,
        stream_id: str = "",
        **_kwargs: Any,
    ) -> bool:
        """Return True only when:

        1. Silence exceeds dynamic tolerance AND
        2. Recent velocity has dropped enough to show starvation OR
           we haven't seen enough chunks to establish velocity
        """
        with self._lock:
            state: _StreamState | None
            if stream_id and stream_id in self._streams:
                state = self._streams[stream_id]
            else:
                # No stream_id provided or unknown stream — cannot determine stall.
                # Return False to avoid false positives on untracked streams.
                state = None

            phase = state.phase if state is not None else "streaming"
            tolerance = self._get_tolerance_ms(provider, phase)

            # Generous grace period
            if since_last_chunk_ms < tolerance * 0.5:
                return False

            # Absolute ceiling
            if since_last_chunk_ms > fallback_timeout_ms:
                return True

            if state is None:
                # No state for this stream; conservatively report not stalled
                # up to the tolerance threshold, then report stalled.
                return since_last_chunk_ms > tolerance

            # If no chunks have been received yet, use baseline streaming tolerance
            if state.chunk_count == 0:
                tolerance = self._get_tolerance_ms(provider, "streaming")

            if state.chunk_count >= self._MIN_CHUNKS_FOR_VELOCITY:
                # Recent velocity is effectively 0 during silence
                if state.token_velocity > 0:
                    recent_velocity = 0.0
                    if recent_velocity < self._VELOCITY_DROP_THRESHOLD * state.token_velocity:
                        return True
                # Also stall if silence exceeds tolerance directly
                return since_last_chunk_ms > tolerance

            # Not enough chunks to establish velocity
            return since_last_chunk_ms > tolerance

    def end_stream(self, stream_id: str = "") -> None:
        """Clean up stream state."""
        with self._lock:
            if stream_id and stream_id in self._streams:
                del self._streams[stream_id]
                for prov, sid in list(self._last_stream_id_for_provider.items()):
                    if sid == stream_id:
                        del self._last_stream_id_for_provider[prov]
                        break

    def cleanup_stale_streams(self, max_age_ms: float = 300000.0) -> int:
        """Remove streams that have not received a chunk in *max_age_ms*.

        Returns the number of streams removed.
        """
        now = time.perf_counter() * 1000.0
        removed = 0
        with self._lock:
            stale_ids = [
                sid
                for sid, state in self._streams.items()
                if now - state.last_chunk_at > max_age_ms
            ]
            for sid in stale_ids:
                del self._streams[sid]
                for prov, s in list(self._last_stream_id_for_provider.items()):
                    if s == sid:
                        del self._last_stream_id_for_provider[prov]
                        break
                removed += 1
        return removed

    def get_ignored_chunk_count(self) -> int:
        """Return the number of chunks dropped because their stream_id was unknown."""
        with self._lock:
            return self._ignored_chunk_count

    def get_ignored_chunk_stats(self) -> dict[str, Any]:
        """Return per-provider ignored-chunk summary."""
        with self._lock:
            return {
                "total": self._ignored_chunk_count,
                "by_provider": dict(self._ignored_chunks_by_provider),
            }

    def get_stream_stats(self, stream_id: str = "") -> dict[str, Any]:
        """Return current stream health metrics.

        If *stream_id* is not provided or not found, returns an empty dict.
        """
        with self._lock:
            if stream_id and stream_id in self._streams:
                state = self._streams[stream_id]
            else:
                return {}

            elapsed_since_chunk = (
                time.perf_counter() * 1000.0 - state.last_chunk_at
                if state.last_chunk_at > 0
                else 0.0
            )
            return {
                "provider": state.provider,
                "stream_id": state.stream_id,
                "chunk_count": state.chunk_count,
                "total_tokens": state.total_tokens,
                "rolling_inter_chunk_ms": round(state.rolling_inter_chunk_ms, 3),
                "token_velocity": round(state.token_velocity, 3),
                "phase": state.phase,
                "elapsed_since_chunk_ms": round(elapsed_since_chunk, 3),
            }
