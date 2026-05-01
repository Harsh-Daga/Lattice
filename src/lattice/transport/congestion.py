"""Token-aware congestion control for provider HTTP traffic."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import heapq
import time
from collections import deque
from typing import Any

_EWMA_ALPHA = 0.125
_HIGH_LATENCY_MULTIPLIER = 3.0


@dataclasses.dataclass(slots=True)
class ProviderCongestionState:
    """Per-provider congestion control state."""

    provider: str
    window_size: float = 1.0
    ssthresh: float = 16.0
    rtt_estimate: float = 0.0
    token_rate_estimate: float = 0.0
    last_adjustment: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    in_slow_start: bool = True
    active_requests: int = 0
    active_token_pressure: int = 0
    blocked_until: float = 0.0
    last_stall_detected: bool = False
    pending_waiters: list[_AdmissionRequest] = dataclasses.field(
        default_factory=list,
        repr=False,
    )
    next_waiter_sequence: int = 0
    pending_token_reservations: deque[int] = dataclasses.field(
        default_factory=deque,
        repr=False,
    )

    # New control inputs
    last_ttft_ms: float = 0.0
    token_velocity: float = 0.0
    retry_after: float = 0.0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    batch_pressure: int = 0
    stall_detected: bool = False
    last_decision: str = ""
    last_decision_reason: str = ""

    @property
    def window_limit(self) -> int:
        """Integer concurrency limit derived from the congestion window."""
        return max(1, int(self.window_size))

    def token_window_limit(self, token_budget_per_request: int) -> int:
        """Return the current soft token budget for in-flight requests."""
        return max(1, self.window_limit * max(1, token_budget_per_request))


class AdmissionDecision(enum.Enum):
    ADMIT = "admit"
    DELAY = "delay"
    REJECT = "reject"
    PRIORITY_DOWNGRADE = "priority_downgrade"


@dataclasses.dataclass(order=True, slots=True)
class _AdmissionRequest:
    """Queued admission request for one provider."""

    sort_key: tuple[int, int]
    provider: str = dataclasses.field(compare=False)
    estimated_tokens: int = dataclasses.field(compare=False)
    priority: int = dataclasses.field(compare=False)
    cancelled: bool = dataclasses.field(default=False, compare=False)


class TACCController:
    """AIMD-style token-aware congestion controller for LLM providers."""

    def __init__(
        self,
        enabled: bool = True,
        *,
        token_budget_per_request: int = 4096,
        priority_boost_step: float = 0.05,
    ) -> None:
        self._enabled = enabled
        self._token_budget_per_request = max(1, token_budget_per_request)
        self._priority_boost_step = max(0.0, priority_boost_step)
        self._states: dict[str, ProviderCongestionState] = {}
        self._condition = asyncio.Condition()

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def before_request(
        self,
        provider: str,
        estimated_tokens: int | None = None,
        priority: int = 0,
    ) -> bool:
        """Return True if request can proceed within provider window."""
        if not self._enabled:
            return True
        now = time.monotonic()
        async with self._condition:
            state = self._get_or_create(provider)
            if state.blocked_until > now:
                return False
            if state.active_requests >= state.window_limit:
                return False
            reservation = max(1, int(estimated_tokens or 1))
            priority = max(0, int(priority))
            token_window = state.token_window_limit(self._token_budget_per_request)
            token_window = max(token_window, reservation)
            if priority:
                token_window = int(
                    token_window * (1.0 + min(priority, 10) * self._priority_boost_step)
                )
                token_window = max(token_window, reservation)
            if state.active_token_pressure + reservation > token_window:
                return False
            state.active_requests += 1
            state.active_token_pressure += reservation
            state.pending_token_reservations.append(reservation)
            return True

    async def acquire_request(
        self,
        provider: str,
        estimated_tokens: int | None = None,
        priority: int = 0,
    ) -> bool:
        """Wait until the request can be admitted under the provider window.

        This is the queue-backed admission path used by the transport. It
        preserves the existing AIMD state while avoiding busy-spin loops.
        """
        if not self._enabled:
            return True

        reservation = max(1, int(estimated_tokens or 1))
        priority = max(0, int(priority))

        async with self._condition:
            state = self._get_or_create(provider)
            entry = _AdmissionRequest(
                sort_key=(-priority, state.next_waiter_sequence),
                provider=provider,
                estimated_tokens=reservation,
                priority=priority,
            )
            state.next_waiter_sequence += 1
            heapq.heappush(state.pending_waiters, entry)
            state.queue_depth = len(state.pending_waiters)
            self._condition.notify_all()

            try:
                while True:
                    self._prune_pending_waiters(state)
                    if entry.cancelled:
                        return False

                    now = time.monotonic()
                    if state.blocked_until > now:
                        await self._wait_for_admission(state.blocked_until - now)
                        continue

                    if not state.pending_waiters or state.pending_waiters[0] is not entry:
                        await self._wait_for_admission()
                        continue

                    token_window = state.token_window_limit(self._token_budget_per_request)
                    token_window = max(token_window, reservation)
                    if priority:
                        token_window = int(
                            token_window * (1.0 + min(priority, 10) * self._priority_boost_step)
                        )
                        token_window = max(token_window, reservation)

                    if state.active_requests >= state.window_limit:
                        await self._wait_for_admission()
                        continue
                    if state.active_token_pressure + reservation > token_window:
                        await self._wait_for_admission()
                        continue

                    heapq.heappop(state.pending_waiters)
                    state.queue_depth = len(state.pending_waiters)
                    state.active_requests += 1
                    state.active_token_pressure += reservation
                    state.pending_token_reservations.append(reservation)
                    self._condition.notify_all()
                    return True
            except asyncio.CancelledError:
                entry.cancelled = True
                self._prune_pending_waiters(state)
                self._condition.notify_all()
                raise

    async def release_request(self, provider: str) -> None:
        """Release one in-flight slot without changing congestion estimates."""
        if not self._enabled:
            return
        async with self._condition:
            state = self._states.get(provider)
            if state is None:
                return
            if state.active_requests > 0:
                state.active_requests -= 1
            self._release_token_reservation(state)
            self._condition.notify_all()

    async def after_response(
        self,
        provider: str,
        latency_ms: float,
        tokens_generated: int,
        status_code: int,
        retry_after: float | None = None,
        *,
        cache_hit: bool = False,
        batch_size: int = 1,
    ) -> None:
        """Update provider congestion state after one HTTP response.

        Args:
            provider: Provider name.
            latency_ms: Response latency in milliseconds.
            tokens_generated: Number of completion tokens.
            status_code: HTTP status code.
            retry_after: Retry-After header value in seconds.
            cache_hit: Whether the provider reported a cache hit.
            batch_size: Number of requests batched together (affects pressure).
        """
        now = time.monotonic()
        async with self._condition:
            state = self._get_or_create(provider)
            if state.active_requests > 0:
                state.active_requests -= 1
            self._release_token_reservation(state)

            # Cache hits reduce effective load — treat as partial success credit
            effective_latency = latency_ms
            if cache_hit and state.rtt_estimate > 0:
                effective_latency = max(latency_ms * 0.5, state.rtt_estimate * 0.8)

            if retry_after is not None:
                state.retry_after = retry_after

            if status_code == 200:
                # Congestion signal: latency spike compared to EWMA baseline.
                if state.rtt_estimate > 0 and effective_latency > _HIGH_LATENCY_MULTIPLIER * state.rtt_estimate:
                    self._on_timeout_like_signal(state, now)
                else:
                    self._on_success(state, effective_latency, tokens_generated, now)
                # Batch pressure: large batches are heavier
                if batch_size > 1:
                    state.active_requests = max(0, state.active_requests - (batch_size - 1))
                self._condition.notify_all()
                return

            if status_code in (429, 503):
                self._on_backpressure(state, retry_after, now)
                return

            if status_code >= 500:
                if state.last_stall_detected:
                    self._on_stall_signal(state, now)
                    state.last_stall_detected = False
                else:
                    self._on_timeout_like_signal(state, now)
                self._condition.notify_all()
                return

            # 4xx other than 429 should not aggressively collapse the window.
            state.consecutive_failures += 1
            state.consecutive_successes = 0
            state.last_adjustment = now
            self._condition.notify_all()

    def record_stall_state(self, provider: str, stalled: bool) -> None:
        """Record whether a stall was detected for the current stream."""
        if not self._enabled:
            return
        state = self._get_or_create(provider)
        state.last_stall_detected = stalled
        state.stall_detected = stalled

    async def record_stream_velocity(
        self,
        provider: str,
        tokens_per_second: float,
    ) -> None:
        """Record streaming velocity as a token-rate sample.

        High velocity suggests low load; very low velocity suggests congestion.
        """
        if not self._enabled:
            return
        async with self._condition:
            state = self._get_or_create(provider)
            state.token_velocity = tokens_per_second
            state.token_rate_estimate = self._ewma(
                state.token_rate_estimate, tokens_per_second
            )
            self._condition.notify_all()

    def should_downgrade_priority(
        self,
        provider: str,
        priority: int,
    ) -> bool:
        """Return True if the controller recommends downgrading priority.

        This happens when the provider is under sustained pressure and
        lower-priority requests should be deferred.
        """
        if not self._enabled:
            return False
        state = self._states.get(provider)
        if state is None:
            return False
        # Downgrade if window is collapsed and there are pending waiters
        if state.window_size <= 2.0 and len(state.pending_waiters) > 0:
            return priority < 5
        return False

    def evaluate_admission(
        self,
        provider: str,
        estimated_tokens: int,
        priority: int,
        *,
        cache_hit_expected: bool = False,
        is_batch: bool = False,
        is_speculative: bool = False,
    ) -> tuple[AdmissionDecision, str]:
        """Return decision and reason for admitting a request.

        Decision logic:
        1. If blocked_until > now -> REJECT ("provider_blocked")
        2. If stall_detected and not cache_hit_expected -> DELAY ("post_stall_cooldown")
        3. If window_size <= 1.0 and queue_depth > 5 -> REJECT ("window_collapsed")
        4. If active_token_pressure + estimated > token_window * 1.5 -> DELAY ("token_pressure")
        5. If is_speculative and window_size < 4.0 -> PRIORITY_DOWNGRADE ("speculative_degraded")
        6. If is_batch and batch_pressure > window_size * 2 -> DELAY ("batch_pressure")
        7. Otherwise -> ADMIT ("ok")
        """
        if not self._enabled:
            return AdmissionDecision.ADMIT, "disabled"
        state = self._get_or_create(provider)
        now = time.monotonic()

        if state.blocked_until > now:
            state.last_decision = AdmissionDecision.REJECT.value
            state.last_decision_reason = "provider_blocked"
            return AdmissionDecision.REJECT, "provider_blocked"

        if state.stall_detected and not cache_hit_expected:
            state.last_decision = AdmissionDecision.DELAY.value
            state.last_decision_reason = "post_stall_cooldown"
            return AdmissionDecision.DELAY, "post_stall_cooldown"

        if state.window_size <= 1.0 and state.queue_depth > 5:
            state.last_decision = AdmissionDecision.REJECT.value
            state.last_decision_reason = "window_collapsed"
            return AdmissionDecision.REJECT, "window_collapsed"

        token_window = state.token_window_limit(self._token_budget_per_request)
        if state.active_token_pressure + estimated_tokens > token_window * 1.5:
            state.last_decision = AdmissionDecision.DELAY.value
            state.last_decision_reason = "token_pressure"
            return AdmissionDecision.DELAY, "token_pressure"

        if is_speculative and state.window_size < 4.0:
            state.last_decision = AdmissionDecision.PRIORITY_DOWNGRADE.value
            state.last_decision_reason = "speculative_degraded"
            return AdmissionDecision.PRIORITY_DOWNGRADE, "speculative_degraded"

        if is_batch and state.batch_pressure > state.window_size * 2:
            state.last_decision = AdmissionDecision.DELAY.value
            state.last_decision_reason = "batch_pressure"
            return AdmissionDecision.DELAY, "batch_pressure"

        state.last_decision = AdmissionDecision.ADMIT.value
        state.last_decision_reason = "ok"
        return AdmissionDecision.ADMIT, "ok"

    async def record_ttft(self, provider: str, ttft_ms: float) -> None:
        """Record time-to-first-token for a provider."""
        if not self._enabled:
            return
        async with self._condition:
            state = self._get_or_create(provider)
            state.last_ttft_ms = self._ewma(state.last_ttft_ms, ttft_ms)
            self._condition.notify_all()

    async def record_cache_hit_rate(self, provider: str, hit_rate: float) -> None:
        """Record cache hit rate (0.0-1.0) for a provider."""
        if not self._enabled:
            return
        async with self._condition:
            state = self._get_or_create(provider)
            state.cache_hit_rate = self._ewma(state.cache_hit_rate, hit_rate)
            self._condition.notify_all()

    async def record_batch_pressure(self, provider: str, batch_size: int) -> None:
        """Record current batch size for a provider."""
        if not self._enabled:
            return
        async with self._condition:
            state = self._get_or_create(provider)
            state.batch_pressure = batch_size
            self._condition.notify_all()

    def window_size(self, provider: str) -> int:
        """Current integer window size for a provider."""
        state = self._states.get(provider)
        return state.window_limit if state else 1

    def stats(self, provider: str) -> dict[str, Any]:
        """Return current congestion state metrics for a provider."""
        state = self._states.get(provider)
        if not state:
            return {}
        return {
            "window_size": state.window_limit,
            "window_size_float": round(state.window_size, 3),
            "token_window_limit": state.token_window_limit(self._token_budget_per_request),
            "active_token_pressure": state.active_token_pressure,
            "ssthresh": round(state.ssthresh, 3),
            "rtt_estimate_ms": round(state.rtt_estimate, 3),
            "token_rate_estimate": round(state.token_rate_estimate, 3),
            "in_slow_start": state.in_slow_start,
            "active_requests": state.active_requests,
            "pending_requests": len(state.pending_waiters),
            "pending_token_pressure": sum(
                waiter.estimated_tokens
                for waiter in state.pending_waiters
                if not waiter.cancelled
            ),
            "consecutive_successes": state.consecutive_successes,
            "consecutive_failures": state.consecutive_failures,
            "blocked_until": state.blocked_until,
            "last_stall_detected": state.last_stall_detected,
            "last_ttft_ms": round(state.last_ttft_ms, 3),
            "token_velocity": round(state.token_velocity, 3),
            "cache_hit_rate": round(state.cache_hit_rate, 3),
            "batch_pressure": state.batch_pressure,
            "stall_detected": state.stall_detected,
            "last_decision": state.last_decision,
            "last_decision_reason": state.last_decision_reason,
        }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Return congestion stats for all observed providers."""
        return {provider: self.stats(provider) for provider in sorted(self._states)}

    def _get_or_create(self, provider: str) -> ProviderCongestionState:
        state = self._states.get(provider)
        if state is None:
            state = ProviderCongestionState(provider=provider)
            self._states[provider] = state
        return state

    async def _wait_for_admission(self, timeout: float | None = None) -> None:
        try:
            if timeout is None or timeout <= 0:
                await self._condition.wait()
            else:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
        except TimeoutError:
            return

    @staticmethod
    def _prune_pending_waiters(state: ProviderCongestionState) -> None:
        while state.pending_waiters and state.pending_waiters[0].cancelled:
            heapq.heappop(state.pending_waiters)
        state.queue_depth = len(state.pending_waiters)

    @staticmethod
    def _ewma(previous: float, sample: float) -> float:
        if previous <= 0:
            return sample
        return (1.0 - _EWMA_ALPHA) * previous + _EWMA_ALPHA * sample

    def _on_success(
        self,
        state: ProviderCongestionState,
        latency_ms: float,
        tokens_generated: int,
        now: float,
    ) -> None:
        state.rtt_estimate = self._ewma(state.rtt_estimate, latency_ms)
        if latency_ms > 0 and tokens_generated > 0:
            token_rate = (tokens_generated * 1000.0) / latency_ms
            state.token_rate_estimate = self._ewma(state.token_rate_estimate, token_rate)

        state.consecutive_successes += 1
        state.consecutive_failures = 0

        if state.in_slow_start:
            state.window_size += 1.0
            if state.window_size >= state.ssthresh:
                state.in_slow_start = False
        else:
            state.window_size += 1.0 / max(state.window_size, 1.0)

        state.window_size = max(1.0, state.window_size)
        state.last_adjustment = now

    def _on_timeout_like_signal(self, state: ProviderCongestionState, now: float) -> None:
        state.ssthresh = max(state.window_size / 2.0, 1.0)
        state.window_size = 1.0
        state.in_slow_start = True
        state.consecutive_failures += 1
        state.consecutive_successes = 0
        state.last_adjustment = now

    def _on_stall_signal(self, state: ProviderCongestionState, now: float) -> None:
        # Stronger congestion signal than a plain timeout
        state.ssthresh = max(state.window_size / 4.0, 1.0)
        state.window_size = 1.0
        state.in_slow_start = True
        state.consecutive_failures += 1
        state.consecutive_successes = 0
        state.last_adjustment = now
        state.blocked_until = max(state.blocked_until, now + 2.0)

    def _on_backpressure(
        self,
        state: ProviderCongestionState,
        retry_after: float | None,
        now: float,
    ) -> None:
        state.window_size = max(1.0, state.window_size - 2.0)
        state.ssthresh = max(state.window_size, 1.0)
        state.in_slow_start = state.window_size <= 1.0
        state.consecutive_failures += 1
        state.consecutive_successes = 0
        state.last_adjustment = now
        if retry_after and retry_after > 0:
            state.blocked_until = max(state.blocked_until, now + retry_after)
        self._condition.notify_all()

    @staticmethod
    def _release_token_reservation(state: ProviderCongestionState) -> None:
        if state.pending_token_reservations:
            reservation = state.pending_token_reservations.popleft()
        else:
            reservation = 1
        state.active_token_pressure = max(0, state.active_token_pressure - reservation)
