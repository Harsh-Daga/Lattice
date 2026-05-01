"""Deterministic load simulation for TACC validation."""

from __future__ import annotations

import asyncio
import dataclasses
import random

from lattice.transport.congestion import TACCController


@dataclasses.dataclass(slots=True, frozen=True)
class SimulationConfig:
    """Configuration for synthetic request-load simulation."""

    duration_seconds: float = 20.0
    step_seconds: float = 0.02
    arrival_rate_qps: float = 60.0
    service_capacity: int = 6
    static_concurrency: int = 12
    base_latency_ms: float = 120.0
    seed: int = 0


@dataclasses.dataclass(slots=True, frozen=True)
class SimulationMetrics:
    """Simulation result summary."""

    throughput_qps: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    total_requests: int
    completed_requests: int
    failed_requests: int
    blocked_requests: int


@dataclasses.dataclass(slots=True)
class _InFlightRequest:
    finish_time: float
    latency_ms: float
    status_code: int
    tokens_generated: int
    retry_after: float | None


def run_static_concurrency_simulation(config: SimulationConfig) -> SimulationMetrics:
    """Run synthetic simulation with fixed static concurrency."""
    return asyncio.run(_run_simulation(config, use_tacc=False))


def run_tacc_simulation(config: SimulationConfig) -> SimulationMetrics:
    """Run synthetic simulation with TACC-based admission control."""
    return asyncio.run(_run_simulation(config, use_tacc=True))


async def _run_simulation(config: SimulationConfig, *, use_tacc: bool) -> SimulationMetrics:
    rng = random.Random(config.seed)
    controller = TACCController() if use_tacc else None
    inflight: list[_InFlightRequest] = []

    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    blocked_requests = 0
    success_latencies_ms: list[float] = []
    pending_requests = 0

    arrival_budget = 0.0
    now = 0.0
    provider_name = "default"

    while now < config.duration_seconds:
        completed_now = [req for req in inflight if req.finish_time <= now]
        if completed_now:
            inflight = [req for req in inflight if req.finish_time > now]
            for req in completed_now:
                if controller is not None:
                    await controller.after_response(
                        provider_name,
                        req.latency_ms,
                        req.tokens_generated,
                        req.status_code,
                        retry_after=req.retry_after,
                    )
                if req.status_code == 200:
                    completed_requests += 1
                    success_latencies_ms.append(req.latency_ms)
                else:
                    failed_requests += 1

        arrival_budget += config.arrival_rate_qps * config.step_seconds
        arrivals = int(arrival_budget)
        arrival_budget -= arrivals
        total_requests += arrivals
        pending_requests += arrivals

        while pending_requests > 0:
            if controller is not None:
                can_send = await controller.before_request(provider_name)
            else:
                can_send = len(inflight) < config.static_concurrency

            if not can_send:
                break

            active_after_dispatch = len(inflight) + 1
            outcome = _sample_request_outcome(config, rng, active_after_dispatch)
            inflight.append(
                _InFlightRequest(
                    finish_time=now + (outcome.latency_ms / 1000.0),
                    latency_ms=outcome.latency_ms,
                    status_code=outcome.status_code,
                    tokens_generated=outcome.tokens_generated,
                    retry_after=outcome.retry_after,
                )
            )
            pending_requests -= 1

        now += config.step_seconds

    blocked_requests += pending_requests

    # Drain requests dispatched before duration cutoff.
    while inflight:
        now += config.step_seconds
        completed_now = [req for req in inflight if req.finish_time <= now]
        if not completed_now:
            continue
        inflight = [req for req in inflight if req.finish_time > now]
        for req in completed_now:
            if controller is not None:
                await controller.after_response(
                    provider_name,
                    req.latency_ms,
                    req.tokens_generated,
                    req.status_code,
                    retry_after=req.retry_after,
                )
            if req.status_code == 200:
                completed_requests += 1
                success_latencies_ms.append(req.latency_ms)
            else:
                failed_requests += 1

    sent_requests = max(1, total_requests - blocked_requests)
    p95 = _percentile(success_latencies_ms, 0.95)
    p99 = _percentile(success_latencies_ms, 0.99)
    return SimulationMetrics(
        throughput_qps=completed_requests / max(config.duration_seconds, 1e-6),
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        error_rate=failed_requests / sent_requests,
        total_requests=total_requests,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        blocked_requests=blocked_requests,
    )


@dataclasses.dataclass(slots=True, frozen=True)
class _Outcome:
    latency_ms: float
    status_code: int
    tokens_generated: int
    retry_after: float | None


def _sample_request_outcome(
    config: SimulationConfig,
    rng: random.Random,
    active_requests: int,
) -> _Outcome:
    overload = max(0, active_requests - config.service_capacity)
    jitter = rng.uniform(0.9, 1.1)

    latency_ms = config.base_latency_ms * (1.0 + 0.22 * overload) * jitter
    backpressure_probability = min(0.35, 0.012 * (overload**1.2))
    if overload > 0 and rng.random() < backpressure_probability:
        return _Outcome(
            latency_ms=max(10.0, latency_ms * 0.55),
            status_code=429,
            tokens_generated=0,
            retry_after=None,
        )

    tokens_generated = int(rng.uniform(80, 220))
    return _Outcome(
        latency_ms=latency_ms,
        status_code=200,
        tokens_generated=tokens_generated,
        retry_after=None,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(len(ordered) * percentile)
    idx = max(0, min(len(ordered) - 1, idx - 1))
    return ordered[idx]
