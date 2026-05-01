"""Simulation tests for TACC vs static concurrency."""

from __future__ import annotations

from lattice.transport.simulation import (
    SimulationConfig,
    run_static_concurrency_simulation,
    run_tacc_simulation,
)


def test_tacc_reduces_tail_latency_and_errors_under_load() -> None:
    config = SimulationConfig(
        duration_seconds=20.0,
        step_seconds=0.02,
        arrival_rate_qps=60.0,
        service_capacity=6,
        static_concurrency=30,
        base_latency_ms=120.0,
        seed=42,
    )

    static_metrics = run_static_concurrency_simulation(config)
    tacc_metrics = run_tacc_simulation(config)

    assert tacc_metrics.p99_latency_ms < static_metrics.p99_latency_ms
    assert tacc_metrics.error_rate < static_metrics.error_rate
    assert tacc_metrics.throughput_qps >= static_metrics.throughput_qps * 0.9


def test_simulation_metrics_are_well_formed() -> None:
    config = SimulationConfig(
        duration_seconds=5.0,
        step_seconds=0.02,
        arrival_rate_qps=10.0,
        service_capacity=6,
        static_concurrency=6,
        base_latency_ms=80.0,
        seed=7,
    )

    metrics = run_tacc_simulation(config)
    assert metrics.total_requests > 0
    assert metrics.completed_requests >= 0
    assert metrics.completed_requests <= metrics.total_requests
    assert metrics.p99_latency_ms >= metrics.p95_latency_ms
