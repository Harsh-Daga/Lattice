"""Unit tests for core metrics collector."""

from __future__ import annotations

from lattice.core.metrics import MetricsCollector
from lattice.transport.congestion import ProviderCongestionState


def test_tacc_metrics_exports_provider_gauges() -> None:
    metrics = MetricsCollector()
    state = ProviderCongestionState(
        provider="openai",
        window_size=4.0,
        ssthresh=8.0,
        rtt_estimate=120.5,
        token_rate_estimate=42.0,
        in_slow_start=False,
        active_requests=2,
    )

    metrics.tacc_metrics("openai", state)

    assert metrics.get_gauge("lattice_tacc_window", {"provider": "openai"}) == 4.0
    assert metrics.get_gauge("lattice_tacc_ssthresh", {"provider": "openai"}) == 8.0
    assert metrics.get_gauge("lattice_tacc_rtt_ms", {"provider": "openai"}) == 120.5
    assert metrics.get_gauge("lattice_tacc_token_rate", {"provider": "openai"}) == 42.0
    assert metrics.get_gauge("lattice_tacc_active_requests", {"provider": "openai"}) == 2.0
    assert metrics.get_gauge("lattice_tacc_pending_requests", {"provider": "openai"}) == 0.0
    assert metrics.get_gauge("lattice_tacc_active_token_pressure", {"provider": "openai"}) == 0.0
    assert metrics.get_gauge("lattice_tacc_in_slow_start", {"provider": "openai"}) == 0.0
