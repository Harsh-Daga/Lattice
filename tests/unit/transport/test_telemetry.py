"""Transport telemetry headers and dispatcher recording."""

from __future__ import annotations

from lattice.transport.telemetry import TransportTelemetry


def test_telemetry_headers() -> None:
    tel = TransportTelemetry(
        provider="openai",
        model="gpt-4",
        rtt_ms=120.5,
        attempt=2,
        pool_utilization=0.25,
    )
    headers = tel.to_headers()
    assert headers["x-lattice-transport-rtt-ms"] == "120.50"
    assert headers["x-lattice-transport-attempt"] == "2"
    assert headers["x-lattice-transport-pool-utilization"] == "0.250"
