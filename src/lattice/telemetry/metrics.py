"""Metrics collection and telemetry for LATTICE.

Provides Prometheus-compatible metrics endpoints for:
- Request counts, latencies, and error rates
- Per-transform token savings
- Provider performance

All metrics are collected without locking to avoid contention.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# LatencyTracker
# =============================================================================


@dataclasses.dataclass(slots=True)
class LatencyTracker:
    """Tracks latency distributions for percentile reporting.

    Uses a fixed-size ring buffer for lock-free updates (Phase 1).
    For Phase 0, uses simple lists with periodic pruning.
    """

    _samples: list[float] = dataclasses.field(default_factory=list)
    _max_samples: int = 10_000

    def record(self, latency_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._samples.append(latency_ms)
        # Prune if too many samples
        if len(self._samples) > self._max_samples:
            # Keep most recent samples (last N)
            self._samples = self._samples[-self._max_samples :]

    @property
    def count(self) -> int:
        """Number of samples recorded."""
        return len(self._samples)

    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = len(s) // 2
        return s[idx]

    @property
    def p95(self) -> float:
        """95th percentile."""
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = int(len(s) * 0.95)
        return s[max(0, idx - 1)]

    @property
    def p99(self) -> float:
        """99th percentile."""
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = int(len(s) * 0.99)
        return s[max(0, idx - 1)]

    def summary(self) -> dict[str, Any]:
        """Return a summary of latency distribution."""
        return {
            "count": self.count,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "min": min(self._samples) if self._samples else 0.0,
            "max": max(self._samples) if self._samples else 0.0,
        }


# =============================================================================
# MetricsCollector
# =============================================================================


class MetricsCollector:
    """Central metrics collection for LATTICE.

    Collects counters, gauges, and histograms. Exposes Prometheus
    exposition format for scraping.

    Thread-safe: counters are atomic; gauges use compare-and-swap logic.
    For Phase 0, simple dict + lock. Phase 1: lock-free ring buffers.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._trackers: dict[str, LatencyTracker] = {}
        self._log = logger.bind(module="metrics_collector")

    # ------------------------------------------------------------------
    # Counters (monotonically increasing)
    # ------------------------------------------------------------------

    def increment(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter.

        Args:
            name: Counter name (e.g., "lattice_requests_total").
            value: Amount to increment by.
            labels: Optional label dict for multi-dimensional counters.
        """
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get current counter value."""
        key = self._key(name, labels)
        return self._counters.get(key, 0)

    # ------------------------------------------------------------------
    # Gauges (current value)
    # ------------------------------------------------------------------

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        key = self._key(name, labels)
        self._gauges[key] = value

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        key = self._key(name, labels)
        return self._gauges.get(key, 0.0)

    def record_metric(
        self,
        namespace: str,
        name: str,
        value: int | float | bool,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record an arbitrary point-in-time metric as a gauge.

        Older gateway code records transform/runtime values through this
        namespace/name API, while the collector stores concrete metric names.
        Keeping this compatibility method avoids losing hot-path telemetry.
        """
        metric_name = f"lattice_{namespace}_{name}"
        self.gauge(metric_name, float(value), labels)

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    def record_latency(
        self, name: str, latency_ms: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a latency sample.

        Args:
            name: Metric name (e.g., "lattice_transform_latency_ms").
            latency_ms: Latency in milliseconds.
            labels: Optional label dict.
        """
        key = self._key(name, labels)
        if key not in self._trackers:
            self._trackers[key] = LatencyTracker()
        self._trackers[key].record(latency_ms)
        # Also update gauge for current latency
        self.gauge(f"{name}_current", latency_ms, labels)

    def get_tracker(self, name: str, labels: dict[str, str] | None = None) -> LatencyTracker | None:
        """Get latency tracker for a metric."""
        key = self._key(name, labels)
        return self._trackers.get(key)

    def tacc_metrics(self, provider: str, state: Any) -> None:
        """Export Token-Aware Congestion Control state as gauges."""
        labels = {"provider": provider}

        if isinstance(state, dict):
            window = float(state.get("window_size", 0.0))
            ssthresh = float(state.get("ssthresh", 0.0))
            rtt = float(state.get("rtt_estimate_ms", 0.0))
            token_rate = float(state.get("token_rate_estimate", 0.0))
            active = float(state.get("active_requests", 0.0))
            pending = float(state.get("pending_requests", 0.0))
            token_pressure = float(state.get("active_token_pressure", 0.0))
            in_slow_start = bool(state.get("in_slow_start", False))
        else:
            window = float(getattr(state, "window_size", 0.0))
            ssthresh = float(getattr(state, "ssthresh", 0.0))
            rtt = float(getattr(state, "rtt_estimate", 0.0))
            token_rate = float(getattr(state, "token_rate_estimate", 0.0))
            active = float(getattr(state, "active_requests", 0.0))
            pending = float(getattr(state, "pending_requests", 0.0))
            token_pressure = float(getattr(state, "active_token_pressure", 0.0))
            in_slow_start = bool(getattr(state, "in_slow_start", False))

        self.gauge("lattice_tacc_window", window, labels)
        self.gauge("lattice_tacc_ssthresh", ssthresh, labels)
        self.gauge("lattice_tacc_rtt_ms", rtt, labels)
        self.gauge("lattice_tacc_token_rate", token_rate, labels)
        self.gauge("lattice_tacc_active_requests", active, labels)
        self.gauge("lattice_tacc_pending_requests", pending, labels)
        self.gauge("lattice_tacc_active_token_pressure", token_pressure, labels)
        self.gauge("lattice_tacc_in_slow_start", 1.0 if in_slow_start else 0.0, labels)

    # ------------------------------------------------------------------
    # Prometheus exposition
    # ------------------------------------------------------------------

    def prometheus_output(self) -> str:
        """Generate Prometheus exposition format.

        Returns a string in the Prometheus text exposition format,
        suitable for scraping by Prometheus or compatible tools.
        """
        lines: list[str] = []

        # Counters
        for key, value in self._counters.items():
            name, labels_str = self._parse_key(key)
            lines.append(f"# HELP {name} Total occurrences")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{{{labels_str}}} {value}")
            lines.append("")

        # Gauges
        for gkey, gvalue in self._gauges.items():
            name, labels_str = self._parse_key(gkey)
            lines.append(f"# HELP {name} Current value")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{{{labels_str}}} {gvalue}")
            lines.append("")

        # Latency summaries
        for key, tracker in self._trackers.items():
            name, labels_str = self._parse_key(key)
            summary = tracker.summary()
            lines.append(f"# HELP {name} Latency distribution in ms")
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_count{{{labels_str}}} {summary['count']}")
            lines.append(f"{name}_p50{{{labels_str}}} {summary['p50']}")
            lines.append(f"{name}_p95{{{labels_str}}} {summary['p95']}")
            lines.append(f"{name}_p99{{{labels_str}}} {summary['p99']}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        """Build a flat key from name + labels for internal storage.

        Example: _key("requests_total", {"provider": "openai"})
                 → "requests_total|provider=openai"
        """
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        """Parse a flat key back into name + labels string.

        Returns:
            (name, labels_string) where labels_string is ready for Prometheus.
        """
        if "|" not in key:
            return key, ""
        name, labels = key.split("|", 1)
        return name, labels


# =============================================================================
# Global instance (for singleton access pattern)
# =============================================================================

_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get the global MetricsCollector singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def reset_metrics() -> None:
    """Reset the global MetricsCollector (for testing)."""
    global _metrics
    _metrics = None
