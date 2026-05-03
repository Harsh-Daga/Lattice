"""Rich-based live display for LATTICE proxy mode.

Usage::

    from lattice.ui import ProxyLiveDisplay
    display = ProxyLiveDisplay(metrics_client)
    display.start()
    ...
    display.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class ProxyLiveDisplay:
    """Real-time proxy statistics rendered with Rich.

    Displays:
    - Requests / second
    - Average compression ratio
    - Active sessions
    - Per-provider latency
    - Current compression mode
    """

    REFRESH_INTERVAL = 0.5

    def __init__(self, metrics: Any, config: Any | None = None) -> None:
        self.metrics = metrics
        self.config = config
        self.console = Console()
        self._live: Live | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the live display in a background thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the live display."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Render loop — runs in its own thread."""
        with Live(
            self._render(),
            console=self.console,
            refresh_per_second=2,
            screen=False,
        ) as live:
            self._live = live
            while not self._stop_event.is_set():
                live.update(self._render())
                time.sleep(self.REFRESH_INTERVAL)

    def _render(self) -> Panel:
        """Build the current frame."""
        mode = getattr(self.config, "compression_mode", "balanced") if self.config else "balanced"
        uptime_sec = time.perf_counter() - self._start_time

        # Metrics
        total_requests = getattr(self.metrics, "get_counter", lambda k, d=0: d)(
            "lattice_requests_total"
        )
        rps = total_requests / max(uptime_sec, 1.0)

        latency_ms = getattr(self.metrics, "get_histogram_avg", lambda k, d=0.0: d)(
            "lattice_request_latency_ms"
        )
        llm_latency_ms = getattr(self.metrics, "get_histogram_avg", lambda k, d=0.0: d)(
            "lattice_llm_latency_ms"
        )

        active_sessions = getattr(self.metrics, "get_gauge", lambda k, d=0: d)(
            "lattice_active_sessions"
        )

        # Compression stats
        compression = getattr(self.metrics, "get_gauge", lambda k, d="0%": d)(
            "lattice_last_compression_ratio"
        )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="cyan", justify="right")
        table.add_column("Value", style="white")

        table.add_row(
            "Mode",
            Text(
                mode, style="green" if mode == "safe" else "yellow" if mode == "balanced" else "red"
            ),
        )
        table.add_row("Uptime", f"{uptime_sec:.0f}s")
        table.add_row("Requests", str(total_requests))
        table.add_row("RPS", f"{rps:.1f}")
        table.add_row("Proxy Latency", f"{latency_ms:.0f} ms")
        table.add_row("LLM Latency", f"{llm_latency_ms:.0f} ms")
        table.add_row("Sessions", str(active_sessions))
        table.add_row("Compression", str(compression) if compression is not None else "N/A")

        # Provider latency breakdown
        provider_table = Table(title="Providers", show_header=True, box=None)
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Avg ms", justify="right")
        provider_table.add_column("P99 ms", justify="right")

        provider_names: list[str] = getattr(self.metrics, "provider_names", lambda: [])()
        for provider in provider_names:
            avg = getattr(self.metrics, "get_histogram_avg", lambda k, d=0.0: d)(
                f"lattice_provider_latency_ms{{provider={provider}}}"
            )
            p99 = getattr(self.metrics, "get_histogram_p99", lambda k, d=0.0: d)(
                f"lattice_provider_latency_ms{{provider={provider}}}"
            )
            provider_table.add_row(provider, f"{avg:.0f}", f"{p99:.0f}")

        if not provider_names:
            provider_table.add_row("—", "—", "—")

        return Panel(
            table,
            title="[bold]LATTICE Proxy[/bold]",
            subtitle=f"v{getattr(self.config, '__version__', 'dev')}" if self.config else "dev",
            border_style="blue",
        )
