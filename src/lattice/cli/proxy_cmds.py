from __future__ import annotations

import json
import os
import signal
import sys
from typing import Any

import uvicorn
from rich.panel import Panel
from rich.table import Table

from lattice.cli._console import (
    _get_config,
    _get_pid_mgr,
    _print_banner,
    _start_background,
    console,
)
from lattice.cli.info_cmds import _enabled_transforms


def _parse_proxy_args(args: list[str]) -> dict[str, Any]:
    """Parse proxy subcommand arguments."""
    result: dict[str, Any] = {
        "host": None,
        "port": None,
        "workers": None,
        "reload": False,
        "mode": None,
        "no_ui": False,
    }
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            _print_proxy_help()
            sys.exit(0)
        elif arg == "--host":
            i += 1
            if i < len(args):
                result["host"] = args[i]
        elif arg == "--port":
            i += 1
            if i < len(args):
                result["port"] = int(args[i])
        elif arg == "--workers":
            i += 1
            if i < len(args):
                result["workers"] = int(args[i])
        elif arg == "--mode":
            i += 1
            if i < len(args):
                result["mode"] = args[i]
        elif arg == "--reload":
            result["reload"] = True
        elif arg == "--no-ui":
            result["no_ui"] = True
        else:
            console.print(f"[red]Unknown argument: {arg}[/red]")
            _print_proxy_help()
            sys.exit(1)
        i += 1
    return result


def _print_proxy_help() -> None:
    console.print(
        Panel.fit(
            "[bold]lattice proxy[/bold]\n\n"
            "Start the LATTICE proxy server.\n\n"
            "Usage:\n"
            "  lattice proxy run   [--host HOST] [--port PORT] [--workers N] [--mode MODE] [--no-ui] [--reload]\n"
            "  lattice proxy start [--host HOST] [--port PORT] [--workers N] [--mode MODE]\n"
            "  lattice proxy stop  [--grace N] [--force]\n"
            "  lattice proxy restart\n"
            "  lattice proxy status\n\n"
            "Subcommands:\n"
            "  run     Start in foreground (blocks, Ctrl+C to stop)\n"
            "  start   Start in background (detached daemon)\n"
            "  stop    Stop the background proxy\n"
            "  restart Stop, then start the background proxy\n"
            "  status  Show PID, uptime, and health\n\n"
            "Options:\n"
            "  --host HOST    Bind address (default: from config)\n"
            "  --port PORT    Listen port (default: 8787)\n"
            "  --workers N    Number of workers (default: auto)\n"
            "  --mode MODE    Compression mode: safe | balanced | aggressive (default: balanced)\n"
            "  --no-ui        Disable Rich live display in foreground mode\n"
            "  --reload       Enable auto-reload (development)\n"
            "  --grace N      Seconds for graceful shutdown (default: 10)\n"
            "  --force        Skip graceful period, SIGKILL immediately\n"
            "  -h, --help     Show this message\n",
            title="lattice proxy",
            border_style="green",
        )
    )


def _cmd_proxy(args: list[str]) -> None:
    """Proxy lifecycle management: start / stop / restart / status / run."""
    if not args:
        args = ["run"]
    if args[0] in ("-h", "--help"):
        _print_proxy_help()
        return

    subcmd = args[0]
    sub_args = args[1:]

    if subcmd == "run":
        _cmd_proxy_run(sub_args)
    elif subcmd == "start":
        _cmd_proxy_start(sub_args)
    elif subcmd == "stop":
        _cmd_proxy_stop(sub_args)
    elif subcmd == "restart":
        _cmd_proxy_restart(sub_args)
    elif subcmd == "status":
        _cmd_proxy_status(sub_args)
    else:
        console.print(f"[red]Unknown proxy subcommand: {subcmd}[/red]")
        _print_proxy_help()
        sys.exit(1)


def _cmd_proxy_run(args: list[str]) -> None:
    """Start the proxy in the foreground with optional live UI."""
    parsed = _parse_proxy_args(args)
    config = _get_config()

    host = parsed["host"] or config.proxy_host
    port = parsed["port"] or config.proxy_port
    workers = parsed["workers"] or config.proxy_workers
    reload = parsed["reload"] or config.proxy_reload
    mode = parsed["mode"]
    if mode:
        config.compression_mode = mode
        config.apply_compression_mode()

    _print_banner()
    console.print(
        f"Starting proxy on [bold]{host}:{port}[/bold]"
        + (f" with {workers} workers" if workers else " with auto workers")
    )
    console.print(f"Mode: [bold]{config.compression_mode}[/bold]")
    console.print(f"Provider: [bold]{config.provider_base_url or 'https://api.openai.com'}[/bold]")
    console.print("Transforms:", ", ".join(_enabled_transforms(config)))

    # Live UI (disabled with --no-ui or in reload/dev mode)
    live_display = None
    if not parsed["no_ui"] and not reload and workers is None:
        try:
            from lattice.ui import ProxyLiveDisplay

            # Dummy metrics object for now — real metrics injected at runtime
            class _DummyMetrics:
                def get_counter(self, _key: str, default: int = 0) -> int:
                    return default

                def get_histogram_avg(self, _key: str, default: float = 0.0) -> float:
                    return default

                def get_histogram_p99(self, _key: str, default: float = 0.0) -> float:
                    return default

                def get_gauge(self, _key: str, default: Any = 0) -> Any:
                    return default

                def provider_names(self) -> list[str]:
                    return []

            live_display = ProxyLiveDisplay(_DummyMetrics(), config)
            live_display.start()
            console.print("[dim]Live display enabled (Ctrl+C to stop)[/dim]")
        except Exception:
            pass

    try:
        uvicorn.run(
            "lattice.proxy.server:create_app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
        )
    finally:
        if live_display is not None:
            live_display.stop()


def _cmd_proxy_start(args: list[str]) -> None:
    """Start the proxy in the background (daemon/double-fork)."""
    parsed = _parse_proxy_args(args)
    config = _get_config()
    host = parsed["host"] or config.proxy_host
    port = parsed["port"] or config.proxy_port
    workers = parsed["workers"] or config.proxy_workers
    mode = parsed["mode"]
    if mode:
        config.compression_mode = mode
        config.apply_compression_mode()

    pid_mgr = _get_pid_mgr()
    status = pid_mgr.status()
    if status is not None:
        console.print(
            f"[yellow]Proxy already running (PID {status.pid})[/yellow]"
            f" — use `lattice proxy restart` to cycle."
        )
        sys.exit(1)

    _print_banner()
    console.print(f"Starting proxy in background on [bold]{host}:{port}[/bold]")
    if mode:
        console.print(f"Mode: [bold]{config.compression_mode}[/bold]")
    pid = _start_background(host=host, port=port, workers=workers)
    pid_mgr.write(pid)

    # Wait a moment for the server to bind
    import time
    import urllib.request

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.1)

    console.print(f"[green]✓ Proxy started (PID {pid})[/green]")
    console.print(f"  Health: http://{host}:{port}/healthz")
    console.print(f"  Proxy:  http://{host}:{port}/v1")


def _cmd_proxy_stop(args: list[str]) -> None:
    """Stop the background proxy via PID file."""
    grace_period = 10.0
    force = False
    i = 0
    while i < len(args):
        if args[i] == "--grace":
            i += 1
            if i < len(args):
                grace_period = float(args[i])
        elif args[i] == "--force":
            force = True
        elif args[i] in ("-h", "--help"):
            _print_proxy_help()
            return
        else:
            console.print(f"[red]Unknown argument: {args[i]}[/red]")
            sys.exit(1)
        i += 1

    pid_mgr = _get_pid_mgr()
    if force:
        console.print("[yellow]Force-stopping proxy...[/yellow]")
        pid = pid_mgr.read()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
                pid_mgr.remove()
                console.print(f"[green]✓ Process {pid} killed[/green]")
            except Exception as exc:
                console.print(f"[red]Failed to kill {pid}: {exc}[/red]")
                sys.exit(1)
        else:
            console.print("[yellow]No running proxy found.[/yellow]")
    else:
        result = pid_mgr.stop(grace_period=grace_period)
        if result.stopped:
            console.print(f"[green]✓ {result.message}[/green]")
        else:
            console.print(f"[red]✗ {result.message}[/red]")
            sys.exit(1)


def _cmd_proxy_restart(args: list[str]) -> None:
    """Restart the background proxy."""
    pid_mgr = _get_pid_mgr()
    if pid_mgr.is_running():
        console.print("Stopping existing proxy...")
        result = pid_mgr.stop(grace_period=5.0)
        console.print(f"  {result.message}")

    _cmd_proxy_start(args)


def _cmd_proxy_status(_args: list[str]) -> None:
    """Show proxy status: PID, uptime, and health."""
    pid_mgr = _get_pid_mgr()
    status = pid_mgr.status()

    table = Table(title="LATTICE Proxy Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    if status is None:
        table.add_row("Status", "[yellow]Not running[/yellow]")
        table.add_row("PID", "—")
    else:
        table.add_row("Status", "[green]Running[/green]")
        table.add_row("PID", str(status.pid))
        if status.process_name:
            table.add_row("Process", status.process_name)
        if status.uptime_seconds is not None:
            table.add_row("Uptime", f"{status.uptime_seconds:.1f}s")

        # Health check
        import urllib.request

        config = _get_config()
        try:
            url = f"http://{config.proxy_host}:{config.proxy_port}/healthz"
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                table.add_row("Health", f"[green]{data.get('status', 'unknown')}[/green]")
        except Exception as exc:
            table.add_row("Health", f"[red]Unreachable: {exc}[/red]")

    _print_banner()
    console.print(table)


# =============================================================================
# init command
# =============================================================================


