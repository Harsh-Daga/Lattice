"""Command-line interface for LATTICE.

Provides commands for running the proxy, diagnostics, and configuration.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from typing import Any

import structlog
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lattice._version import __version__

# Heavy imports deferred to avoid triggering pipeline construction at CLI startup.
# LatticeConfig, integrations, and lifecycle are imported inside the commands that
# need them, not at module level.

logger = structlog.get_logger()
console = Console()


def _get_config() -> Any:
    """Lazy-load LatticeConfig to avoid triggering pipeline at import time."""
    from lattice.core.config import LatticeConfig

    return LatticeConfig.auto()


def _get_pid_mgr() -> Any:
    """Lazy-load PIDManager."""
    from lattice.proxy.lifecycle import PIDManager

    return _get_pid_mgr()


def _start_background(host: str, port: int, workers: Any) -> int:
    """Lazy-load start_background_server."""
    from lattice.proxy.lifecycle import start_background_server

    return _start_background(host=host, port=port, workers=workers)


def _list_agents() -> list[str]:
    """Lazy-load agent registry."""
    from lattice.integrations.registry import list_supported_agents

    return _list_agents()


def __detect_init_targets(global_scope: bool = True) -> list[str]:
    from lattice.integrations.init import detect_init_targets

    return _detect_init_targets(global_scope=global_scope)


def __run_init(targets: list[str], port: int, global_scope: bool) -> dict[str, Any]:
    from lattice.integrations.init import run_init

    return _run_init(targets, port=port, global_scope=global_scope)


def __lace_agent(**kwargs: Any) -> int:
    from lattice.integrations.lace import lace_agent

    return _lace_agent(**kwargs)


def _un_lace_agent(agent: str) -> dict[str, Any]:
    from lattice.integrations.lace import unlace_agent

    return un_lace_agent(agent)


def __list_mutated_agents() -> list[str]:
    from lattice.integrations.mutation_store import list_mutated_agents

    return _list_mutated_agents()


def _print_banner() -> None:
    """Print the LATTICE ASCII banner."""
    banner = """
    ┌─────────────────────────────────────────┐
    │  LATTICE — LLM Transport & Efficiency   │
    │  Optimize · Compress · Accelerate       │
    └─────────────────────────────────────────┘
    """
    console.print(banner, style="cyan")


def main() -> None:
    """Main CLI entry point with subcommand dispatch.

    Usage:
        lattice --help
        lattice proxy start --port 8787
        lattice proxy stop
        lattice proxy status
        lattice init
        lattice lace claude
        lattice unlace claude
        lattice info
        lattice config
        lattice benchmark --phase 0
    """
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return
    if args[0] in ("-v", "--version", "version"):
        console.print(f"lattice {__version__}")
        return

    cmd = args[0]
    cmd_args = args[1:]

    if cmd == "proxy":
        _cmd_proxy(cmd_args)
    elif cmd == "init":
        _cmd_init(cmd_args)
    elif cmd == "lace":
        _cmd_lace(cmd_args)
    elif cmd == "unlace":
        _cmd_unlace(cmd_args)
    elif cmd == "info":
        _cmd_info(cmd_args)
    elif cmd == "config":
        _cmd_config(cmd_args)
    elif cmd == "benchmark":
        _cmd_benchmark(cmd_args)
    elif cmd == "health":
        _cmd_health(cmd_args)
    elif cmd == "status":
        _cmd_agent_status(cmd_args)
    elif cmd == "doctor":
        _cmd_doctor(cmd_args)
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    """Show usage help."""
    console.print(
        Panel.fit(
            "[bold]LATTICE CLI[/bold]\n\n"
            "Commands:\n"
            "  [cyan]proxy[/cyan]      Manage the LATTICE proxy server\n"
            "              subcommands: start, stop, restart, status, run\n"
            "  [cyan]init[/cyan]       One-step setup: detect agents and configure\n"
            "  [cyan]lace[/cyan]       Route an agent through the LATTICE proxy\n"
            "              e.g., lattice lace claude, lattice lace codex\n"
            "  [cyan]unlace[/cyan]     Restore agent to original configuration\n"
            "              e.g., lattice unlace claude\n"
            "  [cyan]status[/cyan]     Show agent status\n"
            "  [cyan]doctor[/cyan]     Diagnose proxy connectivity\n"
            "  [cyan]info[/cyan]       Show version and transform status\n"
            "  [cyan]config[/cyan]     Display resolved configuration\n"
            "  [cyan]benchmark[/cyan]  Run phase benchmark suite\n"
            "  [cyan]health[/cyan]     Check proxy health (requires running proxy)\n\n"
            "Options:\n"
            "  -h, --help     Show this message\n"
            "  -v, --version  Show version\n",
            title="Usage",
            border_style="blue",
        )
    )


# =============================================================================
# proxy command
# =============================================================================


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


def _cmd_init(args: list[str]) -> None:
    """Durable init: detect agents and configure them for LATTICE."""
    port = 8787
    global_scope = True
    start_proxy = False
    targets: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            _print_init_help()
            return
        elif arg == "--port":
            i += 1
            if i < len(args):
                port = int(args[i])
        elif arg == "--local":
            global_scope = False
        elif arg == "--global":
            global_scope = True
        elif arg == "--start-proxy":
            start_proxy = True
        elif not arg.startswith("-"):
            targets.append(arg)
        else:
            console.print(f"[red]Unknown argument: {arg}[/red]")
            _print_init_help()
            sys.exit(1)
        i += 1

    _print_banner()

    if not targets:
        console.print("[cyan]Detecting agents...[/cyan]")
        targets = _detect_init_targets(global_scope=global_scope)
        if not targets:
            console.print(_format_init_empty_error(global_scope))
            sys.exit(1)
        console.print(f"[dim]Auto-detected: {', '.join(targets)}[/dim]\n")

    result = _run_init(targets, port=port, global_scope=global_scope)

    table = Table(title="LATTICE Init Results")
    table.add_column("Agent", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    for target, info in result.get("results", {}).items():
        if info.get("success"):
            table.add_row(target, "[green]✓[/green]", info.get("message", ""))
        else:
            table.add_row(target, "[red]✗[/red]", info.get("message", ""))

    console.print(table)

    # Optionally start proxy
    if start_proxy:
        console.print("\n[cyan]Starting proxy...[/cyan]")
        _cmd_proxy_start(["--port", str(port)])


def _print_init_help() -> None:
    console.print(
        Panel.fit(
            "[bold]lattice init[/bold]\n\n"
            "Install durable LATTICE integrations for supported agents.\n\n"
            "Usage:\n"
            "  lattice init                    # Auto-detect agents\n"
            "  lattice init --start-proxy      # Auto-detect + start proxy\n"
            "  lattice init claude codex       # Configure specific agents\n"
            "  lattice init --port 9999        # Use custom proxy port\n\n"
            "Options:\n"
            "  --port PORT       Proxy port to configure (default: 8787)\n"
            "  --start-proxy     Start proxy after init\n"
            "  --local           Local scope only\n"
            "  --global          Global scope (default)\n"
            "  -h, --help        Show this message\n"
            "  lattice init claude codex       # Configure specific agents\n"
            "  lattice init --port 9999        # Custom proxy port\n"
            "  lattice init --local            # Local scope only\n\n"
            "Agents:\n"
            "  claude, codex, opencode, cursor, copilot\n",
            title="lattice init",
            border_style="green",
        )
    )


def _format_init_empty_error(global_scope: bool) -> str:
    from lattice.integrations.init import _format_empty_detection_error

    return _format_empty_detection_error(global_scope)


# =============================================================================
# lace command
# =============================================================================


def _cmd_lace(args: list[str]) -> None:
    """Route an agent through the LATTICE proxy."""
    if not args or args[0] in ("-h", "--help"):
        _print_lace_help()
        return

    port = 8787
    no_start = False
    no_patch = False
    no_tunnel = False
    dry_run = False
    agent_args: list[str] = []
    agent_name: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--port":
            i += 1
            if i < len(args):
                port = int(args[i])
            else:
                console.print("[red]--port requires a port number[/red]")
                sys.exit(1)
        elif arg == "--no-start":
            no_start = True
        elif arg == "--no-patch":
            no_patch = True
        elif arg == "--no-tunnel":
            no_tunnel = True
        elif arg == "--dry-run":
            dry_run = True
        elif arg in ("-h", "--help"):
            _cmd_lace([])
            return
        elif agent_name is None:
            agent_name = arg
        else:
            agent_args.append(arg)
        i += 1

    if agent_name is None:
        console.print("[red]No agent specified.[/red]")
        _cmd_lace([])
        sys.exit(1)

    valid_agents = _list_agents()
    if agent_name not in valid_agents:
        console.print(f"[red]Unknown agent: {agent_name}[/red]")
        console.print(f"[dim]Supported agents: {', '.join(valid_agents)}[/dim]")
        sys.exit(1)

    exit_code = _lace_agent(
        agent=agent_name,
        args=agent_args,
        port=port,
        no_start=no_start,
        no_patch=no_patch,
        no_tunnel=no_tunnel,
        dry_run=dry_run,
    )
    sys.exit(exit_code)


def _print_lace_help() -> None:
    console.print(
        Panel.fit(
            "[bold]lattice lace[/bold]\n\n"
            "Route an agent through the LATTICE proxy.\n\n"
            "Starts or reuses a LATTICE proxy, starts a persistent sidecar\n"
            "tunnel, configures the agent to route all API calls through it,\n"
            "launches the agent, and cleans up on exit.\n\n"
            "Usage:\n"
            "  lattice lace <agent> [agent_args...]\n"
            "  lattice lace --port 9999 claude\n"
            "  lattice lace --no-start codex\n"
            "  lattice lace --no-patch claude\n"
            "  lattice lace --no-tunnel claude\n"
            "  lattice lace --dry-run claude\n\n"
            "Options:\n"
            "  --port PORT       Proxy port (default: 8787)\n"
            "  --no-start        Assume proxy is already running\n"
            "  --no-patch        Don't modify agent config (env vars only)\n"
            "  --no-tunnel       Skip persistent sidecar (agent → proxy direct)\n"
            "  --dry-run         Show what would happen without executing\n\n"
            "Agents:\n"
            "  claude      Claude Code\n"
            "  codex       OpenAI Codex\n"
            "  cursor      Cursor\n"
            "  opencode    OpenCode\n"
            "  copilot     GitHub Copilot CLI\n"
            "  generic     Any OpenAI-compatible client\n\n"
            "Examples:\n"
            "  lattice lace claude\n"
            "  lattice lace codex --model o4-mini\n"
            "  lattice lace --port 9999 claude\n"
            "  lattice lace --no-start claude",
            title="lattice lace",
            border_style="green",
        )
    )


# =============================================================================
# unlace command
# =============================================================================


def _cmd_unlace(args: list[str]) -> None:
    """Restore an agent's original configuration."""
    if not args or args[0] in ("-h", "--help"):
        _print_unlace_help()
        return

    agent = args[0]
    valid_agents = _list_agents()
    if agent not in valid_agents:
        console.print(f"[red]Unknown agent: {agent}[/red]")
        console.print(f"[dim]Supported agents: {', '.join(valid_agents)}[/dim]")
        sys.exit(1)

    result = un_lace_agent(agent)
    if result.get("success"):
        console.print(f"[green]{result.get('message', f'{agent} unlaced successfully')}[/green]")
    else:
        console.print(f"[red]{result.get('message', f'Failed to unlace {agent}')}[/red]")
        sys.exit(1)


def _print_unlace_help() -> None:
    console.print(
        Panel.fit(
            "[bold]lattice unlace[/bold]\n\n"
            "Restore an agent to its original configuration.\n"
            "Reverses the effects of ``lattice lace``.\n\n"
            "Usage:\n"
            "  lattice unlace <agent>\n\n"
            "Agents:\n"
            "  claude, codex, cursor, opencode, copilot, generic\n\n"
            "Example:\n"
            "  lattice unlace claude",
            title="lattice unlace",
            border_style="green",
        )
    )


# =============================================================================
# info command
# =============================================================================


def _enabled_transforms(config) -> list[str]:
    """Return list of currently enabled transforms."""
    transforms: list[str] = []
    if config.transform_content_profiler:
        transforms.append("content_profiler")
    if config.transform_prefix_opt:
        transforms.append("prefix_opt")
    if config.transform_structural_fingerprint:
        transforms.append("structural_fingerprint")
    if config.transform_self_information:
        transforms.append("self_information")
    if config.transform_message_dedup:
        transforms.append("message_dedup")
    if config.transform_reference_sub:
        transforms.append("reference_sub")
    if config.transform_semantic_compress:
        transforms.append("rate_distortion")
    if config.transform_hierarchical_summary:
        transforms.append("hierarchical_summary")
    if config.transform_tool_filter:
        transforms.append("tool_filter")
    if config.transform_format_conversion:
        transforms.append("format_conversion")
    if config.transform_output_cleanup:
        transforms.append("output_cleanup")
    return transforms


def _cmd_info(args: list[str]) -> None:
    """Show version, transforms, and runtime info."""
    if args and args[0] in ("-h", "--help"):
        console.print("Usage: lattice info")
        return

    config = _get_config()
    transforms = _enabled_transforms(config)

    table = Table(title="LATTICE Information", header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Version", __version__)
    table.add_row("Config Source", "env / lattice.yaml" if config else "defaults")
    table.add_row("Enabled Transforms", ", ".join(transforms) or "none")
    table.add_row("Session Store", config.session_store)
    table.add_row("Session TTL", f"{config.session_ttl_seconds}s")
    table.add_row("Compression Timeout", f"{config.compression_timeout_ms}ms")
    table.add_row("Graceful Degradation", "on" if config.graceful_degradation else "off")

    _print_banner()
    console.print(table)


# =============================================================================
# config command
# =============================================================================


def _cmd_config(args: list[str]) -> None:
    """Display current configuration."""
    if args and args[0] in ("-h", "--help"):
        console.print("Usage: lattice config [--json]")
        return

    config = _get_config()
    as_json = "--json" in args

    if as_json:
        console.print_json(config.model_dump_json())
    else:
        table = Table(title="LatticeConfig", header_style="bold magenta")
        table.add_column("Field", style="magenta")
        table.add_column("Value")

        for key, value in config.model_dump().items():
            if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                value = "***" if value else ""
            table.add_row(key, str(value))

        console.print(table)


# =============================================================================
# benchmark command
# =============================================================================


def _cmd_benchmark(args: list[str]) -> None:
    """Benchmark entrypoint now routes to the production eval suite."""
    _ = args
    console.print("[bold]Benchmarking has moved to benchmarks/evals/cli.py[/bold]")
    console.print("Run: uv run python benchmarks/evals/cli.py --suite all")


# =============================================================================
# health command
# =============================================================================


def _cmd_health(args: list[str]) -> None:
    """Check if proxy is healthy."""
    host = "localhost"
    port = 8787
    i = 0
    while i < len(args):
        if args[i] == "--host":
            i += 1
            if i < len(args):
                host = args[i]
        elif args[i] == "--port":
            i += 1
            if i < len(args):
                port = int(args[i])
        i += 1

    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "healthy":
                console.print(f"[green]Proxy is healthy[/green] at {host}:{port}")
            else:
                console.print(f"[red]Proxy unhealthy[/red]: {data}")
    except Exception as exc:
        console.print(f"[red]Could not reach proxy[/red] at {host}:{port}: {exc}")


# =============================================================================
# status command
# =============================================================================


def _cmd_agent_status(args: list[str]) -> None:
    """Show proxy health, detected agents, and LATTICE routing info."""
    if args and args[0] in ("-h", "--help"):
        console.print("Usage: lattice status")
        return

    _print_banner()
    config = _get_config()
    proxy_url = f"http://{config.proxy_host}:{config.proxy_port}"

    # Proxy health
    proxy_healthy = False
    proxy_detail = "Not running"
    try:
        import urllib.request

        with urllib.request.urlopen(f"{proxy_url}/healthz", timeout=2) as resp:
            if resp.status == 200:
                proxy_healthy = True
                proxy_detail = proxy_url
            else:
                proxy_detail = f"Health check returned {resp.status}"
    except Exception as exc:
        proxy_detail = str(exc)

    table = Table(title="LATTICE Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    table.add_row(
        "Proxy",
        "[green]✓[/green]" if proxy_healthy else "[red]✗[/red]",
        proxy_detail,
    )
    table.add_row("Version", "[green]✓[/green]", __version__)
    table.add_row("Mode", "[green]✓[/green]", config.compression_mode)

    # Mutated agents (durable init)
    

    mutated = _list_mutated_agents()
    if mutated:
        for agent in mutated:
            table.add_row(f"Init: {agent}", "[green]configured[/green]", f"lattice unlace {agent}")

    # Detected agents
    detected = _detect_init_targets(global_scope=True)
    if detected:
        for agent in detected:
            status = "[green]detected[/green]"
            detail = f"lattice lace {agent}"
            if agent in mutated:
                status = "[green]ready[/green]"
                detail = "init + lace ready"
            table.add_row(f"Agent: {agent}", status, detail)
    else:
        table.add_row("Agents", "[yellow]–[/yellow]", "No supported agents found on PATH")

    console.print(table)
    console.print(
        "\n[dim]Use [cyan]lattice lace <agent>[/cyan] to route an agent through LATTICE.[/dim]"
    )


# =============================================================================
# doctor command
# =============================================================================


def _cmd_doctor(args: list[str]) -> None:
    """Diagnose why an agent isn't routing through LATTICE."""
    if args and args[0] in ("-h", "--help"):
        console.print(
            Panel.fit(
                "[bold]lattice doctor[/bold]\n\n"
                "Diagnose LATTICE proxy routing issues.\n\n"
                "Usage:\n"
                "  lattice doctor codex\n"
                "  lattice doctor claude\n"
                "  lattice doctor opencode\n\n"
                "Checks:\n"
                "  1. Is the agent laced (env file / config file patched)?\n"
                "  2. Are required env vars set in the current shell?\n"
                "  3. Is the LATTICE proxy running and healthy?\n"
                "  4. Can we make a test request through the proxy?",
                title="lattice doctor",
                border_style="blue",
            )
        )
        return

    table = Table(title="LATTICE Doctor — Proxy Connectivity")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    # 1. Is proxy running?
    config = _get_config()
    proxy_url = f"http://{config.proxy_host}:{config.proxy_port}"
    import urllib.request

    try:
        with urllib.request.urlopen(f"{proxy_url}/healthz", timeout=2) as resp:
            if resp.status == 200:
                table.add_row("Proxy running", "[green]✓[/green]", proxy_url)
            else:
                table.add_row("Proxy running", "[yellow]⚠[/yellow]", f"Health check returned {resp.status}")
    except Exception as exc:
        table.add_row("Proxy running", "[red]✗[/red]", f"{exc}\nStart with: lattice proxy run")

    # 2. Test request through proxy
    try:
        req = urllib.request.Request(
            f"{proxy_url}/v1/models",
            headers={"Authorization": "Bearer sk-test"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            table.add_row("Proxy reachable", "[green]✓[/green]", f"Status {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            table.add_row("Proxy reachable", "[green]✓[/green]", f"Responding (HTTP {exc.code})")
        else:
            table.add_row("Proxy reachable", "[red]✗[/red]", f"HTTP {exc.code}")
    except Exception as exc:
        table.add_row("Proxy reachable", "[red]✗[/red]", str(exc))

    _print_banner()
    console.print(table)
    note = (
        "\n[bold]Note:[/bold] CLI tools should use "
        "[cyan]lattice lace <agent>[/cyan]"
    )
    console.print(note)
    console.print("        OAuth tokens are forwarded transparently.")


if __name__ == "__main__":
    main()
