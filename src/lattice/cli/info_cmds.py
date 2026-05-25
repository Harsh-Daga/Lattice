from __future__ import annotations

import json
import sys

from rich.table import Table

from lattice._version import __version__
from lattice.cli._console import (
    _detect_init_targets,
    _get_config,
    _list_mutated_agents,
    _print_banner,
    console,
)


def _enabled_transforms(config) -> list[str]:
    """Return list of currently enabled transforms."""
    transforms: list[str] = []
    if config.transform_content_profiler:
        transforms.append("content_profiler")
    if config.transform_message_dedup:
        transforms.append("message_dedup")
    if config.transform_reference_sub:
        transforms.append("reference_sub")
    if config.transform_rate_distortion:
        transforms.append("rate_distortion")
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
    """Run the LATTICE benchmark suite (wraps benchmarks/evals/cli.py)."""
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    bench_cli = repo_root / "benchmarks" / "evals" / "cli.py"
    if not bench_cli.exists():
        console.print("[red]benchmarks/ not packaged with this install.[/red]")
        console.print("Run from a source checkout, or install with `pip install -e '.[dev]'`.")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(bench_cli), *args], check=False)
    sys.exit(result.returncode)


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
