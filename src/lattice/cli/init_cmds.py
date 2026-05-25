from __future__ import annotations

import sys

from rich.panel import Panel
from rich.table import Table

from lattice.cli._console import (
    _detect_init_targets,
    _lace_agent,
    _list_agents,
    _print_banner,
    _run_init,
    _unlace_agent,
    console,
)
from lattice.cli.proxy_cmds import _cmd_proxy_start


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

    result = _unlace_agent(agent)
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


