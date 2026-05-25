from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from lattice.cli._console import _get_config, _print_banner, console


def _cmd_doctor(args: list[str]) -> None:
    """Diagnose why an agent isn't routing through LATTICE."""
    from lattice.integrations.agents import (
        get_agent_integration,
        list_agents,
        list_primary_agents,
    )

    if args and args[0] in ("-h", "--help"):
        supported = ", ".join(list_primary_agents())
        console.print(
            Panel.fit(
                "[bold]lattice doctor[/bold]\n\n"
                "Diagnose LATTICE proxy routing for each agent integration.\n\n"
                "Usage:\n"
                "  lattice doctor\n"
                f"  lattice doctor <agent>   ({supported})\n\n"
                "Checks per agent:\n"
                "  1. Is the agent installed (binary / config)?\n"
                "  2. Durable init or transient lace active?\n"
                "  3. Is the LATTICE proxy reachable at /healthz?",
                title="lattice doctor",
                border_style="blue",
            )
        )
        return

    config = _get_config()
    if args:
        agent_names = [args[0].lower()]
    else:
        agent_names = list_primary_agents()

    _print_banner()

    for agent_name in agent_names:
        if agent_name not in list_agents():
            console.print(f"[red]Unknown agent: {agent_name}[/red]")
            console.print(f"[dim]Supported: {', '.join(list_primary_agents())}[/dim]")
            continue

        try:
            integration = get_agent_integration(agent_name, config)
            report = integration.doctor()
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        _print_doctor_report(report)


def _print_doctor_report(report: object) -> None:
    """Render a single agent doctor report."""
    from lattice.integrations.agents import AgentDoctorReport

    if not isinstance(report, AgentDoctorReport):
        return

    table = Table(title=f"LATTICE Doctor — {report.agent}")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    def _row(label: str, ok: bool, detail: str) -> None:
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(label, mark, detail)

    _row("Installed", report.is_installed, "Binary or config present on this machine")
    _row(
        "Durable routing",
        report.is_patched_durable,
        "init / wrap_agent config active",
    )
    _row(
        "Transient lace",
        report.is_patched_transient,
        "active lattice lace session",
    )
    _row(
        "Proxy /healthz",
        report.proxy_reachable,
        f"http://127.0.0.1:{_get_config().proxy_port}/healthz",
    )

    console.print(table)
    for line in report.diagnostic_lines:
        console.print(f"  [dim]•[/dim] {line}")
    console.print()
