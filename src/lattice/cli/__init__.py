"""Command-line interface for LATTICE."""

from __future__ import annotations

import sys
from typing import Any

from lattice._version import __version__
from lattice.cli._console import (
    _get_config,
    _get_pid_mgr,
    _lace_agent,
    _list_agents,
    _list_mutated_agents,
    _print_banner,
    _run_init,
    _start_background,
    _unlace_agent,
    console,
    logger,
)
from lattice.cli.doctor_cmds import _cmd_doctor, _print_doctor_report
from lattice.cli.info_cmds import (
    _cmd_agent_status,
    _cmd_benchmark,
    _cmd_config,
    _cmd_health,
    _cmd_info,
    _enabled_transforms,
)
from lattice.cli.init_cmds import (
    _cmd_init,
    _cmd_lace,
    _cmd_unlace,
    _format_init_empty_error,
    _print_init_help,
    _print_lace_help,
    _print_unlace_help,
)
from lattice.cli.proxy_cmds import (
    _cmd_proxy,
    _cmd_proxy_restart,
    _cmd_proxy_run,
    _cmd_proxy_start,
    _cmd_proxy_status,
    _cmd_proxy_stop,
    _parse_proxy_args,
    _print_proxy_help,
)


def _print_help() -> None:
    from rich.panel import Panel

    console.print(
        Panel.fit(
            "[bold]lattice[/bold] — LLM transport proxy\n\n"
            "  lattice proxy start|stop|status|run\n"
            "  lattice init [agent]\n"
            "  lattice lace <agent>\n"
            "  lattice unlace <agent>\n"
            "  lattice info | config | benchmark | health\n"
            "  lattice status | doctor",
            title="Usage",
        )
    )


def main() -> None:
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


__all__ = ["main"]
