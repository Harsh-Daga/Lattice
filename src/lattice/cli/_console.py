"""Command-line interface for LATTICE.

Provides commands for running the proxy, diagnostics, and configuration.
"""

from __future__ import annotations

from typing import Any

import structlog
from rich.console import Console

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

    return PIDManager()


def _start_background(host: str, port: int, workers: Any) -> int:
    """Lazy-load start_background_server."""
    from lattice.proxy.lifecycle import start_background_server

    return start_background_server(host=host, port=port, workers=workers)


def _list_agents() -> list[str]:
    """Lazy-load agent registry."""
    from lattice.integrations.registry import list_supported_agents

    return list_supported_agents()


def _detect_init_targets(global_scope: bool = True) -> list[str]:
    from lattice.integrations.init import detect_init_targets as _fn

    return _fn(global_scope=global_scope)


def _run_init(targets: list[str], port: int, global_scope: bool) -> dict[str, Any]:
    from lattice.integrations.init import run_init as _fn

    return _fn(targets, port=port, global_scope=global_scope)


def _lace_agent(**kwargs: Any) -> int:
    from lattice.integrations.lace import lace_agent as _fn

    return _fn(**kwargs)


def _unlace_agent(agent: str) -> dict[str, Any]:
    from lattice.integrations.lace import unlace_agent as _fn

    return _fn(agent)


def _list_mutated_agents() -> list[str]:
    from lattice.integrations.mutation_store import list_all_active as _fn

    return _fn()


def _print_banner() -> None:
    """Print the LATTICE ASCII banner."""
    banner = """
    ┌─────────────────────────────────────────┐
    │  LATTICE — LLM Transport & Efficiency   │
    │  Optimize · Compress · Accelerate       │
    └─────────────────────────────────────────┘
    """
    console.print(banner, style="cyan")

