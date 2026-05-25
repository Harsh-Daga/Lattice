from __future__ import annotations

import urllib.error
import urllib.request

import structlog

from lattice.core.config import LatticeConfig
from lattice.integrations.agents.base import AgentIntegration
from lattice.integrations.agents.protocol import AgentDoctorReport
from lattice.integrations.mutation_store import (
    get_mutation,
    list_transient_laced,
)

logger = structlog.get_logger()

_PRIMARY_AGENTS: tuple[str, ...] = ("claude", "codex", "cursor", "opencode", "copilot")


def list_primary_agents() -> list[str]:
    """Return the five product agents (doctor / init targets)."""
    return list(_PRIMARY_AGENTS)


def _proxy_reachable(lattice_config: LatticeConfig) -> bool:
    url = f"http://{lattice_config.proxy_host}:{lattice_config.proxy_port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def build_agent_doctor_report(
    integration: "AgentIntegration",
    *,
    is_installed: bool,
) -> AgentDoctorReport:
    """Shared doctor checks for any integration instance."""
    name = integration.name
    durable = get_mutation(name) is not None or integration.is_patched()
    transient = name in {r.agent for r in list_transient_laced()}
    proxy_ok = _proxy_reachable(integration.lattice_config)
    lines: list[str] = []
    if not is_installed:
        lines.append(f"{name}: agent or config not found on this machine.")
    if durable:
        lines.append(f"{name}: durable routing configured (init or env/config patch).")
    elif transient:
        lines.append(f"{name}: transient lace session active.")
    else:
        lines.append(
            f"{name}: not routed through LATTICE — run `lattice init {name}` or `lattice lace {name}`."
        )
    if proxy_ok:
        lines.append(
            f"Proxy reachable at http://{integration.lattice_config.proxy_host}:"
            f"{integration.lattice_config.proxy_port}/healthz"
        )
    else:
        lines.append(
            "Proxy not reachable — start with `lattice proxy run` or `lattice lace <agent>`."
        )
    return AgentDoctorReport(
        agent=name,
        is_installed=is_installed,
        is_patched_durable=durable,
        is_patched_transient=transient,
        proxy_reachable=proxy_ok,
        diagnostic_lines=lines,
    )
