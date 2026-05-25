from __future__ import annotations

from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.integrations.agents.env_builder import EnvFileIntegration
from lattice.integrations.agents.models import AgentConfig
from lattice.integrations.agents.profiles import JsonFileIntegration
from lattice.integrations.agents.protocol import AgentNotInstalledError
from lattice.integrations.agents.registry import _AGENT_REGISTRY, list_agents

logger = structlog.get_logger()

def wrap_agent(
    agent_name: str,
    lattice_config: LatticeConfig | None = None,
    dry_run: bool = False,
) -> AgentConfig:
    """Route an agent through the LATTICE proxy.

    Args:
        agent_name: Name of the agent (claude, cursor, codex, opencode, …).
        lattice_config: ``LatticeConfig``. Auto-discovers if ``None``.
        dry_run: Show what would change without writing files.

    Returns:
        ``AgentConfig`` describing the result.
    """
    lattice_config = lattice_config or LatticeConfig.auto()
    lower = agent_name.lower()
    if lower not in _AGENT_REGISTRY:
        return AgentConfig(
            agent_name=agent_name,
            patched=False,
            backup_path=None,
            message=f"Unknown agent '{agent_name}'. Supported: {', '.join(list_agents())}",
        )
    return _AGENT_REGISTRY[lower](lattice_config).patch(dry_run=dry_run)


def unwrap_agent(
    agent_name: str,
    lattice_config: LatticeConfig | None = None,
    dry_run: bool = False,
) -> AgentConfig:
    """Restore an agent's original configuration.

    Args:
        agent_name: Name of the agent.
        lattice_config: ``LatticeConfig``.
        dry_run: Show what would restore without writing files.

    Returns:
        ``AgentConfig`` describing the result.
    """
    lattice_config = lattice_config or LatticeConfig.auto()
    lower = agent_name.lower()
    if lower not in _AGENT_REGISTRY:
        return AgentConfig(
            agent_name=agent_name,
            patched=False,
            backup_path=None,
            message=f"Unknown agent '{agent_name}'.",
        )
    return _AGENT_REGISTRY[lower](lattice_config).unpatch(dry_run=dry_run)


def agent_status(
    agent_name: str,
    lattice_config: LatticeConfig | None = None,
) -> dict[str, Any]:
    """Check whether an agent is currently routed through LATTICE."""
    lattice_config = lattice_config or LatticeConfig.auto()
    lower = agent_name.lower()
    result: dict[str, Any] = {"agent": agent_name, "patched": False}

    if lower not in _AGENT_REGISTRY:
        result["message"] = f"Unknown agent '{agent_name}'."
        return result

    integration = _AGENT_REGISTRY[lower](lattice_config)
    result["patched"] = integration.is_patched()

    if isinstance(integration, JsonFileIntegration):
        path = integration._config_path()
        if path:
            result["config_path"] = str(path)
    elif isinstance(integration, EnvFileIntegration):
        result["env_file"] = str(integration._env_file())

    return result


def wrap_all(
    lattice_config: LatticeConfig | None = None,
    dry_run: bool = False,
) -> list[AgentConfig]:
    """Wrap every supported agent in one call.

    Returns a list of ``AgentConfig`` objects so callers can audit what
    changed and what failed.
    """
    lattice_config = lattice_config or LatticeConfig.auto()
    seen: set[str] = set()
    results: list[AgentConfig] = []
    for name in list_agents():
        integration = _AGENT_REGISTRY[name](lattice_config)
        if integration.name in seen:
            continue
        seen.add(integration.name)
        try:
            results.append(integration.patch(dry_run=dry_run))
        except AgentNotInstalledError as exc:
            results.append(
                AgentConfig(
                    agent_name=integration.name,
                    patched=False,
                    backup_path=None,
                    message=str(exc),
                )
            )
    return results


def unwrap_all(
    lattice_config: LatticeConfig | None = None,
    dry_run: bool = False,
) -> list[AgentConfig]:
    """Unwrap every supported agent in one call."""
    lattice_config = lattice_config or LatticeConfig.auto()
    seen: set[str] = set()
    results: list[AgentConfig] = []
    for name in list_agents():
        integration = _AGENT_REGISTRY[name](lattice_config)
        if integration.name in seen:
            continue
        seen.add(integration.name)
        results.append(integration.unpatch(dry_run=dry_run))
    return results
