from __future__ import annotations

from collections.abc import Callable

from lattice.core.config import LatticeConfig
from lattice.integrations.agents.base import AgentIntegration
from lattice.integrations.agents.env_builder import (
    ClaudeCodeIntegration,
    CodexIntegration,
    GenericIntegration,
    VSCodeIntegration,
)
from lattice.integrations.agents.profiles import (
    CopilotIntegration,
    CursorIntegration,
    OpenCodeIntegration,
)

_AGENT_REGISTRY: dict[str, Callable[[LatticeConfig | None], AgentIntegration]] = {
    "claude": ClaudeCodeIntegration,
    "claude-code": ClaudeCodeIntegration,
    "codex": CodexIntegration,
    "cursor": CursorIntegration,
    "opencode": OpenCodeIntegration,
    "copilot": CopilotIntegration,
    "vscode": VSCodeIntegration,
    "generic": GenericIntegration,
}


def list_agents() -> list[str]:
    """Return all supported agent names."""
    return list(_AGENT_REGISTRY.keys())


def get_agent_integration(
    agent_name: str,
    lattice_config: LatticeConfig | None = None,
) -> AgentIntegration:
    """Instantiate a registered integration by CLI name."""
    lattice_config = lattice_config or LatticeConfig.auto()
    lower = agent_name.lower()
    if lower not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Supported: {', '.join(list_agents())}")
    return _AGENT_REGISTRY[lower](lattice_config)
