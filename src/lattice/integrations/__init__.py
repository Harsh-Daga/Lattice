"""Third-party integrations for LATTICE."""

from lattice.integrations.agents import (
    AgentConfig,
    AgentIntegration,
    ClaudeCodeIntegration,
    CodexIntegration,
    CursorIntegration,
    EnvFileIntegration,
    GenericIntegration,
    JsonFileIntegration,
    OpenCodeIntegration,
    agent_status,
    list_agents,
    unwrap_agent,
    unwrap_all,
    wrap_agent,
    wrap_all,
)
from lattice.integrations.init import (
    detect_init_targets,
    run_init,
    run_uninit,
)
from lattice.integrations.lace import lace_agent, unlace_agent
from lattice.integrations.mcp import LatticeMCPTools
from lattice.integrations.registry import list_supported_agents

__all__ = [
    "AgentConfig",
    "AgentIntegration",
    "ClaudeCodeIntegration",
    "CodexIntegration",
    "CursorIntegration",
    "EnvFileIntegration",
    "GenericIntegration",
    "JsonFileIntegration",
    "LatticeMCPTools",
    "OpenCodeIntegration",
    "agent_status",
    "detect_init_targets",
    "lace_agent",
    "list_agents",
    "list_supported_agents",
    "run_init",
    "run_uninit",
    "unlace_agent",
    "unwrap_agent",
    "unwrap_all",
    "wrap_agent",
    "wrap_all",
]
