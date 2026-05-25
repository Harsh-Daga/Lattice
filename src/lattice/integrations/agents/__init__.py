from __future__ import annotations

from lattice.integrations.agents.base import AgentIntegration
from lattice.integrations.agents.doctor import build_agent_doctor_report, list_primary_agents
from lattice.integrations.agents.env_builder import (
    ClaudeCodeIntegration,
    CodexIntegration,
    EnvFileIntegration,
    GenericIntegration,
    VSCodeIntegration,
)
from lattice.integrations.agents.lifecycle import (
    agent_status,
    unwrap_agent,
    unwrap_all,
    wrap_agent,
    wrap_all,
)
from lattice.integrations.agents.models import AgentConfig
from lattice.integrations.agents.profiles import (
    CopilotIntegration,
    CursorIntegration,
    JsonFileIntegration,
    OpenCodeIntegration,
)
from lattice.integrations.agents.protocol import (
    AgentDoctorReport,
    AgentIntegrationProtocol,
    AgentNotInstalledError,
)
from lattice.integrations.agents.registry import _AGENT_REGISTRY, get_agent_integration, list_agents

__all__ = [
    "AgentConfig",
    "AgentDoctorReport",
    "AgentIntegration",
    "AgentIntegrationProtocol",
    "AgentNotInstalledError",
    "ClaudeCodeIntegration",
    "CodexIntegration",
    "CopilotIntegration",
    "CursorIntegration",
    "EnvFileIntegration",
    "GenericIntegration",
    "JsonFileIntegration",
    "OpenCodeIntegration",
    "VSCodeIntegration",
    "_AGENT_REGISTRY",
    "agent_status",
    "build_agent_doctor_report",
    "get_agent_integration",
    "list_agents",
    "list_primary_agents",
    "unwrap_agent",
    "unwrap_all",
    "wrap_agent",
    "wrap_all",
]
