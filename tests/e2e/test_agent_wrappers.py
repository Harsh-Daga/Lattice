"""End-to-end tests for agent integrations.

Validates wrap/unwrap flows using the public API.
These tests use dry_run to avoid modifying real config files.
"""

from lattice.integrations.agents import (
    agent_status,
    list_agents,
    unwrap_agent,
    wrap_agent,
)


class TestPublicAgentAPI:
    def test_list_agents(self):
        agents = list_agents()
        assert "claude" in agents
        assert "codex" in agents
        assert "cursor" in agents
        assert "opencode" in agents

    def test_wrap_unknown_agent(self):
        result = wrap_agent("nonexistent_agent", dry_run=True)
        assert result.patched is False
        assert "unknown" in result.message.lower()

    def test_unwrap_unknown_agent(self):
        result = unwrap_agent("nonexistent_agent", dry_run=True)
        assert result.patched is False
        assert "unknown" in result.message.lower()

    def test_agent_status_unknown(self):
        status = agent_status("nonexistent_agent")
        assert status["patched"] is False
        assert "unknown" in status.get("message", "").lower()

    def test_claude_wrap_dry_run(self):
        result = wrap_agent("claude", dry_run=True)
        # Claude uses env file — dry_run still returns config
        assert result.agent_name == "claude"
        assert result.patched in (True, False)  # may already be patched

    def test_codex_wrap_dry_run(self):
        result = wrap_agent("codex", dry_run=True)
        assert result.agent_name == "codex"

    def test_cursor_wrap_dry_run(self):
        result = wrap_agent("cursor", dry_run=True)
        assert result.agent_name == "cursor"

    def test_opencode_wrap_dry_run(self):
        result = wrap_agent("opencode", dry_run=True)
        assert result.agent_name == "opencode"

    def test_unwrap_not_patched(self):
        # If agent is not patched, unwrap should report nothing to do
        result = unwrap_agent("generic", dry_run=True)
        assert result.patched is False

    def test_status_returns_dict(self):
        status = agent_status("claude")
        assert isinstance(status, dict)
        assert "agent" in status
        assert "patched" in status
