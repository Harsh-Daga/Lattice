"""Registry alignment: primary agents vs wrap aliases."""

from __future__ import annotations

from lattice.integrations.agents import _AGENT_REGISTRY, list_primary_agents
from lattice.integrations.registry import list_supported_agents


def test_list_supported_agents_matches_primary_agents() -> None:
    assert list_supported_agents() == list_primary_agents()


def test_primary_agents_are_in_agent_registry() -> None:
    for name in list_primary_agents():
        assert name in _AGENT_REGISTRY


def test_registry_has_wrap_aliases_beyond_primary() -> None:
    primary = set(list_primary_agents())
    assert primary < set(_AGENT_REGISTRY.keys())
    assert "claude-code" in _AGENT_REGISTRY
    assert "generic" in _AGENT_REGISTRY
