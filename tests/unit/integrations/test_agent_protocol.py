"""Agent integration protocol conformance."""

from __future__ import annotations

from lattice.integrations.agents import (
    AgentIntegrationProtocol,
    ClaudeCodeIntegration,
    CodexIntegration,
    CopilotIntegration,
    CursorIntegration,
    GenericIntegration,
    OpenCodeIntegration,
)


def test_every_subclass_satisfies_protocol() -> None:
    for cls in (
        ClaudeCodeIntegration,
        CodexIntegration,
        CursorIntegration,
        OpenCodeIntegration,
        CopilotIntegration,
        GenericIntegration,
    ):
        instance = cls()
        assert isinstance(instance, AgentIntegrationProtocol), (
            f"{cls.__name__} does not satisfy AgentIntegrationProtocol"
        )
        for method in ("patch", "unpatch", "is_patched", "doctor"):
            assert hasattr(instance, method)
            assert callable(getattr(instance, method))
