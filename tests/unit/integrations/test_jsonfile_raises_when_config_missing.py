"""JsonFileIntegration raises when config is missing."""

from __future__ import annotations

import pytest

from lattice.integrations.agents import AgentNotInstalledError, OpenCodeIntegration


def test_opencode_raises_when_config_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    integration = OpenCodeIntegration()
    with pytest.raises(AgentNotInstalledError):
        integration.patch()
