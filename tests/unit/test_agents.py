"""Tests for agent integration wrap / unwrap / status.

Covers all supported agents:
- claude     -> env file integration
- codex      -> env file integration
- cursor     -> file-based (settings.json)
- opencode   -> file-based (opencode.json)
- generic    -> env file integration
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest

from lattice.core.config import LatticeConfig
from lattice.integrations.agents import (
    ClaudeCodeIntegration,
    CodexIntegration,
    CursorIntegration,
    OpenCodeIntegration,
    agent_status,
    list_agents,
    unwrap_agent,
    unwrap_all,
    wrap_agent,
    wrap_all,
)


@pytest.fixture
def lattice_config() -> LatticeConfig:
    return LatticeConfig(
        proxy_host="127.0.0.1",
        proxy_port=8787,
        provider_api_key="sk-test",
    )


# =============================================================================
# Helpers
# =============================================================================


def _make_opencode_config(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps(
            {
                "provider": {
                    "openadapter": {
                        "options": {"baseURL": "https://api.openadapter.in/v1"}
                    },
                    "ollama": {
                        "options": {"baseURL": "http://127.0.0.1:11434/v1"}
                    },
                }
            }
        )
    )
    return path


def _make_cursor_config(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"cursor.openai.baseUrl": "https://api.openai.com/v1"}))
    return path


# =============================================================================
# OpenCode
# =============================================================================


class TestOpenCodeIntegration:
    def test_patch_no_config(self, lattice_config: LatticeConfig) -> None:
        integration = OpenCodeIntegration(lattice_config)
        with patch.object(
            integration,
            "_config_path",
            return_value=pathlib.Path("/nonexistent/opencode.json"),
        ):
            result = integration.patch(dry_run=True)
        assert result.patched is False
        assert "not found" in result.message.lower()

    def test_patch_and_unpatch(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = _make_opencode_config(tmp_path)
        state_path = tmp_path / "opencode_state.json"
        integration = OpenCodeIntegration(lattice_config)
        with (
            patch.object(integration, "_config_path", return_value=config_path),
            patch.object(integration, "_state_path", return_value=state_path),
        ):
            result = integration.patch(dry_run=False)

        assert result.patched is True
        assert len(result.changes) == 4  # baseURL + header per provider
        assert "provider.openadapter.options.baseURL" in result.changes
        assert "provider.openadapter.options.headers.x-lattice-provider" in result.changes
        assert result.backup_path is not None

        raw = json.loads(config_path.read_text())
        assert raw["provider"]["openadapter"]["options"]["baseURL"] == lattice_config.proxy_url()
        assert raw["provider"]["ollama"]["options"]["baseURL"] == lattice_config.proxy_url()
        assert raw["provider"]["openadapter"]["options"]["headers"]["x-lattice-provider"] == "openadapter"
        assert raw["provider"]["ollama"]["options"]["headers"]["x-lattice-provider"] == "ollama"
        # LATTICE state must NOT be inside opencode.json
        assert "_lattice_wrapped" not in raw
        assert "_lattice_original_urls" not in raw
        # State lives in separate file
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "originals" in state

        with (
            patch.object(integration, "_config_path", return_value=config_path),
            patch.object(integration, "_state_path", return_value=state_path),
        ):
            restored = integration.unpatch(dry_run=False)
        assert restored.patched is False
        raw2 = json.loads(config_path.read_text())
        assert raw2["provider"]["openadapter"]["options"]["baseURL"] == "https://api.openadapter.in/v1"
        assert raw2["provider"]["ollama"]["options"]["baseURL"] == "http://127.0.0.1:11434/v1"
        # Headers should be removed on unpatch
        assert "headers" not in raw2["provider"]["openadapter"]["options"]
        assert "headers" not in raw2["provider"]["ollama"]["options"]
        assert not state_path.exists()

    def test_dry_run(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = _make_opencode_config(tmp_path)
        original = json.loads(config_path.read_text())
        integration = OpenCodeIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.patch(dry_run=True)
        assert result.patched is True
        assert result.backup_path is None  # dry_run should not create backup
        assert json.loads(config_path.read_text()) == original

    def test_idempotent_wrap(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = _make_opencode_config(tmp_path)
        integration = OpenCodeIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            integration.patch(dry_run=False)
            second = integration.patch(dry_run=False)
        assert second.patched is True
        assert "already routed" in second.message.lower()

    def test_unsupported_providers_skipped(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """Unsupported providers are left untouched; supported ones are patched."""
        config_path = tmp_path / "opencode.json"
        config_path.write_text(
            json.dumps(
                {
                    "provider": {
                        "openadapter": {
                            "options": {"baseURL": "https://api.openadapter.in/v1"}
                        },
                        "groq": {
                            "options": {"baseURL": "https://api.groq.com/v1"}
                        },
                        "ollama-cloud": {
                            "options": {"baseURL": "https://ollama.com/v1"}
                        },
                        "mistral": {
                            "options": {"baseURL": "https://api.mistral.ai/v1"}
                        },
                    }
                }
            )
        )
        state_path = tmp_path / "opencode_state.json"
        integration = OpenCodeIntegration(lattice_config)
        with (
            patch.object(integration, "_config_path", return_value=config_path),
            patch.object(integration, "_state_path", return_value=state_path),
        ):
            result = integration.patch(dry_run=False)

        raw = json.loads(config_path.read_text())
        # Supported providers patched
        assert raw["provider"]["openadapter"]["options"]["baseURL"] == lattice_config.proxy_url()
        assert raw["provider"]["ollama-cloud"]["options"]["baseURL"] == lattice_config.proxy_url()
        # Unsupported providers left untouched
        assert raw["provider"]["groq"]["options"]["baseURL"] == "https://api.groq.com/v1"
        assert raw["provider"]["mistral"]["options"]["baseURL"] == "https://api.mistral.ai/v1"
        # Headers injected only for supported
        assert raw["provider"]["openadapter"]["options"]["headers"]["x-lattice-provider"] == "openadapter"
        assert raw["provider"]["ollama-cloud"]["options"]["headers"]["x-lattice-provider"] == "ollama-cloud"
        assert "headers" not in raw["provider"]["groq"]["options"]
        assert "headers" not in raw["provider"]["mistral"]["options"]
        # State records skipped
        state = json.loads(state_path.read_text())
        assert sorted(state["skipped"]) == ["groq", "mistral"]
        # Message reports skipped
        assert "Skipped unsupported" in result.message

    def test_agent_status(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = _make_opencode_config(tmp_path)
        state_path = tmp_path / "opencode_state.json"
        # Not patched yet
        with (
            patch.object(OpenCodeIntegration, "_config_path", return_value=config_path),
            patch.object(OpenCodeIntegration, "_state_path", return_value=state_path),
        ):
            status = agent_status("opencode", lattice_config)
        assert status["patched"] is False

        # After patch
        with (
            patch.object(OpenCodeIntegration, "_config_path", return_value=config_path),
            patch.object(OpenCodeIntegration, "_state_path", return_value=state_path),
        ):
            wrap_agent("opencode", lattice_config)
            status = agent_status("opencode", lattice_config)
        assert status["patched"] is True
        assert status["config_path"] == str(config_path)


# =============================================================================
# Cursor
# =============================================================================


class TestCursorIntegration:
    def test_patch_no_config(self, lattice_config: LatticeConfig) -> None:
        integration = CursorIntegration(lattice_config)
        with patch.object(
            integration,
            "_config_path",
            return_value=pathlib.Path("/nonexistent/settings.json"),
        ):
            result = integration.patch(dry_run=True)
        assert result.patched is False

    def test_patch_and_unpatch(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = _make_cursor_config(tmp_path)
        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.patch(dry_run=False)

        assert result.patched is True
        assert "cursor.openai.baseUrl" in result.changes

        raw = json.loads(config_path.read_text())
        assert raw["cursor.openai.baseUrl"] == lattice_config.proxy_url()
        assert raw.get("_lattice_wrapped") is True

        with patch.object(integration, "_config_path", return_value=config_path):
            restored = integration.unpatch(dry_run=False)
        assert restored.patched is False
        raw2 = json.loads(config_path.read_text())
        assert raw2["cursor.openai.baseUrl"] == "https://api.openai.com/v1"
        assert "_lattice_wrapped" not in raw2

    def test_idempotent_wrap(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = _make_cursor_config(tmp_path)
        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            integration.patch(dry_run=False)
            second = integration.patch(dry_run=False)
        assert "already routed" in second.message.lower()

    def test_patch_multiple_providers(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        config_path = tmp_path / "settings.json"
        config_path.write_text(
            json.dumps({
                "cursor.openai.baseUrl": "https://api.openai.com/v1",
                "cursor.anthropic.baseUrl": "https://api.anthropic.com",
                "cursor.gemini.baseUrl": "https://generativelanguage.googleapis.com",
            })
        )
        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.patch(dry_run=False)

        assert result.patched is True
        raw = json.loads(config_path.read_text())
        assert raw["cursor.openai.baseUrl"] == lattice_config.proxy_url()
        assert raw["cursor.anthropic.baseUrl"] == lattice_config.proxy_url()
        assert raw["cursor.gemini.baseUrl"] == lattice_config.proxy_url()

        with patch.object(integration, "_config_path", return_value=config_path):
            restored = integration.unpatch(dry_run=False)
        assert restored.patched is False
        raw2 = json.loads(config_path.read_text())
        assert raw2["cursor.openai.baseUrl"] == "https://api.openai.com/v1"
        assert raw2["cursor.anthropic.baseUrl"] == "https://api.anthropic.com"
        assert raw2["cursor.gemini.baseUrl"] == "https://generativelanguage.googleapis.com"

    def test_create_settings_from_scratch(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """If Cursor has never been customised, settings.json doesn't exist.
        LATTICE should create it and inject the proxy URL."""
        config_path = tmp_path / "Cursor" / "User" / "settings.json"
        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.patch(dry_run=False)

        assert result.patched is True
        assert config_path.exists()
        raw = json.loads(config_path.read_text())
        assert raw["cursor.openai.baseUrl"] == lattice_config.proxy_url()
        assert raw["cursor.anthropic.baseUrl"] == lattice_config.proxy_url()
        assert raw["cursor.gemini.baseUrl"] == lattice_config.proxy_url()
        assert raw.get("_lattice_created_keys") == [
            "cursor.openai.baseUrl",
            "cursor.anthropic.baseUrl",
            "cursor.gemini.baseUrl",
        ]

    def test_created_keys_removed_on_unpatch(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """Keys that LATTICE created (didn't exist before) must be deleted on unpatch."""
        # Start with ONLY an unrelated key
        config_path = tmp_path / "settings.json"
        config_path.write_text(json.dumps({"editor.fontSize": 14}))

        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            integration.patch(dry_run=False)

        raw = json.loads(config_path.read_text())
        assert "cursor.openai.baseUrl" in raw  # was created
        assert raw.get("_lattice_created_keys")  # marker exists

        with patch.object(integration, "_config_path", return_value=config_path):
            integration.unpatch(dry_run=False)

        raw2 = json.loads(config_path.read_text())
        assert "cursor.openai.baseUrl" not in raw2
        assert "cursor.anthropic.baseUrl" not in raw2
        assert "cursor.gemini.baseUrl" not in raw2
        assert "_lattice_created_keys" not in raw2
        assert "_lattice_wrapped" not in raw2
        # Unrelated key preserved
        assert raw2["editor.fontSize"] == 14

    def test_baseurl_variant_normalization(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """If settings has baseURL (PascalCase), normalize to baseUrl (camelCase)."""
        config_path = tmp_path / "settings.json"
        config_path.write_text(
            json.dumps({
                "cursor.openai.baseURL": "https://api.openai.com/v1",
            })
        )
        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            integration.patch(dry_run=False)

        raw = json.loads(config_path.read_text())
        assert "cursor.openai.baseUrl" in raw
        assert "cursor.openai.baseURL" not in raw  # old variant removed
        assert raw["cursor.openai.baseUrl"] == lattice_config.proxy_url()

        with patch.object(integration, "_config_path", return_value=config_path):
            integration.unpatch(dry_run=False)

        raw2 = json.loads(config_path.read_text())
        assert raw2["cursor.openai.baseUrl"] == "https://api.openai.com/v1"
        assert "cursor.openai.baseURL" not in raw2

    def test_dry_run_no_changes(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """dry_run=True must not write anything."""
        config_path = tmp_path / "settings.json"
        config_path.write_text(json.dumps({"cursor.openai.baseUrl": "https://api.openai.com/v1"}))

        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.patch(dry_run=True)

        assert result.patched is True  # would succeed
        raw = json.loads(config_path.read_text())
        # File unchanged
        assert raw["cursor.openai.baseUrl"] == "https://api.openai.com/v1"
        assert "_lattice_wrapped" not in raw

    def test_unpatch_from_backup_when_marker_missing(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """If marker is missing but backup exists, restore from backup."""
        config_path = tmp_path / "settings.json"
        config_path.write_text(json.dumps({"cursor.openai.baseUrl": "https://api.openai.com/v1"}))

        integration = CursorIntegration(lattice_config)
        with patch.object(integration, "_config_path", return_value=config_path):
            integration.patch(dry_run=False)

        # Simulate marker corruption (remove markers but keep patched URL)
        raw = json.loads(config_path.read_text())
        raw.pop("_lattice_original_urls", None)
        raw.pop("_lattice_created_keys", None)
        raw.pop("_lattice_wrapped", None)
        config_path.write_text(json.dumps(raw))

        with patch.object(integration, "_config_path", return_value=config_path):
            result = integration.unpatch(dry_run=False)

        # Should restore from backup
        assert result.patched is False
        raw2 = json.loads(config_path.read_text())
        assert raw2["cursor.openai.baseUrl"] == "https://api.openai.com/v1"


# =============================================================================
# Claude Code (env-based)
# =============================================================================


class TestClaudeCodeIntegration:
    def test_patch_creates_env_file(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        integration = ClaudeCodeIntegration(lattice_config)
        env_path = tmp_path / "claude.env"
        with patch.object(integration, "_env_file", return_value=env_path):
            result = integration.patch()
        assert result.patched is True
        assert "source" in result.message.lower()

    def test_unpatch_removes_env_file(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        integration = ClaudeCodeIntegration(lattice_config)
        env_path = tmp_path / "claude.env"
        env_path.write_text("OPENAI_BASE_URL=...")
        with patch.object(integration, "_env_file", return_value=env_path):
            result = integration.unpatch()
        assert result.patched is False
        assert not env_path.exists()

    def test_is_patched(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        integration = ClaudeCodeIntegration(lattice_config)
        env_path = tmp_path / "claude.env"
        with patch.object(integration, "_env_file", return_value=env_path):
            assert integration.is_patched() is False
            env_path.write_text("OPENAI_BASE_URL=...")
            assert integration.is_patched() is True

    def test_agent_status(self, lattice_config: LatticeConfig) -> None:
        status = agent_status("claude", lattice_config)
        assert status["patched"] is False
        assert "env_file" in status


# =============================================================================
# Codex (env-based)
# =============================================================================


class TestCodexIntegration:
    def test_patch_creates_env_file(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        integration = CodexIntegration(lattice_config)
        env_path = tmp_path / "codex.env"
        state_path = tmp_path / "codex_state.json"
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=tmp_path / "nonexistent.toml"),
        ):
            result = integration.patch()
        assert result.patched is True
        assert "source" in result.message.lower()

    def test_wrap_and_unwrap_roundtrip(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        integration = CodexIntegration(lattice_config)
        env_path = tmp_path / "codex.env"
        state_path = tmp_path / "codex_state.json"
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=tmp_path / "nonexistent.toml"),
        ):
            wrap_result = integration.patch()
            assert wrap_result.patched is True
            unwrap_result = integration.unpatch()
            assert unwrap_result.patched is False
            assert not env_path.exists()

    def test_patch_top_level_openai_base_url(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """Top-level openai_base_url is patched and restored."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('openai_base_url = "https://api.openai.com/v1"\n')
        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            result = integration.patch()
        assert result.patched is True
        assert "codex /logout" in result.message
        text = config_path.read_text()
        assert f'openai_base_url = "{lattice_config.proxy_url()}"' in text

        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.unpatch()
        assert 'openai_base_url = "https://api.openai.com/v1"' in config_path.read_text()

    def test_patch_sets_openai_base_url_when_missing(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """If openai_base_url is absent, it is created."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('model = "gpt-4o"\n')
        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.patch()
        assert f'openai_base_url = "{lattice_config.proxy_url()}"' in config_path.read_text()
        # Unpatch should remove it since original was None
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.unpatch()
        assert "openai_base_url" not in config_path.read_text()

    def test_patch_model_providers(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """[model_providers.*].base_url is patched for ALL custom providers."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[model_providers.proxy]\nbase_url = "https://proxy.example.com"\n'
            '[model_providers.ollama]\nbase_url = "http://localhost:11434/v1"\n'
        )
        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.patch()
        text = config_path.read_text()
        # Top-level openai_base_url was also added (was absent)
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in text
        # Both model_providers patched
        assert '[model_providers.proxy]\nbase_url = "http://127.0.0.1:8787/v1"' in text
        assert '[model_providers.ollama]\nbase_url = "http://127.0.0.1:8787/v1"' in text

        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.unpatch()
        text = config_path.read_text()
        assert 'base_url = "https://proxy.example.com"' in text
        assert 'base_url = "http://localhost:11434/v1"' in text

    def test_patch_profiles(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """[profiles.*].openai_base_url is patched for ALL profiles."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[profiles.deep-review]\nmodel = "gpt-5-pro"\nopenai_base_url = "https://us.api.openai.com/v1"\n'
            '[profiles.lightweight]\nmodel = "gpt-4.1"\nopenai_base_url = "https://eu.api.openai.com/v1"\n'
        )
        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.patch()
        text = config_path.read_text()
        assert text.count('openai_base_url = "http://127.0.0.1:8787/v1"') == 2

        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.unpatch()
        text = config_path.read_text()
        assert 'openai_base_url = "https://us.api.openai.com/v1"' in text
        assert 'openai_base_url = "https://eu.api.openai.com/v1"' in text

    def test_patch_model_provider_reference(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """If model_provider references a custom provider, that provider is patched."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'model_provider = "proxy"\n'
            '[model_providers.proxy]\nbase_url = "https://proxy.example.com"\n'
        )
        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.patch()
        text = config_path.read_text()
        assert 'base_url = "http://127.0.0.1:8787/v1"' in text
        # model_provider itself is NOT changed, only tracked
        assert 'model_provider = "proxy"' in text

        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.unpatch()
        text = config_path.read_text()
        assert 'base_url = "https://proxy.example.com"' in text

    def test_patch_oss_provider(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """oss_provider references are tracked and the provider base_url patched."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'oss_provider = "ollama"\n'
            '[model_providers.ollama]\nbase_url = "http://localhost:11434/v1"\n'
        )
        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.patch()
        text = config_path.read_text()
        assert 'base_url = "http://127.0.0.1:8787/v1"' in text
        assert 'oss_provider = "ollama"' in text

        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=config_path),
            patch.object(integration, "_codex_project_configs", return_value=[]),
        ):
            integration.unpatch()
        text = config_path.read_text()
        assert 'base_url = "http://localhost:11434/v1"' in text

    def test_patch_project_level_config(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """Project-level .codex/config.toml is discovered and patched."""
        project_dir = tmp_path / "myproject"
        codex_dir = project_dir / ".codex"
        codex_dir.mkdir(parents=True)
        config_path = codex_dir / "config.toml"
        config_path.write_text('openai_base_url = "https://project.example.com"\n')

        state_path = tmp_path / "codex_state.json"
        env_path = tmp_path / "codex.env"
        integration = CodexIntegration(lattice_config)
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=tmp_path / "nonexistent.toml"),
            patch("pathlib.Path.cwd", return_value=project_dir),
        ):
            integration.patch()
        assert f'openai_base_url = "{lattice_config.proxy_url()}"' in config_path.read_text()

    def test_env_file_contains_http_proxy(self, tmp_path: pathlib.Path) -> None:
        """HTTP_PROXY is injected into env file when configured."""
        config = LatticeConfig(
            proxy_host="127.0.0.1",
            proxy_port=8787,
            http_proxy="http://corp-proxy:8080",
        )
        integration = CodexIntegration(config)
        env_path = tmp_path / "codex.env"
        state_path = tmp_path / "codex_state.json"
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=tmp_path / "nonexistent.toml"),
        ):
            integration.patch()
        content = env_path.read_text()
        assert 'export HTTP_PROXY="http://corp-proxy:8080"' in content
        assert 'export HTTPS_PROXY="http://corp-proxy:8080"' in content
        assert 'export http_proxy="http://corp-proxy:8080"' in content

    def test_patch_message_contains_oauth_warning(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """Patch message warns about OAuth logout."""
        integration = CodexIntegration(lattice_config)
        env_path = tmp_path / "codex.env"
        state_path = tmp_path / "codex_state.json"
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
            patch.object(integration, "_codex_user_config", return_value=tmp_path / "nonexistent.toml"),
        ):
            result = integration.patch()
        assert "codex /logout" in result.message
        assert "OAuth" in result.message

    def test_unpatch_with_no_state_is_clean(self, tmp_path: pathlib.Path, lattice_config: LatticeConfig) -> None:
        """Unpatch without state file just removes env file gracefully."""
        integration = CodexIntegration(lattice_config)
        env_path = tmp_path / "codex.env"
        env_path.write_text("export OPENAI_BASE_URL=...\n")
        state_path = tmp_path / "codex_state.json"
        with (
            patch.object(integration, "_env_file", return_value=env_path),
            patch.object(integration, "_state_path", return_value=state_path),
        ):
            result = integration.unpatch()
        assert result.patched is False
        assert not env_path.exists()


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    def test_list_agents(self) -> None:
        agents = list_agents()
        assert "opencode" in agents
        assert "cursor" in agents
        assert "claude" in agents
        assert "codex" in agents

    def test_wrap_unknown_agent(self, lattice_config: LatticeConfig) -> None:
        result = wrap_agent("unknown-agent", lattice_config)
        assert result.patched is False
        assert "Unknown agent" in result.message

    def test_unwrap_unknown_agent(self, lattice_config: LatticeConfig) -> None:
        result = unwrap_agent("unknown-agent", lattice_config)
        assert result.patched is False

    def test_wrap_all(self, lattice_config: LatticeConfig) -> None:
        results = wrap_all(lattice_config, dry_run=True)
        # wrap_all deduplicates by integration name (claude-code → claude)
        assert len(results) < len(list_agents())
        names = {r.agent_name for r in results}
        assert "claude" in names
        assert "vscode" in names

    def test_unwrap_all(self, lattice_config: LatticeConfig) -> None:
        results = unwrap_all(lattice_config, dry_run=True)
        # unwrap_all deduplicates by integration name
        assert len(results) < len(list_agents())

    def test_wrap_agent_dry_run_opencode(
        self, tmp_path: pathlib.Path, lattice_config: LatticeConfig
    ) -> None:
        config_path = _make_opencode_config(tmp_path)
        with patch.object(OpenCodeIntegration, "_config_path", return_value=config_path):
            result = wrap_agent("opencode", lattice_config, dry_run=True)
        assert result.patched is True
        assert json.loads(config_path.read_text()) == {
            "provider": {
                "openadapter": {"options": {"baseURL": "https://api.openadapter.in/v1"}},
                "ollama": {"options": {"baseURL": "http://127.0.0.1:11434/v1"}},
            }
        }
