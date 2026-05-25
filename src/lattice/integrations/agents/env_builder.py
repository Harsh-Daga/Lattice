from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import sys
from typing import Any

import structlog

from lattice.integrations.agents.base import (
    AgentIntegration,
    _backup_dir,
    _save_json,
    _timestamped_backup,
)
from lattice.integrations.agents.models import AgentConfig

logger = structlog.get_logger()

class EnvFileIntegration(AgentIntegration):
    """Agents controlled via ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``.

    We write a persistent ``.env`` file to ``~/.config/lattice/<name>.env``.
    The user sources it before running the agent, or we print the commands
    for manual use.
    """

    def _env_file(self) -> pathlib.Path:
        return pathlib.Path.home() / ".config" / "lattice" / f"{self.name}.env"

    def _write_env(self) -> pathlib.Path:
        proxy_base = f"http://{self.lattice_config.proxy_host}:{self.lattice_config.proxy_port}"
        lines = [
            f"# LATTICE proxy for {self.name}",
            f'export OPENAI_BASE_URL="{self.proxy_url}"',
            f'export OPENAI_API_BASE="{self.proxy_url}"',  # fallback for older SDKs
            f'export ANTHROPIC_BASE_URL="{proxy_base}"',
            f'export ANTHROPIC_API_BASE="{proxy_base}"',  # fallback
        ]
        # Pass through any existing API keys so the agent doesn't lose them
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_ORG_ID"):
            val = os.environ.get(key)
            if val:
                lines.append(f'export {key}="{val}"')
        # Enterprise HTTP/HTTPS proxy support
        http_proxy = self.lattice_config.http_proxy
        if http_proxy:
            lines.append(f'export HTTP_PROXY="{http_proxy}"')
            lines.append(f'export HTTPS_PROXY="{http_proxy}"')
            # Also set lowercase variants for tools that prefer them
            lines.append(f'export http_proxy="{http_proxy}"')
            lines.append(f'export https_proxy="{http_proxy}"')
        # NOTE: We intentionally do NOT overwrite the agent's API key.
        # LATTICE is a transparent transport proxy; the agent keeps its
        # own API key and LATTICE uses its configured upstream key.
        path = self._env_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    def _delete_env(self) -> None:
        path = self._env_file()
        if path.exists():
            path.unlink()

    def is_patched(self) -> bool:
        return self._env_file().exists()

    def patch(self, dry_run: bool = False) -> AgentConfig:
        env_path = self._env_file()
        if self.is_patched():
            return AgentConfig(
                agent_name=self.name,
                patched=True,
                backup_path=str(env_path),
                changes=[],
                message=(f"{self.name} is already routed through LATTICE.\nEnv file: {env_path}"),
            )

        backup: pathlib.Path | None = None
        if env_path.exists():
            backup = _timestamped_backup(env_path, self.name)

        if not dry_run:
            self._write_env()

        commands = (
            f"export OPENAI_BASE_URL={self.proxy_url!r}\n"
            f"export ANTHROPIC_BASE_URL='http://{self.lattice_config.proxy_host}:{self.lattice_config.proxy_port}'"
        )
        return AgentConfig(
            agent_name=self.name,
            patched=True,
            backup_path=str(backup) if backup else None,
            changes=["OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"],
            message=(
                f"# To use {self.name} with LATTICE, source the env file:\n"
                f"source {env_path}\n\n"
                f"# Or run these commands manually:\n{commands}"
            ),
        )

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        env_path = self._env_file()
        existed = env_path.exists()
        if not dry_run:
            self._delete_env()

        if existed:
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                changes=["OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"],
                message=(
                    f"Removed env file for {self.name}: {env_path}\n\n"
                    f"# If the variables were exported in your shell, also run:\n"
                    f"unset OPENAI_BASE_URL ANTHROPIC_BASE_URL"
                ),
            )
        return AgentConfig(
            agent_name=self.name,
            patched=False,
            backup_path=None,
            changes=[],
            message=(f"No env file found for {self.name} at {env_path}. Nothing to restore."),
        )


class ClaudeCodeIntegration(EnvFileIntegration):
    """Claude Code — routes via ``OPENAI_BASE_URL`` env var."""

    @property
    def name(self) -> str:
        return "claude"

    def _agent_binary_name(self) -> str | None:
        return "claude"


class CodexIntegration(EnvFileIntegration):
    """Codex CLI — comprehensive TOML + env-file integration.

    Codex reads config from multiple layers (user-level, project-level,
    profiles, custom model providers).  We patch *every* layer that can
    route around LATTICE and store the originals in a single state file
    so ``unpatch`` is fully reversible.

    Patched targets
    ---------------
    * Top-level ``openai_base_url``
    * Top-level ``model_provider`` (if it references a custom provider,
      that provider's ``base_url`` is also patched)
    * Top-level ``oss_provider`` (the referenced local provider's
      ``base_url`` is patched)
    * ``[model_providers.<id>].base_url`` — ALL custom providers
    * ``[profiles.<name>].openai_base_url`` — ALL profiles
    * ``[profiles.<name>].model_provider`` — tracked, provider patched
    * Project-level ``.codex/config.toml`` (closest to CWD wins)

    State format
    ------------
    ``~/.config/lattice/codex_state.json`` stores a list of configs::

        {
          "configs": [
            {
              "path": "...",
              "top_level": {"openai_base_url": "...", ...},
              "model_providers": {"proxy": {"base_url": "..."}},
              "profiles": {"deep-review": {"openai_base_url": "..."}}
            }
          ]
        }
    """

    @property
    def name(self) -> str:
        return "codex"

    def _agent_binary_name(self) -> str | None:
        return "codex"

    def _is_agent_installed(self) -> bool:
        if shutil.which("codex") is not None:
            return True
        return self._codex_user_config().exists()

    # ------------------------------------------------------------------ paths

    def _codex_user_config(self) -> pathlib.Path:
        """Path to Codex user-level config.toml."""
        return pathlib.Path.home() / ".config" / "codex" / "config.toml"

    def _codex_project_configs(self) -> list[pathlib.Path]:
        """Find all project-level ``.codex/config.toml`` files from CWD upward."""
        configs: list[pathlib.Path] = []
        cwd = pathlib.Path.cwd().resolve()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / ".codex" / "config.toml"
            if candidate.exists():
                configs.append(candidate)
        return configs

    def _state_path(self) -> pathlib.Path:
        """Path to LATTICE's separate state file for Codex."""
        return _backup_dir().parent / "codex_state.json"

    # ------------------------------------------------------------------ toml

    def _load_toml(self, path: pathlib.Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            if sys.version_info >= (3, 11):
                import tomllib

                with path.open("rb") as f:
                    data: dict[str, Any] = tomllib.load(f)
                    return data
            else:
                import tomli

                with path.open("rb") as f:
                    return dict(tomli.load(f))
        except Exception:
            return {}

    @staticmethod
    def _set_toml_key(path: pathlib.Path, key: str, value: str) -> None:
        """Set a top-level string key in a TOML file, preserving other content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f'{key} = "{value}"\n')
            return

        text = path.read_text()
        pattern = re.compile(rf'^({re.escape(key)}\s*=\s*)["\'].*?["\']', re.MULTILINE)
        new_line = f'{key} = "{value}"'
        if pattern.search(text):
            text = pattern.sub(rf'\g<1>"{value}"', text)
        else:
            section_match = re.search(r"^\[", text, re.MULTILINE)
            if section_match:
                insert_pos = section_match.start()
                text = text[:insert_pos] + new_line + "\n\n" + text[insert_pos:]
            else:
                text = text.rstrip() + "\n" + new_line + "\n"
        path.write_text(text)

    @staticmethod
    def _remove_toml_key(path: pathlib.Path, key: str) -> None:
        """Remove a top-level key from a TOML file."""
        if not path.exists():
            return
        text = path.read_text()
        pattern = re.compile(rf'^\s*{re.escape(key)}\s*=\s*["\'].*?["\']\s*\n?', re.MULTILINE)
        text = pattern.sub("", text)
        path.write_text(text)

    @staticmethod
    def _set_toml_table_key(path: pathlib.Path, table: str, key: str, value: str) -> None:
        """Set ``key = "value"`` inside a TOML table like ``[table]`` or ``[[table]]``.

        Uses regex to find the table header, then searches within that
        section (until the next ``[`` header) for the key.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f'[{table}]\n{key} = "{value}"\n')
            return

        text = path.read_text()
        # Find the table header — match [table] but not [table.something]
        header_re = re.compile(rf"^\[{re.escape(table)}\]\s*$", re.MULTILINE)
        header_match = header_re.search(text)

        if not header_match:
            # Table doesn't exist — append at end
            text = text.rstrip() + f'\n\n[{table}]\n{key} = "{value}"\n'
            path.write_text(text)
            return

        section_start = header_match.end()
        # Find the end of this section (next [ or EOF)
        next_section = re.search(r"^(\[|\[\[)", text[section_start:], re.MULTILINE)
        section_end = section_start + next_section.start() if next_section else len(text)

        section_text = text[section_start:section_end]
        # Look for key inside section
        key_re = re.compile(rf'^({re.escape(key)}\s*=\s*)["\'].*?["\']', re.MULTILINE)
        new_line = f'{key} = "{value}"'
        if key_re.search(section_text):
            section_text = key_re.sub(rf'\g<1>"{value}"', section_text)
        else:
            section_text = section_text.rstrip() + "\n" + new_line + "\n"

        text = text[:section_start] + section_text + text[section_end:]
        path.write_text(text)

    @staticmethod
    def _remove_toml_table_key(path: pathlib.Path, table: str, key: str) -> None:
        """Remove a key from inside a TOML table."""
        if not path.exists():
            return
        text = path.read_text()
        header_re = re.compile(rf"^\[{re.escape(table)}\]\s*$", re.MULTILINE)
        header_match = header_re.search(text)
        if not header_match:
            return

        section_start = header_match.end()
        next_section = re.search(r"^(\[|\[\[)", text[section_start:], re.MULTILINE)
        section_end = section_start + next_section.start() if next_section else len(text)

        section_text = text[section_start:section_end]
        key_re = re.compile(rf'^\s*{re.escape(key)}\s*=\s*["\'].*?["\']\s*\n?', re.MULTILINE)
        section_text = key_re.sub("", section_text)
        text = text[:section_start] + section_text + text[section_end:]
        path.write_text(text)

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict[str, Any]:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            data: dict[str, Any] = json.loads(path.read_text())
            return data
        except Exception:
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        _save_json(self._state_path(), state)

    # ------------------------------------------------------------------ patch helpers

    def _patch_single_config(
        self,
        path: pathlib.Path,
        data: dict[str, Any],
        dry_run: bool,
    ) -> dict[str, Any] | None:
        """Patch a single Codex config file.  Returns a state dict or None."""
        original: dict[str, Any] = {
            "path": str(path),
            "top_level": {},
            "model_providers": {},
            "profiles": {},
        }
        changed = False

        # 1. Top-level openai_base_url
        top_url = data.get("openai_base_url")
        if top_url is None or top_url != self.proxy_url:
            original["top_level"]["openai_base_url"] = top_url
            if not dry_run:
                self._set_toml_key(path, "openai_base_url", self.proxy_url)
            changed = True

        # 2. [model_providers.*].base_url — patch ALL custom providers
        providers = data.get("model_providers", {})
        if isinstance(providers, dict):
            for provider_id, pcfg in providers.items():
                if not isinstance(pcfg, dict):
                    continue
                base_url = pcfg.get("base_url")
                if base_url is None or base_url != self.proxy_url:
                    original["model_providers"][provider_id] = {"base_url": base_url}
                    table = f"model_providers.{provider_id}"
                    if not dry_run:
                        self._set_toml_table_key(path, table, "base_url", self.proxy_url)
                    changed = True

        # 3. [profiles.*].openai_base_url
        profiles = data.get("profiles", {})
        if isinstance(profiles, dict):
            for profile_name, pcfg in profiles.items():
                if not isinstance(pcfg, dict):
                    continue
                profile_url = pcfg.get("openai_base_url")
                if profile_url is None or profile_url != self.proxy_url:
                    original["profiles"][profile_name] = original["profiles"].get(profile_name, {})
                    original["profiles"][profile_name]["openai_base_url"] = profile_url
                    table = f"profiles.{profile_name}"
                    if not dry_run:
                        self._set_toml_table_key(path, table, "openai_base_url", self.proxy_url)
                    changed = True

        # 4. model_provider tracking — if it references a custom provider,
        #    we already patched that provider's base_url above.  Record it.
        model_provider = data.get("model_provider")
        if model_provider and isinstance(model_provider, str):
            original["top_level"]["model_provider"] = model_provider
            # If the referenced provider exists, ensure its base_url is patched
            if isinstance(providers, dict) and model_provider in providers:
                ref_cfg = providers[model_provider]
                if isinstance(ref_cfg, dict):
                    ref_url = ref_cfg.get("base_url")
                    if ref_url is None or ref_url != self.proxy_url:
                        original["model_providers"][model_provider] = {"base_url": ref_url}
                        table = f"model_providers.{model_provider}"
                        if not dry_run:
                            self._set_toml_table_key(path, table, "base_url", self.proxy_url)
                        changed = True

        # 5. oss_provider tracking — ensure the referenced local provider is patched
        oss_provider = data.get("oss_provider")
        if oss_provider and isinstance(oss_provider, str):
            original["top_level"]["oss_provider"] = oss_provider
            if isinstance(providers, dict) and oss_provider in providers:
                ref_cfg = providers[oss_provider]
                if isinstance(ref_cfg, dict):
                    ref_url = ref_cfg.get("base_url")
                    if ref_url is None or ref_url != self.proxy_url:
                        original["model_providers"][oss_provider] = {"base_url": ref_url}
                        table = f"model_providers.{oss_provider}"
                        if not dry_run:
                            self._set_toml_table_key(path, table, "base_url", self.proxy_url)
                        changed = True

        return original if changed else None

    def _unpatch_single_config(self, state_entry: dict[str, Any], dry_run: bool) -> bool:
        """Restore a single config file from state.  Returns True if restored."""
        path = pathlib.Path(state_entry["path"])
        if not path.exists():
            return False

        # Top-level keys
        for key, value in state_entry.get("top_level", {}).items():
            if value is None:
                if not dry_run:
                    self._remove_toml_key(path, key)
            else:
                if not dry_run:
                    self._set_toml_key(path, key, str(value))

        # model_providers
        for provider_id, pcfg in state_entry.get("model_providers", {}).items():
            for key, value in pcfg.items():
                table = f"model_providers.{provider_id}"
                if value is None:
                    if not dry_run:
                        self._remove_toml_table_key(path, table, key)
                else:
                    if not dry_run:
                        self._set_toml_table_key(path, table, key, str(value))

        # profiles
        for profile_name, pcfg in state_entry.get("profiles", {}).items():
            for key, value in pcfg.items():
                table = f"profiles.{profile_name}"
                if value is None:
                    if not dry_run:
                        self._remove_toml_table_key(path, table, key)
                else:
                    if not dry_run:
                        self._set_toml_table_key(path, table, key, str(value))

        return True

    # ------------------------------------------------------------------ public

    def is_patched(self) -> bool:
        return self._env_file().exists() or self._state_path().exists()

    def patch(self, dry_run: bool = False) -> AgentConfig:
        # 1. Do the env-file patch (standard behaviour)
        result = super().patch(dry_run=dry_run)

        # 2. Discover configs to patch
        all_configs = [self._codex_user_config()] + self._codex_project_configs()
        state: dict[str, Any] = {"configs": []}
        codex_messages: list[str] = []

        for cfg_path in all_configs:
            data = self._load_toml(cfg_path)
            if not data:
                continue
            if not dry_run:
                _timestamped_backup(cfg_path, self.name)
            entry = self._patch_single_config(cfg_path, data, dry_run)
            if entry:
                state["configs"].append(entry)
                codex_messages.append(f"  {cfg_path}\n    openai_base_url → {self.proxy_url}")

        if codex_messages:
            if not dry_run:
                self._save_state(state)
            codex_msg = "\n\nAlso patched Codex config(s):\n" + "\n".join(codex_messages)
            all_changes = list(result.changes) + ["codex_config"]
            result = AgentConfig(
                agent_name=result.agent_name,
                patched=result.patched,
                backup_path=result.backup_path,
                changes=all_changes,
                message=result.message + codex_msg,
            )

        # 3. OAuth warning (critical for Codex)
        oauth_note = (
            "\n\n[IMPORTANT] If Codex is currently authenticated via OAuth, "
            "run `codex /logout` before using LATTICE.  "
            "OAuth mode ignores OPENAI_API_KEY and will bypass the proxy."
        )
        result = AgentConfig(
            agent_name=result.agent_name,
            patched=result.patched,
            backup_path=result.backup_path,
            changes=list(result.changes),
            message=result.message + oauth_note,
        )

        return result

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        # 1. Remove env file
        result = super().unpatch(dry_run=dry_run)

        # 2. Restore all configs from state
        state = self._load_state()
        configs = state.get("configs", [])
        restored_msgs: list[str] = []

        for entry in configs:
            if self._unpatch_single_config(entry, dry_run):
                restored_msgs.append(f"  {entry['path']}")

        if restored_msgs:
            if not dry_run:
                self._state_path().unlink(missing_ok=True)
            codex_msg = "\n\nAlso restored Codex config(s):\n" + "\n".join(restored_msgs)
            all_changes = list(result.changes) + ["codex_config"]
            result = AgentConfig(
                agent_name=result.agent_name,
                patched=result.patched,
                backup_path=result.backup_path,
                changes=all_changes,
                message=result.message + codex_msg,
            )

        return result


class VSCodeIntegration(EnvFileIntegration):
    """VS Code — routes via standard OpenAI env vars."""

    @property
    def name(self) -> str:
        return "vscode"


class GenericIntegration(EnvFileIntegration):
    """Catch-all for any tool using standard OpenAI env vars."""

    @property
    def name(self) -> str:
        return "generic"


# =============================================================================
# 2. JSON file integrations (Cursor, OpenCode)
# =============================================================================


