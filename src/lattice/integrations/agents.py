"""Agent integrations for LATTICE.

Architecture
------------
Every supported AI coding agent gets an ``AgentIntegration`` subclass.  The
subclass only has to answer three questions:

1. **Where is your config?** → ``_config_path()``
2. **How do we route you through LATTICE?** → ``patch()``
3. **How do we put you back the way you were?** → ``unpatch()``

Two reusable bases cover 90 % of agents:

* ``EnvFileIntegration`` — for agents driven by ``OPENAI_BASE_URL`` env var
  (Claude Code, Codex, generic).  We persist a ``.env`` file that the user
  sources before launching the agent.

* ``JsonFileIntegration`` — for agents driven by JSON config files (Cursor,
  OpenCode).  We rewrite ``base_url`` / ``baseURL`` fields and store the
  *original* value inside the file itself as ``_lattice_original``.  This
  makes ``unpatch`` work even when timestamped disk backups have been
  cleaned up.

Public API
----------
``wrap_agent(name)``      — route an agent through the LATTICE proxy.
``unwrap_agent(name)``    — restore original routing.
``agent_status(name)``    — check whether an agent is currently routed.
``wrap_all()``            — batch wrap every supported agent.
``unwrap_all()``          — batch unwrap every supported agent.

Supported agents
----------------
+-------------+----------------+----------------------------------+
| Agent       | Driver         | Config location                  |
+=============+================+==================================+
| claude      | env var        | ``~/.config/lattice/claude.env`` |
| claude-code | env var        | ``~/.config/lattice/claude.env`` |
| codex       | env var        | ``~/.config/lattice/codex.env``   |
| cursor      | JSON file      | ``...Cursor/User/settings.json`` |
| opencode    | JSON file      | ``~/.config/opencode/opencode.json`` |
| generic     | env var        | ``~/.config/lattice/generic.env`` |
+-------------+----------------+----------------------------------+

How to add a new agent
----------------------
1. Subclass ``EnvFileIntegration`` or ``JsonFileIntegration``.
2. Override ``_config_path()`` (and ``_extract_url`` / ``_inject_url`` for JSON).
3. Register in ``_AGENT_REGISTRY``.
4. Add tests in ``tests/unit/test_agents.py``.
5. Update the table above.

Safety guarantees
-----------------
* **Backups** — every wrap creates a timestamped copy in
  ``~/.config/lattice/backups/``.
* **In-place reversibility** — the original URL is embedded inside the agent's
  own config as ``_lattice_original``.  ``unwrap`` prefers this over disk
  backups, so restoration works even if backups are deleted.
* **Non-destructive** — only ``base_url`` / ``baseURL`` / env vars are touched.
  API keys, models, MCP servers, and other metadata are never modified.
* **Idempotent** — wrapping twice is a no-op (no duplicate backups).
* **Cross-platform** — macOS / Linux / Windows paths resolved automatically.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shutil
import sys
import time
from collections.abc import Callable
from typing import Any

import structlog

from lattice.core.config import LatticeConfig

logger = structlog.get_logger()


# =============================================================================
# Data model
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class AgentConfig:
    """Result of a wrap / unwrap operation."""

    agent_name: str
    patched: bool
    backup_path: str | None
    changes: list[str] = dataclasses.field(default_factory=list)
    message: str = ""


# =============================================================================
# Shared helpers
# =============================================================================


def _backup_dir() -> pathlib.Path:
    p = pathlib.Path.home() / ".config" / "lattice" / "backups"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _timestamped_backup(source: pathlib.Path, prefix: str) -> pathlib.Path | None:
    """Copy *source* to ``~/.config/lattice/backups/{prefix}-{ts}.json``."""
    if not source.exists():
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = _backup_dir() / f"{prefix}-{ts}.json"
    shutil.copy2(source, dest)
    return dest


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except Exception:
        return {}


def _save_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


# =============================================================================
# Base class
# =============================================================================


class AgentIntegration:
    """Abstract base for every agent integration.

    The *only* public contract required by the registry is:

    * ``name`` – human-readable identifier (also the CLI argument).
    * ``patch(dry_run) -> AgentConfig``
    * ``unpatch(dry_run) -> AgentConfig``
    * ``is_patched() -> bool``
    """

    def __init__(self, lattice_config: LatticeConfig | None = None) -> None:
        self.lattice_config = lattice_config or LatticeConfig.auto()
        self.proxy_url = self.lattice_config.proxy_url()
        self._log = logger.bind(module="agent_integration")

    @property
    def name(self) -> str:
        return ""

    def patch(self, dry_run: bool = False) -> AgentConfig:
        """Route this agent through the LATTICE proxy."""
        raise NotImplementedError

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        """Restore the agent's original routing."""
        raise NotImplementedError

    def is_patched(self) -> bool:
        """Return ``True`` if currently routed through LATTICE."""
        raise NotImplementedError


# =============================================================================
# 1. Env-file integrations (Claude Code, Codex, generic)
# =============================================================================


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


class JsonFileIntegration(AgentIntegration):
    """Generic base for agents that store config in JSON files.

    Subclasses override three hooks:

    * ``_config_path()`` → ``pathlib.Path`` to the config file.
    * ``_extract_url(data)`` → read the current provider base URL from *data*.
    * ``_inject_url(data, url)`` → return *data* with the URL rewritten.

    The base class handles backup, marker management, and restoration.
    """

    _MARKER_ORIGINAL = "_lattice_original"
    _MARKER_WRAPPED = "_lattice_wrapped"

    # ------------------------------------------------------------------ hooks

    def _config_path(self) -> pathlib.Path | None:
        raise NotImplementedError

    def _extract_url(self, data: dict[str, Any]) -> str | None:
        raise NotImplementedError

    def _inject_url(self, data: dict[str, Any], url: str) -> dict[str, Any]:
        raise NotImplementedError

    # -------------------------------------------------------------- lifecycle

    def is_patched(self) -> bool:
        path = self._config_path()
        if path is None or not path.exists():
            return False
        return _load_json(path).get(self._MARKER_WRAPPED) is True

    def patch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None:
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Could not determine config path for {self.name}.",
            )
        if not path.exists():
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Config not found at {path}.",
            )

        if self.is_patched():
            return AgentConfig(
                agent_name=self.name,
                patched=True,
                backup_path=None,
                changes=[],
                message=f"{self.name} is already routed through LATTICE. No changes made.",
            )

        backup: pathlib.Path | None = None
        if not dry_run:
            backup = _timestamped_backup(path, self.name)
        data = _load_json(path)
        original_url = self._extract_url(data)

        data[self._MARKER_ORIGINAL] = original_url
        data[self._MARKER_WRAPPED] = True
        data = self._inject_url(data, self.proxy_url)

        if not dry_run:
            _save_json(path, data)

        return AgentConfig(
            agent_name=self.name,
            patched=True,
            backup_path=str(backup) if backup else None,
            changes=["base_url"],
            message=(
                f"Patched {self.name} config → {self.proxy_url}\n"
                f"Backup: {backup}\n"
                f"Original URL preserved inside config under '{self._MARKER_ORIGINAL}'"
            ),
        )

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None or not path.exists():
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Config not found at {path}.",
            )

        data = _load_json(path)

        # 1. In-place restore from marker (preferred — always works)
        if self._MARKER_ORIGINAL in data:
            original = data.pop(self._MARKER_ORIGINAL)
            data.pop(self._MARKER_WRAPPED, None)
            data = self._inject_url(data, original)
            if not dry_run:
                _save_json(path, data)
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                changes=["base_url"],
                message=f"Restored {self.name} config to original URL: {original}",
            )

        # 2. Fallback: restore from latest timestamped backup
        backups = sorted(_backup_dir().glob(f"{self.name}-*.json"), reverse=True)
        if backups:
            latest = backups[0]
            if not dry_run:
                shutil.copy2(latest, path)
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=str(latest),
                changes=["base_url"],
                message=f"Restored {self.name} config from backup: {latest}",
            )

        return AgentConfig(
            agent_name=self.name,
            patched=False,
            backup_path=None,
            changes=[],
            message=f"No marker or backup found for {self.name}. Nothing to restore.",
        )


class CursorIntegration(JsonFileIntegration):
    """Cursor IDE — patches provider base URLs in settings.json.

    Cursor's chat/composer interface speaks OpenAI-compatible API. Users
    configure a custom OpenAI base URL in Cursor settings; the IDE then
    sends all chat requests to that endpoint. LATTICE acts as the proxy.

    Cursor also supports direct Anthropic and Gemini adapters. We patch
    ALL provider base URLs so every model routes through LATTICE regardless
    of which provider the user has selected in the UI.

    Research references:
    - https://github.com/pezzos/cursor-openrouter-proxy (proxy approach)
    - https://cursor.com/docs/api (Cursor enterprise APIs, not IDE)

    Important: Cursor should keep "GPT-4o" selected in the UI. LATTICE's
    proxy routes the request to the actual model based on the ``model``
    parameter in the request body.
    """

    # Canonical provider keys we inject. We use ``baseUrl`` (camelCase) as
    # the canonical form and delete conflicting ``baseURL`` variants.
    _CANONICAL_PROVIDERS: tuple[str, ...] = (
        "cursor.openai.baseUrl",
        "cursor.anthropic.baseUrl",
        "cursor.gemini.baseUrl",
    )

    # All known variants (camelCase + PascalCase) that we clean up
    _ALL_VARIANTS: tuple[str, ...] = (
        "cursor.openai.baseUrl",
        "cursor.openai.baseURL",
        "cursor.anthropic.baseUrl",
        "cursor.anthropic.baseURL",
        "cursor.gemini.baseUrl",
        "cursor.gemini.baseURL",
    )

    # Marker keys stored inside settings.json
    _MARKER_ORIGINAL_URLS = "_lattice_original_urls"  # key → original URL
    _MARKER_CREATED_KEYS = "_lattice_created_keys"  # keys we added

    @property
    def name(self) -> str:
        return "cursor"

    def _config_path(self) -> pathlib.Path | None:
        home = pathlib.Path.home()
        if sys.platform == "darwin":
            return home / "Library" / "Application Support" / "Cursor" / "User" / "settings.json"
        elif sys.platform.startswith("linux"):
            return home / ".config" / "Cursor" / "User" / "settings.json"
        elif sys.platform == "win32":
            return (
                pathlib.Path(os.environ.get("APPDATA", str(home / "AppData" / "Roaming")))
                / "Cursor"
                / "User"
                / "settings.json"
            )
        return None

    def _extract_url(self, data: dict[str, Any]) -> str | None:
        # Cursor may use either baseUrl or baseURL — check both
        return data.get("cursor.openai.baseUrl") or data.get("cursor.openai.baseURL")

    def _inject_url(self, data: dict[str, Any], url: str) -> dict[str, Any]:
        # Normalize OpenAI key and remove conflicting baseURL
        data["cursor.openai.baseUrl"] = url
        data.pop("cursor.openai.baseURL", None)
        return data

    def is_patched(self) -> bool:
        path = self._config_path()
        if path is None or not path.exists():
            return False
        return _load_json(path).get(self._MARKER_WRAPPED) is True

    def patch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None:
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Could not determine config path for {self.name}.",
            )

        # Create settings.json if it doesn't exist yet (user never customised)
        created_new_file = False
        if not path.exists():
            if dry_run:
                return AgentConfig(
                    agent_name=self.name,
                    patched=False,
                    backup_path=None,
                    message=f"Config not found at {path}. Would create new file.",
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            created_new_file = True

        if self.is_patched():
            return AgentConfig(
                agent_name=self.name,
                patched=True,
                backup_path=None,
                changes=[],
                message=f"{self.name} is already routed through LATTICE. No changes made.",
            )

        backup: pathlib.Path | None = None
        if not dry_run and not created_new_file:
            backup = _timestamped_backup(path, self.name)
        data = _load_json(path)

        # ------------------------------------------------------------------
        # 1. Record original URLs for keys that already exist
        # ------------------------------------------------------------------
        originals: dict[str, str] = {}
        created_keys: list[str] = []

        for canonical in self._CANONICAL_PROVIDERS:
            # Find any existing variant (baseUrl or baseURL)
            existing_variant = None
            for variant in self._ALL_VARIANTS:
                if variant.startswith(canonical.rsplit(".", 1)[0]) and variant in data:
                    existing_variant = variant
                    break

            if existing_variant is not None and data[existing_variant] != self.proxy_url:
                # Existing key with a non-LATTICE URL → save original, then replace
                originals[canonical] = data[existing_variant]
                # Delete all variants for this provider
                for v in self._ALL_VARIANTS:
                    if v.startswith(canonical.rsplit(".", 1)[0]):
                        data.pop(v, None)
                data[canonical] = self.proxy_url
            elif existing_variant is None:
                # No variant exists → create the canonical key
                created_keys.append(canonical)
                data[canonical] = self.proxy_url
            # else: already set to our URL (shouldn't happen because is_patched checked)

        # ------------------------------------------------------------------
        # 2. Store markers
        # ------------------------------------------------------------------
        if originals:
            data[self._MARKER_ORIGINAL_URLS] = originals
        if created_keys:
            data[self._MARKER_CREATED_KEYS] = created_keys
        data[self._MARKER_WRAPPED] = True

        if not dry_run:
            _save_json(path, data)

        changed = list(originals.keys()) + created_keys
        msg_parts = [f"Patched {self.name} config → {self.proxy_url}"]
        if created_new_file:
            msg_parts.append(f"Created new settings.json at {path}")
        if originals:
            msg_parts.append(f"Modified {len(originals)} existing provider key(s)")
        if created_keys:
            msg_parts.append(f"Added {len(created_keys)} new provider key(s)")
        msg_parts.append(
            "Tip: Keep GPT-4o selected in Cursor's model dropdown. "
            "LATTICE routes to the actual model from the request."
        )

        return AgentConfig(
            agent_name=self.name,
            patched=True,
            backup_path=str(backup) if backup else None,
            changes=changed or ["cursor.openai.baseUrl"],
            message="\n".join(msg_parts),
        )

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None or not path.exists():
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Config not found at {path}.",
            )

        data = _load_json(path)
        restored: list[str] = []
        deleted: list[str] = []
        had_marker = False

        # ------------------------------------------------------------------
        # 1. Restore modified keys
        # ------------------------------------------------------------------
        originals = data.pop(self._MARKER_ORIGINAL_URLS, None)
        if isinstance(originals, dict):
            had_marker = True
            for key, orig_url in originals.items():
                # Remove all variants first, then restore canonical
                prefix = key.rsplit(".", 1)[0]
                for v in self._ALL_VARIANTS:
                    if v.startswith(prefix):
                        data.pop(v, None)
                data[key] = orig_url
                restored.append(key)

        # ------------------------------------------------------------------
        # 2. Delete keys we created
        # ------------------------------------------------------------------
        created = data.pop(self._MARKER_CREATED_KEYS, None)
        if isinstance(created, list):
            had_marker = True
            for key in created:
                if key in data:
                    del data[key]
                    deleted.append(key)
                # Also clean up any lingering variants
                prefix = key.rsplit(".", 1)[0]
                for v in self._ALL_VARIANTS:
                    if v.startswith(prefix) and v in data:
                        del data[v]
                        if v not in deleted:
                            deleted.append(v)

        # Always remove the wrapped flag if any marker was present
        if had_marker:
            data.pop(self._MARKER_WRAPPED, None)

        if restored or deleted:
            if not dry_run:
                _save_json(path, data)
            msg_parts = [f"Restored {self.name} config"]
            if restored:
                msg_parts.append(f"Restored original URLs: {', '.join(restored)}")
            if deleted:
                msg_parts.append(f"Removed created keys: {', '.join(deleted)}")
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                changes=restored + deleted,
                message="\n".join(msg_parts),
            )

        # 3. Fallback: restore from latest timestamped backup
        backups = sorted(_backup_dir().glob(f"{self.name}-*.json"), reverse=True)
        if backups:
            latest = backups[0]
            if not dry_run:
                shutil.copy2(latest, path)
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=str(latest),
                changes=["base_url"],
                message=f"Restored {self.name} config from backup: {latest}",
            )

        return AgentConfig(
            agent_name=self.name,
            patched=False,
            backup_path=None,
            changes=[],
            message=f"No marker or backup found for {self.name}. Nothing to restore.",
        )


class OpenCodeIntegration(JsonFileIntegration):
    """OpenCode — patches supported ``provider.<name>.options.baseURL`` entries.

    Only providers that LATTICE has adapters for are modified.  Unsupported
    providers are left completely untouched so OpenCode can still use them
    natively.

    Because OpenCode validates its config file and rejects unknown keys,
    LATTICE state (original URLs, wrapped flag) is stored in a **separate**
    file at ``~/.config/lattice/opencode_state.json``.  Only the provider
    URLs are rewritten inside the actual OpenCode config.
    """

    # Provider slugs that LATTICE has adapters for.  Only these are patched.
    _SUPPORTED_PROVIDERS: tuple[str, ...] = (
        "openai",
        "anthropic",
        "ollama",
        "ollama-cloud",
        "azure",
        "bedrock",
        "openadapter",
    )

    @property
    def name(self) -> str:
        return "opencode"

    def _config_path(self) -> pathlib.Path | None:
        home = pathlib.Path.home()
        if sys.platform == "win32":
            return (
                pathlib.Path(os.environ.get("APPDATA", str(home / "AppData" / "Roaming")))
                / "opencode"
                / "opencode.json"
            )
        return home / ".config" / "opencode" / "opencode.json"

    def _state_path(self) -> pathlib.Path:
        """Path to LATTICE's separate state file for OpenCode."""
        return _backup_dir().parent / "opencode_state.json"

    def _extract_url(self, _data: dict[str, Any]) -> str | None:
        return None  # not a single URL — handled below

    def _inject_url(self, data: dict[str, Any], _url: str) -> dict[str, Any]:
        return data  # overridden in patch / unpatch

    def is_patched(self) -> bool:
        return self._state_path().exists()

    def patch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None or not path.exists():
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Config not found at {path}.",
            )

        if self.is_patched():
            return AgentConfig(
                agent_name=self.name,
                patched=True,
                backup_path=None,
                changes=[],
                message=f"{self.name} is already routed through LATTICE. No changes made.",
            )

        backup: pathlib.Path | None = None
        if not dry_run:
            backup = _timestamped_backup(path, self.name)
        data = _load_json(path)
        providers = data.setdefault("provider", {})
        originals: dict[str, str] = {}
        changes: list[str] = []
        skipped: list[str] = []

        for p_name, p_cfg in providers.items():
            # Only touch providers LATTICE supports
            if p_name not in self._SUPPORTED_PROVIDERS:
                skipped.append(p_name)
                continue

            opts = p_cfg.setdefault("options", {})
            old_url = opts.get("baseURL") or opts.get("baseUrl")
            if old_url and old_url != self.proxy_url:
                originals[p_name] = old_url
                opts["baseURL"] = self.proxy_url
                opts.pop("baseUrl", None)
                changes.append(f"provider.{p_name}.options.baseURL")

            # Inject x-lattice-provider header so proxy knows which provider
            # this request originated from (critical for unambiguous routing)
            headers = opts.setdefault("headers", {})
            headers["x-lattice-provider"] = p_name
            changes.append(f"provider.{p_name}.options.headers.x-lattice-provider")

        if not dry_run:
            _save_json(path, data)
            # Write LATTICE state to separate file (not inside opencode.json)
            _save_json(
                self._state_path(),
                {"originals": originals, "skipped": skipped},
            )

        backup_msg = f"Backup: {backup}\n" if backup else ""
        skipped_msg = f"Skipped unsupported: {', '.join(skipped)}\n" if skipped else ""
        return AgentConfig(
            agent_name=self.name,
            patched=True,
            backup_path=str(backup) if backup else None,
            changes=changes,
            message=(
                f"Patched OpenCode config ({len(changes)} supported provider(s))\n"
                f"{backup_msg}"
                f"{skipped_msg}"
                f"Original URLs preserved in LATTICE state file: {self._state_path()}"
            ),
        )

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None or not path.exists():
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Config not found at {path}.",
            )

        data = _load_json(path)
        state_path = self._state_path()

        # 1. In-place restore from LATTICE state file
        state = _load_json(state_path)
        originals = state.get("originals")
        if isinstance(originals, dict):
            providers = data.setdefault("provider", {})
            for p_name, orig_url in originals.items():
                p_cfg = providers.setdefault(p_name, {})
                opts = p_cfg.setdefault("options", {})
                opts["baseURL"] = orig_url
                # Remove injected x-lattice-provider header
                headers = opts.get("headers", {})
                headers.pop("x-lattice-provider", None)
                if not headers:
                    opts.pop("headers", None)
            if not dry_run:
                _save_json(path, data)
                state_path.unlink(missing_ok=True)
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                changes=list(originals.keys()),
                message=f"Restored OpenCode providers: {', '.join(originals.keys())}",
            )

        # 2. Fallback to timestamped backup
        backups = sorted(_backup_dir().glob("opencode-*.json"), reverse=True)
        if backups:
            latest = backups[0]
            if not dry_run:
                shutil.copy2(latest, path)
                state_path.unlink(missing_ok=True)
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=str(latest),
                changes=["restored from backup"],
                message=f"Restored OpenCode config from backup: {latest}",
            )

        return AgentConfig(
            agent_name=self.name,
            patched=False,
            backup_path=None,
            changes=[],
            message="No marker or backup found for OpenCode. Nothing to restore.",
        )


# =============================================================================
# 3. Registry
# =============================================================================

_AGENT_REGISTRY: dict[str, Callable[[LatticeConfig | None], AgentIntegration]] = {
    "claude": ClaudeCodeIntegration,
    "claude-code": ClaudeCodeIntegration,
    "codex": CodexIntegration,
    "cursor": CursorIntegration,
    "opencode": OpenCodeIntegration,
    "vscode": VSCodeIntegration,
    "generic": GenericIntegration,
}


# =============================================================================
# 4. Public API
# =============================================================================


def list_agents() -> list[str]:
    """Return all supported agent names."""
    return list(_AGENT_REGISTRY.keys())


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
        results.append(integration.patch(dry_run=dry_run))
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
