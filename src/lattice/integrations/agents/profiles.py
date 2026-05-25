from __future__ import annotations

import os
import pathlib
import shutil
import sys
from typing import Any

import structlog

from lattice.integrations.agents.base import (
    AgentIntegration,
    _backup_dir,
    _load_json,
    _save_json,
    _timestamped_backup,
)
from lattice.integrations.agents.models import AgentConfig
from lattice.integrations.agents.protocol import AgentNotInstalledError
from lattice.integrations.mutation_store import (
    get_mutation,
)

logger = structlog.get_logger()

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

    def _is_agent_installed(self) -> bool:
        path = self._config_path()
        return path is not None and path.exists()

    def patch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None or not path.exists():
            if dry_run:
                return AgentConfig(
                    agent_name=self.name,
                    patched=False,
                    backup_path=None,
                    message=f"Config not found at {path}.",
                )
            raise AgentNotInstalledError(
                f"{self.name}: config file not found "
                f"({path or 'no path returned'}). "
                f"Is the agent installed? "
                f"Run `which {self.name}` to verify."
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

    def _is_agent_installed(self) -> bool:
        path = self._config_path()
        if path is None:
            return False
        return path.parent.parent.exists()

    def is_patched(self) -> bool:
        path = self._config_path()
        if path is None or not path.exists():
            return False
        return _load_json(path).get(self._MARKER_WRAPPED) is True

    def patch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None:
            raise AgentNotInstalledError(
                f"{self.name}: could not determine config path on this platform."
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

    def _is_agent_installed(self) -> bool:
        path = self._config_path()
        return path is not None and path.exists()

    def is_patched(self) -> bool:
        return self._state_path().exists()

    def patch(self, dry_run: bool = False) -> AgentConfig:
        path = self._config_path()
        if path is None or not path.exists():
            if dry_run:
                return AgentConfig(
                    agent_name=self.name,
                    patched=False,
                    backup_path=None,
                    message=f"Config not found at {path}.",
                )
            raise AgentNotInstalledError(
                f"{self.name}: config file not found "
                f"({path or 'no path returned'}). "
                f"Is the agent installed? "
                f"Run `which {self.name}` to verify."
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


class CopilotIntegration(AgentIntegration):
    """GitHub Copilot — durable hooks in ``~/.copilot/config.json`` via init."""

    @property
    def name(self) -> str:
        return "copilot"

    def _config_path(self) -> pathlib.Path:
        return pathlib.Path.home() / ".copilot" / "config.json"

    def _agent_binary_name(self) -> str | None:
        return "copilot"

    def is_patched(self) -> bool:
        mutation = get_mutation("copilot")
        if mutation is not None:
            return True
        path = self._config_path()
        if not path.exists():
            return False
        hooks = _load_json(path).get("hooks")
        return isinstance(hooks, dict) and hooks.get("lattice_init") is True

    def patch(self, dry_run: bool = False) -> AgentConfig:
        if not dry_run and not self._is_agent_installed():
            raise AgentNotInstalledError(
                f"{self.name}: `copilot` binary not found on PATH. "
                "Install GitHub Copilot CLI, then retry."
            )
        from lattice.integrations.copilot.install import apply_provider_scope

        if dry_run:
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message=f"Would patch Copilot config at {self._config_path()}",
            )
        from lattice.integrations.mutation_store import store_mutation

        mutation = apply_provider_scope(port=self.lattice_config.proxy_port)
        if mutation:
            store_mutation("copilot", mutation)
        return AgentConfig(
            agent_name=self.name,
            patched=True,
            backup_path=None,
            changes=["hooks"],
            message=f"Patched Copilot config at {self._config_path()}",
        )

    def unpatch(self, dry_run: bool = False) -> AgentConfig:
        from lattice.integrations.copilot.install import revert_provider_scope

        mutation = get_mutation("copilot")
        if mutation is None and not self.is_patched():
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message="Copilot is not routed through LATTICE.",
            )
        if dry_run:
            return AgentConfig(
                agent_name=self.name,
                patched=False,
                backup_path=None,
                message="Would restore Copilot configuration",
            )
        if mutation:
            revert_provider_scope(mutation)
        elif self._config_path().exists():
            revert_provider_scope(
                {"target": "copilot", "kind": "json-hooks", "path": str(self._config_path())}
            )
        return AgentConfig(
            agent_name=self.name,
            patched=False,
            backup_path=None,
            changes=["hooks"],
            message="Restored Copilot configuration",
        )


# =============================================================================
# 3. Registry
# =============================================================================
