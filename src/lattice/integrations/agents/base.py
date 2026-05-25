from __future__ import annotations

import json
import pathlib
import shutil
import time
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.integrations.agents.protocol import AgentDoctorReport

logger = structlog.get_logger()


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

from lattice.integrations.agents.models import AgentConfig


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

    def _agent_binary_name(self) -> str | None:
        """Optional PATH binary used to detect installation."""
        return None

    def _is_agent_installed(self) -> bool:
        binary = self._agent_binary_name()
        if binary is None:
            return True
        return shutil.which(binary) is not None

    def doctor(self) -> AgentDoctorReport:
        from lattice.integrations.agents.doctor import build_agent_doctor_report

        return build_agent_doctor_report(self, is_installed=self._is_agent_installed())


# =============================================================================
# 1. Env-file integrations (Claude Code, Codex, generic)
# =============================================================================


