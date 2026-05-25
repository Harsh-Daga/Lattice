from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import time
from typing import Any

import structlog

logger = structlog.get_logger()

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

