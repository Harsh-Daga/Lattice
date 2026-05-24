"""Persistent mutation store for LATTICE init/uninit and transient lace.

Durable mutations (``lattice init``) live in ``~/.lattice/mutations.json``.
Transient lace sessions are tracked in ``~/.config/lattice/transient_laces.json``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MUTATIONS_PATH = Path.home() / ".lattice" / "mutations.json"
_TRANSIENT_PATH = Path.home() / ".config" / "lattice" / "transient_laces.json"


@dataclass(frozen=True, slots=True)
class TransientLaceRecord:
    """In-flight ``lattice lace`` session."""

    agent: str
    started_at: float
    pid: int


def _ensure_mutations_dir() -> None:
    _MUTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _ensure_transient_dir() -> None:
    _TRANSIENT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _file_lock(path: Path) -> int:
    import fcntl

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _file_unlock(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _read_transient_raw() -> dict[str, Any]:
    if not _TRANSIENT_PATH.exists():
        return {}
    try:
        return json.loads(_TRANSIENT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_transient_raw(data: dict[str, Any]) -> None:
    _ensure_transient_dir()
    tmp = _TRANSIENT_PATH.with_suffix(".tmp")
    fd = _file_lock(_TRANSIENT_PATH)
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(_TRANSIENT_PATH)
    finally:
        _file_unlock(fd)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class MutationStore:
    """Durable init mutations plus transient lace sessions."""

    def record(self, agent: str, mutation: dict[str, Any]) -> None:
        """Store a durable mutation for an agent."""
        store_mutation(agent, mutation)

    def list_mutated_agents(self) -> list[str]:
        """Return agent names with durable init mutations."""
        return list_mutated_agents()

    def remove(self, agent: str) -> None:
        """Remove a durable mutation entry."""
        remove_mutation(agent)

    def record_transient_lace(self, agent: str, pid: int) -> None:
        """Record an active ``lattice lace`` session."""
        data = _read_transient_raw()
        data[agent] = {
            "agent": agent,
            "started_at": time.time(),
            "pid": pid,
        }
        _write_transient_raw(data)

    def clear_transient_lace(self, agent: str) -> None:
        """Clear transient lace state for an agent."""
        data = _read_transient_raw()
        if agent in data:
            data.pop(agent, None)
            _write_transient_raw(data)

    def list_transient_laced(self) -> list[TransientLaceRecord]:
        """Return live transient lace records (prunes dead PIDs)."""
        data = _read_transient_raw()
        live: dict[str, Any] = {}
        records: list[TransientLaceRecord] = []
        for agent, entry in data.items():
            if not isinstance(entry, dict):
                continue
            pid = int(entry.get("pid", 0))
            if not _pid_alive(pid):
                continue
            live[agent] = entry
            records.append(
                TransientLaceRecord(
                    agent=agent,
                    started_at=float(entry.get("started_at", 0.0)),
                    pid=pid,
                )
            )
        if live != data:
            _write_transient_raw(live)
        return sorted(records, key=lambda r: r.agent)

    def list_all_active(self) -> list[str]:
        """Union of durable init mutations and live transient lace sessions."""
        durable = set(self.list_mutated_agents())
        transient = {r.agent for r in self.list_transient_laced()}
        return sorted(durable | transient)


_default_store = MutationStore()


def get_store() -> MutationStore:
    """Return the process-wide mutation store."""
    return _default_store


def load_mutations() -> dict[str, Any]:
    """Load the full durable mutation store."""
    if not _MUTATIONS_PATH.exists():
        return {}
    try:
        return json.loads(_MUTATIONS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_mutations(mutations: dict[str, Any]) -> None:
    """Persist the full durable mutation store atomically."""
    _ensure_mutations_dir()
    tmp = _MUTATIONS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(mutations, indent=2) + "\n")
    tmp.replace(_MUTATIONS_PATH)


def store_mutation(agent: str, mutation: dict[str, Any] | None) -> None:
    """Store a durable mutation for an agent (``None`` removes the entry)."""
    mutations = load_mutations()
    if mutation is None:
        mutations.pop(agent, None)
    else:
        mutations[agent] = mutation
    save_mutations(mutations)


def get_mutation(agent: str) -> dict[str, Any] | None:
    """Retrieve the stored durable mutation for an agent, or ``None``."""
    return load_mutations().get(agent)


def list_mutated_agents() -> list[str]:
    """Return all agent names that have stored durable mutations."""
    return sorted(load_mutations().keys())


def remove_mutation(agent: str) -> None:
    """Remove a durable mutation entry for an agent."""
    mutations = load_mutations()
    mutations.pop(agent, None)
    save_mutations(mutations)


def record_transient_lace(agent: str, pid: int) -> None:
    """Record an active transient lace session."""
    _default_store.record_transient_lace(agent, pid)


def clear_transient_lace(agent: str) -> None:
    """Clear transient lace state for an agent."""
    _default_store.clear_transient_lace(agent)


def list_transient_laced() -> list[TransientLaceRecord]:
    """Return live transient lace records."""
    return _default_store.list_transient_laced()


def list_all_active() -> list[str]:
    """Union of durable and transient active agents."""
    return _default_store.list_all_active()


def transient_record_to_dict(record: TransientLaceRecord) -> dict[str, Any]:
    """Serialize a transient record for JSON persistence."""
    return asdict(record)
