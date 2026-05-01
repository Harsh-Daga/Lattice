"""Persistent mutation store for LATTICE init/uninit.

Stores the state returned by ``apply_provider_scope`` so that
``revert_provider_scope`` can restore the original configuration.

File layout (JSON)::~

    {
      "claude": {
        "target": "claude",
        "kind": "json-env",
        "path": "/Users/.../.claude/settings.json",
        "previous": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
      },
      "codex": { ... },
      ...
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MUTATIONS_PATH = Path.home() / ".lattice" / "mutations.json"


def _ensure_dir() -> None:
    _MUTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_mutations() -> dict[str, Any]:
    """Load the full mutation store."""
    if not _MUTATIONS_PATH.exists():
        return {}
    try:
        return json.loads(_MUTATIONS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_mutations(mutations: dict[str, Any]) -> None:
    """Persist the full mutation store atomically."""
    _ensure_dir()
    tmp = _MUTATIONS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(mutations, indent=2) + "\n")
    tmp.replace(_MUTATIONS_PATH)


def store_mutation(agent: str, mutation: dict[str, Any] | None) -> None:
    """Store a mutation for an agent.

    If *mutation* is None the agent entry is removed.
    """
    mutations = load_mutations()
    if mutation is None:
        mutations.pop(agent, None)
    else:
        mutations[agent] = mutation
    save_mutations(mutations)


def get_mutation(agent: str) -> dict[str, Any] | None:
    """Retrieve the stored mutation for an agent, or None."""
    return load_mutations().get(agent)


def list_mutated_agents() -> list[str]:
    """Return all agent names that have stored mutations."""
    return sorted(load_mutations().keys())


def remove_mutation(agent: str) -> None:
    """Remove a mutation entry for an agent."""
    mutations = load_mutations()
    mutations.pop(agent, None)
    save_mutations(mutations)
