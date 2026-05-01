"""Tests for lattice.integrations.mutation_store."""

from __future__ import annotations

import json
from pathlib import Path

from lattice.integrations.mutation_store import (
    get_mutation,
    list_mutated_agents,
    load_mutations,
    remove_mutation,
    save_mutations,
    store_mutation,
)


def _clear_store() -> None:
    """Remove the mutations file so tests start clean."""
    from lattice.integrations.mutation_store import _MUTATIONS_PATH

    _MUTATIONS_PATH.unlink(missing_ok=True)


def test_round_trip() -> None:
    _clear_store()
    mutation = {"target": "claude", "kind": "json-env", "path": "/tmp/test.json"}
    store_mutation("claude", mutation)
    assert get_mutation("claude") == mutation


def test_get_missing() -> None:
    _clear_store()
    assert get_mutation("nonexistent") is None


def test_remove_mutation() -> None:
    _clear_store()
    store_mutation("codex", {"target": "codex"})
    remove_mutation("codex")
    assert get_mutation("codex") is None


def test_list_mutated_agents() -> None:
    _clear_store()
    store_mutation("claude", {"target": "claude"})
    store_mutation("codex", {"target": "codex"})
    assert list_mutated_agents() == ["claude", "codex"]


def test_save_and_load() -> None:
    _clear_store()
    data = {"a": 1, "b": {"nested": True}}
    save_mutations(data)
    assert load_mutations() == data


def test_store_none_removes() -> None:
    _clear_store()
    store_mutation("claude", {"target": "claude"})
    store_mutation("claude", None)
    assert get_mutation("claude") is None
