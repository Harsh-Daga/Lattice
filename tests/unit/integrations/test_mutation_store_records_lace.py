"""Transient lace tracking in mutation_store."""

from __future__ import annotations

import os

from lattice.integrations.mutation_store import MutationStore


def test_transient_lace_record_and_clear(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MutationStore()
    store.record_transient_lace("claude", pid=os.getpid())
    assert "claude" in {r.agent for r in store.list_transient_laced()}
    store.clear_transient_lace("claude")
    assert "claude" not in {r.agent for r in store.list_transient_laced()}


def test_list_all_active_unions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MutationStore()
    store.record("codex", {"timestamp": 1})
    store.record_transient_lace("claude", pid=os.getpid())
    assert set(store.list_all_active()) == {"codex", "claude"}
