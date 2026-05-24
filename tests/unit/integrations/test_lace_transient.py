"""Lace records and clears transient mutation store state."""

from __future__ import annotations

from typing import Any

from lattice.integrations import mutation_store as ms
from lattice.integrations.lace import lace_agent


def test_lace_records_and_clears_transient(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    events: list[tuple[str, str]] = []

    def track_record(agent: str, pid: int) -> None:
        events.append(("record", agent))
        ms.record_transient_lace(agent, pid)

    def track_clear(agent: str) -> None:
        events.append(("clear", agent))
        ms.clear_transient_lace(agent)

    monkeypatch.setattr("lattice.integrations.lace.record_transient_lace", track_record)
    monkeypatch.setattr("lattice.integrations.lace.clear_transient_lace", track_clear)
    monkeypatch.setattr(
        "lattice.integrations.lace._ensure_proxy",
        lambda *args, **kwargs: {"started": False, "message": "ok"},
    )
    monkeypatch.setattr("lattice.integrations.lace._find_agent_binary", lambda _a: "/bin/echo")

    class _FakeProc:
        def wait(self) -> int:
            return 0

    monkeypatch.setattr("lattice.integrations.lace.subprocess.Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(
        "lattice.integrations.lace.build_launch_env",
        lambda *a, **k: ({}, []),
    )

    assert ms.list_transient_laced() == []
    exit_code = lace_agent("claude", no_tunnel=True, no_start=True)
    assert exit_code == 0
    assert ("record", "claude") in events
    assert ("clear", "claude") in events
    assert ms.list_transient_laced() == []
