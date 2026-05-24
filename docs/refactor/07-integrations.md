# Phase 7 (REFACTOR_PLAN) / STATUS Phase 8 — Agent Integrations

> **STATUS Phase 8** (this file is REFACTOR_PLAN Phase 7 in the original numbering).

> **Goal.** The five agent integrations (Claude Code, Codex, Cursor, OpenCode, GitHub Copilot) are already well-decomposed via `EnvFileIntegration` / `JsonFileIntegration` base classes in `agents.py`. Phase 7 verifies that decomposition holds, surfaces a couple of latent issues found in the audit (`mutation_store` not consistently used by every concrete class; one agent integration silently no-ops on patch when its config file is missing instead of raising), tightens the `AgentIntegration` Protocol so mypy can prove every subclass is complete, ensures `lattice doctor <agent>` covers all five, and decides the fate of `core/tunnel_sidecar.py` (784 LoC).
>
> **Outcome.** `from lattice.integrations import AgentIntegration, ClaudeCodeIntegration, CodexIntegration, CursorIntegration, OpenCodeIntegration, CopilotIntegration` works. Each concrete integration has identical lifecycle (patch / unpatch / is_patched / status). `lattice doctor <agent>` runs a per-agent health check matrix. `tunnel_sidecar.py` moves to `integrations/tunnel.py`.
>
> **Estimated effort.** 1 day.

---

## 1. Why this phase exists

The audit found `integrations/` is already well-decomposed (per Phase 6's audit of `integrations/agents.py` at 1444 LoC with clean inheritance). What's *not* clean:

1. **`mutation_store.py` is inconsistently consumed.** `init.py` writes mutation records when patching; `lace.py` doesn't (it's transient). But `agent_status()` reads from `mutation_store` to decide whether an agent is "configured". The result: an agent laced but never `init`-ed shows as "not configured" in `lattice status` even though `lattice lace claude` would route requests through LATTICE. Fix the asymmetry: `lattice status` should reflect both durable (`init`) and transient (`lace`) states.

2. **One agent silently no-ops when its config file doesn't exist.** Per the audit's `JsonFileIntegration` base class behaviour: if `_config_path()` returns a path that doesn't exist, `patch(dry_run=False)` returns success without doing anything. This is wrong — the user typed `lattice init cursor` and expected Cursor to be patched. Should raise `AgentNotInstalledError`.

3. **`AgentIntegration` Protocol vs. concrete classes drift.** The current `agents.py` declares the base methods but they aren't `Protocol`-enforced. Adding `@runtime_checkable` Protocol + abstract method declarations makes mypy enforce every concrete class is complete.

4. **`lattice doctor` only knows about `claude`, `codex`, `opencode`.** Audit confirmed `cursor` and `copilot` are missing from doctor's case statements. Fix.

5. **`core/tunnel_sidecar.py` (784 LoC) is the sidecar process that runs when `lattice lace <agent>` is invoked without `--no-tunnel`.** It implements local Unix socket / HTTP proxy / WebSocket tunnel. It's logically an integrations concern, not a core primitive. Move it.

---

## 2. Files touched

### 2.1 Moved

| Current path | New path |
|---|---|
| `src/lattice/core/tunnel_sidecar.py` | `src/lattice/integrations/tunnel.py` |

### 2.2 Modified

- `src/lattice/integrations/agents.py` — strengthen `AgentIntegration` Protocol; add `AgentNotInstalledError`; raise it from `JsonFileIntegration.patch()` when `_config_path()` doesn't exist; standardise `mutation_store` usage.
- `src/lattice/integrations/mutation_store.py` — add `record_transient_lace(agent)` / `clear_transient_lace(agent)` methods so `lace.py` can track in-flight lacing for `lattice status` to see.
- `src/lattice/integrations/init.py` — call `mutation_store.record(...)` consistently.
- `src/lattice/integrations/lace.py` — call `mutation_store.record_transient_lace(...)` on entry; `clear_transient_lace(...)` on exit (including signal-handler cleanup).
- `src/lattice/integrations/registry.py` — verify the list of supported agents matches `agents.py`'s subclasses exactly (no drift).
- `src/lattice/cli.py` — update `_cmd_doctor` to handle all five agents.

### 2.3 Created

```
tests/unit/integrations/test_agent_protocol.py
tests/unit/integrations/test_mutation_store_records_lace.py
tests/unit/integrations/test_jsonfile_raises_when_config_missing.py
tests/unit/integrations/test_doctor_covers_all_agents.py
```

### 2.4 Deleted

Nothing. (`tunnel_sidecar.py` is moved, not deleted.)

---

## 3. Step-by-step

### 3.1 Move tunnel_sidecar

```bash
git mv src/lattice/core/tunnel_sidecar.py src/lattice/integrations/tunnel.py
sd 'from lattice\.core\.tunnel_sidecar import' 'from lattice.integrations.tunnel import' $(rg -l "from lattice.core.tunnel_sidecar import")
```

Update `pyproject.toml`'s ruff per-file-ignores:

```toml
[tool.ruff.lint.per-file-ignores]
# Was:  "src/lattice/core/tunnel_sidecar.py" = ["E402"]
"src/lattice/integrations/tunnel.py" = ["E402"]
```

Update `pyproject.toml` test ignores:

```toml
# Was:  "tests/unit/test_tunnel_sidecar.py" = ["F841"]
"tests/unit/integrations/test_tunnel.py" = ["F841"]
```

Move the test:

```bash
mkdir -p tests/unit/integrations
git mv tests/unit/test_tunnel_sidecar.py tests/unit/integrations/test_tunnel.py
```

### 3.2 Strengthen the `AgentIntegration` Protocol

In `integrations/agents.py`, add at module top:

```python
from typing import Protocol, runtime_checkable

class AgentNotInstalledError(Exception):
    """Raised when an integration target (config file, env file, executable) is not present."""

@runtime_checkable
class AgentIntegrationProtocol(Protocol):
    """Stable Protocol every integration subclass must satisfy."""
    @property
    def name(self) -> str: ...
    @property
    def proxy_url(self) -> str: ...
    def patch(self, dry_run: bool = False) -> "AgentConfig": ...
    def unpatch(self, dry_run: bool = False) -> "AgentConfig": ...
    def is_patched(self) -> bool: ...
    def doctor(self) -> "AgentDoctorReport": ...   # NEW — see §3.6
```

Update the abstract base class `AgentIntegration` to implement the Protocol; mark `name` as `abstract` so subclasses must override.

### 3.3 Raise on missing config file

In `integrations/agents.py`'s `JsonFileIntegration`:

```python
class JsonFileIntegration(AgentIntegration):
    def _config_path(self) -> Path | None:
        """Subclass returns the JSON config path or None if agent not installed."""
        raise NotImplementedError

    def patch(self, dry_run: bool = False) -> AgentConfig:
        cfg_path = self._config_path()
        if cfg_path is None or not cfg_path.exists():
            raise AgentNotInstalledError(
                f"{self.name}: config file not found "
                f"({cfg_path or 'no path returned'}). "
                f"Is the agent installed? "
                f"Run `which {self.name}` to verify."
            )
        # ... existing patch logic ...
```

The caller (`init.py` or `lace.py`) catches `AgentNotInstalledError` and reports a clear failure rather than a silent success.

### 3.4 Standardise mutation_store usage

`integrations/mutation_store.py` today tracks "durable" mutations (those from `lattice init`). Add transient tracking:

```python
@dataclass
class TransientLaceRecord:
    agent: str
    started_at: float
    pid: int

class MutationStore:
    def record(self, agent: str, mutation: dict) -> None: ...   # existing, durable
    def list_mutated_agents(self) -> list[str]: ...             # existing
    def remove(self, agent: str) -> None: ...                   # existing

    # NEW:
    def record_transient_lace(self, agent: str, pid: int) -> None: ...
    def clear_transient_lace(self, agent: str) -> None: ...
    def list_transient_laced(self) -> list[TransientLaceRecord]: ...
    def list_all_active(self) -> list[str]:
        """Union of durable + transient."""
        return list(set(self.list_mutated_agents()) | {r.agent for r in self.list_transient_laced()})
```

Backing storage: in-process `dict` + optional file at `~/.config/lattice/transient_laces.json` updated atomically.

### 3.5 Hook `lace.py` into transient mutation_store

In `integrations/lace.py`, the function `lace_agent(...)`:

```python
def lace_agent(agent: str, args: list[str], port: int, no_start: bool, no_patch: bool, no_tunnel: bool, dry_run: bool) -> int:
    ...
    if not dry_run and not no_patch:
        mutation_store.record_transient_lace(agent, pid=os.getpid())
    try:
        # ... existing flow: maybe start proxy, maybe start tunnel, launch agent ...
        return exit_code
    finally:
        if not dry_run and not no_patch:
            mutation_store.clear_transient_lace(agent)
```

Now `lattice status` (in `cli.py`'s `_cmd_agent_status`) can call `mutation_store.list_all_active()` and report both kinds.

### 3.6 Add `doctor()` method to each integration

The `AgentIntegrationProtocol` declares `doctor() -> AgentDoctorReport`. Implement on each concrete class. Report dataclass:

```python
@dataclass
class AgentDoctorReport:
    agent: str
    is_installed: bool                        # config file or binary exists
    is_patched_durable: bool                  # init was run
    is_patched_transient: bool                # lace currently active
    proxy_reachable: bool
    diagnostic_lines: list[str]               # human-readable hints
```

Each concrete subclass's `doctor()` implementation runs the four checks above and returns the report. The CLI's `_cmd_doctor` then prints it.

### 3.7 Update `cli.py:_cmd_doctor` to cover all five agents

In `cli.py`:

```python
def _cmd_doctor(args: list[str]) -> None:
    """Diagnose why an agent isn't routing through LATTICE."""
    if not args:
        # No agent specified: run doctor for every supported agent
        agents = _list_agents()
    else:
        agents = [args[0]]

    for agent_name in agents:
        if agent_name not in _list_agents():
            console.print(f"[red]Unknown agent: {agent_name}[/red]")
            console.print(f"[dim]Supported: {', '.join(_list_agents())}[/dim]")
            continue
        report = _run_doctor(agent_name)
        _print_doctor_report(report)
```

Where `_run_doctor(name)` calls `agents.{ClaudeCodeIntegration|CodexIntegration|...}().doctor()`.

### 3.8 Verify `registry.list_supported_agents()` is accurate

```bash
# In a Python REPL or test:
from lattice.integrations.registry import list_supported_agents
from lattice.integrations.agents import _AGENT_REGISTRY
assert set(list_supported_agents()) == set(_AGENT_REGISTRY.keys())
```

If the two diverge today, fix `registry.py` to reflect `agents.py`'s registry (single source of truth).

### 3.9 Final verification

```bash
uv run ruff check src/ tests/
uv run mypy src/lattice/
uv run pytest tests/ -q
uv run pytest tests/contract/ -q

# Manual integration sanity (skipped in CI):
# 1. With Claude Code installed:
uv run lattice doctor claude
# Expect: is_installed=True, is_patched_durable depends on prior init

# 2. With no agents installed:
uv run lattice doctor cursor
# Expect: clear "config not found" message, NOT silent success

# 3. Lace and check status from another shell:
uv run lattice lace --dry-run claude
# Expect: a transient_lace record visible to `lattice status` (in real flow, not dry-run)
```

---

## 4. Per-file disposition

| File | LoC | Action |
|---|---|---|
| `integrations/agents.py` | 1444 | MODIFY — add `AgentNotInstalledError`, `AgentIntegrationProtocol`, `AgentDoctorReport`, `doctor()` on each subclass, raise on missing config |
| `integrations/init.py` | ~80 | MODIFY — catch `AgentNotInstalledError`, report cleanly |
| `integrations/lace.py` | ~50 | MODIFY — call `mutation_store.record_transient_lace` / `clear_transient_lace` |
| `integrations/unlace.py` | (small) | UNCHANGED |
| `integrations/registry.py` | (small) | VERIFY — `list_supported_agents()` reflects `_AGENT_REGISTRY` |
| `integrations/mutation_store.py` | ~120 | MODIFY — add transient lace tracking |
| `integrations/tunnel.py` | 784 | MOVED from `core/tunnel_sidecar.py`; no code change |
| `integrations/claude/{install.py,runtime.py}` | small | UNCHANGED |
| `integrations/codex/{install.py,runtime.py,auth.py,ws_handler.py}` | varies | UNCHANGED |
| `integrations/cursor/{install.py,runtime.py}` | small | UNCHANGED |
| `integrations/opencode/{install.py,runtime.py}` | small | UNCHANGED |
| `integrations/copilot/{install.py,runtime.py}` | small | UNCHANGED |
| `cli.py` | 1011 | MODIFY — `_cmd_doctor` covers all agents |
| `core/tunnel_sidecar.py` | — | **MOVED** to `integrations/tunnel.py` |

---

## 5. Tests

### 5.1 New tests

**`tests/unit/integrations/test_agent_protocol.py`**:

```python
def test_every_subclass_satisfies_protocol():
    from lattice.integrations.agents import (
        AgentIntegrationProtocol,
        ClaudeCodeIntegration, CodexIntegration,
        CursorIntegration, OpenCodeIntegration,
        CopilotIntegration, GenericIntegration,
    )
    for cls in [ClaudeCodeIntegration, CodexIntegration, CursorIntegration,
                OpenCodeIntegration, CopilotIntegration, GenericIntegration]:
        instance = cls()
        assert isinstance(instance, AgentIntegrationProtocol), (
            f"{cls.__name__} does not satisfy AgentIntegrationProtocol"
        )
        # Check each required method is concrete
        for method in ("patch", "unpatch", "is_patched", "doctor"):
            assert hasattr(instance, method)
            assert callable(getattr(instance, method))
```

**`tests/unit/integrations/test_mutation_store_records_lace.py`**:

```python
def test_transient_lace_record_and_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from lattice.integrations.mutation_store import MutationStore
    store = MutationStore()
    store.record_transient_lace("claude", pid=12345)
    assert "claude" in {r.agent for r in store.list_transient_laced()}
    store.clear_transient_lace("claude")
    assert "claude" not in {r.agent for r in store.list_transient_laced()}

def test_list_all_active_unions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from lattice.integrations.mutation_store import MutationStore
    store = MutationStore()
    store.record("codex", {"timestamp": 1})        # durable
    store.record_transient_lace("claude", pid=1)   # transient
    assert set(store.list_all_active()) == {"codex", "claude"}
```

**`tests/unit/integrations/test_jsonfile_raises_when_config_missing.py`**:

```python
def test_cursor_raises_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from lattice.integrations.agents import CursorIntegration, AgentNotInstalledError
    integration = CursorIntegration()
    import pytest
    with pytest.raises(AgentNotInstalledError):
        integration.patch()
```

**`tests/unit/integrations/test_doctor_covers_all_agents.py`**:

```python
import subprocess
import pytest

@pytest.mark.parametrize("agent", ["claude", "codex", "cursor", "opencode", "copilot"])
def test_doctor_runs_for_each(agent):
    """Even if the agent isn't installed, `lattice doctor <agent>` must run cleanly
    (exit 0 with a clear report; not crash)."""
    result = subprocess.run(
        ["lattice", "doctor", agent],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"{agent}: stderr={result.stderr}"
    assert agent in result.stdout
    # Should NOT show a generic 'Unknown agent' message
    assert "Unknown agent" not in result.stdout
```

### 5.2 Updated contract tests

`tests/contract/test_cli_contract.py` (Phase 0) — extend with:

```python
def test_doctor_no_args_lists_all_agents():
    """`lattice doctor` with no arg should report on every supported agent."""
    result = subprocess.run(["lattice", "doctor"], capture_output=True, text=True)
    assert result.returncode == 0
    for agent in ("claude", "codex", "cursor", "opencode", "copilot"):
        assert agent in result.stdout
```

---

## 6. Symbol migration table

| Old | New |
|---|---|
| `lattice.core.tunnel_sidecar.TunnelSidecar` | `lattice.integrations.tunnel.TunnelSidecar` |
| `lattice.core.tunnel_sidecar.SidecarThread` | `lattice.integrations.tunnel.SidecarThread` |
| `lattice.core.tunnel_sidecar.TunnelState` | `lattice.integrations.tunnel.TunnelState` |
| `lattice.integrations.agents.AgentNotInstalledError` | **NEW** |
| `lattice.integrations.agents.AgentIntegrationProtocol` | **NEW** |
| `lattice.integrations.agents.AgentDoctorReport` | **NEW** |
| `mutation_store.record_transient_lace` | **NEW** |
| `mutation_store.clear_transient_lace` | **NEW** |
| `mutation_store.list_transient_laced` | **NEW** |
| `mutation_store.list_all_active` | **NEW** |

---

## 7. Acceptance criteria

- [x] `src/lattice/core/tunnel_sidecar.py` does not exist.
- [x] `src/lattice/integrations/tunnel.py` exists; identical content (modulo header).
- [x] `pyproject.toml`'s `per-file-ignores` references the new path.
- [x] `from lattice.integrations.tunnel import TunnelSidecar, TunnelState, SidecarThread` works.
- [x] `from lattice.integrations.agents import AgentNotInstalledError, AgentIntegrationProtocol, AgentDoctorReport` works.
- [x] Every integration class (Claude, Codex, Cursor, OpenCode, Copilot, Generic) implements `patch()`, `unpatch()`, `is_patched()`, `doctor()`, `name`, `proxy_url`.
- [x] `JsonFileIntegration.patch()` raises `AgentNotInstalledError` when `_config_path()` returns a path that doesn't exist.
- [x] `lattice doctor <agent>` exits 0 for every one of the five supported agents.
- [x] `lattice doctor` (no args) reports on all five.
- [x] `mutation_store.list_all_active()` returns the union of durable + transient lacing.
- [x] `lattice lace` (real, not `--dry-run`) records into `transient_laces.json`; on exit (Ctrl-C, SIGTERM, normal exit), the record is cleared.
- [x] All new tests in §5.1 pass.
- [x] `uv run ruff check src/ tests/` clean.
- [x] `uv run mypy src/lattice/` clean (the new Protocol catches any missing method on subclasses).
- [x] `uv run pytest tests/ -q` passes.
- [x] `uv run pytest tests/contract/ -q` passes.

---

## 8. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Raising `AgentNotInstalledError` on missing config breaks an existing user's `lattice init cursor` flow that worked silently | This was the silent bug. The error message is actionable ("Is the agent installed?"); document in CHANGELOG. |
| `transient_laces.json` file at `~/.config/lattice/` becomes stale if the lace process is killed without trap | The atexit / signal handler in `lace.py` clears the record. Additionally, `mutation_store.list_transient_laced()` checks `os.kill(pid, 0)` for liveness and prunes dead entries automatically. |
| Concurrent `lattice lace` runs race on `transient_laces.json` | Use a file lock (`fcntl.flock`) when reading/writing. The write window is tiny. |
| The new `doctor()` method on each integration may take longer than the old monolithic check | Per-agent doctor is fast (4 checks: file exists, init record, transient record, HTTP /healthz). Total wall time <2 s even on slow disks. |
| `core/tunnel_sidecar.py` is imported by `sdk/` code (per audit hint) | Phase 6 already pinned the import path. Verify with `rg "tunnel_sidecar" src/` after move — should match only the new `integrations/tunnel.py` references. |
| `Protocol` with `@runtime_checkable` is slower than ABC for `isinstance()` checks | These checks happen once at startup, not per-request. Imperceptible. |

---

## 9. Rollback plan

```bash
git revert <phase-7-merge-commit>
```

Restores tunnel_sidecar to `core/`, restores silent success on missing configs, removes doctor for cursor/copilot, removes transient lace tracking. No data loss; `transient_laces.json` becomes orphaned (cleanup script in Phase 11 if needed).

---

## 10. PR shape

```
refactor(integrations): move tunnel; raise on missing config; doctor covers all agents [Phase 7]

- Move core/tunnel_sidecar.py → integrations/tunnel.py (784 LoC; no code change)
- New AgentNotInstalledError raised by JsonFileIntegration.patch() when config file is missing
- New @runtime_checkable AgentIntegrationProtocol; mypy enforces every subclass is complete
- Add doctor() method to every integration; AgentDoctorReport dataclass
- mutation_store: transient-lace tracking; list_all_active() unions durable + transient
- CLI: _cmd_doctor handles all 5 agents (was: 3 of 5); no-arg form runs doctor for every supported agent
- registry.py audited to match agents.py _AGENT_REGISTRY exactly
- pyproject.toml ruff per-file-ignores updated

Net: +1 file moved, +1 error type, +1 protocol, +1 dataclass, ~80 LoC across integrations/agents.py.
All 1600+ tests green. Contract tests cover doctor + lace status.
```
