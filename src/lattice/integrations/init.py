"""Durable LATTICE agent initialization.

One-step setup that persists across reboots. Modifies agent config files,
registers provider blocks, and tracks mutations so ``lattice unlace`` can
fully restore the original state.

Usage::

    lattice init                    # Auto-detect and configure all agents
    lattice init claude             # Configure Claude Code only
    lattice init codex opencode     # Configure Codex + OpenCode
    lattice init --global           # User-scope (default)
    lattice init --port 9999        # Custom proxy port
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.integrations.mutation_store import get_mutation, remove_mutation, store_mutation
from lattice.integrations.registry import (
    apply_provider_scope,
    build_install_target_envs,
    list_supported_agents,
    revert_provider_scope,
)

logger = structlog.get_logger()

_SUPPORTED_TARGETS = ("claude", "codex", "opencode", "cursor", "copilot")
_LOCAL_TARGETS = {"claude", "codex", "opencode"}
_GLOBAL_TARGETS = {"claude", "codex", "opencode", "cursor", "copilot"}


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def _probe_init_targets(global_scope: bool) -> list[tuple[str, str | None]]:
    """Return ``[(target, which_result)]`` for every in-scope supported target."""
    allowed = _GLOBAL_TARGETS if global_scope else _LOCAL_TARGETS
    probes: list[tuple[str, str | None]] = []
    for target in _SUPPORTED_TARGETS:
        if target not in allowed:
            continue
        path = shutil.which(target)
        probes.append((target, path))
    return probes


def detect_init_targets(global_scope: bool = True) -> list[str]:
    """Return agent names in scope for which a binary was found on PATH."""
    return [name for name, path in _probe_init_targets(global_scope) if path]


def _format_empty_detection_error(global_scope: bool) -> str:
    """Build the error message shown when no in-scope targets were detected."""
    probes = _probe_init_targets(global_scope)
    scope_label = "user" if global_scope else "local"
    lines: list[str] = [
        f"No supported {scope_label}-scope agents were found on PATH.",
        "",
        "LATTICE probed the following agents via shutil.which():",
    ]
    for name, path in probes:
        status = f"found at {path}" if path else "not found"
        lines.append(f"  - {name}: {status}")
    lines.extend(
        [
            "",
            "Install the agent you want first, then re-run with an explicit target:",
            "",
        ]
    )
    for name, _path in probes:
        lines.append(f"  lattice init {name}")
    lines.extend(["", "Tip: run `lattice init --help` to see all options."])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-agent init
# ---------------------------------------------------------------------------


def _init_claude(*, port: int) -> dict[str, Any]:
    mutation = apply_provider_scope("claude", port=port)
    return {"success": True, "message": f"Configured Claude Code (ANTHROPIC_BASE_URL=http://127.0.0.1:{port})", "mutation": mutation}


def _init_codex(*, port: int) -> dict[str, Any]:
    mutation = apply_provider_scope("codex", port=port)
    return {"success": True, "message": f"Configured Codex (OPENAI_BASE_URL=http://127.0.0.1:{port}/v1)", "mutation": mutation}


def _init_opencode(*, port: int) -> dict[str, Any]:
    mutation = apply_provider_scope("opencode", port=port)
    return {"success": True, "message": f"Configured OpenCode (OPENAI_BASE_URL=http://127.0.0.1:{port}/v1)", "mutation": mutation}


def _init_cursor(*, port: int) -> dict[str, Any]:
    from lattice.integrations.cursor.runtime import render_setup_lines

    lines = render_setup_lines(port)
    return {"success": True, "message": "\n".join(lines), "mutation": None}


def _init_copilot(*, port: int) -> dict[str, Any]:
    mutation = apply_provider_scope("copilot", port=port)
    return {"success": True, "message": f"Configured Copilot (COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:{port}/v1)", "mutation": mutation}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_init(
    targets: list[str],
    *,
    port: int = 8787,
    global_scope: bool = True,
) -> dict[str, Any]:
    """Run durable init for the given targets.

    Returns a dict with 'success', 'results' (per-agent), and 'mutations'.
    Mutations are persisted to ``~/.lattice/mutations.json`` so that
    ``lattice unlace`` can fully restore the original state.
    """
    results: dict[str, dict[str, Any]] = {}
    mutations: list[dict[str, Any]] = []

    envs = build_install_target_envs(port=port, targets=targets)
    for target in targets:
        logger.debug("init_target", target=target, port=port)
        if target == "claude":
            result = _init_claude(port=port)
        elif target == "codex":
            result = _init_codex(port=port)
        elif target == "opencode":
            result = _init_opencode(port=port)
        elif target == "cursor":
            result = _init_cursor(port=port)
        elif target == "copilot":
            result = _init_copilot(port=port)
        else:
            result = {"success": False, "message": f"Unknown target: {target}", "mutation": None}
        results[target] = result
        mutation = result.get("mutation")
        if mutation:
            store_mutation(target, mutation)
            mutations.append(mutation)

    return {"success": True, "results": results, "mutations": mutations}


def run_uninit(targets: list[str]) -> dict[str, Any]:
    """Revert durable init for the given targets.

    Reads the stored mutation from ``~/.lattice/mutations.json`` and
    calls the agent-specific reverter. Removes the entry on success.
    """
    results: dict[str, dict[str, Any]] = {}
    for target in targets:
        logger.debug("uninit_target", target=target)
        mutation = get_mutation(target)
        if mutation is None:
            results[target] = {
                "success": True,
                "message": f"No stored mutation for {target} — nothing to revert",
            }
            continue
        revert_provider_scope(target, mutation)
        remove_mutation(target)
        results[target] = {"success": True, "message": f"Restored {target} configuration"}
    return {"success": True, "results": results}
