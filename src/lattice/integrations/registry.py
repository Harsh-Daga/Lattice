"""Provider registry for LATTICE agent integrations.

Declarative pattern inspired by Headroom: register env builders and
provider-scope handlers per agent, then the init/lace system
automatically supports them.

To add a new agent:
    1. Create lattice/integrations/<agent>/install.py with build_install_env,
       apply_provider_scope, revert_provider_scope.
    2. Create lattice/integrations/<agent>/runtime.py with build_launch_env.
    3. Register both in _ENV_BUILDERS and _PROVIDER_SCOPE_HANDLERS below.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_InstallEnvBuilder = Callable[..., dict[str, str]]
_ProviderScopeApplier = Callable[..., Any]
_ProviderScopeReverter = Callable[..., None]

# ---------------------------------------------------------------------------
# Env builders — called by both durable init and transient lace
# ---------------------------------------------------------------------------

_ENV_BUILDERS: dict[str, _InstallEnvBuilder] = {}

# We lazy-import to avoid circular deps and keep startup fast.


def _ensure_builders() -> dict[str, _InstallEnvBuilder]:
    """Lazy-load env builders."""
    global _ENV_BUILDERS
    if _ENV_BUILDERS:
        return _ENV_BUILDERS

    try:
        from lattice.integrations.claude.runtime import build_launch_env as _build_claude

        _ENV_BUILDERS["claude"] = _build_claude
    except ImportError:
        pass

    try:
        from lattice.integrations.codex.runtime import build_launch_env as _build_codex

        _ENV_BUILDERS["codex"] = _build_codex
    except ImportError:
        pass

    try:
        from lattice.integrations.opencode.runtime import build_launch_env as _build_opencode

        _ENV_BUILDERS["opencode"] = _build_opencode
    except ImportError:
        pass

    try:
        from lattice.integrations.cursor.runtime import build_launch_env as _build_cursor

        _ENV_BUILDERS["cursor"] = _build_cursor
    except ImportError:
        pass

    try:
        from lattice.integrations.copilot.runtime import build_launch_env as _build_copilot

        _ENV_BUILDERS["copilot"] = _build_copilot
    except ImportError:
        pass

    return _ENV_BUILDERS


# ---------------------------------------------------------------------------
# Provider-scope handlers — for durable config-file mutation
# ---------------------------------------------------------------------------

_PROVIDER_SCOPE_HANDLERS: dict[str, tuple[_ProviderScopeApplier, _ProviderScopeReverter]] = {}


def _ensure_handlers() -> dict[str, tuple[_ProviderScopeApplier, _ProviderScopeReverter]]:
    """Lazy-load provider-scope handlers."""
    global _PROVIDER_SCOPE_HANDLERS
    if _PROVIDER_SCOPE_HANDLERS:
        return _PROVIDER_SCOPE_HANDLERS

    try:
        from lattice.integrations.claude.install import (
            apply_provider_scope as _apply_claude,
            revert_provider_scope as _revert_claude,
        )

        _PROVIDER_SCOPE_HANDLERS["claude"] = (_apply_claude, _revert_claude)
    except ImportError:
        pass

    try:
        from lattice.integrations.codex.install import (
            apply_provider_scope as _apply_codex,
            revert_provider_scope as _revert_codex,
        )

        _PROVIDER_SCOPE_HANDLERS["codex"] = (_apply_codex, _revert_codex)
    except ImportError:
        pass

    try:
        from lattice.integrations.opencode.install import (
            apply_provider_scope as _apply_opencode,
            revert_provider_scope as _revert_opencode,
        )

        _PROVIDER_SCOPE_HANDLERS["opencode"] = (_apply_opencode, _revert_opencode)
    except ImportError:
        pass

    try:
        from lattice.integrations.cursor.install import (
            apply_provider_scope as _apply_cursor,
            revert_provider_scope as _revert_cursor,
        )

        _PROVIDER_SCOPE_HANDLERS["cursor"] = (_apply_cursor, _revert_cursor)
    except ImportError:
        pass

    try:
        from lattice.integrations.copilot.install import (
            apply_provider_scope as _apply_copilot,
            revert_provider_scope as _revert_copilot,
        )

        _PROVIDER_SCOPE_HANDLERS["copilot"] = (_apply_copilot, _revert_copilot)
    except ImportError:
        pass

    return _PROVIDER_SCOPE_HANDLERS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_supported_agents() -> list[str]:
    """Return all agent names registered in the env-builder map."""
    return sorted(_ensure_builders().keys())


def list_durable_agents() -> list[str]:
    """Return agents that support durable config-file mutation."""
    return sorted(_ensure_handlers().keys())


def build_install_target_envs(
    port: int,
    targets: list[str],
    backend: str = "openai",
) -> dict[str, dict[str, str]]:
    """Build per-target install environment values via provider slices."""
    builders = _ensure_builders()
    result: dict[str, dict[str, str]] = {}
    for target in targets:
        builder = builders.get(target)
        if builder is None:
            continue
        result[target] = builder(port=port, backend=backend)
    return result


def build_launch_env(agent: str, port: int, backend: str = "openai") -> tuple[dict[str, str], list[str]]:
    """Build the environment dict and display lines for a single agent."""
    builders = _ensure_builders()
    builder = builders.get(agent)
    if builder is None:
        return {}, []
    env = builder(port=port, backend=backend)
    display = [f"{k}={v}" for k, v in env.items()]
    return env, display


def apply_provider_scope(agent: str, port: int = 8787) -> Any:
    """Apply durable config mutation for an agent. Returns mutation state or None."""
    handlers = _ensure_handlers()
    pair = handlers.get(agent)
    if pair is None:
        return None
    return pair[0](port=port)


def revert_provider_scope(agent: str, mutation: Any) -> None:
    """Revert durable config mutation for an agent."""
    handlers = _ensure_handlers()
    pair = handlers.get(agent)
    if pair is None:
        return
    pair[1](mutation)
