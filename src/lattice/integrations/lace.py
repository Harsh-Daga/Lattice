"""LATTICE lace command — route an agent through the LATTICE proxy.

Like ``headroom wrap``, this command:
1. Starts or reuses a running LATTICE proxy
2. Configures the agent to route through the proxy (env vars only — transient)
3. Launches the agent as a subprocess
4. Cleans up on exit (stops proxy if we started it)

Usage::

    lattice lace claude          # Route Claude Code through LATTICE
    lattice lace codex           # Route Codex through LATTICE
    lattice lace cursor          # Route Cursor through LATTICE
    lattice lace --port 9999 claude  # Use a specific proxy port
    lattice lace --no-start claude    # Assume proxy already running
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.core.tunnel_sidecar import SidecarThread, TunnelSidecar
from lattice.integrations.mutation_store import get_mutation
from lattice.integrations.registry import build_launch_env
from lattice.proxy.lifecycle import PIDManager

logger = structlog.get_logger()

_PROXY_WAIT_TIMEOUT = 45
_PROXY_CHECK_INTERVAL = 0.5


def _check_proxy(port: int) -> bool:
    """Check if LATTICE proxy is running on given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", port))
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def _ensure_proxy(port: int, no_start: bool = False) -> dict[str, Any]:
    """Ensure a LATTICE proxy is running on the given port.

    Returns dict with 'started' (bool), 'pid' (int|None), 'url' (str).
    """
    pid_mgr = PIDManager()
    status = pid_mgr.status()

    if status is not None and status.running:
        return {
            "started": False,
            "pid": status.pid,
            "url": f"http://127.0.0.1:{port}",
            "message": f"Proxy already running (PID {status.pid})",
        }

    if _check_proxy(port):
        return {
            "started": False,
            "pid": None,
            "url": f"http://127.0.0.1:{port}",
            "message": f"Proxy already running on port {port}",
        }

    if no_start:
        return {
            "started": False,
            "pid": None,
            "url": f"http://127.0.0.1:{port}",
            "message": "Proxy not running and --no-start specified. Assuming proxy at "
            f"http://127.0.0.1:{port}",
        }

    from lattice.proxy.lifecycle import start_background_server

    pid = start_background_server(port=port)
    pid_mgr.write(pid)

    for _ in range(int(_PROXY_WAIT_TIMEOUT / _PROXY_CHECK_INTERVAL)):
        try:
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            return {
                "started": True,
                "pid": pid,
                "url": f"http://127.0.0.1:{port}",
                "message": f"Proxy started (PID {pid})",
            }
        except Exception:
            time.sleep(_PROXY_CHECK_INTERVAL)

    return {
        "started": True,
        "pid": pid,
        "url": f"http://127.0.0.1:{port}",
        "message": f"Proxy started (PID {pid}) but health check timed out. "
        "The proxy may still be initializing.",
    }


def _find_agent_binary(agent: str) -> str | None:
    """Find the binary path for an agent."""
    binary_names = {
        "claude": "claude",
        "claude-code": "claude",
        "codex": "codex",
        "cursor": None,
        "opencode": "opencode",
        "vscode": "code",
        "copilot": "copilot",
        "generic": None,
    }
    name = binary_names.get(agent)
    if name is None:
        return None
    return shutil.which(name)


def lace_agent(
    agent: str,
    args: list[str] | None = None,
    port: int = 8787,
    no_start: bool = False,
    no_patch: bool = False,
    dry_run: bool = False,
    no_tunnel: bool = False,
) -> int:
    """Lace an agent: start proxy, configure routing, launch agent, cleanup.

    Args:
        agent: Agent name (claude, codex, cursor, opencode, copilot, etc.)
        args: Extra arguments to pass to the agent binary.
        port: Proxy port (default 8787).
        no_start: Assume proxy is already running.
        no_patch: No-op for transient lace (lace never mutates config files).
        dry_run: Show what would happen without executing.
        no_tunnel: Skip the persistent sidecar tunnel (agent talks directly to proxy).

    Returns:
        Exit code from the agent subprocess, or 1 on error.
    """
    del no_patch  # Transient lace only sets env vars; config mutation is init.
    proxy_url = f"http://127.0.0.1:{port}"

    if dry_run:
        print(f"[lattice] DRY RUN: Would lace {agent}")
        print(f"[lattice] DRY RUN: Proxy URL: {proxy_url}")
        sidecar_port = 8788 if not no_tunnel else port
        env, display = build_launch_env(agent, sidecar_port)
        print(f"[lattice] DRY RUN: Would set env vars: {env}")
        if not no_tunnel:
            print(f"[lattice] DRY RUN: Sidecar tunnel on port {sidecar_port}")
        binary = _find_agent_binary(agent)
        if binary:
            print(f"[lattice] DRY RUN: Would launch: {binary} {' '.join(args or [])}")
        return 0

    we_started_proxy = False
    sidecar_runner: SidecarThread | None = None

    try:
        # 1. Ensure proxy is running
        proxy_info = _ensure_proxy(port, no_start=no_start)
        we_started_proxy = proxy_info.get("started", False)
        print(f"[lattice] {proxy_info['message']}")

        # 2. Start sidecar tunnel (unless disabled)
        agent_connect_port = port
        if not no_tunnel:
            config = LatticeConfig.auto()
            sidecar = TunnelSidecar(
                config=config,
                tcp_port=8788,
                unix_socket_path=None,
            )
            sidecar_runner = SidecarThread(sidecar)
            sidecar_runner.start()
            # Allow sidecar to bind before agent connects
            time.sleep(0.2)
            agent_connect_port = 8788
            print(f"[lattice] Sidecar tunnel listening on {sidecar.connect_url}")

        # 3. Build environment via registry
        env = os.environ.copy()
        agent_env, env_vars_display = build_launch_env(agent, agent_connect_port)
        env.update(agent_env)
        for line in env_vars_display:
            print(f"[lattice] {line}")

        # 4. Find agent binary
        binary = _find_agent_binary(agent)
        if binary is None:
            print(f"[lattice] ERROR: Could not find '{agent}' binary in PATH.")
            print(f"[lattice] Please install {agent} first.")
            return 1

        # 5. Launch agent
        cmd_args = [binary] + (args or [])
        print(f"[lattice] Launching: {' '.join(cmd_args)}")
        print("[lattice] Press Ctrl+C to stop and cleanup")

        proc = subprocess.Popen(cmd_args, env=env)
        exit_code = proc.wait()
        return exit_code

    except KeyboardInterrupt:
        print("\n[lattice] Interrupted, cleaning up...")
        return 130

    finally:
        # Stop sidecar if running
        if sidecar_runner is not None:
            try:
                sidecar_runner.stop(timeout=3.0)
                print("[lattice] Sidecar stopped.")
            except Exception as e:
                logger.warning("sidecar_stop_failed", error=str(e))

        # Stop proxy if we started it
        if we_started_proxy:
            try:
                pid_mgr = PIDManager()
                pid_mgr.stop()
                print("[lattice] Proxy stopped.")
            except Exception as e:
                logger.warning("proxy_stop_failed", error=str(e))


def unlace_agent(agent: str) -> dict[str, Any]:
    """Unlace an agent: restore original config.

    Reverts durable init mutations via the registry using the stored
    mutation state from ``~/.lattice/mutations.json``.
    """
    from lattice.integrations.mutation_store import remove_mutation
    from lattice.integrations.registry import revert_provider_scope

    mutation = get_mutation(agent)
    if mutation is None:
        return {
            "success": True,
            "message": f"No stored mutation for {agent} — nothing to revert",
        }

    revert_provider_scope(agent, mutation)
    remove_mutation(agent)
    return {"success": True, "message": f"{agent} unlaced successfully"}
