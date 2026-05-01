"""Process lifecycle management for the LATTICE proxy.

Provides production-grade PID file handling for background daemon mode:
- idempotent start
- stale PID cleanup (``kill -0`` detection)
- graceful shutdown with fallback to SIGKILL
- atomic PID file writing (write + fsync + rename)
- cross-platform (macOS, Linux; Windows graceful degradation)

Usage
-----
    pid_mgr = PIDManager("~/.config/lattice/proxy.pid")
    if not pid_mgr.is_running():
        pid = start_background_server(...)
        pid_mgr.write(pid)
    else:
        print(f"Already running (PID {pid_mgr.read()})")
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PIDStatus:
    """Result of a PID status query."""

    pid: int
    running: bool
    process_name: str | None = None
    uptime_seconds: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class StopResult:
    """Result of a stop operation."""

    stopped: bool
    was_running: bool
    pid: int | None = None
    message: str = ""


# ------------------------------------------------------------------------------
# PIDManager
# ------------------------------------------------------------------------------


class PIDManager:
    """Manages a PID file for the LATTICE proxy background process.

    Implements the `pidfile + kill -0` pattern used by nginx, redis,
    gunicorn, etc.  The file is written atomically and validated
    before any destructive operation.
    """

    def __init__(self, *, pid_path: str | Path | None = None) -> None:
        """
        Args:
            pid_path: Path to PID file.  Defaults to
                ``~/.config/lattice/proxy.pid``.
        """
        if pid_path is None:
            self.pid_path = Path.home() / ".config" / "lattice" / "proxy.pid"
        else:
            self.pid_path = Path(pid_path)
        self._log = logger.bind(module="pid_manager", pid_file=str(self.pid_path))

    def write(self, pid: int) -> None:
        """Write *pid* to the PID file atomically.

        Steps:
        1. Ensure parent directory exists.
        2. Write to a temp file next to the PID file.
        3. ``fsync`` the descriptor.
        4. Atomic ``replace`` (rename).
        """
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pid_path.with_suffix(f".pid.tmp.{pid}")
        try:
            with tmp.open("w") as f:
                f.write(f"{pid}\n")
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.pid_path)
            self._log.info("pid_file_written", pid=pid, path=str(self.pid_path))
        except Exception:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    def read(self) -> int | None:
        """Read the PID from file.  Returns ``None`` if file missing or invalid."""
        try:
            raw = self.pid_path.read_text().strip()
            return int(raw)
        except (FileNotFoundError, ValueError, PermissionError):
            return None

    def _is_alive(self, pid: int) -> bool:
        """Check whether process *pid* is alive using ``kill -0``."""
        if platform.system() == "Windows":
            # Windows fallback — try to open the process
            try:
                import ctypes

                kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
                handle = kernel.OpenProcess(1, False, pid)  # PROCESS_TERMINATE
                if handle:
                    kernel.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _process_name(self, pid: int) -> str | None:
        """Best-effort process name resolution (``ps`` or ``psutil``)."""
        try:
            import psutil

            name: str = psutil.Process(pid).name()
            return name
        except Exception:
            return None

    def _uptime(self, pid: int) -> float | None:
        """Best-effort process uptime via ``psutil``."""
        try:
            import psutil

            p = psutil.Process(pid)
            uptime: float = time.time() - p.create_time()
            return uptime
        except Exception:
            return None

    def status(self) -> PIDStatus | None:
        """Return ``PIDStatus`` or ``None`` if no PID file / stale."""
        pid = self.read()
        if pid is None:
            return None
        alive = self._is_alive(pid)
        if not alive:
            # Stale PID file — clean it up
            self._remove_stale()
            return None
        return PIDStatus(
            pid=pid,
            running=True,
            process_name=self._process_name(pid),
            uptime_seconds=self._uptime(pid),
        )

    def is_running(self) -> bool:
        """Convenience: is the proxy currently running?"""
        return self.status() is not None

    def remove(self) -> None:
        """Remove the PID file (called by the server at shutdown)."""
        with contextlib.suppress(FileNotFoundError):
            self.pid_path.unlink()
            self._log.info("pid_file_removed")

    def _remove_stale(self) -> None:
        """Remove a stale PID file after confirming the process is dead."""
        with contextlib.suppress(FileNotFoundError):
            self.pid_path.unlink()
            self._log.info("stale_pid_file_removed")

    def stop(
        self,
        grace_period: float = 10.0,
        signal_grace: int = signal.SIGTERM,
        signal_kill: int = signal.SIGKILL,
    ) -> StopResult:
        """Stop the managed process and its process group.

        Algorithm:
        1. Read PID.
        2. Kill the process group (negative PID) with SIGTERM.
        3. Poll ``kill -0`` every 0.5s for *grace_period*.
        4. If still alive: SIGKILL the process group.
        5. Remove PID file.

        Args:
            grace_period: Seconds to wait for graceful shutdown.
            signal_grace: Signal to send first (default SIGTERM).
            signal_kill: Signal to send if grace expires (default SIGKILL).

        Returns:
            StopResult describing what happened.
        """
        pid = self.read()
        if pid is None:
            return StopResult(stopped=True, was_running=False, message="No PID file found")

        if not self._is_alive(pid):
            self._remove_stale()
            return StopResult(stopped=True, was_running=False, pid=pid, message="Process not running (stale PID removed)")

        self._log.info("stopping_process", pid=pid, grace=grace_period)
        # Kill the process group (negative PID) to catch all children
        pgid = -pid if platform.system() != "Windows" else pid
        try:
            os.kill(pgid, signal_grace)
        except OSError as exc:
            # Fallback: try individual PID if process group kill fails
            try:
                os.kill(pid, signal_grace)
            except OSError:
                return StopResult(stopped=False, was_running=True, pid=pid, message=f"Failed to signal: {exc}")

        # Wait for graceful shutdown
        deadline = time.time() + grace_period
        while time.time() < deadline:
            if not self._is_alive(pid):
                self.remove()
                return StopResult(stopped=True, was_running=True, pid=pid, message=f"Stopped (graceful after {round(grace_period - (deadline - time.time()), 2)}s)")
            time.sleep(0.5)

        # Force kill the process group
        if self._is_alive(pid):
            try:
                os.kill(pgid, signal_kill)
                self._log.warning("force_killed", pid=pid, pgid=pgid)
            except OSError as exc:
                # Fallback: try individual PID
                try:
                    os.kill(pid, signal_kill)
                except OSError:
                    return StopResult(stopped=False, was_running=True, pid=pid, message=f"SIGKILL failed: {exc}")
            # Brief wait for OS to reap
            time.sleep(0.5)

        self.remove()
        return StopResult(stopped=True, was_running=True, pid=pid, message="Stopped (force killed)")


# ------------------------------------------------------------------------------
# Background server starter
# ------------------------------------------------------------------------------


def start_background_server(
    host: str = "0.0.0.0",
    port: int = 8787,
    workers: int | None = None,
    config: Any | None = None,
) -> int:
    """Start the proxy server in the background and return its PID.

    Creates a new process group so that ``Ctrl+C`` in the parent shell
    does not cascade to the background server.  Uses ``uvicorn`` directly
    for maximum compatibility.

    Returns the PID of the **actual server process** (not an intermediate
    fork child), so PID file tracking works correctly.
    """
    if config is not None:
        # LatticeConfig is injected via the factory, but for a background
        # child we need to start via the module path.  Environment variables
        # are the bridge.
        # NOTE: if the user has a config file, it will be auto-discovered.
        pass

    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "lattice.proxy.server:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if workers:
        cmd += ["--workers", str(workers)]

    # Start in a new process group (detached from terminal).
    # preexec_fn=os.setsid creates a new session → the child becomes
    # the session leader and gets its own process group.
    # close_fds=True prevents file descriptor leakage.
    # stdout/stderr go to /dev/null for daemon behaviour.
    kwargs: dict[str, Any] = {"close_fds": True}
    if platform.system() != "Windows":
        kwargs["preexec_fn"] = os.setsid
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    proc = subprocess.Popen(cmd, **kwargs)
    return proc.pid


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

# Back-compat alias (some early code references StartResult)
@dataclasses.dataclass(frozen=True, slots=True)
class _StartResult:
    """Result of a start operation."""
    pid: int
    success: bool
    message: str


# Re-export under the old name
StartResult = _StartResult


__all__ = [
    "PIDManager",
    "PIDStatus",
    "StartResult",
    "StopResult",
    "start_background_server",
]
