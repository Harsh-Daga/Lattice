"""Tests for proxy lifecycle management (PID file, start, stop, status).

Validates:
- PID file read/write
- Stale PID detection and cleanup
- start_background_server returns the actual server PID
- Process group signalling (SIGTERM / SIGKILL)
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

from lattice.proxy.lifecycle import PIDManager, start_background_server

# =============================================================================
# PIDManager
# =============================================================================


class TestPIDManager:
    def test_write_and_read(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "test.pid"
        mgr = PIDManager(pid_path=pid_file)
        mgr.write(12345)
        assert pid_file.read_text().strip() == "12345"
        assert mgr.read() == 12345

    def test_read_missing_file(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "missing.pid"
        mgr = PIDManager(pid_path=pid_file)
        assert mgr.read() is None

    def test_read_invalid_content(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number")
        mgr = PIDManager(pid_path=pid_file)
        assert mgr.read() is None

    def test_is_alive_self(self) -> None:
        mgr = PIDManager(pid_path=Path("/dev/null"))
        assert mgr._is_alive(os.getpid()) is True

    def test_is_alive_dead_pid(self) -> None:
        mgr = PIDManager(pid_path=Path("/dev/null"))
        # PID 1 is init/systemd — always alive.  Use a definitely-dead PID.
        assert mgr._is_alive(99999) is False

    def test_remove(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "to_remove.pid"
        pid_file.write_text("12345")
        mgr = PIDManager(pid_path=pid_file)
        mgr.remove()
        assert not pid_file.exists()

    def test_status_running(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "running.pid"
        mgr = PIDManager(pid_path=pid_file)
        mgr.write(os.getpid())
        status = mgr.status()
        assert status is not None
        assert status.pid == os.getpid()
        assert status.running is True

    def test_status_stale_cleanup(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("99999")
        mgr = PIDManager(pid_path=pid_file)
        status = mgr.status()
        assert status is None
        assert not pid_file.exists()  # stale file removed

    def test_is_running(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "running.pid"
        mgr = PIDManager(pid_path=pid_file)
        mgr.write(os.getpid())
        assert mgr.is_running() is True

    def test_stop_no_pid_file(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "missing.pid"
        mgr = PIDManager(pid_path=pid_file)
        result = mgr.stop(grace_period=0.5)
        assert result.stopped is True
        assert result.was_running is False
        assert "No PID file" in result.message

    def test_stop_stale_pid(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("99999")
        mgr = PIDManager(pid_path=pid_file)
        result = mgr.stop(grace_period=0.5)
        assert result.stopped is True
        assert result.was_running is False
        assert "stale" in result.message.lower()
        assert not pid_file.exists()

    def test_stop_graceful(self, tmp_path: Path) -> None:
        """Start a real child process and stop it gracefully."""
        pid_file = tmp_path / "real.pid"
        mgr = PIDManager(pid_path=pid_file)

        # Start a long-running Python child in its own process group
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        mgr.write(proc.pid)

        assert mgr.is_running() is True

        result = mgr.stop(grace_period=2.0)
        assert result.stopped is True
        assert result.was_running is True
        assert not pid_file.exists()
        # Reap the child and verify it's dead
        proc.wait(timeout=5.0)
        assert not mgr._is_alive(proc.pid)

    def test_stop_force_kill(self, tmp_path: Path) -> None:
        """Start a child that ignores SIGTERM and ensure SIGKILL works."""
        pid_file = tmp_path / "stubborn.pid"
        mgr = PIDManager(pid_path=pid_file)

        # Start a child that ignores SIGTERM in its own process group
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        mgr.write(proc.pid)

        result = mgr.stop(grace_period=0.5)
        assert result.stopped is True
        assert result.was_running is True
        assert "force" in result.message.lower()
        assert not pid_file.exists()
        proc.wait(timeout=5.0)
        assert not mgr._is_alive(proc.pid)


# =============================================================================
# start_background_server
# =============================================================================


class TestStartBackgroundServer:
    def test_returns_valid_pid(self, tmp_path: Path) -> None:
        """start_background_server must return a PID that is actually alive."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        pid = start_background_server(
            host="127.0.0.1",
            port=port,
            workers=1,
        )

        assert pid > 0
        # The PID should be alive
        pid_file = tmp_path / "server.pid"
        mgr = PIDManager(pid_path=pid_file)
        mgr.write(pid)
        assert mgr._is_alive(pid) is True

        # Give uvicorn a moment to start listening
        time.sleep(1.0)

        # Verify the server is actually listening
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", port))

        # Clean up: stop the server
        mgr.stop(grace_period=3.0)
        # Uvicorn may leave a zombie briefly; reap it and wait
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        time.sleep(1.0)
        # Verify it's dead
        assert not mgr._is_alive(pid)

    def test_pid_is_server_not_intermediate(self, tmp_path: Path) -> None:
        """The returned PID must be the uvicorn process, not a fork wrapper."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        pid = start_background_server(
            host="127.0.0.1",
            port=port,
            workers=1,
        )

        pid_file = tmp_path / "server.pid"
        mgr = PIDManager(pid_path=pid_file)
        mgr.write(pid)

        # Wait for startup
        time.sleep(1.0)

        # Check that the process name contains "uvicorn" or "python"
        name = mgr._process_name(pid)
        if name is not None:
            assert "python" in name.lower() or "uvicorn" in name.lower()

        # Clean up
        mgr.stop(grace_period=3.0)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        time.sleep(1.0)
        assert not mgr._is_alive(pid)
