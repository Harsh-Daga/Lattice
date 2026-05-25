"""Shared test infrastructure for unit, integration, and contract suites."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def free_port() -> int:
    """Bind to port 0 and return the OS-assigned ephemeral port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture
def temp_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate HOME / XDG_CONFIG_HOME so tests do not touch the developer machine."""
    cfg = tmp_path / ".config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    return cfg


@pytest.fixture(scope="session")
def proxy_subprocess() -> Generator[int, None, None]:
    """Start one real proxy for the contract suite; tear down after the session."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lattice.cli",
            "proxy",
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-ui",
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = (proc.stderr.read() if proc.stderr else b"").decode()
            raise RuntimeError(f"proxy exited early:\n{stderr}")
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5)
            if response.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.15)
    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        raise TimeoutError(f"proxy did not become healthy on port {port}")
    yield port
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def httpx_mock_provider(monkeypatch: pytest.MonkeyPatch) -> Generator[object, None, None]:
    """Mock an OpenAI-shaped upstream so HTTP contracts do not need API keys."""
    import respx

    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-contract")
        yield mock
