"""Every documented HTTP endpoint responds on a live proxy (Phase 11 matrix)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

_API_SURFACE = Path(__file__).resolve().parents[2] / "docs" / "refactor" / "api-surface.json"

# Bodies for POST probes (upstream may 401/502 without keys — route must exist).
_POST_BODIES: dict[str, dict | None] = {
    "/v1/chat/completions": {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    },
    "/v1/messages": {
        "model": "anthropic/claude-3-5-haiku-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "/v1/responses": {
        "model": "openai/gpt-4o-mini",
        "input": "hi",
    },
    "/lattice/session/start": {"provider": "openai", "model": "gpt-4o-mini"},
    "/lattice/session/append": {
        "session_id": "00000000-0000-0000-0000-000000000001",
        "messages": [{"role": "user", "content": "hi"}],
    },
    "/lattice/session/invalidate": {
        "session_id": "00000000-0000-0000-0000-000000000001",
    },
    "/lattice/gateway": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    },
    "/v1/codex/responses": {
        "model": "openai/gpt-4o-mini",
        "input": "hi",
    },
    "/backend-api/responses": {
        "model": "openai/gpt-4o-mini",
        "input": "hi",
    },
}


def _resolve_path(template: str) -> str:
    return template.replace("{id}", "resp_test").replace(
        "{session_id}", "00000000-0000-0000-0000-000000000001"
    )


def _load_http_cases() -> list[tuple[str, str, dict | None, tuple[int, ...]]]:
    data = json.loads(_API_SURFACE.read_text())
    cases: list[tuple[str, str, dict | None, tuple[int, ...]]] = []
    for ep in data["http"]["endpoints"]:
        method = ep["method"].upper()
        if method == "WS":
            continue
        path = _resolve_path(ep["path"])
        if method == "GET":
            if path.startswith("/v1/"):
                expected: tuple[int, ...] = (200, 401, 403, 404, 502)
            elif path.startswith("/lattice/session/"):
                expected = (200, 404, 422, 502)
            else:
                expected = (200, 503) if path == "/readyz" else (200,)
        else:
            body = _POST_BODIES.get(ep["path"])
            expected = (200, 400, 401, 403, 404, 422, 502)
        cases.append((method, path, body if method != "GET" else None, expected))
    return cases


_HTTP_CASES = _load_http_cases()


def test_api_surface_endpoint_count_locked() -> None:
    """Regression guard: matrix covers every non-WS endpoint in api-surface.json."""
    data = json.loads(_API_SURFACE.read_text())
    non_ws = [e for e in data["http"]["endpoints"] if e["method"].upper() != "WS"]
    assert len(_HTTP_CASES) == len(non_ws)


@pytest.mark.contract
@pytest.mark.parametrize("method,path,body,expected", _HTTP_CASES)
def test_documented_endpoint_responds(
    proxy_subprocess: int,
    method: str,
    path: str,
    body: dict | None,
    expected: tuple[int, ...],
) -> None:
    url = f"http://127.0.0.1:{proxy_subprocess}{path}"
    with httpx.Client(timeout=15) as client:
        if method == "GET":
            response = client.get(url)
        elif method == "DELETE":
            response = client.delete(url)
        else:
            response = client.post(url, json=body)
    assert response.status_code in expected, (
        f"{method} {path}: {response.status_code}\n{response.text[:300]}"
    )
