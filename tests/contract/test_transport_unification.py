"""HTTP client construction and retry logic live only under lattice.transport."""

from __future__ import annotations

import re

from tests.contract._scan import iter_py_under, repo_root

ALLOWED_HTTPX_CLIENT_CONSTRUCTION = {
    "src/lattice/transport/pool.py",
    # Client→proxy or special-case paths (not provider adapter transport)
    "src/lattice/proxy/server.py",
    "src/lattice/sdk/proxy_client.py",
    "src/lattice/integrations/tunnel.py",
    "src/lattice/integrations/codex/ws_handler.py",
}

FORBIDDEN_RETRY_OUTSIDE_TRANSPORT = (
    re.compile(r"for\s+attempt\s+in\s+range"),
    re.compile(r"asyncio\.sleep\s*\([^)]*backoff"),
)


def test_only_pool_constructs_httpx_clients() -> None:
    root = repo_root(__file__)
    lattice = root / "src" / "lattice"
    hits: list[str] = []
    for path in iter_py_under(lattice, skip_rel=("__pycache__",)):
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWED_HTTPX_CLIENT_CONSTRUCTION:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"httpx\.AsyncClient\s*\(", text) or re.search(r"httpx\.Client\s*\(", text):
            hits.append(rel)
    assert not hits, "httpx client construction outside pool:\n" + "\n".join(sorted(hits))


def test_no_adapter_local_retry_loops() -> None:
    root = repo_root(__file__)
    adapters = root / "src" / "lattice" / "providers" / "adapters"
    failures: list[str] = []
    for path in iter_py_under(adapters):
        if path.name in {"__init__.py", "base.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for pat in FORBIDDEN_RETRY_OUTSIDE_TRANSPORT:
            if pat.search(text):
                failures.append(f"{rel}: {pat.pattern}")
    assert not failures, "Adapter-local retry loops:\n" + "\n".join(failures)


def test_single_retry_module() -> None:
    root = repo_root(__file__)
    lattice = root / "src" / "lattice"
    retry_files = [
        p.relative_to(root).as_posix()
        for p in iter_py_under(lattice)
        if p.name == "retry.py" and "transport" in p.as_posix()
    ]
    assert retry_files == ["src/lattice/transport/retry.py"]
