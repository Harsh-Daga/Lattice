"""Adapters must not construct httpx clients or run local retry loops."""

from __future__ import annotations

import re

from tests.contract._scan import iter_py_under, repo_root

FORBIDDEN_IN_ADAPTERS = (
    re.compile(r"httpx\.AsyncClient\s*\("),
    re.compile(r"httpx\.Client\s*\("),
    re.compile(r"async\s+def\s+_retry"),
    re.compile(r"@retry\("),
    re.compile(r"asyncio\.sleep\s*\(\s*backoff"),
)


def test_no_per_adapter_transport_code() -> None:
    root = repo_root(__file__)
    adapters = root / "src" / "lattice" / "providers" / "adapters"
    failures: list[str] = []
    for path in iter_py_under(adapters):
        if path.name in {"__init__.py", "base.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for pat in FORBIDDEN_IN_ADAPTERS:
            if pat.search(text):
                failures.append(f"{rel}: matches {pat.pattern}")
    assert not failures, "Forbidden transport code in adapters:\n" + "\n".join(failures)
