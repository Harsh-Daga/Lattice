"""Stale misleading names must not appear in src/ (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import iter_py_under, repo_root, src_lattice

_STALE = re.compile(
    r"MILV|BatchAccumulator|strategy_selector|constraint_lifting|"
    r"information_theoretic_selector"
)


def test_no_stale_names_in_src() -> None:
    root = repo_root(__file__)
    base = src_lattice(root)
    for path in iter_py_under(base, skip_rel=("core/config.py",)):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert _STALE.search(text) is None, f"stale name in {path.relative_to(root)}"
