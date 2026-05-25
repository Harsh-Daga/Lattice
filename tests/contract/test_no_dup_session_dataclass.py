"""Session domain type has one home (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import count_lines_matching, repo_root, src_lattice


def test_one_session_class_in_state() -> None:
    root = repo_root(__file__)
    n = count_lines_matching(src_lattice(root), re.compile(r"^class Session\b"))
    assert n == 1


def test_session_import_path() -> None:
    from lattice.state.session import Session

    assert Session.__module__ == "lattice.state.session"
