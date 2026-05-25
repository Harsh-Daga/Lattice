"""Token counting goes through utils/token_count (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import count_lines_matching, files_with_line_match, repo_root, src_lattice


def test_tiktoken_import_only_in_token_count_module() -> None:
    root = repo_root(__file__)
    hits = files_with_line_match(
        src_lattice(root),
        re.compile(r"^(import tiktoken|from tiktoken)"),
        repo=root,
    )
    assert hits == {"src/lattice/utils/token_count.py"}, hits


def test_count_tokens_defined_once() -> None:
    root = repo_root(__file__)
    n = count_lines_matching(src_lattice(root), re.compile(r"^def count_tokens\b"))
    assert n == 1
