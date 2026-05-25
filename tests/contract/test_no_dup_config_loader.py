"""LatticeConfig loading is centralized in core/config (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import files_with_line_match, repo_root, src_lattice

_ALLOWLIST = {
    "src/lattice/core/config.py",
    "src/lattice/transforms/format_converter/json_converter.py",
    "src/lattice/integrations/agents/env_builder.py",
    "src/lattice/providers/credentials.py",
}


def test_yaml_safe_load_only_in_allowlisted_modules() -> None:
    root = repo_root(__file__)
    hits = files_with_line_match(
        src_lattice(root),
        re.compile(r"yaml\.safe_load"),
        repo=root,
    )
    offenders = hits - _ALLOWLIST
    assert not offenders, offenders
