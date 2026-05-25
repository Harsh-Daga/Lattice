"""Streaming chunk buffers are not duplicated before Phase 27 canonical home."""

from __future__ import annotations

import re

from tests.contract._scan import files_with_line_match, repo_root, src_lattice

_ALLOWLIST = {"src/lattice/integrations/tunnel.py"}


def test_no_duplicate_streaming_buffer_classes() -> None:
    root = repo_root(__file__)
    hits = files_with_line_match(
        src_lattice(root),
        re.compile(r"^class \w*Buffer\b"),
        repo=root,
    )
    offenders = hits - _ALLOWLIST
    assert not offenders, offenders
