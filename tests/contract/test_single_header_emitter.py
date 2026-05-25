"""Response x-lattice-* headers are emitted only from proxy middleware (Phase 13)."""

from __future__ import annotations

import re

from tests.contract._scan import files_with_line_match, repo_root, src_lattice

_ALLOWLIST = {
    "src/lattice/gateway/compat/headers.py",
    "src/lattice/telemetry/downgrade.py",
    "src/lattice/sdk/proxy_client.py",
    "src/lattice/integrations/agents/profiles.py",
    "src/lattice/gateway/compat/openai_chat.py",
    "src/lattice/transforms/content_profiler/planner_bridge.py",
    "src/lattice/protocol/prefix_canonicalization.py",
}


def test_response_headers_x_lattice_only_in_middleware() -> None:
    root = repo_root(__file__)
    hits = files_with_line_match(
        src_lattice(root),
        re.compile(r"response\.headers\[[^\]]*x-lattice"),
        repo=root,
    )
    assert hits <= {"src/lattice/proxy/middleware.py"}, hits


def test_direct_x_lattice_assignment_allowlisted() -> None:
    root = repo_root(__file__)
    hits = files_with_line_match(
        src_lattice(root),
        re.compile(r'headers\["x-lattice'),
        repo=root,
    )
    assert hits <= _ALLOWLIST, hits - _ALLOWLIST
