"""Response x-lattice-* headers are emitted only from proxy middleware (Phase 13)."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Modules may build header dicts for attach_routing_headers / outbound client requests.
_ALLOWLIST = frozenset(
    {
        "src/lattice/proxy/middleware.py",
        "src/lattice/gateway/compat/headers.py",
        "src/lattice/telemetry/downgrade.py",
        "src/lattice/sdk/proxy_client.py",
        "src/lattice/integrations/agents/profiles.py",
        "src/lattice/gateway/compat/openai_chat.py",
        "src/lattice/transforms/content_profiler/planner_bridge.py",
        "src/lattice/protocol/prefix_canonicalization.py",
    }
)


def test_response_headers_x_lattice_only_in_middleware() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-n", "response\\.headers\\[[^\\]]*x-lattice", "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 1:
        return
    offenders = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not offenders, offenders


def test_direct_x_lattice_assignment_allowlisted() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["rg", "-l", 'headers\\["x-lattice', "src/lattice"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 1:
        return
    paths = {p.strip() for p in proc.stdout.splitlines() if p.strip()}
    assert paths <= _ALLOWLIST, f"x-lattice header builders outside allowlist: {paths - _ALLOWLIST}"
