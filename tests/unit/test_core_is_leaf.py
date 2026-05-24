"""Verify core/ remains a leaf package with no uphill lattice imports."""

from __future__ import annotations

import ast
import pathlib


def test_core_has_no_uphill_imports() -> None:
    """core/ may import lattice.core.*, transport.types, pipeline.runner only."""
    allowed_prefixes = (
        "lattice.core.",
        "lattice.transport.types",
        "lattice.pipeline.runner",
    )
    core_dir = pathlib.Path("src/lattice/core")
    bad: list[str] = []
    for py in core_dir.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("lattice.")
            ):
                if not any(
                    node.module.startswith(p.rstrip(".")) or node.module == p.rstrip(".")
                    for p in allowed_prefixes
                ):
                    bad.append(f"{py.name}: imports {node.module}")
    assert not bad, "core/ imports from non-leaf paths:\n" + "\n".join(bad)
