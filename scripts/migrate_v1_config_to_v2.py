#!/usr/bin/env python3
"""Migrate lattice.toml / env from v1.x toward v2 forward-plan config shape.

Doc reference: docs/refactor/MIGRATION-v1-to-v2.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Renames applied when present in source TOML (v2 phases 16–18, 30)
_KEY_RENAMES: dict[str, str] = {
    "transform_prefix_opt": None,  # removed — no-op in v1.1+
    "transform_constraint_lifting": None,
    "transform_strategy_selector": None,
    "semantic_cache_enabled": "cache.enabled",
    "semantic_cache_ttl_seconds": "cache.ttl_seconds",
}


def migrate_text(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in _KEY_RENAMES:
            target = _KEY_RENAMES[key]
            if target is None:
                notes.append(f"dropped dead flag: {key}")
                continue
            notes.append(f"renamed {key} -> {target}")
            val = stripped.split("=", 1)[1] if "=" in stripped else ""
            out_lines.append(f"{target} = {val}".rstrip())
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate v1 lattice config toward v2 keys")
    parser.add_argument("path", type=Path, help="lattice.toml or .env-style file")
    parser.add_argument("-o", "--output", type=Path, help="Write here; default stdout")
    args = parser.parse_args()
    text = args.path.read_text(encoding="utf-8")
    migrated, notes = migrate_text(text)
    if args.output:
        args.output.write_text(migrated, encoding="utf-8")
    else:
        sys.stdout.write(migrated)
    for n in notes:
        print(f"migrate: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
