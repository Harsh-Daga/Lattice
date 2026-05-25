"""Pure-Python source scans for contract tests (no ripgrep dependency on CI)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


def repo_root(from_file: str) -> Path:
    return Path(from_file).resolve().parents[2]


def src_lattice(root: Path) -> Path:
    return root / "src" / "lattice"


def iter_py_under(base: Path, *, skip_rel: Iterable[str] = ()) -> Iterable[Path]:
    skip = tuple(skip_rel)
    for path in base.rglob("*.py"):
        rel = path.relative_to(base).as_posix()
        if any(rel.startswith(s) or s in rel for s in skip):
            continue
        yield path


def count_lines_matching(
    base: Path, pattern: re.Pattern[str], *, skip_rel: Iterable[str] = ()
) -> int:
    n = 0
    for path in iter_py_under(base, skip_rel=skip_rel):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if pattern.search(line):
                n += 1
    return n


def rel_under_repo(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def files_with_line_match(
    base: Path,
    pattern: re.Pattern[str],
    *,
    skip_rel: Iterable[str] = (),
    repo: Path | None = None,
) -> set[str]:
    root = repo or base.parent.parent
    hits: set[str] = set()
    for path in iter_py_under(base, skip_rel=skip_rel):
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(line) for line in text.splitlines()):
            hits.add(rel_under_repo(path, root))
    return hits
