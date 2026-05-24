"""Phase 5 — classify_by_signals heuristics."""

from __future__ import annotations

from lattice.transforms.content_profiler import ContentProfile, classify_by_signals
from lattice.transport.types import Message, Request


def test_classify_code_heavy_from_fenced_blocks() -> None:
    body = (
        "Fix this implementation and explain the failure.\n"
        + "```python\n"
        + "\n".join(f"def fn_{i}():\n    return {i}\n" for i in range(8))
        + "```\n"
        + "Also check inline `foo` and `bar` usage.\n"
    )
    request = Request(messages=[Message(role="user", content=body)])
    assert classify_by_signals(request) == ContentProfile.CODE_HEAVY


def test_classify_short_below_threshold() -> None:
    request = Request(messages=[Message(role="user", content="hi")])
    assert classify_by_signals(request) == ContentProfile.SHORT


def test_classify_table_heavy_from_markdown_table() -> None:
    row = "| col_a | col_b | col_c |\n|-------|-------|-------|\n| 1 | 2 | 3 |\n"
    table = row * 12
    request = Request(messages=[Message(role="user", content="Summarize this dataset:\n" + table)])
    profile = classify_by_signals(request)
    assert profile in (ContentProfile.TABLE_HEAVY, ContentProfile.MIXED)
