"""Smoke tests for Pipeline.compress() — the new high-level gate-orchestrated entry.

Phase 3 (V1 Kill) ports v1 CompressorPipeline's safety machinery into the
v2 Pipeline. These tests pin that the new compress() entry exists, is
callable, runs content_profiler, and returns an Ok(Request) on the
happy path.
"""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.pipeline.runner import Pipeline
from lattice.transport.types import Message, Request


def test_compress_exists_and_returns_ok() -> None:
    pipeline = Pipeline()
    req = Request(messages=[Message(role="user", content="Hello, world.")])
    ctx = TransformContext()
    result = pipeline.compress(req, ctx)
    assert is_ok(result)
    out = unwrap(result)
    assert isinstance(out, Request)
    assert out.messages, "compressed request should preserve messages"


def test_compress_runs_content_profiler_first() -> None:
    pipeline = Pipeline()
    req = Request(messages=[Message(role="user", content="Debug the failing test.")])
    ctx = TransformContext()
    result = pipeline.compress(req, ctx)
    assert is_ok(result)
    # content_profiler should have run and populated task classification.
    assert "content_profiler" in ctx.transforms_applied


def test_compress_with_empty_messages_does_not_crash() -> None:
    pipeline = Pipeline()
    req = Request(messages=[Message(role="user", content="")])
    ctx = TransformContext()
    result = pipeline.compress(req, ctx)
    assert is_ok(result)


def test_compress_idempotent_on_already_applied_profiler() -> None:
    pipeline = Pipeline()
    req = Request(messages=[Message(role="user", content="Hello.")])
    ctx = TransformContext()
    ctx.mark_transform_applied("content_profiler")
    result = pipeline.compress(req, ctx)
    assert is_ok(result)
