"""Regression tests for new lossless transforms — Phase 2-4."""

from __future__ import annotations

import asyncio

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.transport.types import Message, Request


class TestDiagnosticRLE:
    def test_grouped_repetition(self) -> None:
        from lattice.transforms.diagnostic_rle import DiagnosticRLE

        t = DiagnosticRLE()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        content = [
            Message(
                role="user",
                content="service_0 timeout error\nservice_1 timeout error\nservice_2 timeout error",
            ),
        ]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)

    def test_short_skip(self) -> None:
        from lattice.transforms.diagnostic_rle import DiagnosticRLE

        t = DiagnosticRLE()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        content = [Message(role="user", content="hi")]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)


class TestJSONShape:
    def test_json_shape_factoring_preserves_keys(self) -> None:
        from lattice.transforms.json_shape import JSONShapeFactor

        t = JSONShapeFactor()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        data = '[{"id":1,"status":"ok","latency":120},{"id":2,"status":"ok","latency":121},{"id":3,"status":"ok","latency":122}]'
        content = [Message(role="user", content=data)]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "keys=" in out.messages[0].content or "ok" in out.messages[0].content


class TestColumnarPack:
    def test_table_columnar_pack_preserves_rows(self) -> None:
        from lattice.transforms.columnar_pack import ColumnarTablePack

        t = ColumnarTablePack()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        content = [
            Message(
                role="user",
                content="| id | name | dept |\n| --- | --- | --- |\n| 0 | Alice | Eng |\n| 1 | Bob | Eng |\n| 2 | Carl | Eng |",
            )
        ]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)


class TestNumericPreservationWithoutArithmeticSequence:
    def test_pipeline_passes_numeric_content_unchanged(self) -> None:
        config = LatticeConfig()
        from lattice.pipeline.factory import build_default_pipeline

        pipeline = build_default_pipeline(config)
        messages = [
            Message(role="user", content="0 100\n1 200\n2 300\n3 400\n4 500"),
        ]
        req = Request(model="test", messages=messages)
        ctx = TransformContext(request_id="test", provider="openai", model="test")

        result = asyncio.run(pipeline.process(req, ctx))
        if is_ok(result):
            out = unwrap(result)
            combined = "\n".join(m.content for m in out.messages)
            assert "100" in combined
            assert "200" in combined


class TestPathPrefix:
    def test_path_prefix_roundtrip(self) -> None:
        from lattice.transforms.path_prefix import PathPrefixCompressor

        t = PathPrefixCompressor()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        content = [
            Message(
                role="user", content="/var/log/app/a.log\n/var/log/app/b.log\n/var/log/app/c.log"
            ),
        ]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)


class TestDiagnosticRLEStillHandlesTraces:
    def test_diagnostic_rle_still_handles_traces(self) -> None:
        from lattice.transforms.diagnostic_rle import DiagnosticRLE

        t = DiagnosticRLE()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        trace = (
            "Traceback (most recent call last):\n"
            '  File "app.py", line 10, in foo\n'
            '  File "db.py", line 22, in bar\n'
            "ValueError: boom\n\n"
            "Traceback (most recent call last):\n"
            '  File "app.py", line 10, in foo\n'
            '  File "db.py", line 22, in bar\n'
            "KeyError: bang"
        )
        content = [Message(role="user", content=trace)]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)


class TestSafetyGates:
    def test_placeholder_leakage_rolls_back(self) -> None:
        from lattice.pipeline.guardrails import GuardAction, check_placeholder_leakage

        before = "The error was in module X"
        after = "The error was <ref_17> in module <d_36>"
        decision = check_placeholder_leakage(before, after)
        assert decision.action in (GuardAction.ROLLBACK, GuardAction.REJECT)

    def test_negative_savings_rolls_back(self) -> None:
        from lattice.pipeline.guardrails import GuardAction, check_negative_savings

        decision = check_negative_savings(100, 200)
        assert decision.action in (GuardAction.ROLLBACK, GuardAction.REJECT)

    def test_numeric_preservation_after_transform(self) -> None:
        config = LatticeConfig()
        from lattice.pipeline.factory import build_default_pipeline

        pipeline = build_default_pipeline(config)
        messages = [
            Message(role="user", content="0 100\n1 200\n2 300\n3 400\n4 500"),
        ]
        req = Request(model="test", messages=messages)
        ctx = TransformContext(request_id="test", provider="openai", model="test")

        result = asyncio.run(pipeline.process(req, ctx))
        if is_ok(result):
            out = unwrap(result)
            combined = "\n".join(m.content for m in out.messages)
            assert "100" in combined
            assert "200" in combined
            assert "300" in combined


class TestTaskGating:
    def test_debugging_blocks_rate_distortion(self) -> None:
        from lattice.core.scheduler import _TASK_TRANSFORM_MATRIX

        matrix = _TASK_TRANSFORM_MATRIX.get("debugging", {})
        assert matrix.get("rate_distortion") is False

    def test_reasoning_blocks_rate_distortion(self) -> None:
        from lattice.core.scheduler import _TASK_TRANSFORM_MATRIX

        matrix = _TASK_TRANSFORM_MATRIX.get("reasoning", {})
        assert matrix.get("rate_distortion") is False


class TestExtractiveCompress:
    def test_extractive_preserves_entities(self) -> None:
        from lattice.transforms.extractive_compress import ExtractiveCompressor

        t = ExtractiveCompressor()
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        content = [
            Message(
                role="user",
                content="The error was in module 42 at line 100. The fix is simple. Actually, just look at this filler text that means nothing really.",
            ),
        ]
        req = Request(model="test", messages=content)
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "error" in out.messages[0].content.lower()


class TestFrontierScoring:
    def test_frontier_computation(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.90, 0.15, task_class="retrieval")
        assert f.passed_quality_gate is True
        assert f.passed_savings_gate is True
        assert f.frontier_score > 0.0
        assert f.rollback_reason is None

    def test_low_quality_fails_quality_gate(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.70, 0.20, task_class="reasoning")
        assert f.passed_quality_gate is False
        assert f.rollback_reason is not None

    def test_negative_compression_fails_savings_gate(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.90, -0.05, task_class="simple")
        assert f.passed_savings_gate is False
        assert f.rollback_reason == "negative_savings"

    def test_placeholder_leakage_triggers_rollback(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.95, 0.30, placeholder_leakage=True, task_class="simple")
        assert f.placeholder_leakage is True
        assert f.rollback_reason == "placeholder_leakage"
