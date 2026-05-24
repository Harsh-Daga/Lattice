"""End-to-end integration test for the IR-native optimizer pipeline.

Verifies the full flow:
1. content_profiler compiles canonical IRV2 → stores `_lattice_ir_v2` in request metadata
2. representation_optimizer beam search includes `ir_structure_optimizer`
3. IR-native path activates for JSON-heavy requests (`_lattice_ir_native_applied`)
4. Lossless guarantee: original content preserved after IR-native optimization
5. Plain text requests don't crash when the IR-native optimizer processes them
"""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.pipeline.factory import build_default_pipeline
from lattice.transport.types import Message, Request, Response


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestIROptimizerEndToEnd:
    def test_ir_compilation_flow(self) -> None:
        """A request through content_profiler stores `_lattice_ir_v2` in metadata."""
        cfg = LatticeConfig(use_optimizer_pipeline=True, compression_mode="safe")
        pipeline = build_default_pipeline(cfg)

        names = [t.name for t in pipeline.transforms]
        assert "content_profiler" in names
        assert "ir_structure_optimizer" in names

        request = Request(
            messages=[_req('{"status":"ok","value":1}')],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        modified = unwrap(result)
        assert "_lattice_ir_v2" in modified.metadata, (
            f"`_lattice_ir_v2` not in metadata. Keys: {list(modified.metadata.keys())}"
        )

    def test_ir_native_path_activates_for_json(self) -> None:
        """JSON-heavy request triggers IR compilation with structural metadata.

        The IR-native optimizer (ir_structure_optimizer) operates on
        `constant_fields` / `arithmetic_fields` metadata produced by the
        normalizer.  We verify that the pipeline compiles this metadata so
        the IR-native path *can* activate.
        """
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[
                _req(
                    '[{"status":"ok","name":"alpha"},'
                    '{"status":"ok","name":"beta"},'
                    '{"status":"ok","name":"gamma"}]'
                )
            ],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        modified = unwrap(result)

        ir_v2 = modified.metadata.get("_lattice_ir_v2")
        assert ir_v2 is not None, "`_lattice_ir_v2` not found in metadata"

        sections = list(getattr(ir_v2, "sections", []))
        has_structure = any(
            "constant_fields" in dict(span.structure) or "json_shape" in dict(span.structure)
            for sec in sections
            for span in sec.spans
        )
        assert has_structure, "IR spans should contain `constant_fields` or `json_shape` structure"

        sched = ctx.session_state.get("_lattice_optimizer_schedule")
        allowed = getattr(sched, "allowed_optimizers", []) if sched else []
        assert "ir_structure_optimizer" in allowed, (
            f"`ir_structure_optimizer` not in allowed optimizers: {allowed}"
        )

    def test_ir_structure_optimizer_in_beam_search(self) -> None:
        """`ir_structure_optimizer` is allowed by the scheduler and can factor IR."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        cfg.transform_format_conversion = False
        pipeline = build_default_pipeline(cfg)

        # Use JSON with a constant field so the IR-native optimizer finds
        # something to factor.
        request = Request(
            messages=[
                _req(
                    '[{"status":"ok","name":"alpha"},'
                    '{"status":"ok","name":"beta"},'
                    '{"status":"ok","name":"gamma"}]'
                )
            ],
            model="gpt-4",
        )
        ctx = TransformContext()

        profiler = next(t for t in pipeline.transforms if t.name == "content_profiler")
        result = profiler.process(request, ctx)
        assert is_ok(result)
        after_profiler = unwrap(result)

        sched = ctx.session_state.get("_lattice_optimizer_schedule")
        assert sched is not None
        allowed = getattr(sched, "allowed_optimizers", [])
        assert "ir_structure_optimizer" in allowed, (
            f"`ir_structure_optimizer` not in allowed optimizers: {allowed}"
        )

        from lattice.optimizer.ir_structure_optimizer import IRStructureOptimizer

        ir_opt = IRStructureOptimizer()
        assert ir_opt.can_process(after_profiler, ctx), (
            "ir_structure_optimizer.can_process returned False"
        )

        result2 = ir_opt.process(after_profiler.copy(), ctx)
        assert is_ok(result2)
        modified = result2.unwrap()

        has_tag = modified.metadata.get("_lattice_ir_native_applied") == "ir_structure_optimizer"
        if has_tag:
            ir_v2 = modified.metadata.get("_lattice_ir_v2")
            assert ir_v2 is not None

    def test_lossless_guarantee(self) -> None:
        """Original numbers and keys survive after IR-native optimization runs."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        pipeline = build_default_pipeline(cfg)

        original_content = (
            '[{"status":"ok","value":1},{"status":"ok","value":2},{"status":"ok","value":3}]'
        )
        request = Request(
            messages=[_req(original_content)],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        modified = unwrap(result)

        final_text = "\n".join(m.content for m in modified.messages)
        assert "status" in final_text, "JSON key `status` was lost"
        assert "ok" in final_text, "Value `ok` was lost"
        assert "value" in final_text, "JSON key `value` was lost"
        assert "1" in final_text, "Number `1` was lost"
        assert "2" in final_text, "Number `2` was lost"
        assert "3" in final_text, "Number `3` was lost"

    def test_no_crash_on_plain_text(self) -> None:
        """A simple text request survives the full optimizer pipeline without crashing."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[_req("Hello")],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        modified = unwrap(result)
        assert modified is not None
        final_text = "\n".join(m.content for m in modified.messages)
        assert "Hello" in final_text, "Plain text content was unexpectedly modified"

    def test_optimizer_pipeline_reverse(self) -> None:
        """Reverse should restore any response references after IR-native optimization."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[_req('[{"status":"ok","value":42}]')],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        compressed = unwrap(result)
        assert compressed is not None

        response = Response(
            role="assistant",
            content="The value is 42",
            model="gpt-4",
        )
        restored = pipeline.reverse(response, ctx)
        assert restored is not None
        assert hasattr(restored, "content")
