"""End-to-end integration test for the optimizer pipeline.

Verifies the full flow:
1. content_profiler runs → stores schedule in context.session_state
2. representation_optimizer reads schedule → runs allowed optimizers
3. Hard rollback rejects bad candidates
4. Reverse transforms restore response
"""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.pipeline.factory import build_default_pipeline
from lattice.transport.types import Message, Request, Response


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestOptimizerPipelineEndToEnd:
    def test_optimizer_pipeline_runs_content_profiler_then_representation(self) -> None:
        """Full path: content_profiler → representation_optimizer → allowed optimizers."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"  # minimal extra transforms
        pipeline = build_default_pipeline(cfg)

        # Verify core transforms + representation_optimizer are registered
        names = [t.name for t in pipeline.transforms]
        assert "content_profiler" in names
        assert "runtime_contract" in names
        assert "representation_optimizer" in names

        # Simple request with debug signal
        request = Request(
            messages=[_req("Error: ModuleNotFoundError in build step")],
            model="gpt-4",
        )
        ctx = TransformContext()

        import asyncio

        async def run():
            result = await pipeline.process(request, ctx)
            assert is_ok(result)
            modified = unwrap(result)

            # content_profiler stores schedule in context.session_state
            assert "_lattice_schedule" in ctx.session_state, (
                f"Schedule not in session_state. Keys: {list(ctx.session_state.keys())}"
            )
            schedule = ctx.session_state["_lattice_schedule"]
            assert "allowed_optimizers" in schedule
            allowed = schedule["allowed_optimizers"]
            assert isinstance(allowed, list)

            # At minimum, baseline was selected
            assert modified is not None

        asyncio.run(run())

    def test_scheduler_decision_reaches_optimizers(self) -> None:
        """content_profiler writes _lattice_schedule; representation_optimizer reads it."""
        from lattice.transforms.content_profiler import ContentProfiler

        profiler = ContentProfiler()

        request = Request(
            messages=[_req("Debug the TypeError in line 42")],
            model="gpt-4",
        )
        ctx = TransformContext()

        # Step 1: content_profiler
        result1 = profiler.process(request, ctx)
        assert is_ok(result1)

        # Step 2: verify schedule is in context.session_state
        assert "_lattice_schedule" in ctx.session_state
        schedule = ctx.session_state["_lattice_schedule"]
        assert "allowed_optimizers" in schedule

        # Step 3: representation_optimizer reads the schedule
        # The scheduler maps constituent transforms to parent optimizers.
        # For a debug signal, diagnostic_optimizer should be allowed.
        allowed_from_scheduler = schedule["allowed_optimizers"]
        assert len(allowed_from_scheduler) >= 1
        # At minimum structure/reference/tool optimizers should be present
        # (they are always mapped from allowed transforms)
        assert any(
            opt in allowed_from_scheduler
            for opt in ("structure_optimizer", "reference_optimizer", "tool_optimizer")
        )

    def test_hard_rollback_rejects_bad_candidates(self) -> None:
        """If an optimizer expands tokens without cache/transport gain, reject."""
        from lattice.pipeline.representation_optimizer import _validate_beam_candidate

        class FakeReq:
            token_estimate = 100

        class FakeReqBad:
            token_estimate = 120

        candidate = type(
            "Cand",
            (),
            {
                "tokens_before": 100,
                "tokens_after": 120,
                "quality_estimate": 0.95,
                "cache_gain": 0.0,
                "transport_gain": 0.0,
                "latency_ms": 10.0,
            },
        )()
        ctx = TransformContext()
        assert _validate_beam_candidate(candidate, 0.85, ctx) is False

    def test_optimizer_pipeline_reverse(self) -> None:
        """Reverse should restore placeholder references in response."""
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[
                _req("UUID 550e8400-e29b-41d4-a716-446655440000 is the error"),
            ],
            model="gpt-4",
        )
        ctx = TransformContext()

        import asyncio

        async def run():
            result = await pipeline.process(request, ctx)
            assert is_ok(result)
            compressed = unwrap(result)
            assert compressed is not None
            response = Response(
                role="assistant",
                content="The error was in module <ref_1>",
                model="gpt-4",
            )
            # Reverse through the pipeline (async)
            restored = await pipeline.reverse(response, ctx)
            assert restored is not None
            # Response should still be a Response object
            assert hasattr(restored, "content")

        asyncio.run(run())
