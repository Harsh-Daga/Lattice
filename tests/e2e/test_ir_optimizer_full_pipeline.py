"""End-to-end test for IR optimizer through full pipeline."""

from __future__ import annotations

import asyncio

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.pipeline.factory import build_default_pipeline
from lattice.transport.types import Message, Request


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestIRFullPipeline:
    def test_pipeline_with_json_data(self) -> None:
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[_req('[{"status":"ok","name":"alpha"},{"status":"ok","name":"beta"}]')],
            model="gpt-4",
        )
        ctx = TransformContext()

        async def run() -> None:
            result = await pipeline.process(request, ctx)
            assert is_ok(result)
            modified = unwrap(result)

            assert "_lattice_ir_v2" in modified.metadata, "IRV2 not compiled"

            sched = ctx.session_state.get("_lattice_optimizer_schedule")
            assert sched is not None, "Schedule not created"
            allowed = getattr(sched, "allowed_optimizers", [])
            assert "ir_structure_optimizer" in allowed, (
                f"ir_structure_optimizer not in allowed: {allowed}"
            )

            print(f"Allowed optimizers: {allowed}")
            print(f"Transforms applied: {ctx.transforms_applied}")

        asyncio.run(run())

    def test_pipeline_with_plain_text(self) -> None:
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        cfg.compression_mode = "safe"
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[_req("What is 2+2?")],
            model="gpt-4",
        )
        ctx = TransformContext()

        async def run() -> None:
            result = await pipeline.process(request, ctx)
            assert is_ok(result)
            modified = unwrap(result)
            assert modified is not None

            # Content should still be present
            final_text = "\n".join(m.content for m in modified.messages)
            assert "2+2" in final_text

        asyncio.run(run())
