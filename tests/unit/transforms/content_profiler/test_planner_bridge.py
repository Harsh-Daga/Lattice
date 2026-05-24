"""Phase 5 — planner bridge populates execution plan and IR on context."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.ir.primitives import PromptIRV2
from lattice.planner.runtime_state import get_canonical_state_value
from lattice.transforms.content_profiler import ContentProfiler
from lattice.transport.types import Message, Request


def test_content_profiler_optimize_sets_execution_plan_on_context() -> None:
    request = Request(
        messages=[
            Message(
                role="user",
                content="Analyze JSON: " + '{"status":"ok","items":[1,2,3]}' * 2,
            )
        ],
        model="gpt-4",
    )
    context = TransformContext(provider="openai", model="gpt-4")
    result = ContentProfiler().optimize(PromptIRV2(), request, context)
    assert is_ok(result)
    unwrap(result)

    plan = get_canonical_state_value(context, "_lattice_execution_plan")
    assert plan is not None
    ir = get_canonical_state_value(context, "_lattice_ir_v2")
    assert ir is not None
