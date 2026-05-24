"""Phase 5 — content_profiler metadata keys after optimize()."""

from __future__ import annotations

from lattice.core.context import (
    METADATA_KEY_PROTECTED_SPANS,
    METADATA_KEY_RISK_SCORE,
    METADATA_KEY_SCHEDULE,
    METADATA_KEY_SIG,
    METADATA_KEY_TASK_CLASSIFICATION,
    TransformContext,
)
from lattice.core.result import is_ok, unwrap
from lattice.ir.primitives import PromptIRV2
from lattice.planner.runtime_state import get_canonical_state_value
from lattice.transforms.content_profiler import ContentProfiler
from lattice.transport.types import Message, Request

_EXPECTED_REQUEST_METADATA_KEYS = frozenset(
    {
        "_lattice_profile",
        "_lattice_strategy",
        METADATA_KEY_RISK_SCORE,
        METADATA_KEY_SIG,
        METADATA_KEY_PROTECTED_SPANS,
        METADATA_KEY_TASK_CLASSIFICATION,
        METADATA_KEY_SCHEDULE,
        "_lattice_ir_v2_summary",
    }
)


def test_content_profiler_optimize_sets_metadata_and_ir() -> None:
    profiler = ContentProfiler()
    request = Request(
        messages=[
            Message(
                role="user",
                content="Debug the failing unit test in src/lattice/pipeline/runner.py",
            )
        ]
    )
    context = TransformContext(request_id="profiler-integration", provider="openai")
    ir = PromptIRV2()

    result = profiler.optimize(ir, request, context)
    assert is_ok(result)
    out_ir = unwrap(result)
    assert out_ir is not None

    for key in _EXPECTED_REQUEST_METADATA_KEYS:
        assert key in request.metadata, f"missing request.metadata[{key!r}]"

    stored = get_canonical_state_value(context, "_lattice_ir_v2")
    assert stored is not None
