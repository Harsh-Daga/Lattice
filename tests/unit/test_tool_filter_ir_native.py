"""Unit tests for IR-native tool_filter."""

from __future__ import annotations

import json

from lattice.core.context import TransformContext
from lattice.core.transport import Message, Request
from lattice.ir.builder import build_ir
from lattice.ir.primitives import PromptIRV2, prompt_ir_v2_from_legacy
from lattice.transforms.tool_filter import ToolOutputFilter


def test_tool_filter_optimize_scrubs_tool_output_ir() -> None:
    request = Request(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(
                role="tool",
                content=json.dumps(
                    {
                        "id": "123",
                        "name": "Alice",
                        "created_at": "2024-01-01",
                        "internal_secret": "abc",
                        "metadata": {"extra": "data"},
                    }
                ),
            ),
        ]
    )
    ir = prompt_ir_v2_from_legacy(build_ir(request))
    context = TransformContext(
        request_id="tool-filter-ir",
        provider="openai",
        model="gpt-4",
        session_state={},
    )

    result = ToolOutputFilter().optimize(ir, request, context)
    modified = result.unwrap()
    assert isinstance(modified, PromptIRV2)
    serialized = modified.serialize()

    assert "metadata" not in serialized
    assert "internal_secret" in serialized
    assert dict(modified.metadata)["_lattice_tool_filter_applied"] is True
    assert request.metadata["_lattice_tool_filter_applied"] is True
