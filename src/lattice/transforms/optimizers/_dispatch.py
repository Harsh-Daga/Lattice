"""Constituent dispatch helper for representation optimizers.

When Phase 3 deletes ``process()`` on IR-native transforms, the
representation optimizers can no longer call ``constituent.process(req, ctx)``
directly. This helper dispatches to ``constituent.optimize(ir, req, ctx)``
for IR-native transforms (and serializes the resulting IR back onto the
request) and falls back to ``process()`` for legacy / non-IR-native
constituents.
"""

from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Err, Ok, Result, is_err, is_ok, unwrap, unwrap_err
from lattice.ir.primitives import PromptIRV2
from lattice.planner.runtime_state import get_canonical_state_value
from lattice.transport.types import Request

# Constituents that own a native ``optimize(ir, req, ctx)`` entry point.
# Keep this list aligned with ``Pipeline._IR_NATIVE_TRANSFORMS`` in
# ``lattice.pipeline.runner``.
_IR_NATIVE_CONSTITUENTS = frozenset(
    {
        "cache_arbitrage",
        "causal_chain",
        "content_profiler",
        "format_conversion",
        "message_dedup",
        "path_prefix",
        "rate_distortion",
        "reference_sub",
        "runtime_contract",
        "strategy_selector",
        "tool_filter",
        "tool_projection",
    }
)


def _serialize_ir_to_messages(ir: PromptIRV2, working: Request) -> None:
    """Best-effort mapping of IR sections back to working.messages."""
    if not ir.sections or not working.messages:
        return
    section_texts = ["\n".join(sp.text for sp in sec.spans if sp.text) for sec in ir.sections]
    if len(section_texts) == 1 and working.messages:
        working.messages[0].content = section_texts[0]
        return
    if len(section_texts) >= len(working.messages):
        for i, msg in enumerate(working.messages):
            if i < len(section_texts) - 1:
                msg.content = section_texts[i]
            else:
                msg.content = "\n\n".join(section_texts[i:])
    else:
        for i, text in enumerate(section_texts):
            working.messages[i].content = text


def run_constituent(
    name: str,
    instance: Any,
    request: Request,
    context: TransformContext,
) -> Result[Request, TransformError]:
    """Run a constituent transform, preferring IR-native ``optimize()``.

    For IR-native constituents the IR is read from
    ``context.session_state["_lattice_ir_v2"]`` (created if absent), passed
    to ``optimize``, and the resulting IR is serialized back onto a copy of
    the request before returning. For non-IR-native constituents the
    legacy ``process()`` path is used.
    """
    # Resolve aliases (e.g. ``"format_conv"`` → ``"format_conversion"``,
    # ``"prefix_opt"`` → ``"prefix_optimizer"``) so the lookup matches the
    # canonical IR-native set regardless of which spelling the caller used.
    from lattice.transforms.registry import get_transform_spec

    spec = get_transform_spec(name)
    canonical = spec.canonical_name if spec is not None else name
    if canonical in _IR_NATIVE_CONSTITUENTS and hasattr(instance, "optimize"):
        ir = get_canonical_state_value(context, "_lattice_ir_v2") or PromptIRV2()
        ir_result = instance.optimize(ir, request, context)
        if is_err(ir_result):
            return Err(unwrap_err(ir_result))
        new_ir = unwrap(ir_result)
        if isinstance(new_ir, PromptIRV2):
            context.session_state["_lattice_ir_v2"] = new_ir
            working: Request = request.copy()
            if new_ir.sections:
                _serialize_ir_to_messages(new_ir, working)
            working.metadata["_lattice_ir_v2"] = new_ir
            return Ok(working)
        if isinstance(new_ir, Request):
            return Ok(new_ir)
        return Ok(request)

    legacy_result = instance.process(request, context)
    if is_ok(legacy_result):
        out: Request = unwrap(legacy_result)
        return Ok(out)
    return Err(unwrap_err(legacy_result))
