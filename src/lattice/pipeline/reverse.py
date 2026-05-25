"""Pipeline response-side reverse transform application."""

from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.ir.primitives import ExecutionPlan
from lattice.planner.runtime_state import coerce_execution_plan, get_canonical_state_value
from lattice.transport.types import Response


def pipeline_reverse(
    pipeline: Any,
    response: Response,
    context: TransformContext,
    *,
    plan: ExecutionPlan | None = None,
) -> Response:
    """Reverse transforms in reverse order."""
    from lattice.transforms.registry import is_response_side

    if plan is None:
        plan = coerce_execution_plan(get_canonical_state_value(context, "_lattice_execution_plan"))

    if plan is not None and plan.transforms:
        tx_names: list[str] = list(plan.transforms)
    else:
        tx_names = list(context.transforms_applied)

    for tx_name in reversed(tx_names):
        if is_response_side(tx_name):
            continue
        inst = pipeline.registry.get(tx_name)
        if inst is None or not hasattr(inst, "reverse"):
            continue
        try:
            response = inst.reverse(response, context)
        except Exception:
            pass

    for tx_name in reversed(tx_names):
        if not is_response_side(tx_name):
            continue
        inst = pipeline.registry.get(tx_name)
        if inst is None or not hasattr(inst, "reverse"):
            continue
        try:
            response = inst.reverse(response, context)
        except Exception:
            pass

    for tx_name in pipeline.registry.get_transform_names():
        if not is_response_side(tx_name) or tx_name in tx_names:
            continue
        inst = pipeline.registry.get(tx_name)
        if inst is None or not hasattr(inst, "reverse"):
            continue
        try:
            response = inst.reverse(response, context)
        except Exception:
            pass

    return response
