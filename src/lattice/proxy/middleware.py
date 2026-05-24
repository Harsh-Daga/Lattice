"""Consolidated Lattice response header middleware.

Per-request side-effects are recorded on ``TransformContext`` and/or
``request.state.lattice_response_headers``. After the route handler runs,
this middleware emits headers in one place.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request as FastAPIRequest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_HEADER_KEYS = {
    "compression_pct": "x-lattice-compression",
    "session_id": "x-lattice-session-id",
    "delta_used": "x-lattice-delta",
    "cost_usd": "x-lattice-cost-usd",
    "provider": "x-lattice-provider",
    "transforms_applied": "x-lattice-transforms-applied",
}


def stash_lattice_response_headers(request: Any, headers: dict[str, str]) -> None:
    """Merge routing headers onto ``request.state`` for middleware emission."""
    state = getattr(request, "state", None)
    if state is None:
        return
    pending: dict[str, str] | None = getattr(state, "lattice_response_headers", None)
    if pending is None:
        pending = {}
        state.lattice_response_headers = pending
    pending.update(headers)


def attach_routing_headers(
    request: Any,
    ctx: Any | None,
    headers: dict[str, str],
) -> None:
    """Stash routing headers and mirror canonical keys on ``TransformContext``."""
    stash_lattice_response_headers(request, headers)
    if ctx is None:
        return
    for ctx_key, header_name in _HEADER_KEYS.items():
        if header_name not in headers:
            continue
        value = headers[header_name]
        if ctx_key == "cost_usd":
            try:
                ctx.set(ctx_key, float(value))
            except ValueError:
                ctx.set(ctx_key, value)
        else:
            ctx.set(ctx_key, value)
    if "provider" not in ctx._response_metadata and ctx.provider:
        ctx.set("provider", ctx.provider)
    if ctx.transforms_applied:
        ctx.set("transforms_applied", list(ctx.transforms_applied))


def _format_header_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    return str(value)


class LatticeHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: FastAPIRequest, call_next: Any) -> Response:
        response = await call_next(request)

        pending: dict[str, str] = dict(getattr(request.state, "lattice_response_headers", {}) or {})

        ctx = getattr(request.state, "transform_context", None)
        if ctx is not None:
            summary = ctx.get_summary()
            for ctx_key, header_name in _HEADER_KEYS.items():
                value = summary.get(ctx_key)
                if value is None:
                    continue
                pending[header_name] = _format_header_value(value)

        for header_name, value in pending.items():
            response.headers[header_name] = value

        return response


def install_middleware(app: Any) -> None:
    app.add_middleware(LatticeHeaderMiddleware)
