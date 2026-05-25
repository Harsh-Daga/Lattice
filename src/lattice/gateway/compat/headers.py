from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from lattice.planner.runtime_state import (
    get_canonical_request_value,
)
from lattice.telemetry.cost_estimator import normalize_usage
from lattice.telemetry.downgrade import TransportOutcome
from lattice.transport.types import Request

Handler = Callable[..., Awaitable[Any]]

# =============================================================================
# Shared proxy compatibility helpers
# =============================================================================


def build_routing_headers(
    model_used: str,
    *,
    compressed_tokens: int = 0,
    original_tokens: int = 0,
    session_id: str = "",
    anchor_version: int = 0,
    anchor_hash: str = "",
    used_speculative: bool | None = None,
    prediction_hit: bool | None = None,
    batched: bool | None = None,
    delta_savings_bytes: int = 0,
    cache_hit: bool | None = None,
    cached_tokens: int = 0,
    cost_usd: float = 0.0,
    cache_savings_usd: float = 0.0,
    runtime_tier: str = "",
    runtime_mode: str = "",
    runtime_budget_ms: float = 0.0,
    runtime_actual_ms: float = 0.0,
    runtime_budget_exhausted: bool = False,
    runtime_skipped_count: int = 0,
    framing: str = "",
    delta_mode: str = "",
    http_version: str = "",
    semantic_cache_status: str = "",
    batching_status: str = "",
    speculative_status: str = "",
    fallback_reason: str = "",
    stream_resumed: bool | None = None,
    transport_outcome: TransportOutcome | None = None,
) -> dict[str, str]:
    """Build response routing headers exposed by proxy/gateway.

    When *transport_outcome* is provided, transport-related headers are
    seeded from its canonical ``to_headers()`` output.  Individual legacy
    parameters are then applied on top, so an explicit legacy argument
    always overrides the canonical value.  This lets callers pin a header
    value without reconstructing the entire ``TransportOutcome``.

    Override semantics
    ------------------
    The legacy boolean parameters (used_speculative, batched, cache_hit,
    stream_resumed) use tri-state (None = use canonical, True/False =
    explicitly set).  An explicit ``False`` will **suppress** a canonical
    ``True``, resolving the previous ambiguity where ``False`` was
    indistinguishable from ``not set``.
    """
    compression = (
        f"{round(1 - compressed_tokens / max(original_tokens, 1), 4):.2%}"
        if original_tokens != compressed_tokens and original_tokens > 0
        else "0%"
    )
    headers: dict[str, str] = {
        "x-lattice-model": model_used,
        "x-lattice-compression": compression,
    }
    if session_id:
        headers["x-lattice-session-id"] = session_id
    if anchor_version > 0:
        headers["x-lattice-anchor-version"] = str(anchor_version)
    if anchor_hash:
        headers["x-lattice-anchor-hash"] = anchor_hash
    if delta_savings_bytes > 0:
        headers["x-lattice-delta-savings-bytes"] = str(delta_savings_bytes)
    if cached_tokens > 0:
        headers["x-lattice-cached-tokens"] = str(cached_tokens)
    if cost_usd > 0:
        headers["x-lattice-cost-usd"] = f"{cost_usd:.6f}"
    if cache_savings_usd > 0:
        headers["x-lattice-cache-savings-usd"] = f"{cache_savings_usd:.6f}"
    if runtime_tier:
        headers["x-lattice-runtime-tier"] = runtime_tier
    if runtime_mode:
        headers["x-lattice-runtime-mode"] = runtime_mode
    if runtime_budget_ms > 0:
        headers["x-lattice-runtime-budget-ms"] = f"{runtime_budget_ms:.2f}"
    if runtime_actual_ms > 0:
        headers["x-lattice-runtime-actual-ms"] = f"{runtime_actual_ms:.2f}"
    if runtime_budget_exhausted:
        headers["x-lattice-runtime-budget-exhausted"] = "true"
    if runtime_skipped_count > 0:
        headers["x-lattice-runtime-skipped-transforms"] = str(runtime_skipped_count)

    # Transport-related headers: start from canonical object, then let
    # legacy parameters override (explicit args win over defaults).
    if transport_outcome is not None:
        for k, v in transport_outcome.to_headers().items():
            headers.setdefault(k, v)

    # Tri-state legacy boolean overrides (applied AFTER canonical so
    # explicit False can suppress canonical True values).
    if used_speculative is not None:
        if used_speculative:
            headers["x-lattice-speculative"] = "hit" if prediction_hit else "miss"
        else:
            headers.pop("x-lattice-speculative", None)
            headers.pop("x-lattice-speculative-status", None)
    if batched is not None:
        if batched:
            headers["x-lattice-batched"] = "true"
        else:
            headers.pop("x-lattice-batched", None)
            headers.pop("x-lattice-batching", None)
    if cache_hit is not None:
        if cache_hit:
            headers["x-lattice-cache-hit"] = "true"
        else:
            headers.pop("x-lattice-cache-hit", None)
    if stream_resumed is not None:
        if stream_resumed:
            headers["x-lattice-stream-resumed"] = "true"
        else:
            headers.pop("x-lattice-stream-resumed", None)

    # Legacy individual-parameter path — always applied last so explicit
    # arguments override the canonical object.
    if framing:
        headers["x-lattice-framing"] = framing
    if delta_mode:
        headers["x-lattice-delta"] = delta_mode
    if http_version:
        headers["x-lattice-http-version"] = http_version
    if semantic_cache_status:
        headers["x-lattice-semantic-cache"] = semantic_cache_status
    if batching_status:
        headers["x-lattice-batching"] = batching_status
    if speculative_status:
        headers["x-lattice-speculative-status"] = speculative_status
    if fallback_reason:
        headers["x-lattice-fallback-reason"] = fallback_reason
    return headers


def _runtime_header_values(request: Request) -> dict[str, Any]:
    runtime = get_canonical_request_value(request, None, "_lattice_runtime", {})
    contract = get_canonical_request_value(request, None, "_lattice_runtime_contract", {})
    budget = get_canonical_request_value(request, None, "_lattice_runtime_budget", {})
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(contract, dict):
        contract = {}
    if not isinstance(budget, dict):
        budget = {}
    return {
        "runtime_tier": str(runtime.get("tier") or ""),
        "runtime_mode": str(contract.get("mode") or ""),
        "runtime_budget_ms": float(contract.get("max_transform_latency_ms") or 0.0),
        "runtime_actual_ms": float(budget.get("actual_transform_ms") or 0.0),
        "runtime_budget_exhausted": bool(budget.get("exhausted") or False),
        "runtime_skipped_count": int(budget.get("skipped_count") or 0),
    }


def _extract_cached_tokens(usage: dict[str, Any]) -> int:
    """Extract cached-token count from provider usage dict."""
    return normalize_usage(usage).get("cached_tokens", 0)


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    """Return total logical tokens represented by a usage dict."""
    normalized = normalize_usage(usage)
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(total, int):
        return total
    return normalized["prompt_tokens"] + normalized["completion_tokens"]

