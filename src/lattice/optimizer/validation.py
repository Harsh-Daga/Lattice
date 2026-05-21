"""optimizer/validation.py — Centralized validation for optimizer candidates.

Single source of truth for optimizer safety checks.

Primary path:
- validate_candidate() operates on immutable Candidate nodes.

Compatibility path:
- validate_request_candidate() validates legacy Request pairs.
- validate_beam_candidate() remains for older beam-search tests.
"""

from __future__ import annotations

import re
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.primitives import Candidate
from lattice.core.runtime_state import (
    get_canonical_request_value,
    get_ir_metadata_value,
    thaw_value,
)
from lattice.core.transport import Request


class ValidationResult:
    """Result of validating a candidate."""

    __slots__ = ("ok", "reason", "penalty", "soft_rejected")

    def __init__(
        self,
        ok: bool,
        reason: str = "",
        penalty: float = 0.0,
        soft_rejected: bool = False,
    ) -> None:
        self.ok = ok
        self.reason = reason
        self.penalty = penalty
        self.soft_rejected = soft_rejected

    def __repr__(self) -> str:
        status = "OK" if self.ok else ("SOFT" if self.soft_rejected else "REJECT")
        return f"ValidationResult({status}, reason={self.reason!r}, penalty={self.penalty:.3f})"


def validate_candidate(
    candidate: Candidate,
    context: TransformContext,
    *,
    quality_floor: float = 0.85,
    optimizer_name: str = "unknown",
    max_expansion_ratio: float = 0.15,
    max_compression_ratio: float = 0.70,
) -> ValidationResult:
    """Validate an immutable candidate produced by the optimizer graph.

    This is the canonical path. It uses candidate metrics and execution-plan
    metadata, not mutable request state.
    """
    estimated_quality = _estimate_quality_from_context(context, optimizer_name)
    if estimated_quality < quality_floor:
        return ValidationResult(
            ok=False,
            reason=f"quality_floor: {estimated_quality:.2f} < {quality_floor:.2f}",
        )

    metrics = dict(candidate.metrics)
    before = int(metrics.get("tokens_before", 0))
    after = int(metrics.get("tokens_after", before))
    cache_gain = float(metrics.get("cache_gain", 0.0))
    transport_gain = float(metrics.get("transport_gain", 0.0))
    penalty = 0.0

    if after > before:
        expansion = (after - before) / max(1, before)
        if expansion > max_expansion_ratio:
            return ValidationResult(
                ok=False,
                reason=f"expansion: {expansion:.2%} > {max_expansion_ratio:.2%}",
            )
        if cache_gain <= 0.0 and transport_gain <= 0.0:
            return ValidationResult(
                ok=False,
                reason="expansion: no cache/transport gain",
            )
        penalty += expansion * 2.0

    if before > 0:
        compression = (before - after) / before
        if compression > 0.50:
            overage = compression - 0.50
            penalty += overage * overage * 8.0
        if compression > max_compression_ratio:
            overage = compression - max_compression_ratio
            penalty += overage * overage * 5.0

    soft_rejected = penalty > 0.0
    if soft_rejected:
        return ValidationResult(
            ok=True,
            reason="soft_penalty",
            penalty=penalty,
            soft_rejected=True,
        )

    return ValidationResult(ok=True)


def validate_request_candidate(
    candidate: Request,
    original: Request,
    context: TransformContext,
    *,
    quality_floor: float = 0.85,
    optimizer_name: str = "unknown",
    max_expansion_ratio: float = 0.15,
    max_compression_ratio: float = 0.70,
    check_placeholders: bool = True,
    check_schema_preservation: bool = True,
    check_tool_call_safety: bool = True,
) -> ValidationResult:
    """Compatibility wrapper for request-based validation.

    This keeps the older Request-based callers working while the main search
    path moves to immutable Candidate nodes.
    """
    graph_candidate = Candidate(
        ir=_candidate_ir(candidate, context),
        metrics=frozenset(
            {
                ("tokens_before", original.token_estimate),
                ("tokens_after", candidate.token_estimate),
                ("cache_gain", 0.0),
                ("transport_gain", 0.0),
            }
        ),
    )

    result = validate_candidate(
        graph_candidate,
        context,
        quality_floor=quality_floor,
        optimizer_name=optimizer_name,
        max_expansion_ratio=max_expansion_ratio,
        max_compression_ratio=max_compression_ratio,
    )
    if not result.ok:
        return result

    if check_schema_preservation and _schema_broken(original, candidate):
        return ValidationResult(
            ok=False,
            reason="schema_preservation: JSON/tool structure corrupted",
        )

    if check_tool_call_safety and _tool_calls_corrupted(original, candidate):
        return ValidationResult(
            ok=False,
            reason="tool_call_safety: tool/function call structure corrupted",
        )

    if check_placeholders and _has_placeholder_leakage(candidate):
        context.record_metric(optimizer_name, "placeholder_leakage_blocked", True)
        return ValidationResult(
            ok=False,
            reason="placeholder_leakage: unresolved aliases visible in output",
        )

    return result


def validate_beam_candidate(
    candidate_tokens_before: int,
    candidate_tokens_after: int,
    quality_estimate: float,
    cache_gain: float,
    transport_gain: float,
    context: TransformContext,
    quality_floor: float = 0.85,
    max_compression_ratio: float = 0.70,
) -> tuple[bool, float]:
    """Lightweight compatibility helper for older beam-search tests."""
    if quality_estimate < quality_floor:
        return False, 0.0

    penalty = 0.0
    if candidate_tokens_after > candidate_tokens_before:
        if cache_gain <= 0 and transport_gain <= 0:
            return False, 0.0
        expansion = (candidate_tokens_after - candidate_tokens_before) / max(
            1, candidate_tokens_before
        )
        penalty += expansion * 1.5

    if candidate_tokens_before > 0:
        compression = (candidate_tokens_before - candidate_tokens_after) / candidate_tokens_before
        if compression > max_compression_ratio:
            overage = compression - max_compression_ratio
            penalty += overage * overage * 5.0
        elif compression > 0.50:
            overage = compression - 0.50
            penalty += overage * overage * 8.0

    return True, penalty


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_quality_from_context(context: TransformContext, optimizer_name: str) -> float:
    """Pull quality estimate from context metrics if available."""
    metrics = context.session_state.get("metrics", {})
    for key, val in metrics.items():
        if optimizer_name in key and isinstance(val, dict):
            q = val.get("quality_estimate")
            if q is not None:
                return float(q)

    plan = thaw_value(get_ir_metadata_value(context, "_lattice_execution_plan"))
    if isinstance(plan, dict):
        return float(plan.get("quality_floor", 0.85))
    if plan is not None:
        return float(getattr(plan, "quality_floor", 0.85))
    return 0.90


_PLACEHOLDER_RE = re.compile(r"<ref_\d+>|<file_\d+>|<id_\d+>|<hash_\d+>|<\w+_\d+>")
_PLACEHOLDER_MAP_RE = re.compile(r"\[(MAP|MANIFEST)\s+.*?\]", re.IGNORECASE)


def _candidate_ir(candidate: Request, context: TransformContext) -> Any:
    return get_canonical_request_value(candidate, context, "_lattice_ir_v2")


def _has_placeholder_leakage(request: Request) -> bool:
    """Detect unresolved placeholders in request text."""
    full_text = " ".join(m.content or "" for m in request.messages)
    placeholders = set(_PLACEHOLDER_RE.findall(full_text))
    if not placeholders:
        return False
    return not bool(_PLACEHOLDER_MAP_RE.search(full_text))


def _schema_broken(original: Request, candidate: Request) -> bool:
    """Check if JSON/schema structure was corrupted."""
    orig_text = " ".join(m.content or "" for m in original.messages)
    cand_text = " ".join(m.content or "" for m in candidate.messages)

    def _json_like_count(text: str) -> int:
        return min(text.count("{"), text.count("}"))

    orig_count = _json_like_count(orig_text)
    cand_count = _json_like_count(cand_text)
    return orig_count > 3 and cand_count == 0


def _tool_calls_corrupted(original: Request, candidate: Request) -> bool:
    """Check if tool/function call structure was corrupted."""
    orig_text = " ".join(m.content or "" for m in original.messages)
    cand_text = " ".join(m.content or "" for m in candidate.messages)
    orig_tool_refs = _count_tool_refs(orig_text)
    cand_tool_refs = _count_tool_refs(cand_text)
    return orig_tool_refs > 0 and cand_tool_refs == 0


def _count_tool_refs(text: str) -> int:
    patterns = [
        r'"tool_call_id"',
        r'"tool_use_id"',
        r'"call_id"',
        r'"function"\s*:',
        r"\btool\b.*\bcall\b",
    ]
    count = 0
    for p in patterns:
        count += len(re.findall(p, text, re.IGNORECASE))
    return count
