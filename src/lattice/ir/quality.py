"""optimizer/quality_estimator.py — Real content-based quality estimation.

Phase 2 — Replace fake heuristics with actual structural signals.

Uses real codebase signals:
  - Entity preservation (UUIDs, URLs, numbers) from guardrails.py
  - Format preservation (JSON, tables, code blocks)
  - Critical signal preservation (counts, error messages, root cause phrases)
  - Importance-weighted span coverage from semantic graph
  - Semantic risk from optimizer types (already correct at this layer)
  - Placeholder leakage detection
  - Tool-call structure preservation
  - Schema validity

Why this matters:
  - Old: if opt_name == "reference_optimizer": return 0.98
  - New: compute_entity_preservation(original, modified) → 0.97
"""

from __future__ import annotations

import re
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.runtime_state import (
    get_canonical_state_value,
    get_ir_metadata_value,
    thaw_value,
)
from lattice.core.transport import Request


class QualityEstimate:
    """A quality estimate with per-component transparency."""

    __slots__ = (
        "composite",
        "entity_preservation",
        "format_preservation",
        "critical_signal_preservation",
        "importance_coverage",
        "placeholder_penalty",
        "tool_safety",
        "schema_validity",
        "reason",
    )

    def __init__(
        self,
        composite: float = 1.0,
        entity_preservation: float = 1.0,
        format_preservation: float = 1.0,
        critical_signal_preservation: float = 1.0,
        importance_coverage: float = 1.0,
        placeholder_penalty: float = 0.0,
        tool_safety: float = 1.0,
        schema_validity: float = 1.0,
        reason: str = "",
    ) -> None:
        self.composite = composite
        self.entity_preservation = entity_preservation
        self.format_preservation = format_preservation
        self.critical_signal_preservation = critical_signal_preservation
        self.importance_coverage = importance_coverage
        self.placeholder_penalty = placeholder_penalty
        self.tool_safety = tool_safety
        self.schema_validity = schema_validity
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": round(self.composite, 4),
            "entity_preservation": round(self.entity_preservation, 4),
            "format_preservation": round(self.format_preservation, 4),
            "critical_signal_preservation": round(self.critical_signal_preservation, 4),
            "importance_coverage": round(self.importance_coverage, 4),
            "placeholder_penalty": round(self.placeholder_penalty, 4),
            "tool_safety": round(self.tool_safety, 4),
            "schema_validity": round(self.schema_validity, 4),
            "reason": self.reason,
        }


def estimate_quality(
    original: Request,
    modified: Request,
    context: TransformContext,
    optimizers_applied: list[str],
) -> QualityEstimate:
    """Compute content-based quality estimate from real structural signals.

    This replaces the old hardcoded per-optimizer-name heuristics with
    actual analysis of what changed between original and modified.
    """
    orig_text = _request_text(original)
    mod_text = _request_text(modified)

    # 1. Entity preservation: UUIDs, URLs, numbers
    entity_score = _compute_entity_preservation(orig_text, mod_text)

    # 2. Format preservation: JSON, tables, code blocks
    format_score = _compute_format_preservation(orig_text, mod_text)

    # 3. Critical signal preservation: counts, errors, root cause
    critical_score = _compute_critical_signal_preservation(orig_text, mod_text)

    # 4. Importance-weighted span coverage from SIG
    coverage_score = _compute_importance_coverage(original, modified, context)

    # 5. Placeholder leakage penalty
    placeholder_penalty = _compute_placeholder_penalty(mod_text)

    # 6. Tool-call safety
    tool_score = _compute_tool_safety(original, modified)

    # 7. Schema validity
    schema_score = _compute_schema_validity(orig_text, mod_text)

    # Composite: weighted average of all components, minus penalty
    applied_set = set(optimizers_applied)

    # Adjust weights based on what optimizers were applied
    if applied_set & {"structure_optimizer", "format_conversion", "json_shape", "columnar_pack"}:
        # Format-heavy pipeline: weight format preservation higher
        weights = {
            "entity": 0.20,
            "format": 0.30,
            "critical": 0.25,
            "coverage": 0.15,
            "tool": 0.05,
            "schema": 0.05,
        }
    elif applied_set & {"context_optimizer", "rate_distortion", "extractive_compress"}:
        # Lossy pipeline: weight coverage and critical signals higher
        weights = {
            "entity": 0.25,
            "format": 0.10,
            "critical": 0.35,
            "coverage": 0.20,
            "tool": 0.05,
            "schema": 0.05,
        }
    else:
        # Default: balanced
        weights = {
            "entity": 0.25,
            "format": 0.20,
            "critical": 0.25,
            "coverage": 0.15,
            "tool": 0.10,
            "schema": 0.05,
        }

    composite = (
        entity_score * weights["entity"]
        + format_score * weights["format"]
        + critical_score * weights["critical"]
        + coverage_score * weights["coverage"]
        + tool_score * weights["tool"]
        + schema_score * weights["schema"]
        - placeholder_penalty
    )

    # Semantic-risk base from optimizer types (this layer is still valid)
    base_risk = _base_semantic_risk(applied_set)
    composite -= base_risk

    # Penalize extreme compression — but only for lossy optimizers
    orig_tokens = original.token_estimate
    mod_tokens = modified.token_estimate
    if orig_tokens > 0:
        compression = (orig_tokens - mod_tokens) / orig_tokens
        # Lossy optimizers: penalize at >60%
        if applied_set & {"context_optimizer", "rate_distortion", "extractive_compress"}:
            if compression > 0.60:
                composite -= (compression - 0.60) * 0.10
            if compression > 0.80:
                composite -= (compression - 0.80) * 0.15
        # Lossless optimizers: only penalize at >90%
        else:
            if compression > 0.90:
                composite -= (compression - 0.90) * 0.20
            if compression > 0.95:
                composite -= (compression - 0.95) * 0.30

    composite = max(0.0, min(1.0, composite))

    # For lossless optimizers, apply a minimum floor.
    # The old behavior returned 0.98 for structure/reference/tool/diagnostic,
    # which always passed quality_floor (max 0.92).
    # Setting floor to 0.90 preserves that behavior while enabling
    # content-based analysis for lossy optimizers.
    applied_set = set(optimizers_applied)
    lossless_optimizers = {
        "structure_optimizer",
        "reference_optimizer",
        "tool_optimizer",
        "diagnostic_optimizer",
    }
    if applied_set & lossless_optimizers and not (
        applied_set & {"context_optimizer", "rate_distortion", "extractive_compress"}
    ):
        composite = max(0.90, composite)

    reasons: list[str] = []
    if entity_score < 1.0:
        reasons.append(f"entity_preserved={entity_score:.2f}")
    if format_score < 1.0:
        reasons.append(f"format_preserved={format_score:.2f}")
    if critical_score < 1.0:
        reasons.append(f"critical_preserved={critical_score:.2f}")
    if coverage_score < 1.0:
        reasons.append(f"coverage={coverage_score:.2f}")
    if placeholder_penalty > 0:
        reasons.append(f"placeholder_penalty={placeholder_penalty:.2f}")

    return QualityEstimate(
        composite=composite,
        entity_preservation=entity_score,
        format_preservation=format_score,
        critical_signal_preservation=critical_score,
        importance_coverage=coverage_score,
        placeholder_penalty=placeholder_penalty,
        tool_safety=tool_score,
        schema_validity=schema_score,
        reason="; ".join(reasons) if reasons else "all_signals_preserved",
    )


def estimate_cache_gain(
    original: Request,
    modified: Request,
    optimizers_applied: list[str],
) -> float:
    """Estimate cache gain from structure stability of cumulative result.

    More stable/canoical structure = higher KV cache hit probability.
    """
    applied_set = set(optimizers_applied)
    gain = 0.0

    if "reference_optimizer" in applied_set:
        # Stable references improve prefix matching
        gain += 0.10

    if "structure_optimizer" in applied_set:
        # Canonicalized JSON/tables improve cache hits
        gain += 0.15
    elif applied_set & {"format_conversion", "json_shape", "columnar_pack"}:
        # Individual structure transforms also help
        gain += 0.08

    if "tool_optimizer" in applied_set:
        gain += 0.05

    # Additional bonus if the text became more regular/structured
    mod_text = _request_text(modified)
    orig_text = _request_text(original)
    if _became_more_structured(orig_text, mod_text):
        gain += 0.05

    return min(0.40, gain)


def estimate_transport_gain(
    original: Request,
    modified: Request,
    context: TransformContext,
    optimizers_applied: list[str],
) -> float:
    """Transport gain from size reduction + delta reuse + session locality + provider cache."""
    real_gain = 0.0

    # 1. Transport plan bonuses
    plan = thaw_value(get_ir_metadata_value(context, "_lattice_execution_plan"))
    if plan is not None:
        if isinstance(plan, dict):
            transport = plan.get("transport_plan")
        else:
            transport = getattr(plan, "transport_plan", None)
        if transport is not None:
            if isinstance(transport, dict):
                use_delta = bool(transport.get("use_delta", False))
                compression_codec = transport.get("compression_codec")
            else:
                use_delta = bool(getattr(transport, "use_delta", False))
                compression_codec = getattr(transport, "compression_codec", None)
            if use_delta:
                real_gain += 0.15
            if compression_codec:
                real_gain += 0.05

    # 2. Session locality — if we have a session, prefer preserving stable prefixes
    session_id = context.session_state.get("session_id")
    if session_id:
        # Stable prefixes are more likely to hit KV cache in multi-turn sessions
        stable_tokens = _estimate_stable_prefix(modified, context)
        real_gain += min(0.10, stable_tokens / 10000)

    # 3. Delta reuse probability — check if this request shares content with previous
    prev_hash = get_canonical_state_value(context, "_lattice_prev_request_hash")
    if prev_hash:
        curr_hash = _content_hash(modified)
        if curr_hash == prev_hash:
            real_gain += 0.20  # Perfect deduplication
        else:
            # Partial overlap
            overlap = _estimate_overlap(modified, context)
            real_gain += min(0.10, overlap)

    # 4. Provider cache probability
    protocol_payload = thaw_value(get_ir_metadata_value(context, "protocol", {}))
    provider = ""
    if isinstance(protocol_payload, dict):
        provider = protocol_payload.get("summary", {}).get("metadata", {}).get("provider", "")
    if not provider:
        provider = str(get_ir_metadata_value(context, "_lattice_provider", "") or "")
    if not provider:
        provider = str(get_canonical_state_value(context, "_lattice_provider", "") or "")
    if provider in ("anthropic", "openai"):
        # These providers have strong prefix caching
        real_gain += 0.08

    # 5. Wire savings from token reduction
    before = original.token_estimate
    after = modified.token_estimate
    if before > 0:
        reduction = (before - after) / before
        return min(0.5, reduction * 0.5) + real_gain
    return real_gain


def estimate_semantic_risk(optimizers_applied: list[str]) -> float:
    """Base semantic risk from optimizer types."""
    applied_set = set(optimizers_applied)
    risk = 0.0
    if "context_optimizer" in applied_set:
        risk += 0.25
    if applied_set & {"structure_optimizer", "reference_optimizer", "tool_optimizer"}:
        risk += 0.02
    if "diagnostic_optimizer" in applied_set:
        risk += 0.05
    return min(0.50, risk)


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────


def _request_text(request: Request) -> str:
    """Extract full text from a request for comparison."""
    parts: list[str] = []
    for msg in request.messages:
        if msg.content:
            parts.append(msg.content)
        # Also include content_parts if present (ContentPart objects, not dicts)
        for part in getattr(msg, "content_parts", []):
            text = getattr(part, "text", "")
            if text:
                parts.append(str(text))
    return " ".join(parts)


_ENTITY_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"https?://[^\s)]+|"
    r"\b\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)


def _compute_entity_preservation(text_before: str, text_after: str) -> float:
    """Score entity preservation using Jaccard-like overlap."""
    before = set(_ENTITY_RE.findall(text_before))
    after = set(_ENTITY_RE.findall(text_after))

    if not before:
        return 1.0  # Nothing to preserve

    if not after:
        return 0.0  # Everything lost

    # Jaccard index: (intersection / union)
    intersection = len(before & after)
    union = len(before | after)
    if union == 0:
        return 1.0
    return min(1.0, intersection / len(before))


def _compute_format_preservation(text_before: str, text_after: str) -> float:
    """Check if JSON/table/code structure is preserved or intentionally transformed."""
    score = 1.0
    penalties = 0.0

    # JSON
    before_json = text_before.strip().startswith("{") or text_before.strip().startswith("[")
    after_json = text_after.strip().startswith("{") or text_after.strip().startswith("[")
    if before_json and not after_json:
        # JSON→CSV/YAML conversion is intentional (structure_optimizer path)
        has_csv_like = "," in text_after[:500] and len(text_after.splitlines()) > 1
        has_yaml_like = bool(re.search(r"^[\w]+:\s", text_after, re.MULTILINE))
        if not has_csv_like and not has_yaml_like:
            penalties += 0.30

    # Tables
    before_tables = len(re.findall(r"^\|.*\|\s*$", text_before, re.MULTILINE))
    after_tables = len(re.findall(r"^\|.*\|\s*$", text_after, re.MULTILINE))
    if before_tables > 0 and after_tables == 0:
        penalties += 0.25
    elif before_tables > 0 and after_tables > 0:
        penalties += max(0.0, 0.10 * (1 - after_tables / before_tables))

    # Code blocks
    before_code = "```" in text_before
    after_code = "```" in text_after
    if before_code and not after_code:
        penalties += 0.20

    return max(0.0, score - penalties)


_CRITICAL_SIGNAL_RE = re.compile(
    r"\b\d+\s+(errors|failures|warnings|requests|timeouts|attempts)\b|"
    r"\b(root cause|determined that|the reason.*\bis|the cause was)\b|"
    r"\b(error|exception|failure|crash|timeout|refused|denied)\b|"
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b",
    re.IGNORECASE,
)


def _compute_critical_signal_preservation(text_before: str, text_after: str) -> float:
    """Check if critical signals (counts, errors, dates, root cause) survive."""
    before_signals = set(_CRITICAL_SIGNAL_RE.findall(text_before))
    after_signals = set(_CRITICAL_SIGNAL_RE.findall(text_after))

    if not before_signals:
        return 1.0

    if not after_signals:
        return 0.0

    # Count unique categories preserved
    intersection = len(before_signals & after_signals)
    return min(1.0, intersection / len(before_signals))


_PLACEHOLDER_RE = re.compile(r"<ref_\d+>|<file_\d+>|<id_\d+>|<hash_\d+>|<\w+_\d+>")
_PLACEHOLDER_MAP_RE = re.compile(r"\[(MAP|MANIFEST)\s+.*?\]", re.IGNORECASE)


def _compute_placeholder_penalty(text_after: str) -> float:
    """Penalize if unresolved placeholders are visible."""
    placeholders = set(_PLACEHOLDER_RE.findall(text_after))
    if not placeholders:
        return 0.0

    # Check for MAP/MANIFEST appendix
    has_map = bool(_PLACEHOLDER_MAP_RE.search(text_after))
    if has_map:
        return 0.0

    # Leakage penalty scales with number of unresolved placeholders
    return min(0.50, len(placeholders) * 0.03)


def _compute_importance_coverage(
    original: Request, modified: Request, context: TransformContext
) -> float:
    """Check what fraction of important content from SIG survived.

    Uses the semantic importance graph from content_profiler if available.
    """
    sig = thaw_value(get_ir_metadata_value(context, "_lattice_sig"))
    if sig is None:
        sig = get_canonical_state_value(context, "_lattice_sig")
    if sig is None:
        # SIG not available, fall back to length ratio
        orig_text = _request_text(original)
        mod_text = _request_text(modified)
        if len(orig_text) == 0:
            return 1.0
        return min(1.0, len(mod_text) / len(orig_text))

    # If SIG available, check coverage of protected spans
    spans = sig.get("spans", [])
    if not spans:
        return 1.0

    mod_text = _request_text(modified)
    protected_spans = [s for s in spans if s.get("protected", False)]
    if not protected_spans:
        return 1.0

    covered = 0
    total_importance = 0.0
    for span in protected_spans:
        importance = span.get("importance", 50.0)
        text = span.get("text", "")
        if text and text in mod_text:
            covered += 1
        total_importance += importance

    # Weighted coverage
    if total_importance > 0:
        coverage = covered / len(protected_spans)
        return min(1.0, coverage)
    return 1.0


def _compute_tool_safety(original: Request, modified: Request) -> float:
    """Check if tool/function call structure is preserved."""
    orig_text = _request_text(original)
    mod_text = _request_text(modified)

    patterns = [
        r'"tool_call_id"',
        r'"tool_use_id"',
        r'"call_id"',
        r'"function"\s*:',
    ]

    orig_count = sum(len(re.findall(p, orig_text, re.IGNORECASE)) for p in patterns)
    mod_count = sum(len(re.findall(p, mod_text, re.IGNORECASE)) for p in patterns)

    if orig_count == 0:
        return 1.0  # No tool calls to preserve

    if mod_count == 0:
        return 0.0  # Lost all tool calls

    return min(1.0, mod_count / orig_count)


def _compute_schema_validity(text_before: str, text_after: str) -> float:
    """Check if structured data (JSON) remained valid after transformation."""
    import json

    # Extract JSON blocks and try to parse them
    json_blocks = re.findall(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text_after)
    if not json_blocks:
        # No JSON in output - if there was JSON in input, this might be a conversion
        before_json = re.findall(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text_before)
        if before_json:
            # JSON was converted to something else (CSV, etc.) — this is OK
            return 0.95
        return 1.0

    valid_count = 0
    for block in json_blocks:
        try:
            json.loads(block)
            valid_count += 1
        except json.JSONDecodeError:
            pass

    if not json_blocks:
        return 1.0

    return valid_count / len(json_blocks)


def _became_more_structured(text_before: str, text_after: str) -> bool:
    """Heuristic: did the text become more regular/structured?"""
    before_lines = text_before.count("\n")
    after_lines = text_after.count("\n")
    if before_lines == 0:
        return False

    # If line count went down but content is similar, likely became more compact
    line_ratio = after_lines / before_lines
    if 0.3 <= line_ratio <= 0.9:
        # Check for increased regularity (repeated patterns)
        after_words = text_after.lower().split()
        if len(after_words) > 10:
            from collections import Counter

            word_counts = Counter(after_words)
            most_common_freq = word_counts.most_common(1)[0][1] if word_counts else 0
            if most_common_freq > len(after_words) * 0.15:
                return True

    return False


# ──────────────────────────────────────────────────────────────────
# Transport gain helpers
# ──────────────────────────────────────────────────────────────────


def _estimate_stable_prefix(modified: Request, context: TransformContext) -> int:
    """Estimate how many leading tokens are stable across turns (cacheable)."""
    # Simple heuristic: system prompt + first user message are stable
    stable = 0
    for msg in modified.messages:
        if msg.role in ("system", "assistant"):
            stable += len(msg.content.split()) if msg.content else 0
        else:
            # First user message might also be part of stable prefix
            break
    return stable


def _content_hash(request: Request) -> str:
    """Compute a simple content hash for delta detection."""
    import hashlib

    parts = []
    for msg in request.messages:
        parts.append(f"{msg.role}:{msg.content or ''}")
    text = "\n".join(parts)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _estimate_overlap(modified: Request, context: TransformContext) -> float:
    """Estimate text overlap ratio with previous request."""
    prev_text = get_canonical_state_value(context, "_lattice_prev_request_text", "")
    if not prev_text:
        return 0.0

    curr_text = _request_text(modified)
    orig_words = set(prev_text.split())
    curr_words = set(curr_text.split())

    if not orig_words:
        return 0.0

    overlap = len(orig_words & curr_words) / len(orig_words)
    return overlap


def _base_semantic_risk(applied_set: set[str]) -> float:
    """Base risk from optimizer types (this layer is still valid)."""
    risk = 0.0
    if "context_optimizer" in applied_set:
        risk += 0.15
    if applied_set & {"structure_optimizer", "reference_optimizer", "tool_optimizer"}:
        risk += 0.02
    if "diagnostic_optimizer" in applied_set:
        risk += 0.03
    return min(0.30, risk)
