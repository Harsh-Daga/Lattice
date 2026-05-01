"""Semantic safety and risk scoring for conservative transform gating.

Provides:
1. SafetyProfile — lightweight binary safety flags
2. SemanticRiskScore — numeric 0-100 risk score for gating transforms
3. TransformSafetyBucket — SAFE / CONDITIONAL / DANGEROUS classification
4. Risk-aware gating helpers for the pipeline and individual transforms
"""

from __future__ import annotations

import dataclasses
import enum
import json
import re
from typing import Any

from lattice.core.transport import Request

_STRICT_INSTRUCTION_PATTERNS = (
    r"\bdo not\b",
    r"\bdo n't\b",
    r"\bmust\b",
    r"\bexactly\b",
    r"\bpreserve\b",
    r"\bkeep\b.*\bmeaning\b",
    r"\bwithout changing\b",
    r"\bunchanged\b",
    r"\breturn json\b",
    r"\bformat\b",
    r"\btable\b",
    r"\bcode\b",
)

_STRUCTURED_PREFIXES = ("{", "[", "<")

# =============================================================================
# Risk factor keywords
# =============================================================================

_SENSITIVE_DOMAIN_PATTERNS = (
    r"\blegal\b",
    r"\blawyer\b",
    r"\battorney\b",
    r"\bmedical\b",
    r"\bdiagnos\b",
    r"\bpatient\b",
    r"\bfinancial\b",
    r"\baccount\b.*\bnumber\b",
    r"\bcredit card\b",
    r"\bsafety\b",
    r"\bsecurity\b",
    r"\bcompliance\b",
    r"\breligion\b",
    r"\bpolitics\b",
    r"\belection\b",
    r"\bconfidential\b",
    r"\bsensitive\b",
    r"\bclassified\b",
)

_REASONING_HEAVY_PATTERNS = (
    r"\breason\b.*\bstep\b",
    r"\bstep\b.*\breason\b",
    r"\bthink\b.*\bcarefully\b",
    r"\bcarefully\b.*\bthink\b",
    r"\bexplain\b.*\bwhy\b",
    r"\bwhy\b.*\bexplain\b",
    r"\bmulti-step\b",
    r"\bmulti step\b",
    r"\bchain of thought\b",
    r"\bchain-of-thought\b",
    r"\bdeduce\b",
    r"\binfer\b",
    r"\bsolve\b",
)


# =============================================================================
# Safety profile (unchanged — still used for fast binary checks)
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class SafetyProfile:
    """Conservative content safety profile for a request."""

    has_structured_content: bool
    has_code_blocks: bool
    has_strict_instructions: bool
    has_tool_calls: bool
    has_high_stakes_entities: bool
    long_form: bool


# =============================================================================
# Semantic risk score
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class SemanticRiskScore:
    """Numeric 0–100 risk score for lossy-transform safety.

    Higher = riskier.  Transforms should be gated by explicit thresholds.

    Risk factors (each 0–100 scaled to weight):
      - strict_instructions (0–30)
      - sensitive_domain (0–20)
      - structured_output (0–15)
      - high_stakes_entities (0–15)
      - reasoning_heavy (0–20)
      - intentional_repetition (0–10)
      - tool_call_dependency (0–10)
      - formatting_constraints (0–10)

    Thresholds:
      - LOW    (0–20):  All transforms safe
      - MEDIUM (20–40): Most transforms safe; CONDITIONAL gated
      - HIGH   (40–60): Only SAFE transforms; CONDITIONAL blocked
      - CRITICAL (>60):  Only SAFE transforms; no CONDITIONAL or DANGEROUS
    """

    strict_instructions: float = 0.0
    sensitive_domain: float = 0.0
    structured_output: float = 0.0
    high_stakes_entities: float = 0.0
    reasoning_heavy: float = 0.0
    intentional_repetition: float = 0.0
    tool_call_dependency: float = 0.0
    formatting_constraints: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.strict_instructions
            + self.sensitive_domain
            + self.structured_output
            + self.high_stakes_entities
            + self.reasoning_heavy
            + self.intentional_repetition
            + self.tool_call_dependency
            + self.formatting_constraints,
            1,
        )

    @property
    def level(self) -> str:
        t = self.total
        if t <= 20:
            return "LOW"
        if t <= 40:
            return "MEDIUM"
        if t <= 60:
            return "HIGH"
        return "CRITICAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "level": self.level,
            "strict_instructions": round(self.strict_instructions, 1),
            "sensitive_domain": round(self.sensitive_domain, 1),
            "structured_output": round(self.structured_output, 1),
            "high_stakes_entities": round(self.high_stakes_entities, 1),
            "reasoning_heavy": round(self.reasoning_heavy, 1),
            "intentional_repetition": round(self.intentional_repetition, 1),
            "tool_call_dependency": round(self.tool_call_dependency, 1),
            "formatting_constraints": round(self.formatting_constraints, 1),
        }


def compute_risk_score(request: Request) -> SemanticRiskScore:
    """Compute a numeric semantic risk score for a request."""
    text = "\n".join(msg.content for msg in request.messages)
    lowered = text.lower()

    # Strict instructions (0–30)
    si_score = 0.0
    for pattern in _STRICT_INSTRUCTION_PATTERNS:
        if re.search(pattern, lowered):
            si_score += 3.0
    si_score = min(si_score, 30.0)

    # Sensitive domain (0–20)
    sd_score = 0.0
    for pattern in _SENSITIVE_DOMAIN_PATTERNS:
        if re.search(pattern, lowered):
            sd_score += 5.0
    sd_score = min(sd_score, 20.0)

    # Structured output (0–15)
    so_score = 0.0
    if any(_looks_structured(msg.content) for msg in request.messages):
        so_score += 8.0
    if request.tools:
        so_score += 7.0
    so_score = min(so_score, 15.0)

    # High-stakes entities (0–15)
    hse_score = 0.0
    if re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", text, re.IGNORECASE
    ):
        hse_score += 5.0
    if re.search(r"https?://[^\s)]+", text, re.IGNORECASE):
        hse_score += 3.0
    number_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    hse_score += min(number_count * 0.5, 7.0)
    hse_score = min(hse_score, 15.0)

    # Reasoning heavy (0–20)
    rh_score = 0.0
    for pattern in _REASONING_HEAVY_PATTERNS:
        if re.search(pattern, lowered):
            rh_score += 5.0
    rh_score = min(rh_score, 20.0)

    # Intentional repetition (0–10)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 3:
        from collections import Counter

        line_counts = Counter(lines)
        repeated_count = sum(c for line, c in line_counts.items() if c >= 3)
        ir_score = min(repeated_count * 2.0, 10.0) if repeated_count >= 3 else 0.0
    else:
        ir_score = 0.0

    # Tool call dependency (0–10)
    tc_score = 0.0
    tool_call_count = sum(1 for m in request.messages if bool(m.tool_calls))
    if tool_call_count > 0:
        tc_score = min(tool_call_count * 2.5, 10.0)

    # Formatting constraints (0–10)
    fc_score = 0.0
    if re.search(r"\btable\b", lowered) and re.search(r"\bformat\b", lowered):
        fc_score += 4.0
    if re.search(r"\bjson\b", lowered):
        fc_score += 3.0
    if re.search(r"\bcsv\b", lowered):
        fc_score += 3.0
    fc_score = min(fc_score, 10.0)

    return SemanticRiskScore(
        strict_instructions=si_score,
        sensitive_domain=sd_score,
        structured_output=so_score,
        high_stakes_entities=hse_score,
        reasoning_heavy=rh_score,
        intentional_repetition=ir_score,
        tool_call_dependency=tc_score,
        formatting_constraints=fc_score,
    )


# =============================================================================
# Transform safety buckets
# =============================================================================


class TransformSafetyBucket(enum.Enum):
    """Safety classification for transforms.

    SAFE:        Can run on any input without meaning drift.
    CONDITIONAL: Safe when risk is LOW or MEDIUM; blocked at HIGH+.
    DANGEROUS:   Only safe on LOW risk, specific profiles, or with explicit
                 proof of safety.  Blocked at MEDIUM+.
    """

    SAFE = "safe"
    CONDITIONAL = "conditional"
    DANGEROUS = "dangerous"


# Transform name → safety bucket mapping
_TRANSFORM_SAFETY_MAP: dict[str, TransformSafetyBucket] = {
    # SAFE — structural transforms that don't change meaning
    "content_profiler": TransformSafetyBucket.SAFE,
    "tool_filter": TransformSafetyBucket.SAFE,
    "tool_output_filter": TransformSafetyBucket.SAFE,  # alias
    "context_selector": TransformSafetyBucket.SAFE,
    "prefix_optimizer": TransformSafetyBucket.SAFE,
    "prefix_opt": TransformSafetyBucket.SAFE,  # alias
    "output_cleanup": TransformSafetyBucket.SAFE,
    "delta_encoder": TransformSafetyBucket.SAFE,
    "batching": TransformSafetyBucket.SAFE,
    "speculative": TransformSafetyBucket.SAFE,
    "cache_arbitrage": TransformSafetyBucket.SAFE,
    "strategy_selector": TransformSafetyBucket.SAFE,
    "runtime_contract": TransformSafetyBucket.SAFE,
    # CONDITIONAL — lossy but recoverable; risk-gated
    "reference_sub": TransformSafetyBucket.CONDITIONAL,
    "message_dedup": TransformSafetyBucket.CONDITIONAL,
    "message_deduplicator": TransformSafetyBucket.CONDITIONAL,  # alias
    "format_conversion": TransformSafetyBucket.CONDITIONAL,
    "semantic_compress": TransformSafetyBucket.CONDITIONAL,
    "semantic_compressor": TransformSafetyBucket.CONDITIONAL,  # alias
    "dictionary_compress": TransformSafetyBucket.CONDITIONAL,
    "dictionary_compressor": TransformSafetyBucket.CONDITIONAL,  # alias
    "grammar_compress": TransformSafetyBucket.CONDITIONAL,
    "grammar_compressor": TransformSafetyBucket.CONDITIONAL,  # alias
    "rate_distortion": TransformSafetyBucket.CONDITIONAL,
    "information_theoretic_selector": TransformSafetyBucket.CONDITIONAL,
    "self_information": TransformSafetyBucket.CONDITIONAL,
    # DANGEROUS — can replace meaning-bearing content with placeholders
    "structural_fingerprint": TransformSafetyBucket.DANGEROUS,
    "hierarchical_summary": TransformSafetyBucket.DANGEROUS,
    "hierarchical_summarizer": TransformSafetyBucket.DANGEROUS,  # alias
}

# Unknown transforms default to CONDITIONAL — they must prove safety, not
# assume it.  This prevents alias-based bypass of the classification.
_UNKNOWN_DEFAULT = TransformSafetyBucket.CONDITIONAL


def get_transform_safety_bucket(name: str) -> TransformSafetyBucket:
    """Return the safety bucket for a transform by name.

    Unknown names default to CONDITIONAL — they must prove safety,
    not assume it.  This prevents alias-based bypass.
    """
    return _TRANSFORM_SAFETY_MAP.get(name, _UNKNOWN_DEFAULT)


def transform_allowed_at_risk(
    transform_name: str,
    risk: SemanticRiskScore,
) -> tuple[bool, str]:
    """Return (allowed, reason) for a transform at a given risk level."""
    bucket = get_transform_safety_bucket(transform_name)
    if bucket == TransformSafetyBucket.SAFE:
        return True, "safe_transform"
    if bucket == TransformSafetyBucket.CONDITIONAL:
        if risk.level in ("LOW", "MEDIUM"):
            return True, "conditional_allowed"
        return False, f"conditional_blocked_at_{risk.level.lower()}_risk"
    if bucket == TransformSafetyBucket.DANGEROUS:
        if risk.level == "LOW":
            return True, "dangerous_allowed_at_low_risk"
        return False, f"dangerous_blocked_at_{risk.level.lower()}_risk"
    return True, "unknown_bucket"


# =============================================================================
# Legacy helpers (kept for backward compatibility)
# =============================================================================


def has_code_blocks(text: str) -> bool:
    return "```" in text or bool(re.search(r"`[^`\n]+`", text))


def has_strict_instructions(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _STRICT_INSTRUCTION_PATTERNS)


def _looks_structured(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(_STRUCTURED_PREFIXES):
        return True
    return bool("\n|" in text or re.search(r"^\s*\|.*\|\s*$", text, re.MULTILINE))


def _has_high_stakes_entities(text: str) -> bool:
    if re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", text, re.IGNORECASE
    ):
        return True
    if re.search(r"https?://[^\s)]+", text, re.IGNORECASE):
        return True
    return len(re.findall(r"\b\d+(?:\.\d+)?\b", text)) >= 5


def request_safety_profile(request: Request) -> SafetyProfile:
    """Build a conservative safety profile for a request."""
    text = "\n".join(msg.content for msg in request.messages)
    return SafetyProfile(
        has_structured_content=any(_looks_structured(msg.content) for msg in request.messages)
        or bool(request.tools),
        has_code_blocks=has_code_blocks(text),
        has_strict_instructions=has_strict_instructions(text),
        has_tool_calls=any(bool(msg.tool_calls) for msg in request.messages),
        has_high_stakes_entities=_has_high_stakes_entities(text),
        long_form=request.token_estimate >= 200,
    )


def lossy_transform_allowed(request: Request) -> bool:
    """Return True when a lossy transform is unlikely to change meaning."""
    profile = request_safety_profile(request)
    text = "\n".join(msg.content for msg in request.messages)
    sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
    if profile.has_tool_calls:
        return False
    if profile.has_structured_content or profile.has_code_blocks:
        return False
    if profile.has_strict_instructions:
        return False
    if profile.has_high_stakes_entities and not profile.long_form and sentence_count < 3:
        return False
    return profile.long_form or sentence_count >= 3


def structure_signature(text: str) -> dict[str, Any]:
    """Return a small structural signature for trace/debugging."""
    stripped = text.strip()
    parsed_json = None
    if stripped and stripped[:1] in ("{", "["):
        try:
            parsed_json = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_json = None
    signature: dict[str, Any] = {
        "chars": len(text),
        "lines": text.count("\n") + 1 if text else 0,
        "has_code": has_code_blocks(text),
        "has_table": bool(re.search(r"^\s*\|.*\|\s*$", text, re.MULTILINE)),
        "uuid_count": len(
            re.findall(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                text,
                re.IGNORECASE,
            )
        ),
        "url_count": len(re.findall(r"https?://[^\s)]+", text, re.IGNORECASE)),
        "number_count": len(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
        "json_keys": sorted(parsed_json.keys()) if isinstance(parsed_json, dict) else [],
    }
    return signature


__all__ = [
    "SafetyProfile",
    "SemanticRiskScore",
    "TransformSafetyBucket",
    "compute_risk_score",
    "get_transform_safety_bucket",
    "transform_allowed_at_risk",
    "has_code_blocks",
    "has_strict_instructions",
    "lossy_transform_allowed",
    "request_safety_profile",
    "structure_signature",
]
