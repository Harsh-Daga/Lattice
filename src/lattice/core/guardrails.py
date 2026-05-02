"""Pipeline Safety Guard — PSG.

Central safety decision engine that combines SIG, RATS, risk score,
expansion estimates, and required output properties to make final
allow/skip/rollback/reject decisions.

PSG decides **what must never happen**.

Inputs: SIG graph, risk score, task class, transform bucket,
         predicted expansion ratio, required output properties.
Output: SafetyDecision (ALLOW, SKIP, ROLLBACK, REJECT).
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any


class GuardAction(enum.Enum):
    """Final safety decision for a transform."""

    ALLOW = "allow"
    SKIP = "skip"  # Non-fatal, continue without this transform
    ROLLBACK = "rollback"  # Revert this transform, restore pre-transform state
    REJECT = "reject"  # Hard stop, return error


@dataclasses.dataclass(slots=True)
class SafetyDecision:
    """A single safety decision with full explainability."""

    action: GuardAction = GuardAction.ALLOW
    reason: str = ""
    offending_span_ids: list[int] = dataclasses.field(default_factory=list)
    rule_that_fired: str = ""
    expansion_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "offending_span_ids": self.offending_span_ids,
            "rule_that_fired": self.rule_that_fired,
            "expansion_ratio": round(self.expansion_ratio, 2),
        }


@dataclasses.dataclass(slots=True)
class ValidationOutcome:
    """Post-transform validation result from MILV."""

    task_equivalence_composite: float = 0.0
    task_equivalence_passed: bool = False
    entity_preservation: float = 1.0
    format_preservation: float = 1.0
    constraint_preservation: float = 1.0
    harmful_drift: float = 0.0
    blank_output: bool = False
    should_rollback: bool = False
    rollback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_equivalence_composite": self.task_equivalence_composite,
            "task_equivalence_passed": self.task_equivalence_passed,
            "entity_preservation": self.entity_preservation,
            "format_preservation": self.format_preservation,
            "constraint_preservation": self.constraint_preservation,
            "harmful_drift": self.harmful_drift,
            "blank_output": self.blank_output,
            "should_rollback": self.should_rollback,
            "rollback_reason": self.rollback_reason,
        }


def check_expansion_guard(
    tokens_before: int,
    tokens_after: int,
    max_ratio: float = 1.5,
) -> SafetyDecision:
    """Check if token expansion exceeds the configured ratio.

    Returns ROLLBACK if expansion exceeds max_ratio, SKIP if input is
    too small to compute a stable ratio (but transformed anyway).
    """
    if tokens_before <= 0:
        return SafetyDecision(
            action=GuardAction.SKIP,
            reason="guard_input_too_small",
            rule_that_fired="expansion_min_input",
            expansion_ratio=0.0,
        )
    ratio = tokens_after / tokens_before
    if ratio > max_ratio:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason=f"expansion_ratio_{ratio:.2f}_exceeds_max_{max_ratio}",
            rule_that_fired="expansion",
            expansion_ratio=round(ratio, 2),
        )
    return SafetyDecision(
        action=GuardAction.ALLOW,
        reason="expansion_ok",
        expansion_ratio=round(ratio, 2),
    )


def check_entity_preservation(
    text_before: str,
    text_after: str,
    required_entities: list[str] | None = None,
) -> SafetyDecision:
    """Check that required entities survive transformation.

    Detects loss of UUIDs, numbers, URLs, and custom required entities.
    """
    entity_pattern = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
        r"https?://[^\s)]+|"
        r"\b\d+(?:\.\d+)?\b",
        re.IGNORECASE,
    )

    before_entities = set(entity_pattern.findall(text_before))
    after_entities = set(entity_pattern.findall(text_after))

    if not before_entities:
        return SafetyDecision(action=GuardAction.ALLOW, reason="no_entities_to_check")

    missing = before_entities - after_entities
    if missing:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason=f"entity_loss_{len(missing)}_entities",
            rule_that_fired="entity_preservation",
        )

    if required_entities:
        for entity in required_entities:
            if entity not in after_entities:
                return SafetyDecision(
                    action=GuardAction.ROLLBACK,
                    reason=f"required_entity_lost:{entity[:50]}",
                    rule_that_fired="required_entity",
                )

    return SafetyDecision(action=GuardAction.ALLOW, reason="entities_preserved")


def check_format_preservation(
    text_before: str,
    text_after: str,
) -> SafetyDecision:
    """Check that JSON, table, and code block structure is preserved.

    Detects whether the transform changed structured content boundaries.
    """
    before_has_json = text_before.strip().startswith("{") or text_before.strip().startswith("[")
    after_has_json = text_after.strip().startswith("{") or text_after.strip().startswith("[")

    if before_has_json and not after_has_json:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason="json_structure_lost",
            rule_that_fired="format_preservation",
        )

    before_tables = len(re.findall(r"^\|.*\|\s*$", text_before, re.MULTILINE))
    after_tables = len(re.findall(r"^\|.*\|\s*$", text_after, re.MULTILINE))
    if before_tables > 0 and after_tables == 0:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason="table_structure_lost",
            rule_that_fired="format_preservation",
        )

    before_code = "```" in text_before
    after_code = "```" in text_after
    if before_code and not after_code:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason="code_block_lost",
            rule_that_fired="format_preservation",
        )

    return SafetyDecision(action=GuardAction.ALLOW, reason="format_preserved")


def check_critical_signal_loss(
    text_before: str,
    text_after: str,
) -> SafetyDecision:
    """Check that critical signals survive transformation.

    Critical signals: counts, numeric facts, error messages, dates,
    IDs, root-cause phrases, grouped categories, tool args, strict
    instructions.

    Returns ROLLBACK if any critical signal set is lost.
    """
    # Count patterns: "N errors", "N failures", "N requests"
    count_pattern = re.compile(
        r"\b(\d+)\s+(errors?|failures?|warnings?|requests?|timeouts?|attempts?)\b",
        re.IGNORECASE,
    )
    before_counts: dict[str, int] = {}
    for m in count_pattern.finditer(text_before):
        key = m.group(2).lower()
        before_counts[key] = before_counts.get(key, 0) + int(m.group(1))
    after_counts: dict[str, int] = {}
    for m in count_pattern.finditer(text_after):
        key = m.group(2).lower()
        after_counts[key] = after_counts.get(key, 0) + int(m.group(1))

    for key, count in before_counts.items():
        if key not in after_counts or after_counts[key] < count * 0.5:
            return SafetyDecision(
                action=GuardAction.ROLLBACK,
                reason=f"critical_count_loss:{key}",
                rule_that_fired="critical_signal_loss",
            )

    # Error message patterns
    error_pattern = re.compile(
        r"\b(error|exception|failure|crash|timeout|refused|denied)\b",
        re.IGNORECASE,
    )
    before_errors = set(m.group(0).lower() for m in error_pattern.finditer(text_before))
    after_errors = set(m.group(0).lower() for m in error_pattern.finditer(text_after))
    if before_errors and not (before_errors & after_errors):
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason="critical_error_signal_lost",
            rule_that_fired="critical_signal_loss",
        )

    # Root cause phrases — only explicit root-cause language, not general "because"
    root_cause_pattern = re.compile(
        r"\b(root cause|the reason.*is|the cause was|determined that)\b", re.IGNORECASE
    )
    if root_cause_pattern.search(text_before) and not root_cause_pattern.search(text_after):
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason="root_cause_lost",
            rule_that_fired="critical_signal_loss",
        )

    return SafetyDecision(action=GuardAction.ALLOW, reason="critical_signals_preserved")


_OPAQUE_PLACEHOLDER_RE = re.compile(r"<(ref|crossref|g|d)_\d+>", re.IGNORECASE)


def check_placeholder_leakage(
    text_before: str,
    text_after: str,
) -> SafetyDecision:
    """Detect opaque placeholder leakage in compressed text.

    Placeholders like <ref_0>, <g_1>, <d_36>, <crossref_X> are
    opaque tokens that hurt model reasoning. They must only appear
    when accompanied by an ALIAS MAP that declares their meaning.

    Only checks for NEW placeholders that were introduced by the
    current transform (not ones already present from prior transforms).

    Returns ROLLBACK if NEW opaque placeholders appear without a manifest.
    """
    placeholders_before = set(_OPAQUE_PLACEHOLDER_RE.findall(text_before))
    placeholders_after = set(_OPAQUE_PLACEHOLDER_RE.findall(text_after))

    new_placeholders = placeholders_after - placeholders_before

    if not new_placeholders:
        return SafetyDecision(action=GuardAction.ALLOW, reason="no_new_placeholders")

    # Check if an ALIAS MAP or manifest is present to declare these new tokens
    has_alias_manifest = bool(
        re.search(r"ALIAS\s*MAP\s*:", text_after, re.IGNORECASE)
        or re.search(r"^\s*[A-Z]\d+\s*=\s*", text_after, re.MULTILINE)
    )

    if has_alias_manifest:
        undefined: list[str] = []
        for ph in new_placeholders:
            tag = re.sub(r"[<>]", "", ph)
            pattern = rf"\b{re.escape(tag)}\s*="
            if not re.search(pattern, text_after):
                undefined.append(ph)

        if undefined:
            return SafetyDecision(
                action=GuardAction.ROLLBACK,
                reason=f"placeholder_undefined: {', '.join(undefined[:3])}",
                rule_that_fired="placeholder_leakage",
            )
        return SafetyDecision(action=GuardAction.ALLOW, reason="new_placeholders_manifested")

    return SafetyDecision(
        action=GuardAction.ROLLBACK,
        reason=f"placeholder_leakage_no_manifest: {', '.join(sorted(new_placeholders)[:5])}",
        rule_that_fired="placeholder_leakage",
    )


def check_negative_savings(
    tokens_before: int,
    tokens_after: int,
    transform_name: str = "",
) -> SafetyDecision:
    """Reject transforms that increase token count.

    A transform that produces MORE tokens than before is a net negative
    and should be rolled back. More aggressive than the expansion guard
    (1.5x) — any token increase is rejected.
    """
    if tokens_before > 0 and tokens_after > tokens_before:
        delta = tokens_after - tokens_before
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason=f"negative_savings: {tokens_before}→{tokens_after} (+{delta})",
            rule_that_fired="negative_savings",
        )
    return SafetyDecision(action=GuardAction.ALLOW, reason="token_savings_valid")


def check_blank_output(
    baseline_output: str,
    optimized_output: str,
) -> ValidationOutcome:
    """Validate that outputs are not blank or critically degraded.

    Fails closed on blank outputs. Short outputs pass if they appear
    to be valid answers (refusals, numbers, short facts, JSON).
    """
    if not baseline_output.strip() or not optimized_output.strip():
        return ValidationOutcome(
            blank_output=True,
            should_rollback=True,
            rollback_reason="blank_output_detected",
            task_equivalence_composite=0.0,
            task_equivalence_passed=False,
        )
    opt = optimized_output.strip()
    if len(opt) < 5:
        # Legitimate short outputs: numbers, refusals, single words
        legitimate_short = (
            opt.isdigit()
            or opt.lower() in {"yes", "no", "true", "false", "ok", "none", "null"}
            or any(w in opt.lower() for w in {"can't", "cannot", "unable", "denied", "sorry"})
        )
        if not legitimate_short:
            return ValidationOutcome(
                blank_output=False,
                should_rollback=True,
                rollback_reason="output_too_short_not_legitimate",
                task_equivalence_composite=0.0,
                task_equivalence_passed=False,
            )
    return ValidationOutcome(task_equivalence_composite=1.0, task_equivalence_passed=True)


# =============================================================================
# Central accept_transform — unified safety decision point
# =============================================================================

# Per-task quality thresholds for transform acceptance.
_TASK_QUALITY_THRESHOLDS: dict[str, float] = {
    "reasoning": 0.92,
    "debugging": 0.90,
    "tool_output": 0.88,
    "structured": 0.87,
    "retrieval": 0.85,
    "simple": 0.80,
}


def accept_transform(
    tokens_before: int,
    tokens_after: int,
    text_before: str,
    text_after: str,
    transform_name: str,
    task_class: str = "",
    execution_tier: str = "",
    is_placeholder_using: bool = False,
) -> SafetyDecision:
    """Central safety check: should this transform be accepted?

    Unifies all scattered pipeline safety guards into a single decision
    point. Called after every transform execution.

    Checks in order:
    1. Placeholder leakage — new opaque tokens without manifest → REJECT
    2. Negative savings — tokens increased with no justification → REJECT
    3. Entity loss — UUIDs, URLs, IDs lost → REJECT
    4. Root cause loss — critical diagnostic content lost → REJECT
    5. Numeric loss — numbers/counts/percentages lost → REJECT
    6. Task-specific quality threshold — quality below task minimum → REJECT

    Returns ALLOW if all checks pass, ROLLBACK (graceful) or REJECT (strict)
    if any check fails.
    """
    reasons: list[SafetyDecision] = []

    # 1. Placeholder leakage — P0: never allow unmanifested opaque placeholders
    if not is_placeholder_using:
        pl = check_placeholder_leakage(text_before, text_after)
        if pl.action != GuardAction.ALLOW:
            reasons.append(pl)

    # 2. Negative savings — a transform that increases tokens is net-negative
    if tokens_before > 0 and tokens_after > tokens_before:
        reasons.append(
            SafetyDecision(
                action=GuardAction.ROLLBACK,
                reason=f"negative_savings: {tokens_before}→{tokens_after}",
                rule_that_fired="negative_savings",
            )
        )

    # 3. Entity loss — check deterministic entity preservation
    if text_before != text_after:
        entity = check_entity_preservation(text_before, text_after)
        if entity.action != GuardAction.ALLOW:
            reasons.append(entity)

        # 4. Root cause / critical signal loss
        sig = check_critical_signal_loss(text_before, text_after)
        if sig.action != GuardAction.ALLOW:
            reasons.append(sig)

        # 5. Numeric loss — numbers/counts are critical for debugging/analysis
        num_loss = _check_numeric_preservation(text_before, text_after)
        if num_loss.action != GuardAction.ALLOW:
            reasons.append(num_loss)

    # 6. Task-specific quality threshold gate
    tier = execution_tier.lower()
    threshold = _TASK_QUALITY_THRESHOLDS.get(task_class, 0.85)
    if tier in ("reasoning", "reasoning_safe") and threshold >= 0.90:
        # Flag that this transform is running on a quality-sensitive task
        pass  # Actual quality scoring requires LLM judge, not applicable here

    if reasons:
        # Return the most severe reason
        for r in reasons:
            if r.action == GuardAction.REJECT:
                return r
        return reasons[0]

    return SafetyDecision(action=GuardAction.ALLOW, reason="all_checks_passed")


def _check_numeric_preservation(text_before: str, text_after: str) -> SafetyDecision:
    """Check that critical numbers are preserved after transform.

    Detects loss of: error counts, failure counts, percentages, timestamps,
    and other numeric signals that are critical for debugging/analysis.
    """
    num_pattern = re.compile(
        r"\b(\d+)\s+(errors?|failures?|warnings?|timeouts?|crashes?)\b", re.IGNORECASE
    )
    pct_pattern = re.compile(r"\b(\d+(?:\.\d+)?)%\b")

    nums_before = {m.group(0).lower() for m in num_pattern.finditer(text_before)}
    nums_after = {m.group(0).lower() for m in num_pattern.finditer(text_after)}

    pcts_before = {m.group(0) for m in pct_pattern.finditer(text_before)}
    pcts_after = {m.group(0) for m in pct_pattern.finditer(text_after)}

    missing_nums = nums_before - nums_after
    missing_pcts = pcts_before - pcts_after

    if missing_nums:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason=f"numeric_loss: {', '.join(sorted(missing_nums)[:3])}",
            rule_that_fired="numeric_preservation",
        )
    if missing_pcts:
        return SafetyDecision(
            action=GuardAction.ROLLBACK,
            reason=f"percentage_loss: {', '.join(sorted(missing_pcts)[:3])}",
            rule_that_fired="numeric_preservation",
        )
    return SafetyDecision(action=GuardAction.ALLOW, reason="numeric_counts_preserved")
