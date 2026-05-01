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


def check_blank_output(
    baseline_output: str,
    optimized_output: str,
) -> ValidationOutcome:
    """Validate that outputs are not blank or critically degraded.

    Returns a ValidationOutcome that fails closed on blank outputs.
    """
    if not baseline_output.strip() or not optimized_output.strip():
        return ValidationOutcome(
            blank_output=True,
            should_rollback=True,
            rollback_reason="blank_output_detected",
            task_equivalence_composite=0.0,
            task_equivalence_passed=False,
        )
    if len(optimized_output.strip()) < 10:
        return ValidationOutcome(
            blank_output=False,
            should_rollback=True,
            rollback_reason="output_too_short",
            task_equivalence_composite=0.0,
            task_equivalence_passed=False,
        )
    return ValidationOutcome(task_equivalence_composite=1.0, task_equivalence_passed=True)
