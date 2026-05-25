"""Atomic structural checks shared by guardrails, gates, and post_transform_guard."""

from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    score: float
    detail: str = ""


def numbers_preserved(before: str, after: str) -> CheckResult:
    before_numbers = set(re.findall(r"\b\d+\b", before))
    after_numbers = set(re.findall(r"\b\d+\b", after))
    overlap = len(before_numbers & after_numbers) / max(len(before_numbers), 1)
    passed = overlap >= 0.7 or not before_numbers
    return CheckResult(
        "numbers_preserved",
        passed,
        overlap,
        f"overlap={overlap:.2f}",
    )


def uuids_preserved(before: str, after: str) -> CheckResult:
    uuids = re.findall(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        before,
        re.IGNORECASE,
    )
    lost = sum(1 for u in uuids if u not in after)
    passed = lost == 0
    return CheckResult("uuids_preserved", passed, 1.0 - lost * 0.02, f"lost={lost}")


def urls_preserved(before: str, after: str) -> CheckResult:
    urls = re.findall(r"https?://[^\s)]+", before, re.IGNORECASE)
    lost = sum(1 for u in urls if u not in after)
    passed = lost == 0
    return CheckResult("urls_preserved", passed, 1.0 - lost * 0.02, f"lost={lost}")


def file_paths_preserved(before: str, after: str) -> CheckResult:
    paths = re.findall(r"(?:/[\w.-]+)+\.[\w]{1,8}\b", before)
    lost = sum(1 for p in paths if p not in after)
    passed = lost == 0
    return CheckResult("file_paths_preserved", passed, 1.0 - lost * 0.02, f"lost={lost}")


def error_signals_preserved(before: str, after: str, task_class: str) -> CheckResult:
    before_err = re.findall(r"\b(error|exception|failure|warning)\b", before, re.IGNORECASE)
    after_err = re.findall(r"\b(error|exception|failure|warning)\b", after, re.IGNORECASE)
    if not before_err:
        return CheckResult("error_signals_preserved", True, 1.0, "none")
    ratio = len(after_err) / max(len(before_err), 1)
    passed = ratio >= 0.5
    return CheckResult(
        "error_signals_preserved",
        passed,
        ratio,
        f"task={task_class}",
    )


def root_cause_phrases_preserved(before: str, after: str) -> CheckResult:
    patterns = [
        r"root cause",
        r"the cause was",
        r"the reason is",
        r"determined that",
    ]
    for pattern in patterns:
        if re.search(pattern, before, re.IGNORECASE) and not re.search(
            pattern, after, re.IGNORECASE
        ):
            return CheckResult("root_cause_phrases_preserved", False, 0.0, pattern)
    return CheckResult("root_cause_phrases_preserved", True, 1.0, "ok")


def placeholder_leakage(before: str, after: str, *, placeholder_aliasing_used: bool) -> CheckResult:
    if not placeholder_aliasing_used:
        return CheckResult("placeholder_leakage", True, 1.0, "n/a")
    opaque = re.findall(r"<(?:d_|g_|ref_)\d+>", after)
    if opaque and not re.search(r"ALIAS MAP", after, re.IGNORECASE):
        return CheckResult("placeholder_leakage", False, 0.0, "opaque_without_manifest")
    return CheckResult("placeholder_leakage", True, 1.0, "ok")


def expansion_within_ratio(before_tokens: int, after_tokens: int, ratio: float) -> CheckResult:
    if before_tokens <= 0:
        return CheckResult("expansion_within_ratio", True, 1.0, "empty")
    expansion = after_tokens / before_tokens
    passed = expansion <= ratio
    return CheckResult(
        "expansion_within_ratio",
        passed,
        1.0 / max(expansion, 1.0),
        f"expansion={expansion:.2f}",
    )


def negative_savings(before_tokens: int, after_tokens: int) -> CheckResult:
    passed = after_tokens <= before_tokens
    return CheckResult(
        "negative_savings",
        passed,
        1.0 if passed else 0.0,
        f"{before_tokens}->{after_tokens}",
    )
