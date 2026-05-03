"""Detection helpers shared across all provider adapters.

These are pure functions — no side effects, no network calls, no defaults.
Each helper returns a :class:`DetectionConfidence` and a reason string.
"""

from __future__ import annotations

import re

from lattice.gateway.routing import DetectionConfidence, DetectionResult, RequestSignals

# ---------------------------------------------------------------------------
# Explicit signal detection
# ---------------------------------------------------------------------------


def detect_explicit(
    signals: RequestSignals,
    provider: str,
    aliases: set[str] | None = None,
) -> DetectionResult:
    """Check for an explicit provider declaration in body or headers.

    Matches body fields ``provider`` / ``provider_name`` and header
    ``x-lattice-provider`` against *provider* and any *aliases*.
    """
    names = {provider}
    if aliases:
        names.update(aliases)

    for key in ("provider_name", "provider"):
        value = signals.body.get(key)
        if isinstance(value, str) and value.strip().lower() in names:
            return DetectionResult(
                provider=provider,
                confidence=DetectionConfidence.EXPLICIT,
                reason=f"body field '{key}' explicitly names '{value}'",
                detail={"field": key, "value": value},
            )

    header_value = signals.headers.get("x-lattice-provider")
    if header_value and header_value.strip().lower() in names:
        return DetectionResult(
            provider=provider,
            confidence=DetectionConfidence.EXPLICIT,
            reason=f"x-lattice-provider header explicitly names '{header_value}'",
            detail={"header": "x-lattice-provider", "value": header_value},
        )

    return DetectionResult(provider=provider, confidence=DetectionConfidence.NONE)


# ---------------------------------------------------------------------------
# Model-prefix detection
# ---------------------------------------------------------------------------


def detect_model_prefix(
    signals: RequestSignals,
    provider: str,
    aliases: set[str] | None = None,
) -> DetectionResult:
    """Check if the model string carries a ``provider/`` prefix."""
    if "/" not in signals.model:
        return DetectionResult(provider=provider, confidence=DetectionConfidence.NONE)

    prefix = signals.model.split("/", 1)[0].strip().lower()
    names = {provider}
    if aliases:
        names.update(aliases)

    if prefix in names:
        return DetectionResult(
            provider=provider,
            confidence=DetectionConfidence.MODEL,
            reason=f"model prefix '{prefix}/' identifies provider",
            detail={"model": signals.model, "prefix": prefix},
        )

    return DetectionResult(provider=provider, confidence=DetectionConfidence.NONE)


# ---------------------------------------------------------------------------
# Auth-pattern detection
# ---------------------------------------------------------------------------


def detect_auth_pattern(
    signals: RequestSignals,
    provider: str,
    pattern: re.Pattern[str],
    reason: str,
) -> DetectionResult:
    """Check if the ``authorization`` header matches a provider-specific regex."""
    auth = signals.headers.get("authorization", "")
    if auth and pattern.search(auth):
        return DetectionResult(
            provider=provider,
            confidence=DetectionConfidence.AUTH,
            reason=reason,
            detail={"authorization_prefix": auth[:30]},
        )
    return DetectionResult(provider=provider, confidence=DetectionConfidence.NONE)


# ---------------------------------------------------------------------------
# Header-presence detection (non-authorisation headers)
# ---------------------------------------------------------------------------


def detect_header_present(
    signals: RequestSignals,
    provider: str,
    header_name: str,
    reason: str,
) -> DetectionResult:
    """Check if a provider-specific non-auth header is present."""
    if signals.headers.get(header_name.lower()):
        return DetectionResult(
            provider=provider,
            confidence=DetectionConfidence.AUTH,
            reason=reason,
            detail={"header": header_name},
        )
    return DetectionResult(provider=provider, confidence=DetectionConfidence.NONE)


# ---------------------------------------------------------------------------
# Path detection
# ---------------------------------------------------------------------------


def detect_path(
    signals: RequestSignals,
    provider: str,
    paths: set[str],
    reason: str,
) -> DetectionResult:
    """Check if the request path matches a provider-specific endpoint."""
    normalized = signals.path.lower().rstrip("/")
    if normalized in paths:
        return DetectionResult(
            provider=provider,
            confidence=DetectionConfidence.PATH,
            reason=reason,
            detail={"path": signals.path},
        )
    return DetectionResult(provider=provider, confidence=DetectionConfidence.NONE)


# ---------------------------------------------------------------------------
# Composite: run multiple detectors and return the highest-confidence result
# ---------------------------------------------------------------------------


def highest_confidence(
    provider: str,
    *detectors: DetectionResult,
) -> DetectionResult:
    """Return the highest-confidence result among *detectors*.

    If no detector matched, returns a :data:`DetectionConfidence.NONE` result.
    """
    best: DetectionResult = DetectionResult(
        provider=provider, confidence=DetectionConfidence.NONE
    )
    for result in detectors:
        if result.confidence > best.confidence:
            best = result
    return best
