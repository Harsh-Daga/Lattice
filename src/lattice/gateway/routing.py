"""Production-grade provider routing — zero defaults, zero fallbacks, zero guessing.

Every provider routing decision is made by querying every registered adapter and
computing a confidence score.  Ambiguity is an error.  Silence is an error.
There is no "openai by default" anywhere in this module.

Architecture
------------
1.  :class:`RequestSignals` collects every explicit signal from the incoming request.
2.  Each :class:`ProviderAdapter` implements ``detect(signals) → DetectionResult``.
3.  :class:`ProviderRouter` queries ALL adapters, scores results, and enforces:
    * At least one adapter must report a non-zero confidence.
    * The highest-confidence result must be strictly greater than the second-highest.
    * If two adapters tie, :class:`ProviderAmbiguityError` is raised.
    * If no adapter matches, :class:`ProviderNotDetectedError` is raised.
4.  The router never inspects headers, paths, or auth directly — it delegates every
    signal to the adapter that owns the provider namespace.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

# =============================================================================
# Detection confidence levels — ordered, higher is stronger
# =============================================================================


class DetectionConfidence(enum.IntEnum):
    """How strongly an adapter believes the request belongs to it.

    Ordered so that ``EXPLICIT > AUTH > PATH > MODEL > NONE``.
    """

    NONE = 0
    MODEL = 1  # Model prefix matches (e.g. ``anthropic/claude-3``)
    PATH = 2  # Request path is provider-specific (e.g. ``/v1/messages``)
    AUTH = 3  # Auth header pattern matches (e.g. ``sk-ant-*``)
    EXPLICIT = 4  # Body field or header explicitly names the provider


# =============================================================================
# Detection result
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class DetectionResult:
    """The outcome of a single adapter's ``detect()`` call."""

    provider: str
    confidence: DetectionConfidence
    # Human-readable explanation of why this adapter matched (for debugging / logs)
    reason: str = ""
    # Arbitrary structured detail (e.g. the matched auth prefix, the model prefix)
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)


# =============================================================================
# Request signals — everything we know about the incoming request
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class RequestSignals:
    """Immutable snapshot of every signal available for provider detection."""

    # HTTP method (GET, POST, etc.)
    method: str = ""
    # Request path (e.g. ``/v1/chat/completions``)
    path: str = ""
    # Normalised request headers (lower-cased keys)
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    # Parsed request body (may be empty dict for GET / WebSocket)
    body: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Model string from body or query param (may be empty)
    model: str = ""

    @classmethod
    def from_request(
        cls,
        *,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        model: str = "",
    ) -> RequestSignals:
        """Build a :class:`RequestSignals` from raw request data.

        Headers are normalised to lower-case keys so that every adapter sees
        the same view and no canonical-casing bugs creep in.
        """
        normalised_headers = {k.lower(): v for k, v in headers.items()}
        return cls(
            method=method.upper(),
            path=path,
            headers=normalised_headers,
            body=body or {},
            model=model,
        )


# =============================================================================
# ProviderRouter — the single source of truth for provider resolution
# =============================================================================


class ProviderAmbiguityError(Exception):
    """Raised when two or more adapters claim the same request with equal confidence."""

    def __init__(self, results: list[DetectionResult]) -> None:
        providers = ", ".join(f"{r.provider} ({r.confidence.name}: {r.reason})" for r in results)
        super().__init__(
            f"Provider ambiguity: multiple adapters matched with equal confidence. "
            f"Matches: {providers}"
        )
        self.results = results


class ProviderNotDetectedError(Exception):
    """Raised when no adapter reports a non-zero confidence for the request."""

    def __init__(self, signals: RequestSignals) -> None:
        super().__init__(
            "No provider adapter matched the request. "
            "Provide an explicit signal: "
            "1) ``provider`` or ``provider_name`` body field, "
            "2) ``x-lattice-provider`` header, "
            "3) model prefix like ``provider/model``, "
            "4) provider-specific auth header, "
            "5) provider-specific path. "
            f"method={signals.method!r} path={signals.path!r} "
            f"headers={list(signals.headers)} model={signals.model!r}"
        )
        self.signals = signals


class ProviderRouter:
    """Routes incoming requests to the correct provider adapter.

    Zero defaults.  Zero fallbacks.  Every decision is scored and ambiguous
    decisions are rejected with a descriptive error.

    Usage::

        router = ProviderRouter(registry)
        result = router.resolve(signals)
        # result.provider == "anthropic"
        # result.confidence == DetectionConfidence.AUTH
        # result.reason == "Authorization header matches sk-ant-* pattern"
    """

    def __init__(self, registry: Any) -> None:
        """Initialize with a :class:`ProviderRegistry` instance."""
        self._registry = registry

    def resolve(self, signals: RequestSignals) -> DetectionResult:
        """Resolve the provider for *signals*.

        Queries every registered adapter, collects non-:data:`DetectionConfidence.NONE`
        results, and returns the unique highest-confidence match.

        Raises
        ------
        ProviderNotDetectedError
            When no adapter reports a match.
        ProviderAmbiguityError
            When the top two adapters have the same confidence level.
        """
        results: list[DetectionResult] = []
        for adapter in self._registry.iter_adapters():
            result = adapter.detect(signals)
            if result.confidence > DetectionConfidence.NONE:
                results.append(result)

        if not results:
            raise ProviderNotDetectedError(signals)

        # Sort by confidence descending, then alphabetically by provider for determinism
        results.sort(key=lambda r: (-r.confidence.value, r.provider))

        best = results[0]
        if len(results) >= 2 and results[1].confidence == best.confidence:
            # Tie at the top — ambiguity is an error
            tied = [r for r in results if r.confidence == best.confidence]
            raise ProviderAmbiguityError(tied)

        return best
