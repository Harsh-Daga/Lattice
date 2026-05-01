"""Formal downgrade telemetry categories for LATTICE.

Every non-ideal path (compatibility downgrade, optimization bypass, cache miss)
is classified into a DowngradeCategory so that /stats and response headers
report a consistent, searchable taxonomy.

Core rule
---------
A downgrade is acceptable only if it is visible and intentional.
A silent downgrade is a bug.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class DowngradeCategory(enum.Enum):
    """Taxonomy of every non-ideal runtime path."""

    # Transport compatibility
    BINARY_TO_JSON = "binary_to_json"
    DELTA_TO_FULL_PROMPT = "delta_to_full_prompt"
    HTTP2_TO_HTTP11 = "http2_to_http11"
    STREAM_RESUME_TO_FULL = "stream_resume_to_full"

    # Optimization bypass
    BATCHING_BYPASSED = "batching_bypassed"
    SPECULATION_BYPASSED = "speculation_bypassed"
    CACHE_ARBITRAGE_SKIPPED = "cache_arbitrage_skipped"
    TRANSFORM_SKIPPED = "transform_skipped"

    # Cache
    SEMANTIC_CACHE_EXACT_HIT = "semantic_cache_exact_hit"
    SEMANTIC_CACHE_APPROXIMATE_HIT = "semantic_cache_approximate_hit"
    SEMANTIC_CACHE_MISS = "semantic_cache_miss"
    SEMANTIC_CACHE_DISABLED = "semantic_cache_disabled"

    # Provider / routing
    PROVIDER_ROUTING_FAILURE = "provider_routing_failure"
    PROVIDER_CAPABILITY_MISMATCH = "provider_capability_mismatch"

    # Stall / health
    STREAM_STALL_DETECTED = "stream_stall_detected"
    TACC_DELAYED = "tacc_delayed"
    TACC_REJECTED = "tacc_rejected"


@dataclasses.dataclass(slots=True)
class DowngradeTelemetry:
    """Mutable accumulator for downgrade events.

    One instance lives on the proxy runtime and is queried by /stats.
    Individual requests may also carry a DowngradeSnapshot in metadata.
    """

    _counts: dict[str, int] = dataclasses.field(default_factory=dict)
    _reasons: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    def record(
        self,
        category: DowngradeCategory,
        reason: str = "",
        *,
        max_reasons_per_category: int = 10,
    ) -> None:
        """Record one downgrade event."""
        key = category.value
        self._counts[key] = self._counts.get(key, 0) + 1
        if reason:
            reasons = self._reasons.setdefault(key, [])
            if len(reasons) < max_reasons_per_category:
                reasons.append(reason)

    def snapshot(self) -> dict[str, Any]:
        """Return serializable telemetry for /stats."""
        return {
            "counts": dict(self._counts),
            "recent_reasons": {
                k: v[:5] for k, v in self._reasons.items() if v
            },
        }

    def reset(self) -> None:
        """Clear all accumulated telemetry."""
        self._counts.clear()
        self._reasons.clear()

    @property
    def total_events(self) -> int:
        return sum(self._counts.values())


@dataclasses.dataclass(slots=True)
class TransportOutcome:
    """Canonical transport negotiation state for a single request.

    Used to derive response headers, /stats entries, and benchmark records
    from one source of truth.
    """

    # Framing
    framing: str = "json"  # "native" | "json"
    framing_fallback_reason: str = ""

    # Delta encoding
    delta_mode: str = "bypassed"  # "delta" | "bypassed"
    delta_fallback_reason: str = ""

    # HTTP version
    http_version: str = ""  # "http/2" | "http/1.1"
    http_fallback_reason: str = ""

    # Semantic cache
    semantic_cache_status: str = ""  # "exact-hit" | "approximate-hit" | "miss" | "disabled"

    # Batching
    batching_status: str = ""  # "batched" | "bypassed"

    # Speculation
    speculative_status: str = ""  # "hit" | "miss" | "bypassed"

    # Stream resume
    stream_resumed: bool = False
    stream_resume_fallback_reason: str = ""

    # Generic fallback
    fallback_reason: str = ""

    def to_headers(self) -> dict[str, str]:
        """Produce response headers from this outcome.

        Header alias strategy
        ---------------------
        ``x-lattice-speculative-status`` is the canonical header emitted by
        this method.  ``x-lattice-speculative`` (without ``-status``) is a
        legacy alias maintained for backward compatibility and is added by
        ``build_routing_headers`` when ``used_speculative=True``.
        """
        headers: dict[str, str] = {}
        if self.framing:
            headers["x-lattice-framing"] = self.framing
        if self.delta_mode:
            headers["x-lattice-delta"] = self.delta_mode
        if self.http_version:
            headers["x-lattice-http-version"] = self.http_version
        if self.semantic_cache_status:
            headers["x-lattice-semantic-cache"] = self.semantic_cache_status
        if self.batching_status:
            headers["x-lattice-batching"] = self.batching_status
        if self.speculative_status:
            headers["x-lattice-speculative-status"] = self.speculative_status
        if self.fallback_reason:
            headers["x-lattice-fallback-reason"] = self.fallback_reason
        if self.stream_resumed:
            headers["x-lattice-stream-resumed"] = "true"
        if self.stream_resume_fallback_reason:
            headers["x-lattice-stream-resume-fallback-reason"] = self.stream_resume_fallback_reason
        return headers

    def to_stats(self) -> dict[str, Any]:
        """Produce a /stats-compatible dict from this outcome."""
        return {
            "framing": self.framing,
            "framing_fallback_reason": self.framing_fallback_reason,
            "delta_mode": self.delta_mode,
            "delta_fallback_reason": self.delta_fallback_reason,
            "http_version": self.http_version,
            "http_fallback_reason": self.http_fallback_reason,
            "semantic_cache_status": self.semantic_cache_status,
            "batching_status": self.batching_status,
            "speculative_status": self.speculative_status,
            "stream_resumed": self.stream_resumed,
            "stream_resume_fallback_reason": self.stream_resume_fallback_reason,
            "fallback_reason": self.fallback_reason,
        }

    def to_downgrade_categories(self) -> list[DowngradeCategory]:
        """Return the downgrade categories represented by this outcome.

        Each category is determined by its specific state field, not by a
        generic fallback_reason. This keeps classification precise and avoids
        false positives when a subsystem reports a non-ideal state for an
        unrelated reason.
        """
        categories: list[DowngradeCategory] = []
        if self.framing == "json" and self.framing_fallback_reason:
            categories.append(DowngradeCategory.BINARY_TO_JSON)
        if self.delta_mode == "bypassed" and self.delta_fallback_reason:
            categories.append(DowngradeCategory.DELTA_TO_FULL_PROMPT)
        if self.http_version == "http/1.1" and self.http_fallback_reason:
            categories.append(DowngradeCategory.HTTP2_TO_HTTP11)
        if self.stream_resumed and self.stream_resume_fallback_reason:
            categories.append(DowngradeCategory.STREAM_RESUME_TO_FULL)
        if self.semantic_cache_status == "miss":
            categories.append(DowngradeCategory.SEMANTIC_CACHE_MISS)
        if self.semantic_cache_status == "disabled":
            categories.append(DowngradeCategory.SEMANTIC_CACHE_DISABLED)
        if self.batching_status == "bypassed":
            categories.append(DowngradeCategory.BATCHING_BYPASSED)
        if self.speculative_status == "bypassed":
            categories.append(DowngradeCategory.SPECULATION_BYPASSED)
        return categories
