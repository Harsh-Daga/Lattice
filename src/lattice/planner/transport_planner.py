"""TransportPlanner — unified transport + protocol optimization.

Phase 8 — Integrate networking/transport into the same objective.

Connects delta, cache alignment, binary framing, multiplex, resume, tunnel,
stream reliability, and provider routing into one TransportPlan consumed by
the proxy server.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(slots=True)
class TransportPlan:
    """Complete transport-level strategy for a single request."""

    # --- Delta / session ---
    delta_mode: bool
    session_id: str | None
    base_sequence: int  # 0 = full request, >0 = delta

    # --- Wire format ---
    use_framing: bool
    compression_codec: str | None  # None, "dictionary", "gzip"
    wire_format: str  # "json", "binary", "sse"

    # --- Resilience ---
    resume_enabled: bool
    retry_budget: int  # max retries before fallback

    # --- Provider routing ---
    provider: str
    model: str
    route_tags: list[str]

    # --- Cache alignment ---
    cache_plan: list[dict[str, Any]]  # provider-specific cache annotations

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_mode": self.delta_mode,
            "session_id": self.session_id,
            "base_sequence": self.base_sequence,
            "use_framing": self.use_framing,
            "compression_codec": self.compression_codec,
            "wire_format": self.wire_format,
            "resume_enabled": self.resume_enabled,
            "retry_budget": self.retry_budget,
            "provider": self.provider,
            "model": self.model,
            "route_tags": self.route_tags,
            "cache_plan": self.cache_plan,
        }


def build_transport_plan(
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    base_sequence: int = 0,
    is_multiplex: bool = False,
    is_streaming: bool = False,
    estimated_tokens: int = 0,
    cache_plan: list[dict[str, Any]] | None = None,
    fallback_strategy: str = "retry",
) -> TransportPlan:
    """Build a transport plan from request context.

    Rules:
    - Delta only when session_id is present and base_sequence > 0
    - Framing for multiplex or binary-capable providers
    - Resume enabled for streaming sessions
    - Compression codec chosen based on payload size
    """
    delta_mode = session_id is not None and base_sequence > 0
    resume_enabled = is_streaming and session_id is not None

    # Provider-specific codec preference
    if provider in ("openai", "anthropic"):
        codec: str | None = None
        if estimated_tokens > 4000:
            codec = "dictionary"
        wire = "json"
        framing = is_multiplex
    elif provider in ("ollama", "ollama-cloud"):
        codec = None
        wire = "json"
        framing = False
    else:
        codec = None
        wire = "json"
        framing = is_multiplex

    retry = 3 if fallback_strategy == "retry" else 0

    return TransportPlan(
        delta_mode=delta_mode,
        session_id=session_id,
        base_sequence=base_sequence,
        use_framing=framing,
        compression_codec=codec,
        wire_format=wire,
        resume_enabled=resume_enabled,
        retry_budget=retry,
        provider=provider,
        model=model,
        route_tags=[provider, "production"],
        cache_plan=cache_plan or [],
    )


__all__ = ["TransportPlan", "build_transport_plan"]
