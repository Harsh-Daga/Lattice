"""Prompt cache arbitrage optimizer — formalizes prefix alignment for cache hit maximization.

Reorders messages to place stable content first (system → tools → static docs → variable
content), maximizing the probability of provider-side KV cache hits. Tracks hit/miss
metadata in session state for a feedback loop, and delegates to existing cache_planner
patterns when the protocol module is available.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response, Role


@dataclasses.dataclass(slots=True)
class CacheArbitrageOutcome:
    """Normalized outcome of cache arbitrage processing.

    Consolidates scattered metadata fields into one structured object
    that is stored on request.metadata["_cache_arbitrage_outcome"].
    """

    manifest_source: str = ""  # "injected" | "reconstructed" | "missing"
    plan_applied: bool = False
    plan_summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    expected_cached_tokens: int = 0
    breakpoints: list[Any] = dataclasses.field(default_factory=list)
    annotations: dict[str, Any] = dataclasses.field(default_factory=dict)
    stability_score: float = 0.0
    stable_tokens: int = 0
    skip_reason: str = ""
    planner_failure: dict[str, str] | None = None
    prefix_hash: str = ""
    reordered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_source": self.manifest_source,
            "plan_applied": self.plan_applied,
            "plan_summary": self.plan_summary,
            "expected_cached_tokens": self.expected_cached_tokens,
            "breakpoints": self.breakpoints,
            "annotations": self.annotations,
            "stability_score": self.stability_score,
            "stable_tokens": self.stable_tokens,
            "skip_reason": self.skip_reason,
            "planner_failure": self.planner_failure,
            "prefix_hash": self.prefix_hash,
            "reordered": self.reordered,
        }

try:
    from lattice.protocol.cache_planner import (
        CachePlan,
        get_cache_planner,
    )
    from lattice.protocol.manifest import Manifest
    from lattice.providers.capabilities import CacheMode, get_capability_registry

    _CACHE_PLANNER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CACHE_PLANNER_AVAILABLE = False


class CacheArbitrageOptimizer(ReversibleSyncTransform):
    """Formalize prefix alignment for provider cache hit maximization.

    Problem: Given a prompt P with mutable order, find ordering π
    that maximizes expected cached tokens subject to:
      - System prompt first
      - Tool definitions before messages
      - Variable content last
      - Cache block boundaries at stable positions
    """

    name = "cache_arbitrage"
    priority = 9  # After content_profiler, before reference_sub

    def __init__(self, track_hits: bool = True) -> None:
        self.track_hits = track_hits

    def process(
        self,
        request: Request,
        context: TransformContext,
    ) -> Result[Request, TransformError]:
        # Skip for tool/assistant tool_call conversations — ordering is structural
        if any(
            (m.tool_calls is not None and m.tool_calls) or m.tool_call_id is not None
            for m in request.messages
        ):
            return Ok(request)

        original_order = [
            (m.role.value if isinstance(m.role, Role) else m.role, m.content, m.name)
            for m in request.messages
        ]

        # Step 1 — classify messages into stability buckets
        system_msgs: list[Message] = []
        tool_msgs: list[Message] = []
        static_doc_msgs: list[Message] = []
        variable_msgs: list[Message] = []

        for msg in request.messages:
            role = msg.role.value if isinstance(msg.role, Role) else msg.role
            if role == "system":
                system_msgs.append(msg)
            elif role == "tool":
                tool_msgs.append(msg)
            elif role == "assistant" and msg.metadata.get("is_static_doc"):
                static_doc_msgs.append(msg)
            else:
                variable_msgs.append(msg)

        # Step 2 — canonical ordering: system → tools → static docs → variable
        ordered = system_msgs + tool_msgs + static_doc_msgs + variable_msgs
        request.messages = [m.copy() for m in ordered]

        # Step 3 — annotate static/stable content metadata on the new copies
        for msg in request.messages:
            role = msg.role.value if isinstance(msg.role, Role) else msg.role
            if role == "system" or role == "tool" or role == "assistant" and msg.metadata.get("is_static_doc"):
                msg.metadata.setdefault("_cache_stable", True)
            else:
                msg.metadata.setdefault("_cache_stable", False)

        # Initialize normalized outcome as the source of truth
        outcome = CacheArbitrageOutcome()

        # Step 4 — provider-specific cache annotations via cache_planner if available
        if _CACHE_PLANNER_AVAILABLE:
            self._apply_provider_cache_plan(request, context, outcome)
        else:
            outcome.skip_reason = "cache_planner_unavailable"

        # Step 5 — compute prefix stability score
        stable_token_count = sum(
            m.token_estimate
            for m in request.messages
            if m.metadata.get("_cache_stable")
        )
        total_tokens = request.token_estimate or 1
        stability_score = stable_token_count / total_tokens
        outcome.stability_score = stability_score
        outcome.stable_tokens = stable_token_count

        # Step 6 — track cache hit/miss for feedback loop
        state = context.get_transform_state(self.name)
        if self.track_hits:
            previous_hash = state.get("prefix_hash")
            current_hash = self._compute_prefix_hash(request)
            cache_hit = bool(previous_hash and previous_hash == current_hash)

            if not cache_hit:
                state["prefix_hash"] = current_hash
                state["miss_count"] = state.get("miss_count", 0) + 1
            else:
                state["hit_count"] = state.get("hit_count", 0) + 1

            context.record_metric(self.name, "cache_hit", cache_hit)
            context.record_metric(self.name, "hit_count", state.get("hit_count", 0))
            context.record_metric(self.name, "miss_count", state.get("miss_count", 0))

        # Save original order for reverse
        new_order = [
            (m.role.value if isinstance(m.role, Role) else m.role, m.content, m.name)
            for m in request.messages
        ]
        state["original_order"] = original_order
        state["reordered"] = new_order != original_order

        outcome.prefix_hash = state.get("prefix_hash", "")
        outcome.reordered = state.get("reordered", False)

        context.record_metric(self.name, "stability_score", stability_score)
        context.record_metric(self.name, "stable_tokens", stable_token_count)

        # Store normalized outcome as the authoritative payload
        request.metadata["_cache_arbitrage_outcome"] = outcome.to_dict()

        # Derive legacy metadata from outcome for backward compatibility
        legacy: dict[str, Any] = {}
        if outcome.manifest_source:
            legacy["manifest_source"] = outcome.manifest_source
        if outcome.plan_applied:
            legacy["plan_applied"] = outcome.plan_applied
        if outcome.plan_summary:
            legacy["plan_summary"] = outcome.plan_summary
        if outcome.expected_cached_tokens:
            legacy["expected_cached_tokens"] = outcome.expected_cached_tokens
        if outcome.breakpoints:
            legacy["breakpoints"] = outcome.breakpoints
        if outcome.annotations:
            legacy["annotations"] = outcome.annotations
        if outcome.stability_score:
            legacy["stability_score"] = outcome.stability_score
        if outcome.stable_tokens:
            legacy["stable_tokens"] = outcome.stable_tokens
        if outcome.skip_reason:
            legacy["skip_reason"] = outcome.skip_reason
        if outcome.planner_failure is not None:
            legacy["planner_failure"] = outcome.planner_failure
        request.metadata["_cache_arbitrage"] = legacy
        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        # Metadata-only transform; nothing to restore in response body.
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_prefix_hash(self, request: Request) -> str:
        """Compute a stable hash over the stable-prefix portion of the request."""
        parts: list[str] = []
        for msg in request.messages:
            if msg.metadata.get("_cache_stable"):
                role = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
                parts.append(f"{role}:{msg.content or ''}")
        if request.tools:
            import json
            parts.append(json.dumps(request.tools, sort_keys=True, ensure_ascii=True))
        text = "\n".join(parts)
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _apply_provider_cache_plan(
        self, request: Request, context: TransformContext, outcome: CacheArbitrageOutcome
    ) -> None:
        """Delegate to existing cache_planner patterns when importable.

        Populates the provided *outcome* object directly; legacy metadata is
        derived from the outcome afterward.
        """
        try:
            provider = context.provider or "generic"
            registry = get_capability_registry()
            capability = registry.get(provider)

            # Validate provider supports cache planning
            if capability is None:
                outcome.skip_reason = "provider_not_in_registry"
                outcome.planner_failure = None
                outcome.plan_applied = False
                context.record_metric(self.name, "plan_skipped", "provider_not_in_registry")
                return

            cache_mode = registry.cache_mode(provider)
            if cache_mode == CacheMode.NONE:
                outcome.skip_reason = "provider_no_cache_support"
                outcome.planner_failure = None
                outcome.plan_applied = False
                context.record_metric(self.name, "plan_skipped", "provider_no_cache_support")
                return

            # Idempotency check
            if outcome.plan_applied:
                context.record_metric(self.name, "plan_skipped", "already_applied")
                return

            planner = get_cache_planner(provider)
            manifest = self._resolve_manifest(request)
            plan: CachePlan = planner.plan(manifest)

            # Determine manifest source without mutating the Manifest object
            outcome.manifest_source = (
                "injected" if request.metadata.get("_lattice_manifest") is not None else "reconstructed"
            )

            # Populate outcome with plan info
            outcome.expected_cached_tokens = plan.expected_cached_tokens
            outcome.breakpoints = plan.breakpoints
            outcome.annotations = plan.annotations
            outcome.plan_summary = {
                "stable_prefix_tokens": plan.expected_cached_tokens,
                "breakpoint_count": len(plan.breakpoints),
                "cache_mode": cache_mode.value,
            }
            outcome.plan_applied = True
            outcome.skip_reason = ""
            outcome.planner_failure = None

            # Apply provider-facing cache controls where they are explicit and
            # supported by the adapter.
            cache = registry.cache_semantics(provider)
            if cache_mode == CacheMode.AUTO_PREFIX and cache.cache_key_param:
                request.metadata.setdefault(cache.cache_key_param, manifest.anchor_hash)
                if cache.retention_param and cache.default_ttl_seconds is not None:
                    request.metadata.setdefault(
                        cache.retention_param,
                        self._format_ttl_seconds(cache.default_ttl_seconds),
                    )
            elif cache_mode == CacheMode.EXPLICIT_BREAKPOINT:
                if provider == "anthropic":
                    request.metadata.setdefault("anthropic_cache_control", True)
                    if cache.default_ttl_seconds is not None:
                        request.metadata.setdefault(
                            "anthropic_cache_ttl_seconds",
                            cache.default_ttl_seconds,
                        )
                elif provider == "bedrock":
                    request.metadata.setdefault("bedrock_prompt_caching", True)
            elif cache_mode == CacheMode.EXPLICIT_CONTEXT:
                existing_cached_content = (
                    request.metadata.get("cachedContent")
                    or request.metadata.get("cached_content")
                    or request.metadata.get("gemini_cached_content")
                    or request.extra_body.get("cachedContent")
                    or request.extra_body.get("cached_content")
                )
                if existing_cached_content is not None:
                    request.metadata.setdefault("cachedContent", existing_cached_content)

            context.record_metric(self.name, "plan_applied", True)

        except ImportError as exc:
            outcome.planner_failure = {"category": "import_error", "detail": str(exc)}
            outcome.skip_reason = ""
            outcome.plan_applied = False
            context.record_metric(self.name, "planner_failure", "import_error")
        except ValueError as exc:
            outcome.planner_failure = {"category": "validation_error", "detail": str(exc)}
            outcome.skip_reason = ""
            outcome.plan_applied = False
            context.record_metric(self.name, "planner_failure", "validation_error")
        except Exception as exc:
            outcome.planner_failure = {"category": "unexpected", "exception": type(exc).__name__, "detail": str(exc)}
            outcome.skip_reason = ""
            outcome.plan_applied = False
            context.record_metric(self.name, "planner_failure", "unexpected")

    @staticmethod
    def _format_ttl_seconds(ttl_seconds: int) -> str:
        if ttl_seconds % 3600 == 0:
            hours = ttl_seconds // 3600
            return f"{hours}h"
        if ttl_seconds % 60 == 0:
            minutes = ttl_seconds // 60
            return f"{minutes}m"
        return f"{ttl_seconds}s"

    def _resolve_manifest(self, request: Request) -> Manifest:
        """Return a Manifest for cache planning, preferring injected over reconstructed."""
        injected = request.metadata.get("_lattice_manifest")
        if isinstance(injected, Manifest):
            return injected
        if isinstance(injected, dict):
            return Manifest.from_dict(injected)
        return self._request_to_manifest(request)

    def _request_to_manifest(self, request: Request) -> Manifest:
        """Build a Manifest from the current request for cache planning."""
        # Avoid direct import at top level to keep _CACHE_PLANNER_AVAILABLE guard clean
        from lattice.protocol.content import TextPart
        from lattice.protocol.manifest import build_manifest
        from lattice.protocol.segments import SegmentType, build_segment

        segments: list[Any] = []
        for msg in request.messages:
            role = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
            if role == "system":
                seg_type = SegmentType.SYSTEM
            elif role == "tool":
                seg_type = SegmentType.TOOLS
            else:
                seg_type = SegmentType.MESSAGES

            segments.append(
                build_segment(
                    segment_type=seg_type,
                    parts=[TextPart(text=msg.content or "")],
                    metadata=dict(msg.metadata),
                )
            )

        if request.tools:
            import json
            segments.insert(
                0,
                build_segment(
                    segment_type=SegmentType.TOOLS,
                    parts=[TextPart(text=json.dumps(request.tools, sort_keys=True))],
                ),
            )

        return build_manifest(
            session_id=request.metadata.get("session_id", "cache_arb"),
            segments=segments,
        )
