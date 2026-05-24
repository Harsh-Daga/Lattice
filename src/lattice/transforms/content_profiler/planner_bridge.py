"""IR build, prefix canon, manifest, planner, SIG, and session metadata."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from lattice.core.context import (
    METADATA_KEY_PROTECTED_SPANS,
    METADATA_KEY_PROTOCOL_MANIFEST,
    METADATA_KEY_PROTOCOL_MANIFEST_SUMMARY,
    METADATA_KEY_RISK_SCORE,
    METADATA_KEY_SCHEDULE,
    METADATA_KEY_SIG,
    METADATA_KEY_SIG_SUMMARY,
    METADATA_KEY_TASK_CLASSIFICATION,
    TransformContext,
)
from lattice.core.segmentation import segment_request, segment_summary
from lattice.ir.builder import build_ir
from lattice.ir.normalizer import normalize_ir
from lattice.ir.primitives import PromptIRV2, freeze_value, prompt_ir_v2_from_legacy
from lattice.ir.semantic_graph import SemanticImportanceGraph, SemanticSpan
from lattice.planner.provider_strategy import (
    build_cache_plan_for_provider,
    simulate_provider_cache,
)
from lattice.planner.runtime_state import (
    get_canonical_request_value,
    get_canonical_state_value,
    persist_execution_plan_state,
)
from lattice.planner.unified_planner import SemanticProfile, UnifiedPlanner
from lattice.transforms.content_profiler.classifier import ContentProfile
from lattice.transforms.content_profiler.task_classifier_bridge import TaskClassification
from lattice.transport.serialization import message_to_dict
from lattice.transport.types import Request
from lattice.utils.validation import SemanticRiskScore

if TYPE_CHECKING:
    from lattice.transforms.content_profiler import ContentProfiler


def apply_content_profiling(
    profiler: ContentProfiler,
    request: Request,
    context: TransformContext,
    *,
    profile: ContentProfile,
    task: TaskClassification,
    strategy: dict[str, Any],
    risk_score: SemanticRiskScore,
) -> PromptIRV2:
    """Run planner bridge side effects; return canonical PromptIRV2."""
    sig = build_importance_graph(request)

    segments = segment_request(request)
    segment_meta = segment_summary(segments)
    context.session_state["_lattice_segments"] = [s.to_dict() for s in segments]
    context.session_state["_lattice_segment_summary"] = segment_meta

    from lattice.protocol.prefix_canonicalization import canonicalize_request_prefix

    provider = get_canonical_state_value(context, "_lattice_provider", "")
    previous_hash = get_canonical_state_value(context, "_prefix_hash")
    prefix_manifest = canonicalize_request_prefix(
        request, previous_hash=previous_hash, provider=provider
    )
    context.session_state["_prefix_hash"] = prefix_manifest.prefix_hash
    context.session_state["_prefix_manifest"] = prefix_manifest.to_dict()
    request.metadata["_prefix_manifest"] = prefix_manifest.to_dict()
    request.metadata["_prefix_hash"] = prefix_manifest.prefix_hash
    request.metadata["_cache_hit"] = prefix_manifest.cache_hit
    request.metadata["_prefix_tokens"] = prefix_manifest.prefix_tokens
    request.metadata["_suffix_tokens"] = prefix_manifest.suffix_tokens

    if not prefix_manifest.cache_hit:
        request.extra_headers["x-lattice-prefix-hash"] = prefix_manifest.prefix_hash[:16]
        if prefix_manifest.provider_hint == "anthropic":
            request.extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

    from lattice.protocol.manifest import manifest_from_messages, manifest_summary

    protocol_session_id = context.session_id or str(
        request.metadata.get("session_id") or "prompt_ir"
    )
    protocol_manifest = manifest_from_messages(
        session_id=protocol_session_id,
        messages=[message_to_dict(msg) for msg in request.messages],
        tools=request.tools,
        model=request.model,
        provider=context.provider or "generic",
    )
    protocol_summary = manifest_summary(protocol_manifest)
    protocol_payload = {
        "manifest": protocol_manifest.to_dict(),
        "summary": protocol_summary,
    }
    context.session_state[METADATA_KEY_PROTOCOL_MANIFEST] = protocol_manifest.to_dict()
    context.session_state[METADATA_KEY_PROTOCOL_MANIFEST_SUMMARY] = protocol_summary
    request.metadata[METADATA_KEY_PROTOCOL_MANIFEST] = protocol_manifest.to_dict()
    request.metadata[METADATA_KEY_PROTOCOL_MANIFEST_SUMMARY] = protocol_summary
    request.metadata["_lattice_manifest"] = protocol_manifest.to_dict()

    ir = normalize_ir(build_ir(request))
    request.metadata["_lattice_ir_summary"] = ir.summary()
    request.metadata["_lattice_ir_sections"] = ir.section_types
    if ir.protected_spans > 0:
        request.metadata[METADATA_KEY_PROTECTED_SPANS] = ir.protected_span_ids()

    plan = coerce_execution_plan(
        get_canonical_request_value(request, context, "_lattice_execution_plan")
    )
    if plan is None:
        profile_v2 = SemanticProfile(
            task_class=task.task_class,
            task_label=task.preferred_strategy,
            risk_total=int(risk_score.total),
            context_length=request.token_estimate,
            has_tool_calls=request.is_tool_conversation,
            is_streaming=request.stream,
            is_conservative=task.is_conservative,
            provider=context.provider or "generic",
            model=request.model,
        )
        plan = UnifiedPlanner().plan(request, profile_v2)

    schedule = derive_schedule_from_plan(request, task, risk_score, sig, plan)
    optimizer_schedule = derive_optimizer_schedule_from_plan(
        task=task,
        risk_total=risk_score.total,
        request=request,
        plan=plan,
    )

    state = context.get_transform_state(profiler.name)
    state["profile"] = profile.value
    state["strategy"] = strategy
    state["risk_score"] = risk_score.to_dict()
    state["task_class"] = task.to_dict()
    state["protected_spans"] = sig.protected_span_ids

    context.record_metric(profiler.name, "profile", profile.value)
    context.record_metric(profiler.name, "total_tokens", request.token_estimate)
    context.record_metric(profiler.name, "risk_score", risk_score.total)
    context.record_metric(profiler.name, "risk_level", risk_score.level)
    context.record_metric(profiler.name, "sig_total_spans", sig.total_spans)
    context.record_metric(profiler.name, "sig_protected", sig.protected_count)
    context.record_metric(profiler.name, "task_class", task.task_class.value)

    request.metadata["_lattice_profile"] = profile.value
    request.metadata["_lattice_strategy"] = strategy
    request.metadata[METADATA_KEY_RISK_SCORE] = risk_score.to_dict()
    request.metadata[METADATA_KEY_SIG] = sig.to_dict()
    request.metadata[METADATA_KEY_SIG_SUMMARY] = sig.summary()
    request.metadata[METADATA_KEY_PROTECTED_SPANS] = sig.protected_span_ids
    request.metadata[METADATA_KEY_TASK_CLASSIFICATION] = task.to_dict()
    request.metadata[METADATA_KEY_SCHEDULE] = schedule
    request.metadata["_lattice_plan_utility"] = getattr(plan, "utility_score", 0.0)

    context.session_state[METADATA_KEY_RISK_SCORE] = risk_score.to_dict()
    context.session_state[METADATA_KEY_SIG] = sig.to_dict()
    context.session_state[METADATA_KEY_SIG_SUMMARY] = sig.summary()
    context.session_state[METADATA_KEY_PROTECTED_SPANS] = sig.protected_span_ids
    context.session_state[METADATA_KEY_TASK_CLASSIFICATION] = task.to_dict()
    context.session_state["_lattice_provider"] = context.provider
    context.session_state["_lattice_model"] = request.model
    context.session_state["_lattice_schedule"] = schedule
    context.session_state["_lattice_plan_utility"] = getattr(plan, "utility_score", 0.0)
    context.session_state["_lattice_optimizer_schedule"] = optimizer_schedule
    request.metadata["_lattice_optimizer_schedule"] = optimizer_schedule

    cache_plan = get_canonical_request_value(request, context, "_lattice_cache_plan")
    if not isinstance(cache_plan, list):
        cache_plan = build_cache_plan_for_provider(
            context.provider or "generic",
            segment_count=len(request.messages),
            estimated_tokens=request.token_estimate,
        )
        request.metadata["_lattice_cache_plan"] = cache_plan
    cache_simulation = None
    if plan is not None:
        cache_simulation = simulate_provider_cache(
            context.provider or "generic",
            request.model,
            estimated_tokens=request.token_estimate,
            cache_plan=cache_plan,
            prefix_manifest=request.metadata.get("_prefix_manifest"),
        )
        persist_execution_plan_state(
            request,
            context,
            plan,
            cache_plan=cache_plan,
            cache_simulation=cache_simulation,
        )

    ir_v2 = prompt_ir_v2_from_legacy(ir).add_metadata(
        protocol=protocol_payload,
        _lattice_protocol_manifest=protocol_manifest.to_dict(),
        _lattice_protocol_manifest_summary=protocol_summary,
        _lattice_execution_plan=plan.to_dict() if plan is not None else {},
        _lattice_profile=profile.value,
        _lattice_strategy=strategy,
        _lattice_risk_score=risk_score.to_dict(),
        _lattice_sig=sig.to_dict(),
        _lattice_sig_summary=sig.summary(),
        _lattice_segments=[s.to_dict() for s in segments],
        _lattice_segment_summary=segment_meta,
        _prefix_manifest=prefix_manifest.to_dict(),
        _lattice_task_classification=task.to_dict(),
        _lattice_schedule=schedule,
        _lattice_plan_utility=getattr(plan, "utility_score", 0.0),
        _lattice_optimizer_schedule=optimizer_schedule,
        _lattice_cache_plan=freeze_value(cache_plan),
        _lattice_cache_simulation=freeze_value(
            cache_simulation.to_dict() if cache_simulation is not None else {}
        ),
    )
    request.metadata["_lattice_ir_v2_summary"] = {
        "sections": len(ir_v2.sections),
        "spans": ir_v2.total_spans,
        "protected": ir_v2.protected_spans,
        "compressible": ir_v2.compressible_spans,
    }
    request.metadata["_lattice_ir_v2"] = ir_v2
    context.session_state["_lattice_ir_v2"] = ir_v2
    return ir_v2


def derive_schedule_from_plan(
    request: Request,
    task: TaskClassification,
    risk_score: SemanticRiskScore,
    sig: SemanticImportanceGraph,
    plan: Any,
) -> dict[str, Any]:
    """Project the canonical plan into legacy scheduler metadata."""
    from lattice.transforms.registry import list_transform_names

    all_registered = list(list_transform_names())
    allowed = list(getattr(plan, "transforms", ()) or ())
    blocked = [name for name in all_registered if name not in allowed]
    schedule_entries = [
        {
            "name": name,
            "bucket": "safe" if name in allowed else "blocked",
            "allowed": name in allowed,
            "reason": "plan_selected" if name in allowed else "plan_excluded",
        }
        for name in all_registered
    ]
    return {
        "task_class": task.to_dict(),
        "risk_level": risk_score.level,
        "risk_total": risk_score.total,
        "blocked": blocked,
        "allowed": allowed,
        "allowed_optimizers": [name for name in allowed if name.endswith("_optimizer")],
        "protected_spans": sig.protected_count,
        "budget_ms": float(getattr(plan, "latency_budget_ms", task.budget_ms)),
        "budget_exhausted": False,
        "schedule": schedule_entries,
        "request_tokens": request.token_estimate,
    }


def derive_optimizer_schedule_from_plan(
    *,
    task: TaskClassification,
    risk_total: float,
    request: Request,
    plan: Any,
) -> dict[str, Any]:
    """Project the canonical plan into optimizer-level scheduling metadata."""
    allowed = [name for name in getattr(plan, "transforms", ()) if name.endswith("_optimizer")]
    blocked = {
        name: "plan_excludes"
        for name in (
            "representation_optimizer",
            "reference_optimizer",
            "tool_optimizer",
            "context_optimizer",
            "diagnostic_optimizer",
            "ir_structure_optimizer",
        )
        if name not in allowed
    }
    return {
        "tier": task.execution_tier.value,
        "latency_budget_ms": float(getattr(plan, "latency_budget_ms", task.budget_ms)),
        "quality_floor": float(getattr(plan, "quality_floor", 0.85)),
        "allowed_optimizers": allowed,
        "blocked_optimizers": blocked,
        "transport_enabled": request.stream or request.token_estimate > 2000,
        "cache_enabled": True,
    }


def coerce_execution_plan(plan: Any) -> Any | None:
    """Accept cached dict payloads or concrete plan objects."""
    if plan is None:
        return None
    if isinstance(plan, dict):
        try:
            from lattice.ir.primitives import ExecutionPlan as CoreExecutionPlan

            return CoreExecutionPlan.from_dict(plan)
        except Exception:
            try:
                from lattice.planner.execution_plan import ExecutionPlan as LegacyExecutionPlan

                return LegacyExecutionPlan.from_dict(plan)
            except Exception:
                return None
    return plan


def build_importance_graph(request: Request) -> SemanticImportanceGraph:
    """Build a semantic importance graph from a request."""
    text = "\n".join(msg.content or "" for msg in request.messages)
    if not text.strip():
        return SemanticImportanceGraph(total_spans=0)

    spans = _segment_spans(text)
    _extract_features(spans, text, request)
    _compute_importance(spans)
    _derive_protected(spans)

    importance_values = [s.importance for s in spans]
    avg_importance = sum(importance_values) / len(importance_values) if importance_values else 0.0

    return SemanticImportanceGraph(
        spans=spans,
        total_spans=len(spans),
        protected_count=sum(1 for s in spans if s.protected),
        average_importance=round(avg_importance, 2),
    )


def _segment_spans(text: str) -> list[SemanticSpan]:
    """Split text into spans by structure boundaries."""
    spans: list[SemanticSpan] = []
    pos = 0

    parts = re.split(r"(`{3}[\s\S]*?`{3})", text)
    for part in parts:
        if part.startswith("```"):
            spans.append(
                SemanticSpan(
                    span_id=len(spans),
                    text=part,
                    start_char=pos,
                    end_char=pos + len(part),
                    structure_type="code",
                )
            )
        elif part.strip():
            sub_spans = _segment_structured(part, pos)
            spans.extend(sub_spans)
        pos += len(part)

    for i, s in enumerate(spans):
        s.span_id = i

    return spans


def _segment_structured(text: str, offset: int) -> list[SemanticSpan]:
    """Segment non-code text into structured spans."""
    spans: list[SemanticSpan] = []
    pos = 0

    parts = re.split(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text)
    for part in parts:
        stripped = part.strip()
        if not stripped:
            pos += len(part)
            continue
        if stripped.startswith("{") or stripped.startswith("["):
            spans.append(
                SemanticSpan(
                    span_id=0,
                    text=part,
                    start_char=offset + pos,
                    end_char=offset + pos + len(part),
                    structure_type="json",
                )
            )
        elif stripped.startswith("|"):
            spans.append(
                SemanticSpan(
                    span_id=0,
                    text=part,
                    start_char=offset + pos,
                    end_char=offset + pos + len(part),
                    structure_type="table",
                )
            )
        else:
            sentences = re.split(r"((?<=[.!?])\s+)", stripped)
            for sent in sentences:
                if sent.strip():
                    spans.append(
                        SemanticSpan(
                            span_id=0,
                            text=sent,
                            start_char=offset + pos,
                            end_char=offset + pos + len(sent),
                            structure_type="narrative",
                        )
                    )
                    pos += len(sent)
        pos += len(part)

    return spans


def _extract_features(
    spans: list[SemanticSpan],
    full_text: str,
    request: Request,
) -> None:
    """Extract per-span features: frequency, entities, position, signals."""
    total_spans = len(spans)
    if total_spans == 0:
        return

    for i, span in enumerate(spans):
        span.frequency = max(1.0, full_text.count(span.text) or 1.0)

        if i == 0:
            span.position_weight = 1.0
        elif i == total_spans - 1:
            span.position_weight = 0.8
        elif i < total_spans * 0.2:
            span.position_weight = 0.6
        else:
            span.position_weight = 0.4

        span_text = span.text
        uuids = len(
            re.findall(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                span_text,
                re.IGNORECASE,
            )
        )
        numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", span_text))
        urls = len(re.findall(r"https?://[^\s)]+", span_text, re.IGNORECASE))
        span.entity_density = min(
            (uuids * 3 + numbers * 0.5 + urls * 2) / max(len(span_text.split()), 1), 1.0
        )

        if i < total_spans - 1:
            later_text = " ".join(s.text for s in spans[i + 1 :])
            span_words = set(span.text.lower().split()) - {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "to",
                "of",
                "in",
                "for",
                "and",
                "or",
            }
            if span_words:
                appearing = sum(1 for w in span_words if w in later_text.lower())
                span.dependency_score = min(appearing / len(span_words), 1.0)

        task_indicators = [
            "error",
            "failure",
            "root cause",
            "mitigation",
            "debug",
            "fix",
            "investigate",
            "analyze",
            "compare",
            "trend",
            "conclusion",
        ]
        span.task_relevance = min(
            sum(0.15 for ti in task_indicators if ti in span_text.lower()), 1.0
        )

        reasoning_markers = [
            "therefore",
            "thus",
            "hence",
            "consequently",
            "as a result",
        ]
        span.reasoning_signal = any(m in span_text.lower() for m in reasoning_markers)


def _compute_importance(spans: list[SemanticSpan]) -> None:
    """Compute importance score per span."""
    max_freq = max((s.frequency for s in spans), default=1.0)
    for span in spans:
        freq_norm = span.frequency / max_freq if max_freq > 0 else 0.0
        score = (
            0.25 * freq_norm
            + 0.20 * span.dependency_score
            + 0.20 * span.entity_density
            + 0.20 * span.task_relevance
            + 0.15 * span.position_weight
        )
        if span.reasoning_signal:
            score *= 1.5
        span.importance = min(round(score * 100, 1), 100.0)


def _derive_protected(spans: list[SemanticSpan]) -> None:
    """Contrastive protection: top-k by importance + hard force-protect rules."""
    if not spans:
        return

    ranked = sorted(spans, key=lambda s: s.importance, reverse=True)
    protected_count = max(1, int(0.20 * len(spans)))

    for span in ranked[:protected_count]:
        span.protected = True

    for span in spans:
        text_lower = span.text.lower()

        if (
            re.search(r"\b\d+(?:\.\d+)?\b", span.text)
            and len(re.findall(r"\b\d+(?:\.\d+)?\b", span.text)) >= 2
        ):
            span.protected = True

        if re.search(
            r"\broot cause\b|\bthe reason.*\bis\b|\bthe cause was\b|\bdetermined that\b", text_lower
        ):
            span.protected = True

        if re.search(r"\b(error|exception|failure|crash|timeout|refused|denied)\b", text_lower):
            span.protected = True

        if re.search(
            r"\b\d+\s+(errors|failures|warnings|requests|timeouts|attempts)\b", text_lower
        ):
            span.protected = True

        if re.search(r"\bat\s+\S+\s*\([^)]+:\d+\)|\bFile\s+\".+?\",\s+line\s+\d+", span.text):
            span.protected = True

        if span.reasoning_signal and span.position_weight >= 0.8:
            span.protected = True

        if re.search(r'"(call_id|tool_call_id|tool_use_id)"|"id"\s*:\s*"call_', span.text):
            span.protected = True

        if re.search(r"\breturn json\b|\boutput format\b|\btable format\b", text_lower):
            span.protected = True

    for span in spans:
        span.compressible = (
            not span.protected
            and span.structure_type in ("narrative", "log_line")
            and not span.reasoning_signal
        )


# Backward-compatible alias for tests importing private name.
_build_importance_graph = build_importance_graph

__all__ = [
    "_build_importance_graph",
    "apply_content_profiling",
    "build_importance_graph",
    "coerce_execution_plan",
    "derive_optimizer_schedule_from_plan",
    "derive_schedule_from_plan",
]
