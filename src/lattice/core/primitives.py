"""core/primitives.py — Unified type system for LATTICE Runtime v2.

This module defines:
- PromptIR (immutable, the ONE canonical IR)
- Candidate (immutable search node)
- CandidateGraph (immutable search frontier)
- CandidateScore (canonical scoring)
- ExecutionNode (unit of execution)
- ExecutionPlan (immutable execution plan)

All types are frozen, hashable, and copy-on-write.
Transforms return new instances; they never mutate in place.
"""
from __future__ import annotations

import dataclasses
from typing import Any

# ---------------------------------------------------------------------------
# PromptIR primitives — immutable, hashable
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SpanV2:
    """Immutable span with copy-on-write mutation."""

    span_id: str
    text: str
    role: str = "data"
    section_type: str = "context"
    entities: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    structure: frozenset[tuple[str, Any]] = dataclasses.field(default_factory=frozenset)
    protected: bool = False
    compressible: bool = False
    compression_modes: tuple[str, ...] = ()
    metadata: frozenset[tuple[str, Any]] = dataclasses.field(default_factory=frozenset)

    def with_text(self, text: str) -> SpanV2:
        return dataclasses.replace(self, text=text)

    def with_structure(self, **kwargs: Any) -> SpanV2:
        items = set(self.structure)
        items.update(kwargs.items())
        return dataclasses.replace(self, structure=frozenset(items))

    def with_protection(self, protected: bool) -> SpanV2:
        return dataclasses.replace(self, protected=protected)

    def with_compressible(self, compressible: bool) -> SpanV2:
        return dataclasses.replace(self, compressible=compressible)

    def add_metadata(self, **kwargs: Any) -> SpanV2:
        items = set(self.metadata)
        items.update((k, freeze_value(v)) for k, v in kwargs.items())
        return dataclasses.replace(self, metadata=frozenset(items))


@dataclasses.dataclass(frozen=True, slots=True)
class SectionV2:
    """Immutable section containing immutable spans."""

    type: str
    spans: tuple[SpanV2, ...] = ()
    metadata: frozenset[tuple[str, Any]] = dataclasses.field(default_factory=frozenset)

    def with_span(self, index: int, span: SpanV2) -> SectionV2:
        lst = list(self.spans)
        if 0 <= index < len(lst):
            lst[index] = span
        return dataclasses.replace(self, spans=tuple(lst))

    def add_span(self, span: SpanV2) -> SectionV2:
        return dataclasses.replace(self, spans=(*self.spans, span))

    def with_spans(self, spans: tuple[SpanV2, ...]) -> SectionV2:
        return dataclasses.replace(self, spans=spans)


@dataclasses.dataclass(frozen=True, slots=True)
class PromptIRV2:
    """Immutable PromptIR — the ONE canonical IR."""

    sections: tuple[SectionV2, ...] = ()
    metadata: frozenset[tuple[str, Any]] = dataclasses.field(default_factory=frozenset)

    def with_section(self, index: int, section: SectionV2) -> PromptIRV2:
        lst = list(self.sections)
        if 0 <= index < len(lst):
            lst[index] = section
        return dataclasses.replace(self, sections=tuple(lst))

    def add_section(self, section: SectionV2) -> PromptIRV2:
        return dataclasses.replace(self, sections=(*self.sections, section))

    def with_sections(self, sections: tuple[SectionV2, ...]) -> PromptIRV2:
        return dataclasses.replace(self, sections=sections)

    def add_metadata(self, **kwargs: Any) -> PromptIRV2:
        items = set(self.metadata)
        items.update((k, freeze_value(v)) for k, v in kwargs.items())
        return dataclasses.replace(self, metadata=frozenset(items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [section_to_dict(section) for section in self.sections],
            "metadata": thaw_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptIRV2:
        return cls(
            sections=tuple(section_from_dict(sec) for sec in data.get("sections", [])),
            metadata=freeze_dict(data.get("metadata", {})),
        )

    @property
    def total_spans(self) -> int:
        return sum(len(s.spans) for s in self.sections)

    @property
    def protected_spans(self) -> int:
        return sum(
            1 for sec in self.sections for sp in sec.spans if sp.protected
        )

    @property
    def compressible_spans(self) -> int:
        return sum(
            1 for sec in self.sections for sp in sec.spans if sp.compressible
        )

    def section_types(self) -> list[str]:
        return [s.type for s in self.sections]

    def canonical_fingerprint(self) -> str:
        """Return a deterministic SHA-256 hash of the IR content.

        Used for replay hardening, trace verification, and cache keys.
        The fingerprint is stable across runs because it only hashes
        ordered, immutable fields — no timestamps, no random values.
        """
        import hashlib

        parts: list[str] = []
        for sec in self.sections:
            parts.append(f"SEC:{sec.type}")
            for sp in sorted(sec.spans, key=lambda s: s.span_id):
                parts.append(
                    f"SP:{sp.span_id}:{sp.role}:{sp.section_type}:"
                    f"P={int(sp.protected)}:C={int(sp.compressible)}:"
                    f"T={sp.text}"
                )
        for key in sorted(self.metadata):
            if isinstance(key, tuple):
                k_str = str(key[0])
                v_str = str(key[1])
            else:
                k_str = str(key)
                v_str = ""
            parts.append(f"META:{k_str}={v_str}")
        text = "\n".join(parts)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def serialize(self) -> str:
        """Serialize IR to LLM-readable text."""
        parts: list[str] = []
        for sec in self.sections:
            if sec.spans:
                parts.append("\n".join(sp.text for sp in sec.spans))
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Candidate primitives — immutable search nodes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateScore:
    """Canonical score for a candidate."""

    composite: float
    quality: float = 1.0
    cost_reduction: float = 0.0
    cache_gain: float = 0.0
    transport_gain: float = 0.0
    semantic_risk: float = 0.0
    latency_ms: float = 0.0
    instability_penalty: float = 0.0
    reason: str = ""

    @property
    def expected_utility(self) -> float:
        return (
            self.cost_reduction
            + self.cache_gain
            + self.transport_gain
            - self.semantic_risk
            - self.latency_ms / 1000.0
            - self.instability_penalty
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    """Immutable snapshot of an optimization state."""

    ir: PromptIRV2
    applied: tuple[str, ...] = ()
    metrics: frozenset[tuple[str, Any]] = dataclasses.field(
        default_factory=frozenset
    )
    provenance: frozenset[tuple[str, Any]] = dataclasses.field(
        default_factory=frozenset
    )

    def apply(self, transform_name: str, new_ir: PromptIRV2) -> Candidate:
        """Return NEW candidate, never mutate self."""
        return dataclasses.replace(
            self,
            ir=new_ir,
            applied=(*self.applied, transform_name),
            provenance=frozenset(
                {
                    *self.provenance,
                    ("parent_applied", self.applied),
                    ("depth", len(self.applied)),
                    ("transform", transform_name),
                }
            ),
        )

    def with_metric(self, key: str, value: Any) -> Candidate:
        items = set(self.metrics)
        items.add((key, value))
        return dataclasses.replace(self, metrics=frozenset(items))

    def score(self) -> CandidateScore:
        """Compute canonical score from metrics."""
        m = dict(self.metrics)

        tokens_before = m.get("tokens_before", 1)
        tokens_after = m.get("tokens_after", tokens_before)
        latency_ms = m.get("latency_ms", 0.0)
        quality_estimate = m.get("quality_estimate", 1.0)
        cache_gain = m.get("cache_gain", 0.0)
        transport_gain = m.get("transport_gain", 0.0)
        semantic_risk = m.get("semantic_risk", 0.0)
        instability = m.get("instability", 0.0)

        # Cost reduction: fraction of tokens saved
        cost_reduction = max(0.0, (tokens_before - tokens_after) / max(1, tokens_before))

        # Latency cost: penalize expensive transforms
        latency_cost = latency_ms / 1000.0

        composite = (
            quality_estimate
            + cost_reduction * 0.5
            + cache_gain * 0.2
            + transport_gain * 0.2
            - semantic_risk
            - latency_cost
            - instability
        )

        return CandidateScore(
            composite=round(composite, 4),
            quality=quality_estimate,
            cost_reduction=round(cost_reduction, 4),
            cache_gain=round(cache_gain, 4),
            transport_gain=round(transport_gain, 4),
            semantic_risk=round(semantic_risk, 4),
            latency_ms=round(latency_ms, 3),
            instability_penalty=round(instability, 4),
            reason="all_signals_preserved",
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateGraph:
    """Immutable search frontier."""

    beam: tuple[Candidate, ...] = ()
    step: int = 0

    def expand(self, new_beam: tuple[Candidate, ...]) -> CandidateGraph:
        return dataclasses.replace(self, beam=new_beam, step=self.step + 1)

    def top_k(self, k: int) -> CandidateGraph:
        scored = sorted(self.beam, key=lambda c: c.score().expected_utility, reverse=True)
        return dataclasses.replace(self, beam=tuple(scored[:k]))

    def prune(self, min_score: float) -> CandidateGraph:
        kept = tuple(c for c in self.beam if c.score().expected_utility >= min_score)
        return dataclasses.replace(self, beam=kept)

    @property
    def best(self) -> Candidate | None:
        if not self.beam:
            return None
        return max(self.beam, key=lambda c: c.score().expected_utility)


# ---------------------------------------------------------------------------
# Execution primitives
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CachePlan:
    use_delta: bool = False
    compression_codec: str | None = None
    provider_cache_hint: str | None = None
    stable_prefix_hash: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class TransportPlan:
    use_delta: bool = False
    compression_codec: str | None = None
    provider_cache_hint: str | None = None
    stable_prefix_hash: str | None = None

    @property
    def delta_encoding(self) -> bool:
        return self.use_delta


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable execution plan produced by UnifiedPlanner."""

    transforms: tuple[str, ...] = ()
    quality_floor: float = 0.85
    latency_budget_ms: float = 100.0
    utility_score: float = 0.0
    beam_width: int = 5
    max_depth: int = 6
    cache_plan: CachePlan | None = None
    transport_plan: TransportPlan | None = None
    provider: str = "generic"
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transforms": list(self.transforms),
            "quality_floor": self.quality_floor,
            "latency_budget_ms": self.latency_budget_ms,
            "utility_score": self.utility_score,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "cache_plan": None
            if self.cache_plan is None
            else {
                "use_delta": self.cache_plan.use_delta,
                "compression_codec": self.cache_plan.compression_codec,
                "provider_cache_hint": self.cache_plan.provider_cache_hint,
                "stable_prefix_hash": self.cache_plan.stable_prefix_hash,
            },
            "transport_plan": None
            if self.transport_plan is None
            else {
                "use_delta": self.transport_plan.use_delta,
                "compression_codec": self.transport_plan.compression_codec,
                "provider_cache_hint": self.transport_plan.provider_cache_hint,
                "stable_prefix_hash": self.transport_plan.stable_prefix_hash,
            },
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        cache = data.get("cache_plan")
        transport = data.get("transport_plan")
        return cls(
            transforms=tuple(data.get("transforms", [])),
            quality_floor=float(data.get("quality_floor", 0.85)),
            latency_budget_ms=float(data.get("latency_budget_ms", 100.0)),
            utility_score=float(data.get("utility_score", 0.0)),
            beam_width=int(data.get("beam_width", 5)),
            max_depth=int(data.get("max_depth", 6)),
            cache_plan=None
            if cache is None
            else CachePlan(
                use_delta=bool(cache.get("use_delta", False)),
                compression_codec=cache.get("compression_codec"),
                provider_cache_hint=cache.get("provider_cache_hint"),
                stable_prefix_hash=cache.get("stable_prefix_hash"),
            ),
            transport_plan=None
            if transport is None
            else TransportPlan(
                use_delta=bool(transport.get("use_delta", transport.get("delta_encoding", False))),
                compression_codec=transport.get("compression_codec"),
                provider_cache_hint=transport.get("provider_cache_hint"),
                stable_prefix_hash=transport.get("stable_prefix_hash"),
            ),
            provider=str(data.get("provider", "generic")),
            model=str(data.get("model", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionNode:
    """Single unit of execution."""

    name: str
    priority: int
    config_flag: str = ""
    safety_bucket: str = "safe"

    def is_enabled(self, config: Any) -> bool:
        if not self.config_flag:
            return True
        return getattr(config, self.config_flag, True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def freeze_dict(d: dict[str, Any]) -> frozenset[tuple[str, Any]]:
    """Convert a dict to an immutable frozenset of items (deep-freezes lists/tuples)."""
    items: set[tuple[str, Any]] = set()
    for k, v in d.items():
        items.add((k, freeze_value(v)))
    return frozenset(items)


def freeze_value(value: Any) -> Any:
    """Recursively convert common container types into hashable equivalents."""
    if isinstance(value, dict):
        return freeze_dict(value)
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


def thaw_dict(items: frozenset[tuple[str, Any]]) -> dict[str, Any]:
    """Convert frozenset back to dict."""
    return dict(items)


def thaw_value(value: Any) -> Any:
    """Recursively thaw frozen container values into plain Python containers."""
    if isinstance(value, frozenset):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {k: thaw_value(v) for k, v in value}
        return [thaw_value(item) for item in value]
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, dict):
        return {k: thaw_value(v) for k, v in value.items()}
    return value


def section_to_dict(section: SectionV2) -> dict[str, Any]:
    return {
        "type": section.type,
        "spans": [span_to_dict(span) for span in section.spans],
        "metadata": thaw_dict(section.metadata),
    }


def span_to_dict(span: SpanV2) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "text": span.text,
        "role": span.role,
        "section_type": span.section_type,
        "entities": list(span.entities),
        "numbers": list(span.numbers),
        "keys": list(span.keys),
        "structure": thaw_value(span.structure),
        "protected": span.protected,
        "compressible": span.compressible,
        "compression_modes": list(span.compression_modes),
        "metadata": thaw_dict(span.metadata),
    }


def section_from_dict(data: dict[str, Any]) -> SectionV2:
    return SectionV2(
        type=str(data.get("type", "context")),
        spans=tuple(span_from_dict(sp) for sp in data.get("spans", [])),
        metadata=freeze_dict(data.get("metadata", {})),
    )


def span_from_dict(data: dict[str, Any]) -> SpanV2:
    return SpanV2(
        span_id=str(data.get("span_id", "0")),
        text=str(data.get("text", "")),
        role=str(data.get("role", "data")),
        section_type=str(data.get("section_type", "context")),
        entities=tuple(str(item) for item in data.get("entities", [])),
        numbers=tuple(str(item) for item in data.get("numbers", [])),
        keys=tuple(str(item) for item in data.get("keys", [])),
        structure=freeze_dict(data.get("structure", {})),
        protected=bool(data.get("protected", False)),
        compressible=bool(data.get("compressible", False)),
        compression_modes=tuple(str(item) for item in data.get("compression_modes", [])),
        metadata=freeze_dict(data.get("metadata", {})),
    )


def prompt_ir_v2_from_legacy(legacy_ir: Any) -> PromptIRV2:
    """Convert legacy PromptIR to immutable PromptIRV2."""
    sections: list[SectionV2] = []
    for sec in getattr(legacy_ir, "sections", []):
        spans: list[SpanV2] = []
        for sp in getattr(sec, "spans", []):
            spans.append(
                SpanV2(
                    span_id=getattr(sp, "span_id", "0"),
                    text=getattr(sp, "text", ""),
                    role=getattr(sp, "role", "data"),
                    section_type=getattr(sec, "type", "context"),
                    entities=tuple(getattr(sp, "entities", [])),
                    numbers=tuple(getattr(sp, "numbers", [])),
                    keys=tuple(getattr(sp, "keys", [])),
                    structure=freeze_dict(getattr(sp, "structure", {})),
                    protected=getattr(sp, "protected", False),
                    compressible=getattr(sp, "compressible", False),
                    compression_modes=tuple(getattr(sp, "compression_modes_allowed", [])),
                    metadata=freeze_dict(getattr(sp, "metadata", {})),
                )
            )
        sections.append(
            SectionV2(
                type=getattr(sec, "type", "context"),
                spans=tuple(spans),
                metadata=freeze_dict(getattr(sec, "metadata", {})),
            )
        )
    return PromptIRV2(
        sections=tuple(sections),
        metadata=freeze_dict(getattr(legacy_ir, "metadata", {})),
    )


def prompt_ir_from_v2(v2_ir: PromptIRV2) -> Any:
    """Convert immutable PromptIRV2 back to legacy PromptIR for adapters."""
    from lattice.core.ir import PromptIR, Section, SectionType, Span, SpanRole

    sections: list[Section] = []
    for sec in v2_ir.sections:
        spans: list[Span] = []
        for sp in sec.spans:
            spans.append(
                Span(
                    span_id=sp.span_id,
                    text=sp.text,
                    role=SpanRole(sp.role),
                    section_type=SectionType(sec.type),
                    entities=list(sp.entities),
                    numbers=list(sp.numbers),
                    keys=list(sp.keys),
                    structure=thaw_value(sp.structure),
                    protected=sp.protected,
                    compressible=sp.compressible,
                    compression_modes_allowed=list(sp.compression_modes),
                    metadata=thaw_dict(sp.metadata),
                )
            )
        sections.append(
            Section(
                type=SectionType(sec.type),
                spans=spans,
                metadata=dict(sec.metadata),
            )
        )
    return PromptIR(sections=sections, metadata=dict(v2_ir.metadata))
