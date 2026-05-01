"""Semantic Importance Graph — SIG.

Models content as spans with edges representing dependencies. Produces
an importance graph used by the scheduler (RATS) and safety guard (PSG)
to determine which content must be preserved and which may be compressed.


SIG decides **what matters**.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(slots=True)
class SemanticSpan:
    """A contiguous segment of content with extracted features.

    Attributes:
        span_id: Unique identifier within the request (0-based index).
        text: The raw text of the span.
        start_char: Character offset of the span start in the original text.
        end_char: Character offset of the span end (exclusive).
        structure_type: Classification of the span's structure type
            (narrative, code, json, table, diff, log_line, phrase).
        frequency: How many times this or a near-duplicate span appears.
        position_weight: Importance based on position (first/last bias).
        entity_density: Density of UUIDs, numbers, URLs, names in the span.
        dependency_score: How many later spans reference or depend on this one.
        task_relevance: Relevance to the inferred task type (0–1).
        reasoning_signal: Whether the span carries reasoning-critical content.
        importance: Combined importance score (0–100).
        protected: Whether this span is above the protection threshold.
    """

    span_id: int
    text: str
    start_char: int = 0
    end_char: int = 0
    structure_type: str = "narrative"
    frequency: float = 0.0
    position_weight: float = 0.0
    entity_density: float = 0.0
    dependency_score: float = 0.0
    task_relevance: float = 0.0
    reasoning_signal: bool = False
    importance: float = 0.0
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text[:200],
            "start_char": self.start_char,
            "end_char": self.end_char,
            "structure_type": self.structure_type,
            "frequency": round(self.frequency, 2),
            "position_weight": round(self.position_weight, 2),
            "entity_density": round(self.entity_density, 2),
            "dependency_score": round(self.dependency_score, 2),
            "task_relevance": round(self.task_relevance, 2),
            "reasoning_signal": self.reasoning_signal,
            "importance": round(self.importance, 2),
            "protected": self.protected,
        }


@dataclasses.dataclass(slots=True)
class SemanticEdge:
    """A directed dependency between two spans."""

    source_id: int
    target_id: int
    weight: float = 1.0
    relation: str = "references"


@dataclasses.dataclass(slots=True)
class SemanticImportanceGraph:
    """A directed graph of content spans with importance scores.

    SIG output — consumed by RATS and PSG to make scheduling and
    safety decisions.
    """

    spans: list[SemanticSpan] = dataclasses.field(default_factory=list)
    edges: list[SemanticEdge] = dataclasses.field(default_factory=list)
    total_spans: int = 0
    protected_count: int = 0
    average_importance: float = 0.0

    @property
    def protected_spans(self) -> list[SemanticSpan]:
        return [s for s in self.spans if s.protected]

    @property
    def protected_span_ids(self) -> list[int]:
        return [s.span_id for s in self.spans if s.protected]

    def summary(self) -> dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "protected_count": self.protected_count,
            "average_importance": round(self.average_importance, 2),
            "protected_span_ids": sorted(self.protected_span_ids),
            "structure_types": sorted({s.structure_type for s in self.spans}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [s.to_dict() for s in self.spans],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "weight": e.weight,
                    "relation": e.relation,
                }
                for e in self.edges
            ],
            "summary": self.summary(),
        }
