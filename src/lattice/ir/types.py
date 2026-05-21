"""Canonical Prompt IR — structured representation of a prompt.

The IR decomposes a prompt into typed sections containing typed spans.
Transforms operate on this structured representation rather than raw strings,
enabling:
- Semantic preservation: spans carry role + entity annotation
- Safe compression: only compressible spans are touched
- Quality gating: protected spans block dangerous transforms
- Serialization control: output format is separate from representation
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class SectionType(str, enum.Enum):
    SYSTEM = "system"
    TASK = "task"
    INSTRUCTION = "instruction"
    CONSTRAINTS = "constraints"
    CONTEXT = "context"
    DATA = "data"
    TOOL_OUTPUT = "tool_output"
    TOOL_SCHEMA = "tool_schema"
    JSON = "json"
    TABLE = "table"
    LOGS = "logs"
    CODE = "code"
    OUTPUT_FORMAT = "output_format"
    ERROR = "error"
    STACK_TRACE = "stack_trace"


class SpanRole(str, enum.Enum):
    REASONING = "reasoning"
    DATA = "data"
    BOILERPLATE = "boilerplate"
    CONSTRAINT = "constraint"
    DIAGNOSTIC = "diagnostic"
    SCHEMA = "schema"
    ENTITY = "entity"
    COUNT = "count"
    CAUSE = "cause"


@dataclasses.dataclass(slots=True)
class Span:
    """A contiguous segment of content with extracted features and role."""

    span_id: str
    text: str
    role: SpanRole
    section_type: SectionType
    entities: list[str] = dataclasses.field(default_factory=list)
    numbers: list[str] = dataclasses.field(default_factory=list)
    keys: list[str] = dataclasses.field(default_factory=list)
    structure: dict[str, Any] = dataclasses.field(default_factory=dict)
    protected: bool = False
    compressible: bool = False
    compression_modes_allowed: list[str] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def short_repr(self) -> str:
        role_flag = "P" if self.protected else "C" if self.compressible else "-"
        return f"[{self.span_id}|{role_flag}|{self.role.value[:4]}] {self.text[:80]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "role": self.role.value,
            "section_type": self.section_type.value,
            "entities": self.entities,
            "numbers": self.numbers,
            "keys": self.keys,
            "structure": self.structure,
            "protected": self.protected,
            "compressible": self.compressible,
            "compression_modes_allowed": self.compression_modes_allowed,
            "metadata": self.metadata,
        }


@dataclasses.dataclass(slots=True)
class Section:
    type: SectionType
    spans: list[Span]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def protected_count(self) -> int:
        return sum(1 for s in self.spans if s.protected)

    @property
    def compressible_count(self) -> int:
        return sum(1 for s in self.spans if s.compressible)

    @property
    def total_text(self) -> str:
        return "\n".join(s.text for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "span_count": len(self.spans),
            "protected": self.protected_count,
            "compressible": self.compressible_count,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclasses.dataclass(slots=True)
class PromptIR:
    sections: list[Section]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def total_spans(self) -> int:
        return sum(len(s.spans) for s in self.sections)

    @property
    def protected_spans(self) -> int:
        return sum(s.protected_count for s in self.sections)

    @property
    def compressible_spans(self) -> int:
        return sum(s.compressible_count for s in self.sections)

    @property
    def section_types(self) -> list[str]:
        return [s.type.value for s in self.sections]

    def protected_span_ids(self) -> list[str]:
        return [sp.span_id for sec in self.sections for sp in sec.spans if sp.protected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_count": len(self.sections),
            "total_spans": self.total_spans,
            "protected": self.protected_spans,
            "compressible": self.compressible_spans,
            "section_types": self.section_types,
            "sections": [s.to_dict() for s in self.sections],
        }

    def summary(self) -> dict[str, Any]:
        """Compact summary for transport metadata."""
        return {
            "sections": len(self.sections),
            "types": self.section_types,
            "spans": self.total_spans,
            "protected": self.protected_spans,
            "compressible": self.compressible_spans,
        }
