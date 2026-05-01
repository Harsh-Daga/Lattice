"""LATTICE protocol layer.

Canonical segments, manifests, delta operations, and provider-aware
cache planning.
"""

from lattice.protocol.boundaries import BoundaryType, SemanticBoundaryDetector
from lattice.protocol.content import (
    ContentPart,
    ImagePart,
    ImageSource,
    ImageSourceType,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    content_part_hash,
    content_parts_hash,
    content_to_parts,
    parts_from_dict_list,
    parts_to_dict_list,
    parts_to_str,
)
from lattice.protocol.manifest import (
    Manifest,
    apply_delta,
    build_manifest,
    canonicalize_segments,
    compute_anchor_hash,
    manifest_from_messages,
    manifest_summary,
    manifest_to_messages,
)
from lattice.protocol.multiplex import (
    MultiStreamMux,
    ReliabilityMode,
    Stream,
    StreamState,
    StreamType,
)
from lattice.protocol.reliability import SelectiveReliability
from lattice.protocol.segments import (
    Segment,
    SegmentType,
    build_messages_segment,
    build_segment,
    build_system_segment,
    build_tools_segment,
)

__all__ = [
    # Content
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageSource",
    "ImageSourceType",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    "content_to_parts",
    "parts_to_str",
    "parts_to_dict_list",
    "parts_from_dict_list",
    "content_part_hash",
    "content_parts_hash",
    # Segments
    "Segment",
    "SegmentType",
    "build_segment",
    "build_system_segment",
    "build_tools_segment",
    "build_messages_segment",
    # Manifest
    "Manifest",
    "build_manifest",
    "canonicalize_segments",
    "compute_anchor_hash",
    "manifest_from_messages",
    "manifest_summary",
    "manifest_to_messages",
    "apply_delta",
    # Multiplex
    "MultiStreamMux",
    "Stream",
    "StreamType",
    "StreamState",
    "ReliabilityMode",
    # Reliability
    "SelectiveReliability",
    # Boundaries
    "SemanticBoundaryDetector",
    "BoundaryType",
]
