"""Core abstractions for LATTICE — leaf primitives only.

After Phase 1 of the v1.0.0 refactor, ``core/`` contains only config,
context, errors, result, transport (types — moves to ``transport/`` in
Phase 2), pipeline (moves to ``pipeline/`` in Phase 2), and the
relocated ``segmentation`` module.

The IR types have moved to ``lattice.ir``. The transport types and
``CompressorPipeline``/``ReversibleSyncTransform`` continue to be
re-exported here so the public Python surface remains stable across
phases — Phase 2 will update the underlying paths transparently.
"""

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import (
    ConfigurationError,
    LatticeError,
    ProviderError,
    ProviderTimeoutError,
    RequestTooLargeError,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionStoreError,
    TransformError,
    TransformNotFoundError,
    ValidationError,
)
from lattice.core.pipeline import CompressorPipeline, ReversibleSyncTransform
from lattice.core.result import Err, Ok, Result, is_err, is_ok, unwrap, unwrap_err
from lattice.core.segmentation import (
    SegmentKind,
    SemanticSegment,
    segment_request,
    segment_summary,
)
from lattice.core.transport import (
    Message,
    Request,
    Response,
    Role,
    SyncTransform,
    Transform,
)

__all__ = [
    "LatticeConfig",
    "TransformContext",
    # errors
    "ConfigurationError",
    "LatticeError",
    "ProviderError",
    "ProviderTimeoutError",
    "RequestTooLargeError",
    "SessionError",
    "SessionExpiredError",
    "SessionNotFoundError",
    "SessionStoreError",
    "TransformError",
    "TransformNotFoundError",
    "ValidationError",
    # result
    "Result",
    "Ok",
    "Err",
    "is_ok",
    "is_err",
    "unwrap",
    "unwrap_err",
    # transport types (re-export buffer; physical move = Phase 2)
    "Message",
    "Request",
    "Response",
    "Role",
    "Transform",
    "SyncTransform",
    # pipeline protocol (re-export buffer; physical move = Phase 2)
    "ReversibleSyncTransform",
    "CompressorPipeline",
    # segmentation (moved from transforms/semantic_segmenter.py in Phase 1)
    "SegmentKind",
    "SemanticSegment",
    "segment_request",
    "segment_summary",
]
