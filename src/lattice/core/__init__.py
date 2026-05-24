"""Core leaf primitives — config, context, errors, result, segmentation.

Anything higher-level (planning, IR, transforms, providers, telemetry, state,
cache, safety, pipeline, transport) lives in its own package. core/ depends
on nothing from those packages — it is a true leaf.
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
from lattice.core.result import (
    Err,
    Ok,
    Result,
    is_err,
    is_ok,
    unwrap,
    unwrap_err,
)
from lattice.core.segmentation import (
    SegmentKind,
    SemanticSegment,
    segment_request,
    segment_summary,
)
from lattice.pipeline.runner import ReversibleSyncTransform
from lattice.transport.types import (
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
    "Result",
    "Ok",
    "Err",
    "is_ok",
    "is_err",
    "unwrap",
    "unwrap_err",
    "SegmentKind",
    "SemanticSegment",
    "segment_request",
    "segment_summary",
    "Message",
    "Request",
    "Response",
    "Role",
    "Transform",
    "SyncTransform",
    "ReversibleSyncTransform",
]
