"""Core abstractions for LATTICE.

This package defines the fundamental types and interfaces used throughout
the LATTICE system: Transport, Session, Compressor, Policy, and Metrics.
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
    "Message",
    "Request",
    "Response",
    "Role",
    "Transform",
    "SyncTransform",
    "TransformContext",
    "ReversibleSyncTransform",
    "CompressorPipeline",
]
