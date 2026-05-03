"""Exception hierarchy for LATTICE.

Every error LATTICE raises is either:
1. A TransformError (recoverable, returned as Result.Err)
2. A LatticeError subclass (unrecoverable, raised as exception)

This separation lets callers distinguish between:
- "Your request was processed with reduced compression" (graceful)
- "Your request could not be processed at all" (exceptional)
"""

from __future__ import annotations

import dataclasses
from typing import Any

# =============================================================================
# LatticeError (base)
# =============================================================================


class LatticeError(Exception):
    """Base exception for all LATTICE errors.

    All unrecoverable errors in LATTICE raise a subclass of LatticeError.
    Recoverable errors are returned as TransformError (Result.Err).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "LATTICE_INTERNAL_ERROR",
        request_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.detail = detail or {}

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.args[0]}"]
        if self.request_id:
            parts.append(f" (request_id={self.request_id})")
        return "".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, "
            f"message={self.args[0]!r}, "
            f"request_id={self.request_id!r})"
        )


# =============================================================================
# Configuration errors
# =============================================================================


class ConfigurationError(LatticeError):
    """Configuration is invalid, incomplete, or contradictory.

    Raised at startup. The proxy will not start until configuration
    errors are resolved.
    """

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="CONFIG_ERROR", detail=detail or {})


class ValidationError(LatticeError):
    """A value failed validation.

    Used when Pydantic validation catches invalid input.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field


# =============================================================================
# Session errors
# =============================================================================


class SessionError(LatticeError):
    """Base for session-related errors."""

    pass


class SessionNotFoundError(SessionError):
    """Session ID was provided but no matching session exists."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session '{session_id}' not found",
            code="SESSION_NOT_FOUND",
        )
        self.session_id = session_id


class SessionExpiredError(SessionError):
    """Session exists but has exceeded TTL."""

    def __init__(self, session_id: str, ttl_seconds: int) -> None:
        super().__init__(
            f"Session '{session_id}' expired after {ttl_seconds}s",
            code="SESSION_EXPIRED",
        )
        self.session_id = session_id
        self.ttl_seconds = ttl_seconds


class SessionStoreError(SessionError):
    """Session storage backend (Redis, disk) failed."""

    def __init__(self, message: str, *, backend: str = "unknown") -> None:
        super().__init__(message, code="SESSION_STORE_ERROR")
        self.backend = backend


# =============================================================================
# Pipeline errors
# =============================================================================


class PipelineError(LatticeError):
    """Base for pipeline errors."""

    pass


class TransformNotFoundError(PipelineError):
    """A transform name was referenced but not registered."""

    def __init__(self, transform_name: str) -> None:
        super().__init__(
            f"Transform '{transform_name}' not found in registry",
            code="TRANSFORM_NOT_FOUND",
        )
        self.transform_name = transform_name


class TransformCircularDependencyError(PipelineError):
    """Transforms specified circular dependency ordering."""

    def __init__(self, cycle: list[str]) -> None:
        cycle_str = " -> ".join(cycle + [cycle[0]])
        super().__init__(
            f"Circular transform dependency: {cycle_str}",
            code="TRANSFORM_CIRCULAR_DEPENDENCY",
        )
        self.cycle = cycle


# =============================================================================
# Proxy errors
# =============================================================================


class ProxyError(LatticeError):
    """Base for proxy errors."""

    pass


class ProviderTimeoutError(ProxyError):
    """Provider did not respond within timeout."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Provider '{provider}' timed out after {timeout_seconds}s",
            code="PROVIDER_TIMEOUT",
        )
        self.provider = provider
        self.timeout_seconds = timeout_seconds


class ProviderDetectionError(ProxyError):
    """Provider cannot be determined from request signals.

    Raised when no explicit provider field, header, path, auth heuristic,
    or model prefix is present to disambiguate the upstream.
    """

    pass


class ProviderError(ProxyError):
    """Provider returned an error response."""

    """Provider returned an error response."""

    def __init__(
        self,
        provider: str,
        status_code: int,
        message: str,
        *,
        response_body: str | None = None,
    ) -> None:
        super().__init__(
            f"Provider '{provider}' returned {status_code}: {message}",
            code="PROVIDER_ERROR",
        )
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body


class RequestTooLargeError(ProxyError):
    """Request body exceeded max_request_size_mb."""

    def __init__(self, size_mb: float, max_mb: int) -> None:
        super().__init__(
            f"Request {size_mb:.1f}MB exceeds max {max_mb}MB",
            code="REQUEST_TOO_LARGE",
        )
        self.size_mb = size_mb
        self.max_mb = max_mb


# =============================================================================
# TransformError (recoverable, returned as Result.Err)
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class TransformError:
    """Structured error for individual transform failures.

    This is NOT an exception — it is a value type returned from transforms.
    The pipeline decides whether to continue (graceful degradation) or abort.

    Attributes:
        transform: Name of the transform that failed.
        code: Machine-readable code (e.g., "REF_SUB_DUPLICATE_KEY").
        message: Human-readable description.
        detail: Arbitrary structured context.
    """

    transform: str
    code: str
    message: str
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.transform}:{self.code}] {self.message}"

    def with_detail(self, **kwargs: Any) -> TransformError:
        """Return a new TransformError with additional detail merged in."""
        new_detail = {**self.detail, **kwargs}
        return dataclasses.replace(self, detail=new_detail)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON responses."""
        return {
            "transform": self.transform,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }
