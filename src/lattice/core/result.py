"""Result type for explicit error handling without exceptions.

Inspired by Rust's Result<T, E>, this module provides a type-safe way to
handle operations that may fail, forcing callers to check for errors rather
than letting exceptions propagate silently.

Usage:
    >>> result = some_operation()
    >>> if is_ok(result):
    ...     value = unwrap(result)
    ... else:
    ...     error = unwrap_err(result)

    Or using match (Python 3.10+):
    >>> match result:
    ...     case Ok(value): print(value)
    ...     case Err(error): print(f"Error: {error}")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast, final

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Ok(Generic[T, E]):
    """A successful result containing a value of type T.

    This is the success variant of the Result type. It wraps a single value
    and provides operations that propagate success while ignoring error paths.

    Attributes:
        value: The successful computation result.
    """

    value: T

    def __bool__(self) -> bool:
        return True

    def map(self, op: Callable[[T], U]) -> Result[U, E]:
        """Apply a function to the contained value.

        Args:
            op: Function to apply to the success value.

        Returns:
            A new Ok with the transformed value.
        """
        return Ok(op(self.value))

    def map_err(self, _op: Callable[[E], Any]) -> Result[T, E]:
        """Ignore error transformation on success variant."""
        return Ok(self.value)

    def and_then(self, op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain another operation that returns a Result.

        Also known as flatmap or bind. This is the key operation for
        composing fallible operations without nesting Results.
        """
        return op(self.value)

    def unwrap(self) -> T:
        """Return the contained value.

        On Ok, this always succeeds.
        """
        return self.value

    def unwrap_err(self) -> Never:
        """Panic when called on Ok.

        Raises RuntimeError because there's no error to unwrap.
        """
        raise RuntimeError(
            "Called unwrap_err on an Ok value: "
            f"{_value_repr(self.value)}"
        )

    def unwrap_or(self, _default: U) -> T | U:
        """Return the contained value, ignoring the default."""
        return self.value

    def unwrap_or_else(self, _op: Callable[[E], U]) -> T | U:
        """Return the contained value, ignoring the fallback."""
        return self.value

    def expect(self, _msg: str) -> T:
        """Return the contained value."""
        return self.value

    def expect_err(self, msg: str) -> Never:
        """Panic with the given message."""
        raise RuntimeError(f"{msg}: {_value_repr(self.value)}")

    def __repr__(self) -> str:
        return f"Ok({_value_repr(self.value)})"


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Err(Generic[T, E]):
    """A failed result containing an error of type E.

    This is the error variant of the Result type. It wraps a single error
    and provides operations that propagate error while ignoring success paths.

    Attributes:
        error: The error that caused the failure.
    """

    error: E

    def __bool__(self) -> bool:
        return False

    def map(self, _op: Callable[[T], U]) -> Result[U, E]:
        """Ignore success transformation on error variant."""
        return cast(Result[U, E], Err(self.error))

    def map_err(self, op: Callable[[E], Any]) -> Result[T, Any]:
        """Apply a function to the contained error.

        Args:
            op: Function to apply to the error value.

        Returns:
            A new Err with the transformed error.
        """
        return Err(op(self.error))

    def and_then(self, _op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Ignore chained operation on error variant."""
        return cast(Result[U, E], Err(self.error))

    def unwrap(self) -> Never:
        """Panic when called on Err.

        Raises RuntimeError with the error message.
        """
        raise RuntimeError(
            "Called unwrap() on an Err value: "
            f"{_value_repr(self.error)}"
        )

    def unwrap_err(self) -> E:
        """Return the contained error."""
        return self.error

    def unwrap_or(self, default: U) -> U:
        """Return the default value."""
        return default

    def unwrap_or_else(self, op: Callable[[E], U]) -> U:
        """Compute and return a fallback value using the error."""
        return op(self.error)

    def expect(self, msg: str) -> Never:
        """Panic with the given message prepended to the error."""
        raise RuntimeError(f"{msg}: {_value_repr(self.error)}")

    def expect_err(self, _msg: str) -> E:
        """Return the contained error."""
        return self.error

    def __repr__(self) -> str:
        return f"Err({_value_repr(self.error)})"


# Result is a union type: Result[T, E] = Ok[T, E] | Err[T, E]
Result = Ok[T, E] | Err[T, E]


class Never:
    """The bottom type — a value that can never be constructed.

    Used for type safety in Result methods that should never return
    (like unwrap_err on Ok, which always panics).
    """

    def __new__(cls) -> Never:
        raise TypeError("Cannot instantiate Never type")


# =============================================================================
# Public API
# =============================================================================

def is_ok(result: Result[T, E]) -> bool:
    """Check if a Result is Ok.

    Args:
        result: The Result to check.

    Returns:
        True if the result is Ok, False if it's Err.

    Example:
        >>> is_ok(Ok(42))
        True
        >>> is_ok(Err("fail"))
        False
    """
    return isinstance(result, Ok)


def is_err(result: Result[T, E]) -> bool:
    """Check if a Result is Err.

    Args:
        result: The Result to check.

    Returns:
        True if the result is Err, False if it's Ok.
    """
    return isinstance(result, Err)


def unwrap(result: Result[T, E]) -> T:
    """Extract the value from a Result, panicking on Err.

    This is a convenience function equivalent to calling `.unwrap()`
    on the result directly.

    Args:
        result: The Result to unwrap.

    Returns:
        The success value.

    Raises:
        RuntimeError: If the result is Err.
    """
    return result.unwrap()  # type: ignore[return-value]


def unwrap_err(result: Result[T, E]) -> E:
    """Extract the error from a Result, panicking on Ok.

    Args:
        result: The Result to inspect.

    Returns:
        The error value.

    Raises:
        RuntimeError: If the result is Ok.
    """
    return result.unwrap_err()  # type: ignore[return-value]


def unwrap_or(result: Result[T, E], default: U) -> T | U:
    """Extract the value or return a default.

    Args:
        result: The Result to unwrap.
        default: Value to return if the result is Err.

    Returns:
        The success value, or default if error.
    """
    return result.unwrap_or(default)


def unwrap_or_else(result: Result[T, E], op: Callable[[E], U]) -> T | U:
    """Extract the value or compute a fallback from the error.

    Args:
        result: The Result to unwrap.
        op: Function to call with the error if Err.

    Returns:
        The success value, or op(error) if error.
    """
    return result.unwrap_or_else(op)


# =============================================================================
# Private helpers
# =============================================================================

def _value_repr(value: Any) -> str:
    """Safe repr that truncates long values."""
    s = repr(value)
    if len(s) > 200:
        s = s[:100] + "... [truncated] ..." + s[-50:]
    return s
