"""Unit tests for the Result monad.

These tests verify that Ok/Err behave like Rust's Result type,
covering all public API functions and edge cases.
"""

import pytest

from lattice.core.result import (
    Err,
    Never,
    Ok,
    Result,
    is_err,
    is_ok,
    unwrap,
    unwrap_err,
    unwrap_or,
    unwrap_or_else,
)

# =============================================================================
# Ok
# =============================================================================


class TestOk:
    """Tests for the Ok variant."""

    def test_bool(self) -> None:
        """Ok is truthy."""
        ok: Result[int, str] = Ok(42)
        assert bool(ok) is True

    def test_unwrap(self) -> None:
        """unwrap returns the value."""
        ok: Result[int, str] = Ok(42)
        assert unwrap(ok) == 42

    def test_unwrap_err_raises(self) -> None:
        """unwrap_err on Ok raises RuntimeError."""
        ok: Result[int, str] = Ok(42)
        with pytest.raises(RuntimeError):
            unwrap_err(ok)

    def test_unwrap_or(self) -> None:
        """unwrap_or on Ok returns value, ignoring default."""
        ok: Result[int, str] = Ok(42)
        assert unwrap_or(ok, 999) == 42

    def test_unwrap_or_else(self) -> None:
        """unwrap_or_else on Ok returns value, ignoring function."""
        ok: Result[int, str] = Ok(42)
        assert unwrap_or_else(ok, lambda _: 999) == 42

    def test_map(self) -> None:
        """map transforms the value."""
        ok: Result[int, str] = Ok(21)
        doubled = ok.map(lambda x: x * 2)
        assert unwrap(doubled) == 42

    def test_map_err(self) -> None:
        """map_err on Ok is identity."""
        ok: Result[int, str] = Ok(42)
        assert unwrap(ok.map_err(lambda _: "ignored")) == 42

    def test_and_then(self) -> None:
        """and_then chains operations."""
        ok: Result[int, str] = Ok(21)
        chained = ok.and_then(lambda x: Ok(x * 2))
        assert unwrap(chained) == 42

    # Edge case: wrapping None
    def test_none_value(self) -> None:
        ok: Result[None, str] = Ok(None)
        assert unwrap(ok) is None


# =============================================================================
# Err
# =============================================================================


class TestErr:
    """Tests for the Err variant."""

    def test_bool(self) -> None:
        """Err is falsy."""
        err: Result[int, str] = Err("fail")
        assert bool(err) is False

    def test_unwrap_raises(self) -> None:
        """unwrap on Err raises RuntimeError."""
        err: Result[int, str] = Err("fail")
        with pytest.raises(RuntimeError):
            unwrap(err)

    def test_unwrap_err(self) -> None:
        """unwrap_err returns the error."""
        err: Result[int, str] = Err("fail")
        assert unwrap_err(err) == "fail"

    def test_unwrap_or(self) -> None:
        """unwrap_or on Err returns default."""
        err: Result[int, str] = Err("fail")
        assert unwrap_or(err, 999) == 999

    def test_unwrap_or_else(self) -> None:
        """unwrap_or_else on Err calls function."""
        err: Result[int, str] = Err("fail")
        assert unwrap_or_else(err, lambda _e: 999) == 999

    def test_map(self) -> None:
        """map on Err ignores transformation."""
        err: Result[int, str] = Err("fail")
        assert is_err(err.map(lambda x: x * 2))  # type: ignore[arg-type]

    def test_map_err(self) -> None:
        """map_err transforms the error."""
        err: Result[int, str] = Err("fail")
        transformed = err.map_err(lambda e: e.upper())
        assert unwrap_err(transformed) == "FAIL"

    def test_and_then(self) -> None:
        """and_then on Err short-circuits."""
        err: Result[int, str] = Err("fail")
        chained = err.and_then(lambda x: Ok(x * 2))  # type: ignore[arg-type]
        assert is_err(chained)


# =============================================================================
# Standalone functions
# =============================================================================


class TestFunctions:
    """Tests for convenience functions."""

    def test_is_ok(self) -> None:
        assert is_ok(Ok(42)) is True
        assert is_ok(Err("fail")) is False

    def test_is_err(self) -> None:
        assert is_err(Ok(42)) is False
        assert is_err(Err("fail")) is True


# =============================================================================
# Never type
# =============================================================================


class TestNever:
    """Tests for the bottom type."""

    def test_never_cannot_be_instantiated(self) -> None:
        """Never.__new__ raises TypeError."""
        with pytest.raises(TypeError):
            Never()


# =============================================================================
# Chaining
# =============================================================================


class TestChaining:
    """Tests for chaining multiple Results."""

    def test_ok_chain(self) -> None:
        """Ok + map + and_then produces final value."""
        result = (
            Ok[int, str](5).map(lambda x: x + 1).and_then(lambda x: Ok(x * 2)).map(lambda x: str(x))
        )
        assert unwrap(result) == "12"

    def test_err_breaks_chain(self) -> None:
        """Err early in chain propagates."""
        result = (
            Ok[int, str](5)
            .map(lambda x: x + 1)
            .and_then(lambda _: Err[int, str]("broke"))
            .map(lambda x: x * 100)  # type: ignore[arg-type]
        )
        assert is_err(result)


# =============================================================================
# repr
# =============================================================================


class TestRepr:
    """Tests for human-readable repr."""

    def test_ok_repr_short(self) -> None:
        assert "42" in repr(Ok(42))

    def test_ok_repr_truncated(self) -> None:
        long_value = "x" * 300
        r = repr(Ok(long_value))
        assert "truncated" in r

    def test_err_repr(self) -> None:
        assert "fail" in repr(Err("fail"))
