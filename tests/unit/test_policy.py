"""Comprehensive unit tests for OptimizationPolicy.

Covers all decision paths, edge cases, and transform-specific rules.
"""

from __future__ import annotations

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.policy import (
    Allow,
    OptimizationPolicy,
    Reject,
    Skip,
    TransformConfig,
)
from lattice.core.transport import Message, Request

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> LatticeConfig:
    return LatticeConfig()


@pytest.fixture
def policy() -> OptimizationPolicy:
    return OptimizationPolicy()


@pytest.fixture
def simple_request() -> Request:
    return Request(
        messages=[
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello!"),
        ],
        model="gpt-4",
    )


@pytest.fixture
def context() -> TransformContext:
    return TransformContext()


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    """Policy construction."""

    def test_default_initialization(self) -> None:
        p = OptimizationPolicy()
        assert p.config is not None

    def test_custom_config(self) -> None:
        config = LatticeConfig(default_context_limit=4096)
        p = OptimizationPolicy(config=config)
        assert p.config.default_context_limit == 4096


# =============================================================================
# should_run — basic checks
# =============================================================================


class TestShouldRun:
    """High-level should_run decisions."""

    def test_disabled_by_config(self, policy: OptimizationPolicy, simple_request: Request, context: TransformContext) -> None:
        """Transform disabled in config → Skip."""
        policy.config.transform_reference_sub = False
        result = policy.should_run("reference_sub", simple_request, context)
        assert isinstance(result, Skip)
        assert result.reason == "disabled_by_config"

    def test_enabled_transform(self, policy: OptimizationPolicy, context: TransformContext) -> None:
        """Enabled transform with matching content → Allow."""
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="The ID is 550e8400-e29b-41d4-a716-446655440000 and also 6ba7b810-9dad-11d1-80b4-00c04fd430c8. We have multiple UUIDs in this content that should be compressed.",
                ),
            ],
        )
        result = policy.should_run("reference_sub", req, context)
        assert isinstance(result, Allow)

    def test_disabled_by_header(self, policy: OptimizationPolicy, simple_request: Request, context: TransformContext) -> None:
        """Per-request header disabling transform."""
        context.session_state["x_lattice_disable_transforms"] = "reference_sub,tool_filter"
        result = policy.should_run("reference_sub", simple_request, context)
        assert isinstance(result, Skip)
        assert result.reason == "disabled_by_header"

    def test_disabled_all_by_header(self, policy: OptimizationPolicy, simple_request: Request, context: TransformContext) -> None:
        """Header value 'all' disables every transform."""
        context.session_state["x_lattice_disable_transforms"] = "all"
        result = policy.should_run("reference_sub", simple_request, context)
        assert isinstance(result, Skip)
        assert result.reason == "disabled_by_header"


# =============================================================================
# Request limits
# =============================================================================


class TestRequestLimits:
    """check_request_limits — hard limit enforcement."""

    def test_within_limits(self, policy: OptimizationPolicy) -> None:
        req = Request(messages=[Message(role="user", content="Short")])
        result = policy.check_request_limits(req)
        assert isinstance(result, Allow)

    def test_exceeds_context_limit(self) -> None:
        config = LatticeConfig(default_context_limit=10)
        p = OptimizationPolicy(config=config)
        # Each message: overhead 3 + content/4. 10 messages ~ >10 tokens
        req = Request(
            messages=[
                Message(role="user", content="This is a very long message with many tokens in it.")
            ] * 20
        )
        result = p.check_request_limits(req)
        assert isinstance(result, Reject)
        assert result.code == "REQUEST_TOO_LARGE"


# =============================================================================
# Transform-specific rules
# =============================================================================


class TestBudgetEnforcement:
    """Per-request token budget checks."""

    def test_no_budget_allowed(self, policy: OptimizationPolicy, simple_request: Request, context: TransformContext) -> None:
        """No budget configured → always allow."""
        result = policy._check_budget(simple_request, context)
        assert isinstance(result, Allow)

    def test_budget_exceeded(self) -> None:
        config = LatticeConfig(default_input_token_budget=5)
        p = OptimizationPolicy(config=config)
        # System + user message ~ >5 tokens total estimate
        req = Request(
            messages=[
                Message(role="system", content="System prompt here."),
                Message(role="user", content="User input with many tokens here."),
            ]
        )
        result = p._check_budget(req, TransformContext())
        assert isinstance(result, Reject)
        assert result.code == "BUDGET_EXCEEDED"


# =============================================================================
# Model-specific rules
# =============================================================================


class TestModelSpecificRules:
    """model_transform_rules returns correct overrides."""

    def test_claude_model(self, policy: OptimizationPolicy) -> None:
        rules = policy.model_transform_rules("claude-3-opus-20240229")
        assert "prefix_opt" in rules
        assert rules["prefix_opt"] is True

    def test_openai_model(self, policy: OptimizationPolicy) -> None:
        rules = policy.model_transform_rules("gpt-4-turbo")
        # No special overrides for GPT-4 Turbo
        assert "prefix_opt" not in rules or rules.get("prefix_opt", True)

    def test_reasoning_model(self, policy: OptimizationPolicy) -> None:
        rules = policy.model_transform_rules("o1-preview")
        assert "tool_filter" in rules
        assert rules["tool_filter"] is True


# =============================================================================
# TransformConfig
# =============================================================================


class TestTransformConfig:
    """TransformConfig dataclass helpers."""

    def test_default_config(self) -> None:
        tc = TransformConfig(name="test_transform")
        assert tc.enabled is True
        assert tc.priority == 50

    def test_is_reason_acceptable(self) -> None:
        tc = TransformConfig(name="test_transform")
        assert tc.is_reason_acceptable("disabled_by_config") is True
        assert tc.is_reason_acceptable("unknown_reason") is False

    def test_custom_skip_reasons(self) -> None:
        tc = TransformConfig(
            name="test_transform",
            skip_reasons={"custom_skip"},
        )
        assert tc.is_reason_acceptable("custom_skip") is True
        assert tc.is_reason_acceptable("disabled_by_config") is False
