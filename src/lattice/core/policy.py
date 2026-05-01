"""Policy controls for LATTICE.

Provides per-request optimization policy decisions that sit between the
pipeline and configuration. While `LatticeConfig` stores static settings,
`OptimizationPolicy` makes dynamic decisions based on:
- Request size and complexity
- Model-specific rules
- Per-request header overrides
- Token budgets and context limits

The policy layer can:
- Allow transforms to run normally
- Skip transforms that won't help
- Enforce hard limits (abort pipeline with error)
- Apply model-specific or request-specific overrides

Usage (internal, called by pipeline):
    policy = OptimizationPolicy(config)
    decision = policy.should_run("reference_sub", request, context)
    if decision == Allow()
        ... apply transform ...
    elif decision == Skip(reason="no_matchable_patterns")
        ... continue without error ...
    elif decision == Error(code="REQUEST_TOO_LARGE")
        ... abort pipeline ...
"""

from __future__ import annotations

import dataclasses
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.transport import Request

logger = structlog.get_logger()


# =============================================================================
# PolicyDecision — tagged union
# =============================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class Allow:
    """Transform is allowed to run."""
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class Skip:
    """Transform should be skipped (non-fatal)."""
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Reject:
    """Request violates a hard limit — abort pipeline."""
    code: str
    message: str
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)


PolicyDecision = Allow | Skip | Reject


# =============================================================================
# OptimizationPolicy
# =============================================================================

class OptimizationPolicy:
    """Per-request optimization policy engine.

    Makes dynamic decisions about which transforms should run,
    enforces request limits, and handles per-request overrides.

    Attributes:
        config: The LatticeConfig used for thresholds and defaults.
    """

    def __init__(self, config: LatticeConfig | None = None) -> None:
        """Initialize policy with configuration.

        Args:
            config: LatticeConfig instance. Defaults to auto() if None.
        """
        self.config = config or LatticeConfig.auto()
        self._log = logger.bind(module="optimization_policy")

    # ------------------------------------------------------------------
    # Core decision method
    # ------------------------------------------------------------------

    def should_run(
        self,
        transform_name: str,
        request: Request,
        context: TransformContext,
    ) -> PolicyDecision:
        """Decide whether a transform should run for this request.

        Algorithm:
        1. Check global config (is transform enabled?)
        2. Check request-level overrides (headers)
        3. Check model-specific rules
        4. Check budget constraints
        5. Transform-specific rules

        Args:
            transform_name: Name of the transform (e.g., "reference_sub").
            request: The current request.
            context: Mutable per-request state.

        Returns:
            Allow(), Skip(), or Reject().
        """
        # 1. Global config
        if not self.config.is_transform_enabled(transform_name):
            return Skip(reason="disabled_by_config")

        # 2. Request-level disable via header
        if context.session_state.get("x_lattice_disable_transforms"):
            disabled = set(
                context.session_state["x_lattice_disable_transforms"].split(",")
            )
            if transform_name in disabled or "all" in disabled:
                return Skip(reason="disabled_by_header")

        # 3. Budget check
        budget_decision = self._check_budget(request, context)
        if isinstance(budget_decision, Reject):
            return budget_decision

        # 4. Transform-specific rules
        return self._transform_specific(transform_name, request, context)

    # ------------------------------------------------------------------
    # Request-level limits
    # ------------------------------------------------------------------

    def check_request_limits(self, request: Request) -> PolicyDecision:
        """Enforce hard limits on the request before any transforms run.

        Checks:
        - Context size (total tokens) vs default_context_limit
        - Message count sanity
        - Request body size (if already parsed)

        Args:
            request: The incoming request.

        Returns:
            Allow if within limits, Reject if exceeded.
        """
        total_estimate = request.token_estimate
        limit = self.config.default_context_limit

        if total_estimate > limit:
            self._log.warning(
                "request_context_too_large",
                tokens=total_estimate,
                limit=limit,
            )
            return Reject(
                code="REQUEST_TOO_LARGE",
                message=f"Request estimate {total_estimate} tokens exceeds limit {limit}",
                detail={"tokens": total_estimate, "limit": limit},
            )

        if total_estimate > int(limit * 0.9):
            self._log.info(
                "request_near_limit",
                tokens=total_estimate,
                limit=limit,
            )

        return Allow()

    # ------------------------------------------------------------------
    # Transform-level decisions
    # ------------------------------------------------------------------

    def _transform_specific(
        self,
        transform_name: str,
        request: Request,
        _context: TransformContext,
    ) -> PolicyDecision:
        """Apply transform-specific policy rules.

        Note: Content detection is the transform's responsibility.
        POLICY only enforces governance rules:
        - Budget constraints
        - Model-specific overrides
        - Known incompatibilities

        We intentionally do NOT duplicate transform detection logic
        (no UUID scan, no JSON parse) because transforms already
        handle that internally and can return Ok(no-op).
        """
        # Model-specific overrides
        model_rules = self.model_transform_rules(request.model)
        if transform_name in model_rules:
            return Allow() if model_rules[transform_name] else Skip(
                reason=f"disabled_for_model_{request.model}")

        runtime_contract = request.metadata.get("_lattice_runtime_contract")
        if isinstance(runtime_contract, dict):
            skipped = runtime_contract.get("skipped_transforms", ())
            if isinstance(skipped, (list, tuple, set)) and transform_name in skipped:
                return Skip(reason="disabled_by_runtime_contract")

        return Allow()






    def _check_budget(
        self, request: Request, _context: TransformContext
    ) -> PolicyDecision:
        """Check per-request token budget constraints.

        Validates two things:
        1. Input token budget (messages + tools) against ``default_input_token_budget``.
        2. ``max_tokens`` lower bound against ``min_max_tokens``.
        """
        # 1. Input token budget check
        if self.config.default_input_token_budget is not None:
            current_estimate = request.token_estimate
            budget = self.config.default_input_token_budget
            if current_estimate > budget:
                return Reject(
                    code="BUDGET_EXCEEDED",
                    message=(
                        f"Request estimate {current_estimate} exceeds "
                        f"per-request input budget {budget}"
                    ),
                    detail={"tokens": current_estimate, "budget": budget},
                )

        # 2. max_tokens lower bound check
        if (
            request.max_tokens is not None
            and request.max_tokens < self.config.min_max_tokens
        ):
            return Reject(
                code="MAX_TOKENS_TOO_LOW",
                message=(
                    f"max_tokens ({request.max_tokens}) is below the minimum "
                    f"allowed ({self.config.min_max_tokens})"
                ),
                detail={
                    "max_tokens": request.max_tokens,
                    "min_allowed": self.config.min_max_tokens,
                },
            )

        return Allow()

    # ------------------------------------------------------------------
    # Model-specific rules
    # ------------------------------------------------------------------

    def model_transform_rules(self, model: str) -> dict[str, bool]:
        """Get model-specific transform overrides.

        Some models handle context differently:
        - Claude (anthropic): prefix optimization has different header
        - o1/o3 reasoning models: context limits differ
        - GPT-4o / GPT-4o-mini: different tokenizer

        Args:
            model: Model identifier string.

        Returns:
            Dict mapping transform name → enabled bool.
            Absence means "use default".
        """
        model_lower = model.lower()
        overrides: dict[str, bool] = {}

        if "claude" in model_lower:
            # Anthropic models use different caching headers
            overrides["prefix_opt"] = True  # still beneficial
            # Reference substitution still works
            overrides["reference_sub"] = True

        if model_lower.startswith("o1") or model_lower.startswith("o3"):
            # Reasoning models have different context handling
            overrides["tool_filter"] = True  # still useful

        if "mini" in model_lower:
            # Smaller models benefit from compression even more
            overrides["reference_sub"] = True
            overrides["tool_filter"] = True

        return overrides


# =============================================================================
# TransformConfig
# =============================================================================

@dataclasses.dataclass(slots=True)
class TransformConfig:
    """Per-transform runtime configuration.

    Carries mutable parameters for a specific transform instance,
    separate from global LatticeConfig.

    Attributes:
        name: Transform name.
        enabled: Is this instance enabled.
        priority: Execution order (lower = earlier).
        max_latency_ms: Abort if transform exceeds this latency.
        skip_reasons: Set of Skip reason strings that are acceptable.
    """

    name: str
    enabled: bool = True
    priority: int = 50
    max_latency_ms: int = 10
    skip_reasons: set[str] = dataclasses.field(
        default_factory=lambda: {"disabled_by_config", "no_matchable_patterns"}
    )

    def is_reason_acceptable(self, reason: str) -> bool:
        """Check if a skip reason is acceptable (no-op, not error)."""
        return reason in self.skip_reasons
