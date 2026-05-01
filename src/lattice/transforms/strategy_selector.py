"""Strategy Selection via LinUCB Contextual Bandit.

Adaptive compression strategy selection using a contextual bandit.
The bandit learns which compression strategy (arm) works best for different
request contexts, balancing exploration and exploitation.

Research basis: LinUCB (Li et al., 2010) achieves optimal regret bounds for
contextual bandits. The key insight: treat each strategy as an arm, extract
context features from the request, and learn a linear reward model per arm.

Arms:
    - "full": Full compression pipeline (all transforms).
    - "submodular": Facility-location-based subset selection.
    - "rd": Rate-distortion heuristic compression.
    - "hybrid": Adaptive hybrid combining submodular + rd.

Session state persistence: bandit matrices survive across requests in the
same session, enabling per-session learning.

Priority: 19 (runs after profilers, before compression).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from typing import Any

import structlog

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

logger = structlog.get_logger()

# =============================================================================
# Constants
# =============================================================================

_DEFAULT_ARMS: tuple[str, ...] = ("full", "submodular", "rd", "hybrid")
_DEFAULT_FEATURE_DIM: int = 8
_DEFAULT_ALPHA: float = 0.5


# =============================================================================
# LinUCB Arm State
# =============================================================================

@dataclasses.dataclass(slots=True)
class _ArmState:
    """Per-arm bandit state for LinUCB.

    Maintains the sufficient statistics for estimating the linear reward model:
        A (d×d): design matrix (A = X^T X + λI)
        b (d):   response vector (b = X^T y)
        θ (d):   parameter estimate (θ = A^{-1} b)

    Instead of storing full matrices, we store them as nested lists and
    compute updates via rank-1 Sherman-Morrison-style update (the matrix
    inversion lemma is implicit — we store A and solve for θ on demand).

    For d ≤ 10 this is very fast without numpy.
    """

    A: list[list[float]]  # d×d symmetric positive definite
    b: list[float]  # d-vector
    pulls: int = 0
    cum_reward: float = 0.0

    @classmethod
    def zero(cls, dim: int, lambda_reg: float = 1.0) -> _ArmState:
        """Create a fresh arm state with regularized identity matrix."""
        a_matrix = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            a_matrix[i][i] = lambda_reg
        return cls(A=a_matrix, b=[0.0] * dim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "A": copy.deepcopy(self.A),
            "b": list(self.b),
            "pulls": self.pulls,
            "cum_reward": self.cum_reward,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _ArmState:
        return cls(
            A=copy.deepcopy(data["A"]),
            b=list(data["b"]),
            pulls=data.get("pulls", 0),
            cum_reward=data.get("cum_reward", 0.0),
        )

    def update(self, features: list[float], reward: float) -> None:
        """Update arm state with a new observation.

        A ← A + x x^T
        b ← b + r x
        """
        dim = len(features)
        for i in range(dim):
            for j in range(dim):
                self.A[i][j] += features[i] * features[j]
            self.b[i] += reward * features[i]
        self.pulls += 1
        self.cum_reward += reward

    def theta(self) -> list[float]:
        """Return θ = A^{-1} b using Gaussian elimination (Cholesky-friendly).

        Since A is symmetric positive definite, we solve Ax = b directly.
        """
        dim = len(self.b)
        if dim == 0:
            return []

        # Create augmented matrix [A | b]
        aug: list[list[float]] = [
            [self.A[i][j] for j in range(dim)] + [self.b[i]]
            for i in range(dim)
        ]

        # Gaussian elimination with partial pivoting
        for col in range(dim):
            # Find pivot
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, dim):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < 1e-12:
                # Singular or near-singular — return zero vector
                return [0.0] * dim

            # Swap
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]

            # Eliminate below
            pivot = aug[col][col]
            for row in range(col + 1, dim):
                factor = aug[row][col] / pivot
                if factor == 0.0:
                    continue
                for j in range(col, dim + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = [0.0] * dim
        for i in range(dim - 1, -1, -1):
            total = aug[i][dim]
            for j in range(i + 1, dim):
                total -= aug[i][j] * x[j]
            if abs(aug[i][i]) < 1e-12:
                x[i] = 0.0
            else:
                x[i] = total / aug[i][i]

        return x

    def ucb_score(self, features: list[float], alpha: float) -> float:
        """Compute UCB score: θ^T x + α √(x^T A^{-1} x).

        We solve A y = x to get A^{-1} x, then:
            x^T A^{-1} x = x^T y
        """
        dim = len(features)
        if dim == 0:
            return float("inf")

        theta = self.theta()

        # Exploitation: θ^T x
        exploit = sum(theta[i] * features[i] for i in range(dim))

        # Exploration: solve A y = x
        y = self._solve_ax_eq_b(features)
        x_ax = sum(features[i] * y[i] for i in range(dim))
        explore = alpha * math.sqrt(max(0.0, x_ax))

        return exploit + explore

    def _solve_ax_eq_b(self, rhs: list[float]) -> list[float]:
        """Solve A x = rhs for x using Gaussian elimination."""
        dim = len(rhs)
        aug: list[list[float]] = [
            [self.A[i][j] for j in range(dim)] + [rhs[i]]
            for i in range(dim)
        ]

        for col in range(dim):
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, dim):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < 1e-12:
                return [0.0] * dim
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            for row in range(col + 1, dim):
                factor = aug[row][col] / pivot
                if factor == 0.0:
                    continue
                for j in range(col, dim + 1):
                    aug[row][j] -= factor * aug[col][j]

        x = [0.0] * dim
        for i in range(dim - 1, -1, -1):
            total = aug[i][dim]
            for j in range(i + 1, dim):
                total -= aug[i][j] * x[j]
            if abs(aug[i][i]) < 1e-12:
                x[i] = 0.0
            else:
                x[i] = total / aug[i][i]
        return x


# =============================================================================
# StrategySelector
# =============================================================================

class StrategySelector(ReversibleSyncTransform):
    """LinUCB-based adaptive compression strategy selector.

    This meta-transform runs before compression engines and selects which
    compression strategy (arm) to apply based on request context features.

    State is stored in ``TransformContext.session_state`` under the key
    ``"strategy_selector"`` and survives across requests in the same session.

    The selected strategy is written to ``request.metadata["_lattice_strategy"]``
    as a string (arm name). Downstream transforms read this to decide their
    behavior:
        - ``"full"`` → run full pipeline
        - ``"submodular"`` → run SubmodularContextSelector (skips rd)
        - ``"rd"`` → run RateDistortionCompressor (skips submodular)
        - ``"hybrid"`` → run both submodular and rd

    Configurable Parameters
    -----------------------
    - arms: Tuple of arm names. Default: ("full", "submodular", "rd", "hybrid").
    - feature_dim: Dimension of context feature vector. Default: 8.
    - alpha: Exploration parameter. Higher = more exploration. Default: 0.5.
    - lambda_reg: Ridge regularization on A matrix. Default: 1.0.
    - warmup_pulls: Minimum pulls per arm before trusting UCB scores.
      During warmup, pulls arms in round-robin. Default: 3.
    - reward_ttl_seconds: How long a reward stays valid before decay.
      Default: 300.

    Priority: 19 (runs after content_profiler, before compression).
    """

    name = "strategy_selector"
    priority = 19

    def __init__(
        self,
        *,
        arms: tuple[str, ...] | list[str] | None = None,
        feature_dim: int = _DEFAULT_FEATURE_DIM,
        alpha: float = _DEFAULT_ALPHA,
        lambda_reg: float = 1.0,
        warmup_pulls: int = 3,
        reward_ttl_seconds: float = 300.0,
        default_strategy: str | None = None,
    ) -> None:
        if arms is not None and len(arms) == 0:
            raise ValueError("at least one arm is required")
        self.arms = tuple(arms) if arms is not None else _DEFAULT_ARMS
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("arm names must be unique")
        self.feature_dim = max(1, feature_dim)
        self.alpha = max(0.0, alpha)
        self.lambda_reg = max(1e-6, lambda_reg)
        self.warmup_pulls = max(0, warmup_pulls)
        self.reward_ttl_seconds = max(1.0, reward_ttl_seconds)
        self.default_strategy = default_strategy or self.arms[0]
        if self.default_strategy not in self.arms:
            raise ValueError(
                f"default_strategy {default_strategy!r} not in arms {self.arms}"
            )
        self._log = logger.bind(transform=self.name)

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Select a compression strategy and record it in request metadata."""
        state = context.get_transform_state(self.name)

        # Initialize bandit state if missing
        bandit = self._get_or_init_bandit(state)

        # Extract context features
        features = self._extract_features(request, context)

        # Select arm
        arm = self._preferred_contract_strategy(request) or self._select_arm(bandit, features)

        # Record decision
        state["last_arm"] = arm
        state["last_features"] = features
        state["last_selection_time"] = context.started_at
        context.record_metric(self.name, "selected_strategy", arm)
        context.record_metric(self.name, "feature_norm", math.sqrt(sum(f * f for f in features)))

        # Write strategy into request metadata for downstream transforms
        request.metadata["_lattice_strategy"] = arm

        # Also write per-strategy enablement flags
        self._set_strategy_flags(request, arm)

        self._log.debug(
            "strategy_selected",
            arm=arm,
            request_id=context.request_id,
            session_id=context.session_id,
            features=features,
        )
        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op on reverse — strategy selection does not alter response."""
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        """Process if enabled and we have content to work with."""
        if not self.enabled:
            return False
        # Skip if request is very short (compression won't help)
        return not request.token_estimate < 20

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, request: Request, _context: TransformContext) -> list[float]:
        """Extract a fixed-dimension feature vector from the request.

        Features (in order):
            0. normalized_request_length   → tokens / 8192
            1. has_tools                   → 1 if tools present, 0 otherwise
            2. reasoning_fraction          → fraction of text that looks like reasoning
            3. code_fraction               → fraction of text in code blocks
            4. message_depth               → min(messages, 20) / 20
            5. streaming                   → 1 if streaming, 0 otherwise
            6. table_density               → fraction of text that looks like tables
            7. narrative_ratio             → fraction of non-code text that is narrative
        """
        dim = self.feature_dim
        features: list[float] = [0.0] * dim

        all_text = "\n".join(m.content for m in request.messages)
        total_chars = max(1, len(all_text))
        token_estimate = max(1, request.token_estimate)

        # 0. Normalized request length
        features[0] = min(1.0, token_estimate / 8192.0)

        # 1. Tool presence
        if dim > 1:
            features[1] = 1.0 if (request.tools and len(request.tools) > 0) else 0.0

        # 2. Reasoning fraction
        if dim > 2:
            reasoning_chars = self._count_reasoning_text(all_text)
            features[2] = min(1.0, reasoning_chars / max(1, total_chars))

        # 3. Code fraction
        if dim > 3:
            code_chars = self._count_code_block_chars(all_text)
            features[3] = min(1.0, code_chars / max(1, total_chars))

        # 4. Message depth
        if dim > 4:
            features[4] = min(1.0, len(request.messages) / 20.0)

        # 5. Streaming
        if dim > 5:
            features[5] = 1.0 if request.stream else 0.0

        # 6. Table density
        if dim > 6:
            features[6] = self._estimate_table_density(all_text)

        # 7. Narrative ratio (derived)
        if dim > 7:
            non_code = self._strip_code_blocks(all_text)
            sentences = len(re.split(r"[.!?]+\s+", non_code))
            features[7] = min(1.0, sentences / max(1, len(request.messages) * 2))

        return features

    @staticmethod
    def _count_reasoning_text(text: str) -> int:
        """Count characters that look like reasoning traces."""
        patterns = [
            r"(?:^|\n)\s*Let me think",
            r"(?:^|\n)\s*First,",
            r"(?:^|\n)\s*Step \d+",
            r"(?:^|\n)\s*Reasoning:",
            r"<thinking>",
            r"<reasoning>",
            r"\btherefore\b",
            r"\bbecause\b",
            r"\bhence\b",
            r"\bconsequently\b",
        ]
        total = 0
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                total += len(match.group(0))
        return total

    @staticmethod
    def _count_code_block_chars(text: str) -> int:
        """Count characters inside code blocks."""
        total = 0
        for match in re.finditer(r"```[\w]*\n(.*?)\n```", text, re.DOTALL):
            total += len(match.group(1))
        # Also count inline code
        for match in re.finditer(r"`[^`\n]+`", text):
            total += len(match.group(0))
        return total

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove code blocks from text."""
        return re.sub(r"```[\w]*\n(.*?)\n```", "", text, flags=re.DOTALL)

    @staticmethod
    def _estimate_table_density(text: str) -> float:
        """Estimate fraction of text that is table-like."""
        md_table_rows = len(re.findall(r"\|[-:]+\|", text))
        json_table_rows = len(re.findall(r"\[\s*\{", text))
        total_rows = md_table_rows + json_table_rows
        lines = max(1, text.count("\n") + 1)
        return min(1.0, total_rows / max(1, lines / 3))

    # ------------------------------------------------------------------
    # Bandit logic
    # ------------------------------------------------------------------

    def _get_or_init_bandit(self, state: dict[str, Any]) -> dict[str, Any]:
        """Initialize bandit state in session_state if not present."""
        if "bandit" in state:
            bandit_state: dict[str, Any] = state["bandit"]
            return bandit_state

        bandit: dict[str, Any] = {
            "arms": {
                arm: _ArmState.zero(self.feature_dim, self.lambda_reg).to_dict()
                for arm in self.arms
            },
            "total_pulls": 0,
            "round_robin_idx": 0,
        }
        state["bandit"] = bandit
        return bandit

    def _select_arm(self, bandit: dict[str, Any], features: list[float]) -> str:
        """Select the best arm using LinUCB.

        During warmup (< warmup_pulls per arm), uses round-robin to ensure
        all arms get some initial pulls.
        """
        arms_data: dict[str, dict[str, Any]] = bandit["arms"]

        # Warmup: pull least-pulled arm in round-robin
        if self.warmup_pulls > 0:
            min_pulls = min(data["pulls"] for data in arms_data.values())
            if min_pulls < self.warmup_pulls:
                # Pick the least-pulled arm deterministically
                candidates = [arm for arm, data in arms_data.items() if data["pulls"] == min_pulls]
                return candidates[0]  # arbitrary but stable

        # Compute UCB scores
        scores: dict[str, float] = {}
        for arm, data in arms_data.items():
            state = _ArmState.from_dict(data)
            scores[arm] = state.ucb_score(features, self.alpha)

        # Select argmax (deterministic: first max if ties)
        max_score = max(scores.values())
        best = [arm for arm, sc in scores.items() if sc == max_score]
        return best[0]

    def _preferred_contract_strategy(self, request: Request) -> str | None:
        """Return runtime-contract preferred strategy when valid."""
        contract = request.metadata.get("_lattice_runtime_contract")
        if not isinstance(contract, dict):
            return None
        preferred = contract.get("preferred_strategy")
        if isinstance(preferred, str) and preferred in self.arms:
            return preferred
        return None

    def _set_strategy_flags(self, request: Request, arm: str) -> None:
        """Write per-transform enablement flags into request metadata."""
        # The strategy string itself is the primary signal.
        # Downstream transforms check request.metadata["_lattice_strategy"].
        # We also set individual flags for compatibility.
        flags: dict[str, bool] = {
            "semantic_compress": True,
            "submodular_select": False,
            "rd_compress": False,
        }

        if arm == "full":
            flags = {"semantic_compress": True, "submodular_select": True, "rd_compress": True}
        elif arm == "submodular":
            flags = {"semantic_compress": True, "submodular_select": True, "rd_compress": False}
        elif arm == "rd":
            flags = {"semantic_compress": True, "submodular_select": False, "rd_compress": True}
        elif arm == "hybrid":
            flags = {"semantic_compress": True, "submodular_select": True, "rd_compress": True}

        # Merge into existing _lattice_strategy dict if present
        existing = request.metadata.get("_lattice_strategy")
        if isinstance(existing, str):
            # Convert old string to dict and set new strategy name
            merged: dict[str, Any] = {"name": arm}
            merged.update(flags)
            request.metadata["_lattice_strategy"] = merged
        elif isinstance(existing, dict):
            existing.update(flags)
            existing["name"] = arm
        else:
            request.metadata["_lattice_strategy"] = {"name": arm, **flags}

    # ------------------------------------------------------------------
    # Reward feedback (called externally, e.g., by response handler)
    # ------------------------------------------------------------------

    def update_reward(
        self,
        context: TransformContext,
        reward: float,
        arm: str | None = None,
        features: list[float] | None = None,
    ) -> None:
        """Update the bandit with a reward observation.

        This is typically called after the response arrives, using metrics
        like token savings, latency reduction, or user feedback.

        Args:
            context: The TransformContext from the original request.
            reward: Scalar reward (higher = better). Suggested range [0, 1].
            arm: The arm that was selected. Defaults to last_arm from state.
            features: The feature vector. Defaults to last_features from state.
        """
        state = context.get_transform_state(self.name)
        bandit = self._get_or_init_bandit(state)

        actual_arm = arm or state.get("last_arm")
        actual_features = features or state.get("last_features")

        if actual_arm is None or actual_arm not in self.arms:
            self._log.warning("no_arm_to_reward", request_id=context.request_id)
            return
        if actual_features is None or len(actual_features) != self.feature_dim:
            self._log.warning(
                "invalid_features_for_reward",
                request_id=context.request_id,
                expected_dim=self.feature_dim,
                actual_dim=len(actual_features) if actual_features else None,
            )
            return

        # Clip reward for numerical stability
        clipped = max(0.0, min(1.0, reward))

        arm_data = bandit["arms"][actual_arm]
        arm_state = _ArmState.from_dict(arm_data)
        arm_state.update(actual_features, clipped)
        bandit["arms"][actual_arm] = arm_state.to_dict()
        bandit["total_pulls"] = bandit.get("total_pulls", 0) + 1

        context.record_metric(self.name, "reward", clipped)
        context.record_metric(self.name, "rewarded_arm", actual_arm)

        self._log.debug(
            "bandit_reward",
            arm=actual_arm,
            reward=clipped,
            pulls=arm_state.pulls,
            request_id=context.request_id,
        )
