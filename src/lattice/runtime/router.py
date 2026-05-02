"""Runtime router for LATTICE.

Formalizes the cost-routing classifier (RouteLLM-style) into a reusable
component that can be used by both the proxy and the SDK.

The router scores requests across multiple features and assigns a tier:
SIMPLE < MEDIUM < COMPLEX < REASONING

This tier is used for:
- Compute contract negotiation
- Latency/quality tradeoff hints
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from lattice.core.transport import Request

# =============================================================================
# Tier definitions
# =============================================================================


class Tier:
    """Workload complexity tier."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"

    ALL = (SIMPLE, MEDIUM, COMPLEX, REASONING)


# =============================================================================
# Routing decision
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Result of route classification."""

    tier: str
    score: int
    features: dict[str, int]
    confidence: float  # 0.0 - 1.0
    contract: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "score": self.score,
            "features": self.features,
            "confidence": round(self.confidence, 4),
            "contract": self.contract,
        }


# =============================================================================
# Router
# =============================================================================


class RuntimeRouter:
    """Classifies requests into tiers for compute/optimization decisions.

    Scoring model (0-100):
    - Length: 0-30 pts
    - Tools: 0-20 pts
    - Reasoning: 0-30 pts
    - Code: 0-15 pts
    - Debugging: 0-25 pts (errors, crashes, log analysis, root cause)
    - Depth: 0-10 pts

    Hard overrides:
    - Root cause + debugging context → REASONING
    - Explicit diagnostic/debugging query → REASONING
    - Heavy error signal (>5 error/crash mentions in query) → REASONING

    Thresholds:
    - REASONING >= 70
    - COMPLEX >= 40
    - MEDIUM >= 20
    - SIMPLE < 20
    """

    def __init__(
        self,
        reasoning_keywords: list[str] | None = None,
        code_indicators: list[str] | None = None,
        debug_indicators: list[str] | None = None,
        root_cause_indicators: list[str] | None = None,
    ) -> None:
        self.reasoning_keywords = reasoning_keywords or [
            "prove",
            "step by step",
            "reasoning",
            "derivation",
            "theorem",
            "mathematical",
            "equation",
            "formal proof",
            "logic",
            "deduction",
            "induction",
            "axiom",
            "lemma",
            "corollary",
        ]
        self.code_indicators = code_indicators or [
            "function",
            "class ",
            "def ",
            "import ",
            "component",
            "architecture",
            "microservice",
            "implementation",
            "refactor",
            "algorithm",
            "data structure",
            "api design",
            "unit test",
        ]
        self.debug_indicators = debug_indicators or [
            "debug",
            "error",
            "crash",
            "failure",
            "traceback",
            "exception",
            "timeout",
            "outage",
            "investigate",
            "diagnose",
            "triage",
        ]
        self.root_cause_indicators = root_cause_indicators or [
            "root cause",
            "what caused",
            "explain why",
        ]

    def classify(self, request: Request) -> RoutingDecision:
        """Classify a request into a tier."""
        combined = " ".join(m.content for m in request.messages).lower()
        total_chars = len(combined)

        # 1. Length score (0-30)
        if total_chars > 4000:
            length_score = 30
        elif total_chars > 2000:
            length_score = 20
        elif total_chars > 500:
            length_score = 10
        else:
            length_score = max(0, total_chars // 100)

        # 2. Tool score (0-20)
        tool_count = len(request.tools) if request.tools else 0
        if tool_count >= 4:
            tool_score = 20
        elif tool_count >= 1:
            tool_score = 10
        else:
            tool_score = 0

        # 3. Reasoning score (0-30)
        reasoning_hits = sum(1 for kw in self.reasoning_keywords if kw in combined)
        reasoning_score = min(30, reasoning_hits * 10)

        # 4. Code score (0-15)
        code_hits = sum(1 for ind in self.code_indicators if ind in combined)
        code_score = min(15, code_hits * 3)

        # 5. Debugging score (0-25) — error patterns, log analysis, diagnostic signals
        # Only count patterns in query content (user/assistant), not tool-output data labels.
        # Use word-boundary matching to avoid substring false positives
        # (e.g. "degraded" should not match "debug", "logically" should not match "log").
        query_text = " ".join(
            m.content for m in request.messages if m.role in ("user", "assistant", "system")
        ).lower()
        debug_hits = 0
        for ind in self.debug_indicators:
            if re.search(rf"\b{re.escape(ind)}\b", query_text):
                debug_hits += 1
        # Also match error/crash/failure in tool output for log-line counts
        error_log_lines = max(
            query_text.count("[error]"),
            query_text.count("error:"),
            query_text.count("error :"),
        )
        # Overrides for error-heavy diagnostic content
        if error_log_lines > 25:
            debug_hits += 5
        debug_score = min(25, debug_hits * 3)

        # Root cause signal (0-15) — explicit root cause / explain-why patterns
        root_cause_hits = sum(1 for ind in self.root_cause_indicators if ind in query_text)
        root_cause_score = min(15, root_cause_hits * 5)

        # 6. Depth score (0-10)
        msg_count = len(request.messages)
        if msg_count >= 10:
            depth_score = 10
        elif msg_count >= 5:
            depth_score = 5
        else:
            depth_score = 0

        total_score = (
            length_score
            + tool_score
            + reasoning_score
            + code_score
            + debug_score
            + root_cause_score
            + depth_score
        )

        # ---- Hard overrides for diagnostic/debugging/reasoning context ----
        # These ensure the tier reflects the actual workload, not just
        # keyword-scoring that misses genuine debugging/reasoning tasks.

        has_debug_context = any(
            ind in query_text
            for ind in ("error", "crash", "failure", "debug", "traceback", "exception")
        ) or bool(
            re.search(r"\blog\b", query_text)
            and not re.search(
                r"\b(logically|catalog|analog|dialogue|prolog|blog|slog)\b", query_text
            )
        )
        has_root_cause_signal = "root cause" in query_text
        hard_override = False

        # Hard rule 1: root cause WITH debugging context → REASONING
        if has_root_cause_signal and has_debug_context:
            tier = Tier.REASONING
            total_score = max(total_score, 70)
            hard_override = True

        # Hard rule 2: explicit diagnostic/debugging query → REASONING
        elif any(
            pat in query_text
            for pat in (
                "debug the",
                "investigate the failure",
                "what caused",
            )
        ) or bool(re.search(r"explain why.*(error|crash|fail(?:ed|ure))", query_text)):
            tier = Tier.REASONING
            total_score = max(total_score, 70)
            hard_override = True

        # Hard rule 3: heavy log-line error signal → REASONING
        elif error_log_lines > 25 and total_score < 40:
            tier = Tier.REASONING
            total_score = max(total_score, 70)
            hard_override = True

        # Score-based tier assignment (only if no hard override fired)
        if hard_override:
            confidence = min(1.0, total_score / 100)
        elif total_score >= 70:
            tier = Tier.REASONING
            confidence = min(1.0, total_score / 100)
        elif total_score >= 40:
            tier = Tier.COMPLEX
            confidence = min(1.0, total_score / 70)
        elif total_score >= 20:
            tier = Tier.MEDIUM
            confidence = min(1.0, total_score / 40)
        else:
            tier = Tier.SIMPLE
            confidence = min(1.0, (20 - total_score) / 20)

        features = {
            "length": length_score,
            "tools": tool_score,
            "reasoning": reasoning_score,
            "code": code_score,
            "debugging": debug_score,
            "root_cause": root_cause_score,
            "depth": depth_score,
        }
        return RoutingDecision(
            tier=tier,
            score=total_score,
            features=features,
            confidence=confidence,
            contract=self.optimization_contract(tier, total_score, features),
        )

    def optimization_contract(
        self,
        tier: str,
        score: int,
        features: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Return the optimization contract for a workload tier.

        This governs transform intensity only. It intentionally does not
        contain a replacement model, because LATTICE must preserve the model
        selected by the caller.
        """
        features = features or {}
        if tier == Tier.SIMPLE:
            skipped = (
                "structural_fingerprint",
                "self_information",
                "information_theoretic_selector",
                "rate_distortion",
                "hierarchical_summary",
            )
            mode = "minimal"
            budget_ms = 2.0
            preferred_strategy = "full"
        elif tier == Tier.MEDIUM:
            skipped = (  # type: ignore[assignment]
                "self_information",
                "hierarchical_summary",
            )
            mode = "balanced"
            budget_ms = 20.0
            preferred_strategy = "submodular"
        elif tier == Tier.COMPLEX:
            skipped = ("hierarchical_summary",)  # type: ignore[assignment]
            mode = "aggressive"
            budget_ms = 20.0
            preferred_strategy = "hybrid"
        else:
            skipped = ()  # type: ignore[assignment]
            mode = "max_fidelity"
            budget_ms = 50.0
            preferred_strategy = "hybrid"

        # Tool-heavy requests should retain tool/filtering and cache planning;
        # code/reasoning-heavy requests get higher-fidelity strategies.
        if features.get("tools", 0) >= 20 and "hierarchical_summary" in skipped:
            skipped = tuple(s for s in skipped if s != "hierarchical_summary")  # type: ignore[assignment]

        return {
            "mode": mode,
            "preferred_strategy": preferred_strategy,
            "max_transform_latency_ms": budget_ms,
            "skipped_transforms": list(skipped),
            "allow_model_override": False,
            "score": score,
        }

    def select_model(self, _request: Request, preferred_model: str) -> str:
        """Return the user's preferred model unchanged.

        LATTICE is a transport optimization layer, not a routing gateway.
        We never override the model selected by the caller.
        """
        return preferred_model


__all__ = [
    "Tier",
    "RoutingDecision",
    "RuntimeRouter",
]
