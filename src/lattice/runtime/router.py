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
    - Depth: 0-10 pts

    Thresholds:
    - REASONING >= 65
    - COMPLEX >= 40
    - MEDIUM >= 20
    - SIMPLE < 20
    """

    def __init__(
        self,
        reasoning_keywords: list[str] | None = None,
        code_indicators: list[str] | None = None,
    ) -> None:
        self.reasoning_keywords = reasoning_keywords or [
            "prove", "step by step", "reasoning", "derivation", "theorem",
            "mathematical", "equation", "formal proof", "logic", "deduction",
            "induction", "axiom", "lemma", "corollary",
        ]
        self.code_indicators = code_indicators or [
            "function", "class ", "def ", "import ", "component",
            "architecture", "microservice", "implementation", "refactor",
            "algorithm", "data structure", "api design", "unit test",
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

        # 5. Depth score (0-10)
        msg_count = len(request.messages)
        if msg_count >= 10:
            depth_score = 10
        elif msg_count >= 5:
            depth_score = 5
        else:
            depth_score = 0

        total_score = length_score + tool_score + reasoning_score + code_score + depth_score

        if total_score >= 65:
            tier = Tier.REASONING
            confidence = min(1.0, total_score / 100)
        elif total_score >= 40:
            tier = Tier.COMPLEX
            confidence = min(1.0, total_score / 65)
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
                "strategy_selector",
                "structural_fingerprint",
                "self_information",
                "context_selector",
                "information_theoretic_selector",
                "rate_distortion",
                "hierarchical_summary",
            )
            mode = "minimal"
            budget_ms = 2.0
            preferred_strategy = "full"
        elif tier == Tier.MEDIUM:
            skipped = (
                "self_information",
                "information_theoretic_selector",
                "hierarchical_summary",
            )
            mode = "balanced"
            budget_ms = 8.0
            preferred_strategy = "submodular"
        elif tier == Tier.COMPLEX:
            skipped = ("hierarchical_summary",)
            mode = "aggressive"
            budget_ms = 20.0
            preferred_strategy = "hybrid"
        else:
            skipped = ()
            mode = "max_fidelity"
            budget_ms = 35.0
            preferred_strategy = "hybrid"

        # Tool-heavy requests should retain tool/filtering and cache planning;
        # code/reasoning-heavy requests get higher-fidelity strategies.
        if features.get("tools", 0) >= 20 and "hierarchical_summary" in skipped:
            skipped = tuple(s for s in skipped if s != "hierarchical_summary")

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
