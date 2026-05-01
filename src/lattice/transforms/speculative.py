"""Speculative Execution for LATTICE.

Predicts likely next steps (e.g., "the model will call tool X") and pre-runs
them in parallel with the main request. If the prediction is correct, the
result is returned instantly. If incorrect, the pre-run result is discarded.

Architecture
------------
```
┌─────────────────┐     ┌──────────────────────┐
│  User Request   │────▶│  SpeculativeExecutor │
└─────────────────┘     │                      │
                        │  1. Predict next step│
                        │  2. Pre-run in parallel│
                        │  3. If correct → instant response
                        │  4. If wrong → discard, run real request
                        └──────────────────────┘
```

Design Principles
-----------------
1. **Never block the real request** — speculation is a sidecar.
2. **Prediction must be cheap** — rule-based or lightweight model, not LLM inference.
3. **Discard is always safe** — if prediction is wrong, latency is not worse than baseline.
4. **Metrics-driven** — track prediction accuracy to tune heuristics.

Key: The speculative branch must complete before the real branch for it to save latency.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

logger = structlog.get_logger()


# =============================================================================
# SpeculationResult
# =============================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class SpeculationResult:
    """Outcome of a speculative execution attempt."""

    hit: bool
    predicted_step: str
    actual_step: str | None
    latency_ms: float
    tokens_saved: int = 0


# =============================================================================
# SpeculativeExecutor
# =============================================================================

class SpeculativeExecutor:
    """Pre-runs predicted next steps in parallel with the main request.

    Usage (in proxy server):
        executor = SpeculativeExecutor()
        # On each request:
        prediction = executor.predict(request, context)
        if prediction:
            # Launch speculative and real in parallel
            speculative_task = asyncio.create_task(
                executor.run_speculative(request, prediction)
            )
            real_response = await forward_to_provider(request)
            actual = executor.extract_actual_step(real_response)
            if executor.is_hit(prediction, actual):
                # Use speculative result
                return speculative_task.result()
            else:
                return real_response

    For Phase 2, predictions are rule-based:
    - If the last message requests a known tool → predict tool call
    - If the conversation follows a known pattern → predict completion
    """

    def __init__(
        self,
        max_speculative_tokens: int = 256,
        confidence_threshold: float = 0.7,
        provider_caller: Callable[[Request], Awaitable[Response]] | None = None,
    ) -> None:
        """Initialize speculative executor.

        Args:
            max_speculative_tokens: Max tokens to request in speculative pre-run.
            confidence_threshold: Minimum confidence (0-1) to trigger speculation.
            provider_caller: Async function to call the provider. If None,
                             speculative execution is disabled.
        """
        self.max_speculative_tokens = max_speculative_tokens
        self.confidence_threshold = confidence_threshold
        self.provider_caller = provider_caller

        # Metrics
        self._total: int = 0
        self._hits: int = 0
        self._saved_ms: float = 0.0
        self._log = logger.bind(module="speculative_executor")

    # ------------------------------------------------------------------
    # Prediction (rule-based, fast)
    # ------------------------------------------------------------------

    def predict(
        self, request: Request, _context: TransformContext
    ) -> str | None:
        """Predict the most likely next step for this request.

        Returns:
            A step identifier string (e.g., "tool:search", "completion"),
            or None if no confident prediction can be made.
        """
        if not request.messages:
            return None

        last_msg = request.messages[-1]
        content = (last_msg.content or "").lower()

        # Heuristic 1: explicit tool request
        if request.tools and any(
            t.get("function", {}).get("name", "").lower() in content
            for t in request.tools
        ):
            return "tool_call"

        # Heuristic 2: assistant just asked a question → predict user answer
        if len(request.messages) >= 2:
            prev = request.messages[-2]
            if prev.role in ("assistant", "model") and "?" in (prev.content or ""):
                return "user_answer"

        # Heuristic 3: coding pattern
        if any(kw in content for kw in ("write", "code", "function", "implement")):
            return "code_completion"

        # No confident prediction
        return None

    def confidence(self, prediction: str, request: Request) -> float:
        """Estimate confidence of a prediction (0.0 - 1.0)."""
        if prediction == "tool_call" and request.tools:
            return 0.85
        if prediction == "user_answer":
            return 0.60
        if prediction == "code_completion":
            return 0.55
        return 0.0

    # ------------------------------------------------------------------
    # Speculative execution
    # ------------------------------------------------------------------

    async def run_speculative(
        self, request: Request, prediction: str
    ) -> Response | None:
        """Execute a speculative pre-run.

        Constructs a modified request based on the prediction and sends it
        to the provider in parallel with the real request.

        Args:
            request: The original request.
            prediction: The predicted step.

        Returns:
            A speculative Response, or None if the provider caller is not set.
        """
        if not self.provider_caller:
            return None

        speculative_request = self._build_speculative_request(request, prediction)
        if speculative_request is None:
            return None

        try:
            return await self.provider_caller(speculative_request)
        except Exception as exc:
            self._log.debug("speculative_execution_failed", error=str(exc))
            return None

    def _build_speculative_request(
        self, request: Request, prediction: str
    ) -> Request | None:
        """Build a request for the speculative branch."""
        speculative = request.copy()
        speculative.max_tokens = self.max_speculative_tokens

        if prediction == "tool_call":
            # Add a hint that this is speculative — some providers ignore it
            speculative.metadata["_speculative"] = True
            return speculative

        if prediction == "code_completion":
            # Limit to shorter completion for speed
            speculative.max_tokens = min(self.max_speculative_tokens, 128)
            return speculative

        # Default: just run with limited tokens
        return speculative

    # ------------------------------------------------------------------
    # Hit / Miss detection
    # ------------------------------------------------------------------

    def extract_actual_step(self, response: Response) -> str | None:
        """Determine what step the real response actually took."""
        if response.tool_calls:
            return "tool_call"
        return "completion"

    def is_hit(self, predicted: str, actual: str | None) -> bool:
        """Check if the prediction matches the actual outcome."""
        if actual is None:
            return False
        # Broad match: "tool_call" matches any tool_call
        if predicted == "tool_call" and actual == "tool_call":
            return True
        return predicted == actual

    def record_result(
        self, hit: bool, _predicted: str, _actual: str | None, latency_ms: float
    ) -> None:
        """Record speculation outcome for metrics."""
        self._total += 1
        if hit:
            self._hits += 1
            self._saved_ms += latency_ms

    @property
    def accuracy(self) -> float:
        """Current prediction accuracy."""
        if self._total == 0:
            return 0.0
        return self._hits / self._total

    @property
    def stats(self) -> dict[str, Any]:
        """Return statistics for health/metrics endpoints."""
        return {
            "total": self._total,
            "hits": self._hits,
            "accuracy": round(self.accuracy, 4),
            "saved_ms_total": round(self._saved_ms, 2),
        }


# =============================================================================
# SpeculativeTransform (pipeline integration)
# =============================================================================

class SpeculativeTransform(ReversibleSyncTransform):
    """Pipeline-friendly wrapper for speculative metadata.

    Records whether the request is eligible for speculative execution
    and what the predicted next step is. The actual pre-run happens at
    the proxy layer, not inside the pipeline.
    """

    name = "speculative"
    priority = 2  # Run very early, before batching and delta

    def __init__(self, executor: SpeculativeExecutor | None = None) -> None:
        self.executor = executor or SpeculativeExecutor()

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Record speculative prediction metadata."""
        prediction = self.executor.predict(request, context)
        if prediction:
            conf = self.executor.confidence(prediction, request)
            if conf >= self.executor.confidence_threshold:
                context.record_metric(self.name, "prediction", prediction)
                context.record_metric(self.name, "confidence", round(conf, 4))
                context.session_state["speculative_prediction"] = prediction
        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — speculation is proxy-side."""
        return response
