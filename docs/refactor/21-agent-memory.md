# Phase 21 — Agent Memory: Context GC, Summarization, Token Budgets, Inference-Aware Retry (lightweight)

> **Footprint impact.** **Zero new runtime deps in the default install.** Rule-based relevance scoring (recency × tool-call density × ANSI-token similarity) is the default. Optional embedding-based scoring reuses the **user's provider** embeddings endpoint from [Phase 14](14-hybrid-semantic-cache.md) — still no model download. Summarization optionally uses the **user's cheap model** (e.g. `gpt-4o-mini`, `claude-3-5-haiku`) — no additional vendor relationship, no download. The user already pays for these; we just route through them when summarization is needed.
>
> **Algorithm location.** New `src/lattice/agent/` package. The relevance scorer and budget allocator are pure functions; summarization is a request to the user's provider via the existing dispatcher (so it benefits from the cache, the bandit, and observability for free). Inference-aware retry composes with [Phase 15](15-native-guardrails.md)'s JSON repair and [Phase 19](19-compression-intelligence.md)'s structural repair.
>
> **External-service requirement.** None. The user's provider is the only external service touched; we never bring in our own LLM.
>

> **LoC delta (declared).** +2000 net (`agent/`). Within cap 2000.
> **Transport role.** Pre-dispatch request shaping (GC, budgets). Agent retry strategy ≠ transport retry ([SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) §6 vs §7).
> **Registry.** §6 agent memory.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Outcome.** Agent loops stay within token budgets without truncating critical context. Old messages are scored, summarized (cheap-model) or evicted (free) based on configurable policy. Inference-aware retry distinguishes refusal / truncation / repairable-malformed-JSON and takes the right action without burning a full retry. The session store ([phase-7](07-session-store-segments.md)) gains a `MemoryPolicy` field; agent integrations ([phase-8](08-observability-state.md)) opt in via header or session config.
>
> **Estimated effort.** 5 days (1 PR, ~+2000 LoC — smaller than prior because relevance scoring is rule-based by default; embeddings tier reuses Phase 14 infrastructure).

---

## 1. Why this phase exists, and what changed from the prior draft

### 1.1 The real agent-loop pain

LATTICE compresses individual requests well. It doesn't manage **the cumulative context** an agent loop builds:

| Problem | Today | After this phase |
|---|---|---|
| Long-running agent fills 128k window after ~ 30 tool calls; model degrades or refuses | No-op; agent author has to write their own context-pruning logic | Context GC scores messages by relevance and evicts safely; warns the agent author when the loop will hit the limit |
| Old turns are kept verbatim even when only their conclusion matters | No-op | Optional summarization replaces verbose turns with their summary; uses user's cheap model |
| Hard limits hit mid-loop → 400 from provider; agent crashes | Provider raises; agent must handle | Budget allocator enforces a soft limit (with headroom for response) before the request leaves the proxy |
| Model returns malformed JSON → blanket retry | One full retry cost | Inference-aware retry: tries structural repair first, then minimal-correction prompt, only then a full retry |

### 1.2 The brutal change vs the prior draft

The prior plan defaulted to embedding-based relevance scoring using Phase 14's local SentenceTransformer (which we already replaced with user-provider embeddings in this rewrite). Adding embedding calls to every loop iteration was both costly (per-iteration provider call) and circular (Phase 14 now defaults to no-embedding).

**New default:** rule-based relevance scoring covers the realistic majority of agent workloads:

- **Recency** — message age in turns (exponential decay)
- **Tool-call density** — messages with tool calls / results weighted higher
- **Topic continuity** — last system prompt + most recent user message anchor the topic; older messages share-of-tokens with those anchors
- **Explicit pin** — agent can mark messages as `lattice:pin` (header on append)

Zero deps, < 1 ms per scoring pass. Property: never evicts a `lattice:pin` message; never evicts the current system prompt; never evicts the most recent user turn.

Embedding-based scoring stays as a config option, reusing whatever embedding backend Phase 14 is configured with (default: user's provider; opt-in: local). No new infrastructure.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/agent/__init__.py
src/lattice/agent/context_gc.py             # eviction policy
src/lattice/agent/summarizer.py             # uses user's cheap model
src/lattice/agent/budget.py                 # token budget allocator
src/lattice/agent/retry.py                  # inference-aware retry
src/lattice/agent/relevance.py              # rule-based scorer (default) + optional embedding scorer
src/lattice/agent/policy.py                 # MemoryPolicy dataclass
src/lattice/pipeline/agent_gates.py         # integrates with Pipeline.compress()

tests/unit/agent/test_context_gc.py
tests/unit/agent/test_summarizer.py
tests/unit/agent/test_budget.py
tests/unit/agent/test_retry.py
tests/unit/agent/test_relevance_rule.py
tests/unit/agent/test_relevance_embedding.py
tests/integration/agent/test_loop_eviction_e2e.py
tests/integration/agent/test_summarization_e2e.py
tests/integration/agent/test_inference_retry_e2e.py
tests/contract/test_default_install_no_agent_embeddings.py
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/state/session.py](../../src/lattice/state/session.py) | `Session.memory_policy: MemoryPolicy \| None` field |
| [src/lattice/state/store.py](../../src/lattice/state/store.py) | Persist memory_policy |
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | Insert agent gate after IR build, before transform planning |
| [src/lattice/core/config.py](../../src/lattice/core/config.py) | Add `AgentMemoryConfig` with conservative defaults |
| [src/lattice/proxy/middleware.py](../../src/lattice/proxy/middleware.py) | Headers: `x-lattice-agent-evicted`, `x-lattice-agent-summarized`, `x-lattice-agent-budget`, `x-lattice-agent-retry-strategy` |
| [src/lattice/integrations/agent_stats.py](../../src/lattice/integrations/agent_stats.py) | Track per-session loop iteration count for retry strategy decisions |

### 2.3 Deleted

None.

---

## 3. Step-by-step

### 3.1 Relevance scoring — rule-based default

```python
# src/lattice/agent/relevance.py
@dataclass(frozen=True, slots=True)
class RelevanceScore:
    message_index: int
    score: float                          # 0..1
    components: dict[str, float]          # recency, tool_density, topic_continuity, pin
    pinned: bool                          # cannot be evicted


class RuleRelevanceScorer:
    """Default scorer. Zero deps. ~1 ms per scoring pass on typical sessions."""

    def __init__(self, recency_decay: float = 0.92,
                 tool_call_weight: float = 1.4,
                 topic_continuity_weight: float = 0.6):
        self._decay = recency_decay
        self._tool_w = tool_call_weight
        self._topic_w = topic_continuity_weight

    def score(self, messages: Sequence[Message]) -> Sequence[RelevanceScore]:
        if not messages:
            return ()
        # Anchor topic on the last system prompt + most recent user turn
        anchor_tokens = self._anchor_tokens(messages)
        n = len(messages)
        scores = []
        for i, msg in enumerate(messages):
            pinned = self._is_pinned(msg) or self._is_required(msg, i, n)
            if pinned:
                scores.append(RelevanceScore(i, score=1.0, components={"pin": 1.0}, pinned=True))
                continue
            age = n - 1 - i
            recency = self._decay ** age
            tool_density = self._tool_w if (msg.tool_calls or msg.role == "tool") else 1.0
            topic = self._topic_overlap(msg, anchor_tokens) * self._topic_w
            score = min(1.0, recency * tool_density * (1.0 + topic) / 2.0)
            scores.append(RelevanceScore(i, score=score,
                                          components={"recency": recency, "tool": tool_density, "topic": topic},
                                          pinned=False))
        return scores

    def _is_pinned(self, msg: Message) -> bool:
        return bool(msg.metadata.get("lattice:pin"))

    def _is_required(self, msg: Message, i: int, n: int) -> bool:
        # Last system prompt and last user turn always required
        if msg.role == "system" and self._is_last_of_role(msg, "system", i, n): return True
        if msg.role == "user" and i == self._last_index_of_role(msg, "user", n): return True
        return False

    def _topic_overlap(self, msg: Message, anchor_tokens: frozenset[str]) -> float:
        if not anchor_tokens:
            return 0.0
        toks = frozenset(re.findall(r"\w+", (msg.content or "").lower())[:200])
        if not toks: return 0.0
        return len(toks & anchor_tokens) / len(toks | anchor_tokens)
```

Properties enforced by tests:

- Pinned messages always score 1.0
- Most recent user turn always scores 1.0
- Most recent system prompt always scores 1.0
- Tool-call density boost is monotonic (more tool calls → higher score, holding other factors)
- Score is deterministic — same input → same output

### 3.2 Relevance scoring — optional embedding scorer

```python
# src/lattice/agent/relevance.py (continued)
class EmbeddingRelevanceScorer:
    """OPT-IN higher-quality scorer.

    Reuses Phase 14's EmbeddingBackend — so by default it uses the user's provider
    (no extra dependency, no model download). Falls back to RuleRelevanceScorer if
    the embedding backend is unhealthy.
    """
    def __init__(self, embedding_backend: EmbeddingBackend, fallback: RuleRelevanceScorer):
        self._embed = embedding_backend
        self._fallback = fallback

    def score(self, messages: Sequence[Message]) -> Sequence[RelevanceScore]:
        if not self._embed.health().ok:
            return self._fallback.score(messages)
        anchor_vec = self._anchor_vector(messages)
        rule_scores = self._fallback.score(messages)
        # Re-rank rule scores by anchor similarity for non-pinned messages
        return tuple(self._blend(rs, msg, anchor_vec) for rs, msg in zip(rule_scores, messages))
```

Opt-in via `agent_memory.relevance_scorer: embedding`. Cost: one batch embedding call per loop iteration; tiny on user's cheap provider.

### 3.3 Context GC

```python
# src/lattice/agent/context_gc.py
class ContextGC:
    """Evicts low-relevance messages until token budget fits."""

    def __init__(self, scorer: RelevanceScorerProtocol, tokenizer: Tokenizer):
        self._scorer = scorer
        self._tokenizer = tokenizer

    def evict_to_budget(self, messages: Sequence[Message], target_tokens: int) -> GCResult:
        current = self._tokenizer.count_tokens(messages)
        if current <= target_tokens:
            return GCResult(messages=messages, evicted=(), summarized=(), reason="under_budget")
        scores = self._scorer.score(messages)
        eligible = [s for s in scores if not s.pinned]
        eligible.sort(key=lambda s: s.score)        # lowest first
        kept = list(messages)
        evicted: list[int] = []
        for s in eligible:
            if current <= target_tokens:
                break
            kept[s.message_index] = None            # mark for removal
            current -= self._tokenizer.count_message_tokens(messages[s.message_index])
            evicted.append(s.message_index)
        kept = [m for m in kept if m is not None]
        return GCResult(messages=tuple(kept), evicted=tuple(evicted), summarized=(), reason="evicted")
```

The agent gate calls `evict_to_budget` with `target = min(model_limit - max_tokens_for_response, configured_budget)`.

### 3.4 Summarization (uses user's cheap model — zero new vendor)

```python
# src/lattice/agent/summarizer.py
class Summarizer:
    """Replaces verbose old turns with summaries via the user's cheap model.

    Cost is paid by the user's provider — no new vendor relationship.
    Conservative defaults: only summarizes turns older than `min_age_turns`
    and at least `min_tokens` long. Cached via Phase 14 cache by session +
    message-range key so re-summarization is rare.
    """

    DEFAULT_SUMMARY_PROMPT = (
        "Summarize the following conversation turn for an AI assistant's working "
        "memory. Keep all decisions, file paths, identifiers, and unresolved "
        "questions. Be concise (≤ 150 tokens). Do not add commentary."
    )

    def __init__(self, dispatcher: ProviderDispatcher, *, provider: str, model: str,
                 max_summary_tokens: int = 200):
        self._dispatch = dispatcher
        self._provider = provider
        self._model = model
        self._max = max_summary_tokens

    async def summarize_range(self, messages: Sequence[Message],
                              start: int, end: int) -> SummarizedTurn:
        block = format_for_summary(messages[start:end])
        cache_key = compute_summary_cache_key(block, model=self._model)
        if (cached := await summary_cache.get(cache_key)) is not None:
            return cached
        response = await self._dispatch.chat(
            provider=self._provider, model=self._model,
            messages=[{"role": "system", "content": self.DEFAULT_SUMMARY_PROMPT},
                      {"role": "user", "content": block}],
            max_tokens=self._max, temperature=0.0,
        )
        summary = SummarizedTurn(
            replaced_indices=tuple(range(start, end)),
            text=response.text.strip(),
            tokens=response.usage.completion_tokens,
            source_model=self._model,
        )
        await summary_cache.set(cache_key, summary, ttl=86400)
        return summary
```

Auto-detection of the cheap model: when summarization is enabled and no model specified, the agent gate picks the cheapest model available on the configured upstream that has matching family (OpenAI → `gpt-4o-mini`, Anthropic → `claude-3-5-haiku-latest`, Mistral → `ministral-3b-latest`, etc.). Documented mapping in `agent/summarizer.py`.

Result composition: `GCResult` may include `summarized: tuple[SummarizedTurn, ...]` describing replacements. The agent gate rewrites `messages` accordingly.

### 3.5 Budget allocator

```python
# src/lattice/agent/budget.py
@dataclass(frozen=True, slots=True)
class TokenBudget:
    model_limit: int                      # from provider profile
    reserved_for_response: int            # default 4096 — settable per request
    safety_margin: int = 256              # absolute headroom

    @property
    def input_budget(self) -> int:
        return self.model_limit - self.reserved_for_response - self.safety_margin

    def is_over(self, input_tokens: int) -> bool:
        return input_tokens > self.input_budget
```

Sensible defaults per model (from `providers/profiles.py`): GPT-4o → 128k, Claude 3.5 Sonnet → 200k, Gemini 1.5 Pro → 1M, etc. Per-request override via `x-lattice-token-budget`.

### 3.6 Inference-aware retry

```python
# src/lattice/agent/retry.py
class RetryStrategy(StrEnum):
    NONE = "none"
    STRUCTURAL_REPAIR = "structural_repair"    # Phase 15 / 19 — no LLM call
    MINIMAL_CORRECTION = "minimal_correction"  # tiny continuation prompt — small cost
    REDUCE_CONTEXT = "reduce_context"          # evict more, re-call same model
    FULL_RETRY = "full_retry"                  # last resort


class InferenceAwareRetry:
    def decide(self, response: Response, request: Request, ctx: TransformContext) -> RetryStrategy:
        if response.finish_reason == "length":
            return RetryStrategy.REDUCE_CONTEXT
        if response.has_malformed_json() and not response.has_been_repaired():
            return RetryStrategy.STRUCTURAL_REPAIR
        if response.is_refusal():
            if ctx.retry_count == 0:
                return RetryStrategy.MINIMAL_CORRECTION
            return RetryStrategy.FULL_RETRY
        if response.has_partial_tool_call():
            return RetryStrategy.MINIMAL_CORRECTION
        return RetryStrategy.NONE
```

Composes with [Phase 15](15-native-guardrails.md) output validator and [Phase 19](19-compression-intelligence.md) `repair_v2`. The "minimal correction" prompt is a 1-line continuation (`"Continue from your last token."`) — cheaper than a full retry by ~ 90%.

### 3.7 Integration with `Pipeline.compress()`

```python
# pipeline/runner.py — inside compress()
# After IR built; before transform planning:
agent_result = self._agent_gates.preprocess(request, ir, ctx)
if agent_result.modified:
    request = agent_result.request
    ir = agent_result.ir
    ctx.headers["x-lattice-agent-evicted"] = str(len(agent_result.evicted))
    if agent_result.summarized:
        ctx.headers["x-lattice-agent-summarized"] = str(len(agent_result.summarized))
    ctx.headers["x-lattice-agent-budget"] = f"{agent_result.tokens_in}/{agent_result.budget}"
```

`pipeline/agent_gates.py` orchestrates relevance scoring + GC + summarization.

### 3.8 Default config

```python
# src/lattice/core/config.py
class AgentMemoryConfig(BaseModel):
    enabled: bool = True                            # ON for sessions; no-op for stateless requests
    relevance_scorer: Literal["rule", "embedding"] = "rule"           # zero-dep default
    eviction_enabled: bool = True
    summarization_enabled: bool = False             # OFF by default — pays for itself only on long loops
    summarization_provider: str | None = None       # auto-detect
    summarization_model: str | None = None          # auto-pick cheap model from configured provider
    summarization_min_age_turns: int = 6
    summarization_min_tokens: int = 200
    retry_strategy_enabled: bool = True
    default_reserved_for_response: int = 4096
```

Conservative: eviction default-on (free, lossless aside from dropped low-relevance messages); summarization default-off (costs the user pennies per loop iteration; users opt in once they confirm value).

---

## 4. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/agent -q` | All pass |
| Pin-never-evicted | property | Pinned + last-system + last-user never appear in evicted set |
| Determinism | property | Rule scorer produces same scores for same input |
| Budget enforcement | property | After GC, `count_tokens(messages) ≤ budget` |
| Summary round-trip | `tests/integration/agent/test_summarization_e2e.py` | Summary preserves required fields (paths, identifiers); next loop turn references summary successfully |
| Retry strategy correctness | `tests/integration/agent/test_inference_retry_e2e.py` | Each finish_reason / response shape triggers expected strategy |
| Loop survival | `tests/integration/agent/test_loop_eviction_e2e.py` | 50-iteration agent loop stays within 128k window; no 400s from provider |
| Default-install-no-embeddings contract | `tests/contract/test_default_install_no_agent_embeddings.py` | With default config, agent gates never call embedding backend |
| Footprint | `tests/integration/footprint/test_4gb_laptop.py` | Default config adds ≤ 2 ms median proxy overhead, ≤ 1 MB RSS |
| Canonical bench | usual | ±2% |

---

## 5. Acceptance criteria

1. Default config (rule scorer, eviction on, summarization off) keeps a 50-iteration tool-call agent within the model's context window with no upstream 400s. Verified by E2E test using a fixture provider that returns `length` finish_reason on overrun.
2. Pinned messages (`metadata.lattice:pin = true`) are never evicted. Property test enforces.
3. Most-recent system prompt and most-recent user turn are never evicted. Property test enforces.
4. Optional summarization (`summarization_enabled = true`) uses the user's cheap model (auto-detected from configured upstream provider); summaries are cached by session + message-range; second loop with same prior turns reuses the cached summary.
5. Optional embedding-based scorer reuses Phase 14's embedding backend (defaults to user's provider — no model download); falls back to rule scorer if backend unhealthy.
6. Inference-aware retry: malformed-JSON triggers structural repair (no LLM call) on the first attempt; refusal triggers minimal-correction prompt before a full retry; truncation triggers context reduction before retry.
7. Response headers (`x-lattice-agent-evicted`, `x-lattice-agent-summarized`, `x-lattice-agent-budget`, `x-lattice-agent-retry-strategy`) surface every decision the gate made.
8. Footprint default-install test passes — no new heavy deps loaded.
9. Canonical bench ±2%.

---

## 6. Out of scope

| Topic | Phase / future |
|---|---|
| Long-term memory across sessions ("the model remembers prior conversations") | Future. Out of scope for v2.0. |
| LLM-as-judge for relevance scoring | Future; rule + optional embedding cover the realistic majority. |
| Per-user fine-tuned relevance models | **Cut from plan** — cloud + GPU + training infra. Bandit (Phase 23) covers a slice with zero infra. |
| Auto-pinning learned from agent's own behaviour | Future — depends on a per-tenant decision log. |
