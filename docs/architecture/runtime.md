# LATTICE Runtime Architecture

## Product positioning (read this first)

**LATTICE is the transport / network layer for LLM traffic.**

A single self-hosted process sits between your application and **one** chosen LLM provider per request. It owns:

- **Transport** — connection pooling, HTTP/2 multiplexing, unified retry, timeouts, circuit breaker, backpressure, stream resumption, TACC ([`docs/refactor/27-transport-layer-consolidation.md`](../refactor/27-transport-layer-consolidation.md))
- **Protocol** — LATT binary framing, delta encoding, manifest
- **Policy on the wire** — compression (IR transforms), layered cache, guardrails, agent memory, MCP tool shaping, receipts

Compression is one capability of that layer, not the product definition. SDKs are thin clients ([`docs/refactor/13-python-sdk-quality.md`](../refactor/13-python-sdk-quality.md), [`docs/refactor/17-typescript-sdk.md`](../refactor/17-typescript-sdk.md)); algorithms live in one place ([`docs/refactor/SINGLE_SOURCE_OF_TRUTH.md`](../refactor/SINGLE_SOURCE_OF_TRUTH.md)).

Forward plan constraints (lightweight, no external LLM, self-hosted, code budget): [`docs/refactor/FORWARD_PLAN.md`](../refactor/FORWARD_PLAN.md).

**Eval-driven v2 additions:** [ARCHITECTURE_EVAL_INSIGHTS.md](../refactor/ARCHITECTURE_EVAL_INSIGHTS.md) (what production evals prove, what we reject from the external critique).

---

## Architecture reality (v1.0 production evals)

Pinned benchmark: `benchmarks/results/v1.0.0.json` ([CLAIMS.md](../../benchmarks/results/CLAIMS.md)).

| Layer | Strength | Weakness |
|---|---|---|
| **Structural transforms** | Tables, JSON, grammar scenarios — high reduction with quality preserved | None on structural workloads |
| **Semantic summarization** | — | Long-form / API-doc scenarios — quality ~0.64–0.67 |
| **Orchestration** | Reasoning paths conservative and strong | `features not reached by pipeline` when global tier blocks the right transform |
| **Transport economics** | Delta, prefix, cache transforms exist | Gains not yet fully wired into planner utility (Phases 22, 27) |

**Implication:** v2 optimizes **utility** (cost, latency, cache, stability, quality) — not headline compression %. See [The Scoring Rule](#the-scoring-rule) and [ARCHITECTURE_EVAL_INSIGHTS.md](../refactor/ARCHITECTURE_EVAL_INSIGHTS.md).

**Not re-founding:** Legacy `CompressorPipeline`, `strategy_selector`, and parallel schedulers were removed in refactor Phases 3–5. Do not schedule a second “delete legacy” rewrite.

---

## The Fundamental Insight

LATTICE is not a prompt compressor.
LATTICE is a **semantic operating system for inference** — implemented as a **transport layer** with policy hooks on every request/response byte.

This document is the single source of truth for **lifecycles and module boundaries**. Every module, every class, every function must trace back to one of the lifecycles defined here and to exactly one home in [`SINGLE_SOURCE_OF_TRUTH.md`](../refactor/SINGLE_SOURCE_OF_TRUTH.md).

---

## The Five Lifecycles

```
Request Lifecycle          IR Lifecycle           Candidate Lifecycle
     │                         │                        │
     ▼                         ▼                        ▼
┌─────────┐              ┌─────────┐              ┌─────────────┐
│ Request │ ──compile──▶ │ PromptIR│ ──optimize──▶│ Candidates  │
└─────────┘              └─────────┘              └─────────────┘
     │                                                     │
     │              Execution Lifecycle                   │
     │                    │                               │
     │                    ▼                               │
     │              ┌─────────────┐                       │
     └─────────────▶│ ExecutionPlan│◀───────────────────┘
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │TransportPlan│
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ ProviderExec│
                    └─────────────┘
```

---

## 1. Request Lifecycle

**Purpose:** Transform external API requests into the internal canonical form.

**Phases:**

1. **Deserialize** (O(1)) — JSON/OpenAI format → `Request` dataclass
2. **Validate** (O(1)) —schema, size limits, rate limits
3. **Profile** (O(n)) —semantic analysis, risk scoring, task classification
4. **Compile** (O(n)) —build PromptIR (the ONE canonical IR)

**Output:** `Request` + `PromptIR` + `SemanticProfile`

**Rule:** Nothing after this phase touches `Request.messages` as raw text. Everything operates on `PromptIR`.

---

## 2. IR Lifecycle

**Purpose:** Maintain the ONE canonical structured representation.

**Phases:**

1. **Build** (`ir_builder.py`) — parse messages into typed sections/spans
2. **Normalize** (`ir_normalizer.py`) —canonicalize JSON keys, detect patterns, lift constraints
3. **Annotate** — add compression hints, protection flags, entity refs
4. **Serialize** (`ir_serializer.py`) —render to LLM-readable text

**Key Types:**

```python
@dataclass(frozen=True, slots=True)
class PromptIR:
    """The ONE canonical intermediate representation."""
    sections: tuple[Section, ...]  # Immutable
    metadata: frozendict[str, Any]
    
    def with_section(self, index: int, section: Section) -> PromptIR:
        """Return new PromptIR with replaced section (immutable)."""
        ...
```

**Critical Rule:** `PromptIR` is **immutable**. Every transform returns a NEW `PromptIR`.

---

## 3. Candidate Lifecycle

**Purpose:** Search the space of possible optimizations.

**Key Insight:** Candidates are NOT mutable state. They are **immutable snapshots** in a search graph.

```python
@dataclass(frozen=True, slots=True)
class Candidate:
    """Immutable snapshot of an optimization state."""
    ir: PromptIR                          # The canonical IR
    applied: tuple[str, ...]             # Transform names applied (in order)
    metrics: frozendict[str, float]      # Token counts, latency, quality
    provenance: frozendict[str, Any]     # For replay/debugging
    
    def apply(self, transform_name: str, new_ir: PromptIR) -> Candidate:
        """Return NEW candidate, never mutate self."""
        ...
    
    @property
    def score(self) -> float:
        """Computed from metrics (quality, savings, latency, risk, cache, transport)."""
        ...
```

**Search Algorithm:**

```python
def search(initial: Candidate, transforms: list[Transform]) -> Candidate:
    """True state-space search with immutable candidates."""
    beam = [initial]
    
    for step in range(max_depth):
        expanded = []
        for cand in beam:
            # Option 1: skip this step
            expanded.append(cand)
            
            # Option 2: apply each transform
            for tx in transforms:
                if tx.can_process(cand.ir):
                    new_ir = tx.optimize(cand.ir)
                    if validate(new_ir):
                        new_cand = cand.apply(tx.name, new_ir)
                        if new_cand.score > min_score:
                            expanded.append(new_cand)
        
        beam = top_k(expanded, k=beam_width)
    
    return max(beam, key=lambda c: c.score)
```

**Critical Rule:** No transform mutates in place. `cand.apply()` creates a NEW candidate.

### Segment-aware planning (Phase 23)

Mixed prompts are not one global tier. After [23-segment-aware-planning.md](../refactor/23-segment-aware-planning.md):

```
PromptIR sections → SegmentPlan (per-section policy)
    → allowed_transforms, distortion_budget, transport_hint
    → UnifiedPlanner + pipeline gates + beam search respect per-section allowlists
```

This fixes “transform exists but planner never activated it” without adding transforms.

---

## 4. Execution Lifecycle

**Purpose:** Turn the best candidate into an execution plan.

**Key Insight:** The planner does not decide "what runs." It decides "in what order, with what budget."

```python
@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable plan for executing a candidate."""
    transforms: tuple[str, ...]          # Ordered transform names
    quality_floor: float
    latency_budget_ms: float
    cache_plan: CachePlan | None
    transport_plan: TransportPlan | None
    
    @classmethod
    def from_candidate(cls, candidate: Candidate) -> ExecutionPlan:
        """Derive plan from the winning candidate's provenance."""
        ...
```

**Rule:** The scheduler produces ONE `ExecutionPlan`. The pipeline executes it without re-deciding.

---

## 5. Transport Lifecycle

**Purpose:** Optimize the wire representation.

```python
@dataclass(frozen=True, slots=True)
class TransportPlan:
    """Immutable plan for wire optimization."""
    delta_encoding: bool
    compression_codec: str | None
    framing_scheme: str
    provider_cache_hint: str | None
    stable_prefix_hash: str | None
```

**Phases:**
1. **Delta Detection** — compare to session state
2. **Prefix Canonicalization** — extract stable prefix, compute hash
3. **Cache Alignment** — structure messages for provider KV cache
4. **Framing** — binary/multipart encoding for wire

---

## The Unified Type Hierarchy

```
LatticeObject (abstract base, frozen)
├── SemanticProfile
│   ├── TaskClassification
│   ├── RiskScore
│   └── ContentProfile
│
├── PromptIR
│   ├── Section
│   │   └── Span
│   └── CompressionHint
│
├── Candidate
│   ├── CandidateGraph
│   └── CandidateScore
│
├── ExecutionPlan
│   ├── CachePlan
│   └── TransportPlan
│
└── ExecutionResult
    ├── Response
    └── SessionUpdate
```

**Rule:** Every type in this hierarchy is:
- Frozen (immutable)
- Hashable (for caching)
- Serializable (for replay)
- Copy-on-write (via `.with_*()` methods)

---

## Module Boundaries (Enforced)

| Module | Owns | Must NOT reach into |
|--------|------|---------------------|
| `ir/` | PromptIRV2, Section, Span, quality, validation | transform implementations, providers |
| `core/` | Config, context, errors, result (leaf primitives) | planner, pipeline orchestration |
| `planner/` | UnifiedPlanner, ExecutionPlan, task classification | transform implementations |
| `pipeline/` | Pipeline runner, safety gates, representation beam search | provider HTTP |
| `transforms/optimizers/` | Per-domain optimizer orchestrators | providers |
| `transforms/` | Legacy + IR-native transforms | planner scheduling |
| `runtime/` | TierClassifier (workload complexity tiers) | provider selection |
| `protocol/` | Wire formats, framing, manifests | IR scoring logic |
| `providers/` | Provider adapters, transport, credentials | IR internals |
| `proxy/` | HTTP server | optimization logic |

---

## The Transformation Rule

**Old Model (being replaced):**
```python
class OldTransform:
    def process(self, request: Request, context: TransformContext) -> Result[Request, TransformError]:
        # Mutates request in place
        request.messages[0].content = modify(request.messages[0].content)
        return Ok(request)
```

**Current model:**
```python
class IRTransform:
    def optimize(self, ir: PromptIR, context: TransformContext) -> Result[PromptIR, TransformError]:
        # Returns NEW PromptIR, never mutates
        new_sections = modify_sections(ir.sections)
        return Ok(PromptIR(sections=tuple(new_sections), metadata=ir.metadata))
    
    def reverse(self, response: Response, context: TransformContext) -> Response:
        ...
```

**Migration Path:**
1. Wrap all existing transforms in `IRTransform` adapters
2. Run mixed mode: IR-native where available, text fallback for legacy
3. Phase out text transforms one by one
4. Eventually: optimizer layer only accepts IR transforms

---

## The Scheduler Rule

**Old Model (removed in refactor Phase 4):**
- RATS (`decide_schedule`) decided allowed transforms
- OptimizerScheduler decided allowed optimizers
- ExecutionBuilder reconciled them
- Pipeline gated everything again

**Current model (Phases 3–4):**
- **ONE** `UnifiedPlanner.plan()` produces **ONE** `ExecutionPlan`
- `content_profiler` builds `SemanticProfile` and may call the planner if no plan is pre-set
- `Pipeline.compress()` executes the plan verbatim via safety gates in `pipeline/gates.py`
- No runtime re-decision. No parallel schedulers.

---

## The Scoring Rule

**Old Model (removed or consolidated in Phases 3–5 and 12):**
- Per-optimizer `_Candidate.score` formulas
- `planner/unified_planner._estimate_utility` fixed bonus table (not a real objective — honesty pass in Phase 12 documents this)

**Target model (Phase 13 lands formula home; Phases 22 / 23 / 27 / 14 feed inputs):**
- **ONE** `CandidateScorer` in `ir/quality.py` (or `pipeline/scoring.py` per [SINGLE_SOURCE_OF_TRUTH.md](../refactor/SINGLE_SOURCE_OF_TRUTH.md))
- **ONE** composite:

```
expected_utility =
    + semantic_quality
    + provider_cache_probability    # Phase 22 analyzer → planner
    + transport_gain                # Phase 27 delta_wire + stable prefix metrics
    - normalized_token_cost
    - latency_p95_estimate
    - instability_penalty           # replay / placeholder leakage
```

- Weights are profile-configurable ([Phase 23](../refactor/23-receipts-bandit-profiles.md)), not hard-coded.
- Compression ratio is **one term**, never the sole success metric in docs or CLAIMS.

---

## Validation facade (Phase 12)

All validation entrypoints route through **`runtime/validation_engine.py`** (facade only — no new algorithms):

| Concern | Canonical module |
|---|---|
| IR structural validation | `ir/validation.py` |
| Post-transform guard (rule-based) | `pipeline/post_transform_guard.py` + `pipeline/checks.py` |
| Output / JSON repair | `safety/output/` ([Phase 15](../refactor/15-native-guardrails.md), extended in [Phase 19](../refactor/19-compression-intelligence.md)) |

**Contract:** `tests/contract/test_replay_determinism.py` — same request + profile + plan → same `canonical_fingerprint()` and transform trace (operator replay gate).

---

## Migration Plan

### Phase 1: Refoundation (Week 1-2)
- [x] Create `primitives.py` with unified types
- [x] Make `PromptIR` fully immutable
- [x] Build `Candidate` and `CandidateGraph`
- [x] Create `IRTransform` base class
- [x] Write architecture doc (this document)

### Phase 2: IR Unification (Week 3-4) — MOSTLY DONE
- [x] Merge SIG + semantic segments into PromptIR annotations
- [x] Merge protocol manifest into PromptIR as `protocol` section
- [x] Make `content_profiler` build PromptIR ONLY (no parallel IRs)
- [x] **Update all consumers to read from PromptIR** — DONE via `get_canonical_request_value()` / `get_canonical_state_value()` which check PromptIR metadata first, then request metadata, then session state. 160+ call sites across core, gateway, providers, client, integrations.
- [x] Build canonical state helpers in `runtime_state.py`
- [x] Content profiler writes all canonical state into PromptIRV2 metadata

### Phase 3: Immutable Candidates (Week 5-6) — DONE
- [x] Convert beam search to immutable candidate graph
- [x] Remove nested optimizer execution
- [x] Flatten optimizer hierarchy (no more "optimizers calling optimizers")
- [x] Update validation to operate on candidates

### Phase 4: Unified Planner (Week 7-8) — DONE
- [x] Replace RATS + OptimizerScheduler + ExecutionBuilder with one `UnifiedPlanner`
- [x] Delete `core/scheduler.py` and `core/optimizer_scheduler.py` (refactor Phase 4)
- [x] Move planner modules to `planner/`; optimizers to `transforms/optimizers/`
- [x] Rename `runtime/router.py` → `runtime/tier_classifier.py` (`TierClassifier`)
- [x] Implement utility-based scoring
- [x] Build provider cache simulator

### Phase 5: IR-Native Only (Week 9-10) — DONE
- [x] Move `runtime_contract` and `tool_filter` onto immutable IR
- [x] **Remove duplicate transform execution** — output tail runs once via `ExecutionPlan` + `Pipeline.compress()`
- [x] **Fix placeholder leakage guard** — beam-search `<ref_N>` substitutions are not rolled back by expansion guards
- [x] **Convert transforms to IR-native `optimize()`** — canonical path uses `UnifiedPlanner` + `Pipeline.compress()` gates
- [x] **Wire IR-native transforms into the canonical path** — `UnifiedPlanner._TRANSFORM_ORDER` and tier allowlists match the registry
- [x] **Remove regex-only transforms from canonical path** — `alias_manifest`, `dictionary_compress`, `grammar_compress` are NOT in `_TRANSFORM_ORDER` or any `_TIER_ALLOWED` tier.
- [x] Build stable prefix compiler
- [x] Build session delta planner

### Phase 6: Evaluation & Hardening (Week 11-12) — DONE
- [x] **Canonical fingerprints** — `PromptIRV2.canonical_fingerprint()` deterministic SHA-256 for replay hardening
- [x] **ScenarioResult replay hardening fields** — `request_fingerprint`, `execution_plan_fingerprint`, `determinism_score`, `survivability_score`, `replay_drift`, `longitudinal_index`
- [x] **Benchmark runner integration** — `replay.py` captures `PromptIRV2.canonical_fingerprint()` and `ExecutionPlan` fingerprint on each trace replay
- [x] Determinism validation
- [ ] Task survivability evals (framework ready, data collection deferred)
- [ ] Multi-model evals (framework ready, data collection deferred)
- [ ] Longitudinal replay (framework ready, data collection deferred)
- [ ] Agent benchmark evals (framework ready, data collection deferred)

---

## Enforcement

Every PR must:
1. Trace back to this document
2. Not cross module boundaries (see table above)
3. Not introduce mutable state in candidates or IR
4. Not duplicate scoring logic
5. Not introduce new "scheduling" logic

This document is law.
