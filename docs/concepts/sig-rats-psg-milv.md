# SIG, RATS, PSG, MILV — The Safety Architecture

LATTICE's safety layer consists of four cooperating subsystems with clear contracts. Each layer answers one question and produces metadata consumed by downstream layers.

## Architecture

```
                    ┌─────────────┐
          Request → │  SIG        │ → What MATTERS in this request?
                    │(semantic    │   Spans, importance, protection
                    │ importance) │
                    └──────┬──────┘
                           │ protected spans, risk score, task signals
                    ┌──────▼──────┐
                    │  RATS       │ → What MAY run on this request?
                    │(task-aware  │   Task class, transform gating,
                    │ scheduler)  │   budget, execution order
                    └──────┬──────┘
                           │ schedule (allowed/blocked), task class
                    ┌──────▼──────┐
                    │  PSG        │ → What MUST NEVER happen?
                    │(pipeline    │   Entity preservation, format
                    │ safety)     │   checks, expansion guards
                    └──────┬──────┘
                           │ rollback/reject, safety decisions
                    ┌──────▼──────┐
                    │  MILV       │ → Does the RESULT still work?
                    │(model-in-   │   Task equivalence, blank output
                    │ the-loop)   │   check, production validation
                    └─────────────┘
```

## SIG — Semantic Importance Graph

**Question**: What content matters in this request?

**Files**: `src/lattice/core/semantic_graph.py`, `src/lattice/transforms/content_profiler.py`

SIG segments the request into spans and computes importance scores:

### Span Segmentation

Content is split by structure boundaries — sentences, code blocks, JSON blocks, Markdown tables, diffs, log lines, repeated phrases. Each span gets typed as `narrative`, `code`, `json`, `table`, `diff`, or `log_line`.

### Feature Extraction

| Feature | Computation |
|---------|------------|
| `frequency` | Occurrence count in full text |
| `position_weight` | First/last span bias (1.0 → 0.4) |
| `entity_density` | UUID × 3 + number × 0.5 + URL × 2, normalized |
| `dependency_score` | Word overlap with later spans |
| `task_relevance` | Task indicator words (error, root cause, mitigation...) |
| `reasoning_signal` | Explicit reasoning markers (because, therefore, deduce...) |

### Importance Formula

```
importance = 0.25 × frequency_norm
           + 0.20 × dependency_score
           + 0.20 × entity_density
           + 0.20 × task_relevance
           + 0.15 × position_weight

Boost: × 1.5 if reasoning_signal is true
```

### Protected Spans

Spans with importance ≥ 40.0 are marked protected. Reasoning signals always protect. Entity-dense code/tables get a higher threshold (80.0) to avoid over-protecting boilerplate.

**Metadata keys**: `_lattice_sig`, `_lattice_sig_summary`, `_lattice_protected_spans`

---

## RATS — Runtime-Aware Transform Scheduler

**Question**: What transforms may run, and in what order?

**Files**: `src/lattice/core/task_classifier.py`, `src/lattice/core/scheduler.py`

### Task Classification

Every request is classified into one of five task types:

| Type | Signals | Conservative? |
|------|---------|--------------|
| `retrieval` | "find", "lookup", "search", "select" | No |
| `summarization` | Long text (>500 words), "summarize" | No |
| `analysis` | "compare", "trend", "pattern", "correlation" | No |
| `debugging` | Logs, stack traces, errors, "debug", "crash" | Yes |
| `reasoning` | "why", "deduce", "infer", "solve", "prove" | Yes |

Conservative tasks (debugging, reasoning) get stricter gating — CONDITIONAL and DANGEROUS transforms are blocked, higher transform budget (50ms vs 20ms).

### Transform Gating

```
SAFE:         Always allowed
CONDITIONAL:  Allowed at LOW/MEDIUM risk. Blocked on conservative tasks.
DANGEROUS:    Allowed at LOW risk only. Blocked on conservative tasks + HIGH risk.
UNKNOWN:      Treated as DANGEROUS — must be explicitly registered.
```

### Scheduler Decision

The scheduler produces a `SchedulerDecision` with:
- `allowed_transforms`: what can run
- `blocked_transforms`: what must not run
- `protected_span_count`: number of protected spans from SIG
- `budget_ms`: runtime budget from task classification

**Metadata key**: `_lattice_schedule`

---

## PSG — Pipeline Safety Guard

**Question**: What must never happen?

**Files**: `src/lattice/core/guardrails.py`, `src/lattice/core/pipeline.py`

PSG runs after each irreversible transform and enforces hard safety constraints:

### Checks

| Guard | When | Action on Failure |
|-------|------|-------------------|
| **Risk gate** | Before execution | Skip transform |
| **Scheduler gate** | Before execution | Skip blocked transforms |
| **Protected-span veto** | Before execution (DANGEROUS only) | Skip transform |
| **Expansion guard** | After execution | Rollback if ratio > max |
| **Entity preservation** | After execution (irreversible only) | Rollback |
| **Format preservation** | After execution (irreversible only) | Rollback |
| **Blank output** | Post-provider response | Fail/reject |

### Irreversible Transforms

Entity and format checks only apply to irreversible transforms that genuinely discard content:
- `message_dedup`, `rate_distortion`, `semantic_compress`, `structural_fingerprint`, `hierarchical_summary`

Reversible transforms (`reference_sub`, `dictionary_compress`, `grammar_compress`) store referent mappings and restore them on `reverse()` — no real content loss.

### Fail-Closed Behavior

| Mode | Graceful Degradation | Strict |
|------|---------------------|--------|
| Rollback | Continue to next transform (restore backup) | Return error immediately |
| Entity loss | Continue | Return `PSG_ENTITY_LOSS` error |
| Format loss | Continue | Return `PSG_FORMAT_LOSS` error |

**Metadata keys**: `_lattice_safety_decision`, `_lattice_rollback_reason`

---

## MILV — Model-In-the-Loop Validation

**Question**: Does the optimized request still produce the correct answer?

**Files**: `src/lattice/core/guardrails.py`, `benchmarks/evals/runner.py`, `src/lattice/gateway/compat.py`

### Benchmark Path

Full provider validation:
1. Run baseline prompt through provider
2. Run optimized (compressed) prompt through provider
3. Structural evaluator scores task equivalence across 7 dimensions
4. LLM judge (same model) may supplement structural pass
5. Fail if task equivalence composite < 0.85

### Production Path

Lightweight post-response validation:
- High-risk requests (CRITICAL/HIGH risk, debugging/reasoning task): always flagged
- Low-risk requests: 1% sampled
- Blank-output check runs after provider response
- `check_blank_output()`: fails on truly blank outputs, passes numbers/refusals/short facts

### Task Equivalence Dimensions

| Dimension | 1.0 = Perfect |
|-----------|--------------|
| `constraint_preservation` | Same constraints followed |
| `entity_preservation` | All entities preserved |
| `format_preservation` | Structure unchanged |
| `reasoning_correctness` | Reasoning chain intact |
| `refusal_correctness` | Refusals match |
| `answer_completeness` | Answer as complete |
| `harmful_drift` | No placeholder leakage |

**Metadata key**: `_lattice_validation`

---

## Production Enforcement Flow

```
Request → SIG (content_profiler)
       → RATS (task classification + scheduler)
       → Pipeline:
           for each transform:
             risk gate   → skip unsafe transforms
             scheduler   → skip blocked transforms
             span veto   → skip DANGEROUS on protected content
             execute     → run transform
             expansion   → rollback if token explosion
             PSG check   → rollback on entity/format loss (irreversible only)
       → Provider call
       → MILV check   → blank output? rollback
       → Response
```

## Observability

Every decision is logged and recorded in request metadata:

```json
{
  "_lattice_safety_decision": {
    "applied": ["content_profiler", "runtime_contract", "output_cleanup"],
    "skipped": ["rate_distortion", "hierarchical_summary"],
    "risk_blocked": ["structural_fingerprint"],
    "rollback_reasons": {"message_dedup": "entity_loss_12_entities"}
  }
}
```

All safety decisions appear in the benchmark report under the **PSG Safety Decisions** table.
