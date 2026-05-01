# Safety

LATTICE classifies every transform into a safety bucket and computes a 0-100 semantic risk score before applying lossy transforms. The goal: never let a transform silently change meaning.

## Safety Buckets

| Bucket | Behavior |
|--------|----------|
| **SAFE** | Runs on any input. Structural-only transforms that don't change meaning. |
| **CONDITIONAL** | Lossy but recoverable. Runs at LOW or MEDIUM risk. Blocked at HIGH+. |
| **DANGEROUS** | Can replace meaning-bearing content with placeholders. Only runs at LOW risk. |

Unknown transforms default to DANGEROUS — they must be explicitly registered in the safety map to prove safety.

## Semantic Risk Score

Computed by `content_profiler` (priority 1) from 8 dimensions, each contributing 0 to its max:

| Dimension | Max Score | Signal |
|-----------|-----------|--------|
| **strict_instructions** | 30 | "do not", "must", "exactly", "preserve", "without changing" |
| **sensitive_domain** | 20 | Legal, medical, financial, safety, security, confidential |
| **structured_output** | 15 | JSON/XML prefixes, markdown tables, tool definitions |
| **high_stakes_entities** | 15 | UUIDs, URLs, 5+ distinct numbers |
| **reasoning_heavy** | 20 | "reason step by step", "think carefully", "explain why", "deduce" |
| **intentional_repetition** | 10 | Lines repeated 3+ times (may be intentional) |
| **tool_call_dependency** | 10 | Tool call count × 2.5 |
| **formatting_constraints** | 10 | Requests for tables, JSON, CSV output formats |

### Risk Levels

| Level | Score | What's Allowed |
|-------|-------|---------------|
| **LOW** | 0-20 | All transforms allowed |
| **MEDIUM** | 20-40 | SAFE + CONDITIONAL allowed |
| **HIGH** | 40-60 | Only SAFE transforms |
| **CRITICAL** | >60 | Only SAFE transforms + logged |

### Example: Risk Scores

```
"Hello, how are you?"                    → LOW    (~5)
"Debug this: 550e8400-..."               → LOW    (~15)
"Return valid JSON with these rules..."   → MEDIUM (~30)
"Medical diagnosis. Must preserve format" → HIGH   (~45)
"Legal document. Do not change wording"   → CRITICAL (>60)
```

## Pipeline Gating

Every transform runs through a sequence of gates:

```
config check → policy check → runtime budget → risk gate → expansion guardrail → execution
```

If any gate blocks the transform:
- The pre-transform state is rolled back
- The reason is logged at WARNING level
- Metrics record `risk_blocked: true` and `risk_block_reason`

## Blank Outputs

If either the baseline or optimized output is empty (provider error, timeout, pipeline failure), task-equivalence scoring returns all-zeros automatically. A broken response never passes.

## Intermediate Expansion Guard

Each transform is checked for token expansion. If output tokens > input × `max_transform_expansion_ratio` (default 1.5x), the transform is aborted. The `expansion_ratio` metric is recorded for observability.

## Provider Validation

The benchmark suite runs each scenario through the real provider with both baseline and compressed prompts. A structural evaluator (deterministic, no LLM) scores task equivalence across 7 dimensions. An LLM judge can supplement but never overrule a structural fail. See [Benchmarks](benchmarks.md).
