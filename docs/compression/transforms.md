# Transforms

LATTICE's pipeline runs 18 transforms in priority order on every request. Each transform has a safety classification and runs through config checks, policy gates, risk gating, and expansion guardrails before execution.

## Pipeline Order

| Priority | Transform | Safety Bucket | Purpose |
|----------|-----------|---------------|---------|
| 1 | **content_profiler** | SAFE | Classifies content type, computes risk score, selects compression strategy |
| 2 | **runtime_contract** | SAFE | Enforces transform time budget based on request complexity tier |
| 3 | **strategy_selector** | SAFE | Routes to submodular/adaptive selection strategies |
| 9 | **cache_arbitrage** | SAFE | Reorders messages for KV-cache alignment, sets provider cache hints |
| 10 | **prefix_optimizer** | SAFE | Deduplicates common prefixes across messages |
| 12 | **structural_fingerprint** | DANGEROUS | Detects repeated structural patterns for compression |
| 14 | **self_information** | CONDITIONAL | Entropy-based content filtering |
| 15 | **message_dedup** | CONDITIONAL | Removes exact and near-duplicate messages across turns |
| 18 | **context_selector** | SAFE | Token-budgeted document selection from long contexts |
| 19 | **information_theoretic_selector** | CONDITIONAL | Information-theoretic relevance scoring for content selection |
| 20 | **reference_sub** | CONDITIONAL | Replaces repeated UUIDs, URLs, paths with short inline references |
| 22 | **rate_distortion** | CONDITIONAL | Semantic/extractive compression of long-form text |
| 24 | **grammar_compress** | CONDITIONAL | Grammar-based compression of structured data |
| 25 | **dictionary_compress** | CONDITIONAL | Phrase-level dictionary compression for repeated patterns |
| 28 | **hierarchical_summary** | DANGEROUS | Recursive summarization of nested document structures |
| 30 | **tool_filter** | SAFE | Projects tool outputs to keep only relevant fields |
| 40 | **output_cleanup** | SAFE | Whitespace normalization, trailing boilerplate removal |

---

## SAFE Transforms

These run on any input. They don't change meaning — only structure or packaging.

### Content Profiler (priority 1)

Classifies requests into 12 content types: CODE_HEAVY, TABLE_HEAVY, NARRATIVE_LONG, TOOL_OUTPUT, LOG_OUTPUT, DIFF_OUTPUT, STACK_TRACE, GREP_OUTPUT, FILE_TREE, MCP_OUTPUT, MIXED, SHORT. Computes semantic risk scores and sets per-transform enable/disable flags for downstream transforms.

### Runtime Contract (priority 2)

Assigns a tier (SIMPLE/MEDIUM/COMPLEX/REASONING) based on content depth, tool usage, and length. Sets a max transform latency budget that budget-sensitive transforms must respect.

### Cache Arbitrage (priority 9)

Reorders prompt messages into stability buckets: system → tools → static docs → variable content. Applies provider-specific cache hints (OpenAI prompt_cache_key, Anthropic cache_control breakpoints, Bedrock prompt caching). Maximizes KV-cache hit probability.

### Prefix Optimizer (priority 10)

Identifies common prefix content shared across messages and deduplicates it. Benchmarks show 10-60% savings on multi-turn conversations with stable system prompts.

### Context Selector (priority 18)

When context exceeds token budget, selects the most relevant documents using BM25 and submodular optimization.

### Tool Filter (priority 30)

Strips internal-only fields from tool output JSON. Keeps user-visible schema fields. Configurable to preserve specific paths.

### Output Cleanup (priority 40)

Normalizes whitespace, trims trailing boilerplate, normalizes JSON formatting. Always safe — no content ever removed.

---

## CONDITIONAL Transforms

These are lossy but recoverable. They only run when the semantic risk score is LOW or MEDIUM.

### Reference Substitution (priority 20)

Replaces repeated UUIDs, URLs, file paths, and hash digests with short inline references like `<ref_0>`. Fully reversible — the proxy expands references back on the response path. Example:

```
Before: "Transaction 550e8400-e29b-41d4-a716-446655440000 failed at /api/v2/endpoint/..."
After:  "Transaction <ref_0> failed at <ref_1>..."
```

### Message Dedup (priority 15)

Detects exact and near-duplicate messages across conversation turns. Removes obvious duplicate turn pairs. Conservative — only removes when content is byte-identical or very close.

### Format Conversion (priority 25)

Converts markdown tables, JSON arrays, and structured output to compact CSV/TSV format. Reduces token count significantly for data-heavy prompts. Example:

```
Before: | ID | Name | Dept | Salary |
After:  ID,Name,Dept,Salary\n0,Emp_0,Eng,100000
```

### Rate Distortion / Semantic Compress (priority 22)

Extractive compression of long-form natural language text. Preserves sentences with highest salience scores. Ratio configurable (default 0.6x). Best for narrative long-form content like incident reports, documentation, or story contexts.

### Dictionary Compress (priority 25)

Learns repeated phrases and replaces them with short dictionary references. HPACK-style phrase compression. Example:

```
Before: "The session manifest should remain stable..."
After:  "<d_0> <d_1> <d_2> <d_3>..."
```

### Grammar Compress (priority 24)

Structured data compression using learned grammar patterns. Best for JSON arrays, repetitive API responses, and tabular data.

### Information Theoretic Selector (priority 19)

Information-theoretic document selection using pointwise mutual information and relevance scoring. More aggressive than context_selector.

### Self Information (priority 14)

Filters low-information content using per-token information density estimates. Removes filler words, boilerplate, and redundant qualifiers.

---

## DANGEROUS Transforms

These can replace meaning-bearing content with placeholders. They only run at LOW risk (< 20 on the 0-100 scale). Blocked at MEDIUM+.

### Structural Fingerprint (priority 12)

Detects repeated structural patterns (code blocks, review templates, document shells) and compresses them. Can reduce tokens significantly but must never be applied when the structure carries semantic meaning.

### Hierarchical Summary (priority 28)

Recursive summarization of deeply nested document trees. Compresses parent sections by summarizing children. Dangerous because summaries can lose critical details.

---

## Execution Transforms

These run outside the main pipeline at the execution layer:

| Transform | Purpose |
|-----------|---------|
| **Batching** | Coalesces independent requests for shared provider calls |
| **Speculative Execution** | Pre-runs predicted next-turn prompts in parallel |
| **Delta Encoding** | Session-based incremental message sending |

---

## Risk Gating

Before any CONDITIONAL or DANGEROUS transform runs, the pipeline checks the semantic risk score computed by `content_profiler`:

| Risk Level | Score | SAFE | CONDITIONAL | DANGEROUS |
|------------|-------|------|-------------|-----------|
| LOW | 0-20 | ✅ | ✅ | ✅ |
| MEDIUM | 20-40 | ✅ | ✅ | ❌ |
| HIGH | 40-60 | ✅ | ❌ | ❌ |
| CRITICAL | >60 | ✅ | ❌ | ❌ |

See [Safety](safety.md) for how the risk score is computed.

## Expansion Guardrails

Each transform's output is checked against its input. If tokens expand more than `max_transform_expansion_ratio` (default 1.5x), the transform is aborted and the pre-transform state is restored.

## Adding a Transform

1. Extend `ReversibleSyncTransform` in `src/lattice/core/pipeline.py`
2. Implement `process()` and `reverse()` methods  
3. Assign `name` and `priority`
4. Register in `src/lattice/core/pipeline_factory.py`
5. Add safety bucket entry in `src/lattice/utils/validation.py`
