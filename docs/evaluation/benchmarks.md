# Benchmarks

LATTICE benchmarks evaluate three independent layers: prompt correctness, provider task success, and transport performance.

## Running Benchmarks

```bash
# Local only (no provider keys)
uv run python benchmarks/evals/cli.py --suite feature

# All suites with live provider
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 1 --warmup 0 --provider-warmup 0
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--suite SUITE` | all | One of: `all`, `feature`, `feature-matrix`, `provider`, `protocol`, `transport`, `integration`, `capability`, `replay`, `replay-governance`, `tacc`, `control` |
| `--providers P...` | all | Provider names to evaluate |
| `--provider-model P=M` | catalog | Model override per provider |
| `--scenarios S...` | all | Scenario filter |
| `--iterations N` | 1 | Local eval iterations |
| `--warmup N` | 0 | Local warmup iterations |
| `--provider-iterations N` | 1 | Live provider iterations |
| `--provider-warmup N` | 1 | Live provider warmup |
| `--output-json PATH` | `benchmarks/results/production_evals.json` | JSON output |
| `--output-md PATH` | `benchmarks/results/production_evals.md` | Markdown report |
| `--json-only` | — | Print JSON to stdout |
| `--regression-threshold-quality F` | 0.05 | Max quality drop (5%) |
| `--regression-threshold-latency F` | 1.5 | Max latency multiplier (1.5x) |

## Three Evaluation Layers

### Layer A: Prompt Transform Correctness

Evaluates whether compression preserves meaning without calling any LLM.

```bash
uv run python benchmarks/evals/cli.py --suite feature
```

Tests: token reduction, transform activation, safety profile, lossy transform gating.

### Layer B: Provider-Aligned Task Success

Runs scenarios through real providers with both baseline and compressed prompts, then compares outputs.

```bash
uv run python benchmarks/evals/cli.py --suite provider \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud
```

Tests: task equivalence (7 dimensions), answer correctness, reasoning preservation, format preservation.

### Layer C: Transport and Execution

Tests protocol features and infrastructure.

```bash
uv run python benchmarks/evals/cli.py --suite protocol
uv run python benchmarks/evals/cli.py --suite transport
uv run python benchmarks/evals/cli.py --suite tacc
```

Tests: connection pools, framing roundtrips, manifest generation, TACC congestion control.

## Scenario Categories

The benchmark suite includes 18 scenarios:

| Scenario | Category | What It Tests |
|----------|----------|---------------|
| uuid_deduplication | reference_sub | UUID compression and entity preservation |
| tool_output_filtering | tool_filter | Internal field stripping |
| table_compression | format_conversion | Markdown table → CSV |
| prefix_optimization | prefix_opt | Multi-turn prefix reuse |
| code_review_patterns | structural_fingerprint | Code pattern detection |
| api_docs_summarization | dictionary_compress | Boilerplate removal |
| tool_call_preservation | integration | Tool call roundtrip integrity |
| json_response_format | format_conversion | JSON structure preservation |
| cache_arbitrage_prefix | cache_arbitrage | Cache alignment |
| dictionary_repetition | dictionary_compress | Phrase compression |
| cleanup_noise | output_cleanup | Whitespace normalization |
| message_dedup_turns | message_dedup | Duplicate turn removal |
| semantic_compress_longform | semantic_compress | Narrative compression |
| context_budget_pressure | context_selector | Budgeted selection |
| runtime_contract_pressure | runtime_contract | Complexity tiering |
| grammar_json_table | grammar_compress | Structured data |
| simple_baseline | baseline | No-op check |
| mixed_realworld | mixed | Full pipeline exercise |

## Report Output

The Markdown report includes:

- **Feature Coverage**: Per-feature token savings and ratios
- **Transform Summary**: Per-transform avg before/after/saved
- **Scenario Proof**: Tier matches, feature matches, flaws
- **Feature Matrix**: On/off delta for each feature
- **Provider Validation**: Task equivalence, token pass/fail, safety gates
- **Response Samples**: Baseline vs optimized for inspection
- **Prompt Safety**: Per-scenario safety profiles
- **Usage and Costs**: Token usage and cost estimates

See [Safety](safety.md) for how task equivalence scoring works.
