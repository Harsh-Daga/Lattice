# LATTICE — Development Guide for AI Agents

## Setup

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync

# Run proxy
uv run python -m lattice.proxy.server --port 8787

# Run tests
uv run pytest tests/ -q

# Run benchmarks (local, no keys needed)
uv run python benchmarks/evals/cli.py --suite feature
```

## Architecture

```
Application → [Proxy:8787 or SDK] → Pipeline → DirectHTTPProvider → LLM Provider
```

Key modules:
- `src/lattice/core/` — Request/Response models, pipeline orchestrator, session manager, config
- `src/lattice/providers/` — Per-provider adapters (serialization, streaming, HTTP pooling)
- `src/lattice/transforms/` — 14+ transforms running in priority order through the pipeline
- `src/lattice/proxy/` — FastAPI server with OpenAI-compatible endpoints
- `src/lattice/protocol/` — Canonical segments, cache planners, binary framing
- `src/lattice/gateway/` — HTTP compatibility layer, routing headers

## Code Conventions

- **Result[T,E] monad** for error handling: `Ok(value)` or `Err(error)`
- **ReversibleSyncTransform** base class for all transforms
- **Serialization** via `lattice.core.serialization` as single source of truth
- **Message** uses `content_parts: list[ContentPart]` for multimodal; `content: str` for text
- **mypy strict**, **ruff** for linting; run both before commits

## Adding a Transform

1. Extend `ReversibleSyncTransform` with `name` and `priority`
2. Implement `process(Request, TransformContext) → Result[Request, TransformError]`
3. Implement `reverse(Response, TransformContext) → Response`
4. Register in `lattice.core.pipeline_factory.build_default_pipeline()`
5. Add to `_TRANSFORM_SAFETY_MAP` in `lattice.utils.validation`

## Transform Priority Order

```
 1: content_profiler     (classifies content, computes risk score)
 2: runtime_contract     (enforces transform budget)
 9: cache_arbitrage      (reorders for KV-cache alignment)
10: prefix_optimizer     (deduplicates common prefixes)
12: structural_fingerprint (pattern detection)
14: self_information     (entropy-based filtering)
15: message_dedup        (exact/near-duplicate removal)
20: reference_sub        (UUID/URL/hash substitution)
22: rate_distortion      (semantic compression)
25: dictionary_compress  (phrase dictionary)
28: hierarchical_summary (nested structure summarization)
30: tool_filter          (tool output projection)
40: output_cleanup       (whitespace normalization)
```

## Pipeline Risk Gating

Every transform runs through: config check → policy check → runtime budget → **risk gate** → expansion guardrail → execution. The risk gate reads the semantic risk score (computed by `content_profiler`, priority 1) and blocks CONDITIONAL/DANGEROUS transforms on high-risk inputs.

## Testing

- Unit tests: `tests/unit/` — 1584 tests
- Integration tests: `tests/integration/` — proxy sessions, Redis, session correctness
- Safety gate tests: `tests/unit/test_safety_gates.py` — risk scoring, transform buckets, task equivalence
- Run with `uv run pytest tests/ -q`

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `LATTICE_PROVIDER_BASE_URL` | Default upstream provider base URL |
| `LATTICE_PROVIDER_BASE_URLS` | JSON dict of `{provider: url}` overrides |
| `OPENAI_API_KEY` | Used for OpenAI, Azure, and OpenAI-compatible providers |
| `ANTHROPIC_API_KEY` | Anthropic provider |

## Benchmark CLI

```bash
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama \
  --provider-model ollama=llama3.2 \
  --iterations 3 --warmup 1
```

Suites: `all`, `feature`, `feature-matrix`, `provider`, `protocol`, `transport`, `integration`, `capability`, `replay`, `replay-governance`, `tacc`, `control`.
