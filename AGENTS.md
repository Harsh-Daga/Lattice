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

LATTICE is a **unified optimization + transport system** converging on one canonical v2 runtime:

```
Request → content_profiler → UnifiedPlanner → ExecutionPlan → PipelineV2 → Provider
```

The authoritative architecture document is [`docs/architecture/runtime_v2.md`](docs/architecture/runtime_v2.md).

### Key Modules

| Directory | Responsibility |
|-----------|---------------|
| `core/` | Immutable primitives (PromptIRV2, Candidate, ExecutionPlan), PipelineV2, UnifiedPlanner, config, guardrails |
| `transforms/` | Individual transforms — mostly IR-native via `optimize(ir, ...)`, legacy `process()` kept as compat bridge |
| `optimizer/` | Beam-search orchestrators (representation_optimizer, reference_optimizer, structure_optimizer, etc.) |
| `protocol/` | Prefix canonicalization, cache planners, binary framing, manifest |
| `providers/` | Per-provider adapters (serialization, streaming, HTTP pooling) |
| `proxy/` | FastAPI server with OpenAI-compatible endpoints |
| `gateway/` | HTTP compatibility layer, routing headers |

## Code Conventions

- **Result[T,E] monad** for error handling: `Ok(value)` or `Err(error)`
- **ReversibleSyncTransform** base class for all transforms
- **Immutable PromptIRV2** — transforms return new instances via `.with_sections()` / `.with_spans()` / `.with_text()`
- **Explicit allowlists** — PipelineV2._IR_NATIVE_TRANSFORMS defines which transforms run natively
- **Single scheduler** — UnifiedPlanner is the only scheduling decision maker
- **mypy strict**, **ruff** for linting; run both before commits

## Adding a Transform

1. Extend `ReversibleSyncTransform` with `name` and `priority`
2. Implement `process(Request, TransformContext) → Result[Request, TransformError]` (legacy path)
3. Implement `optimize(PromptIRV2, Request, TransformContext) → Result[PromptIRV2, TransformError]` (v2 IR-native path)
4. Register in `transform_registry.py` and `pipeline/runner.py` `_IR_NATIVE_TRANSFORMS`

## Testing

- Unit tests: `tests/unit/` — 1845 tests
- Integration tests: `tests/integration/` — proxy sessions, Redis, IR optimizer E2E
- E2E tests: `tests/e2e/` — agent wrappers, full pipeline
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

## Latest Benchmark

| Metric | Value |
|--------|-------|
| Tests passed | **1845/1845** |
| ruff errors | **0** |
| mypy errors | **0** |
