# Phase 6 Agent Prompt — Providers & Transport Consolidation

> **Copy this entire document into a new agent session.** Branch from `main` after Phase 5 merge.
>
> **Numbering:** STATUS **Phase 6** = spec file [`05-providers-transport.md`](05-providers-transport.md) (the spec’s internal title says “Phase 5”; ignore that — use STATUS numbering).

---

## Mission

Implement **STATUS Phase 6** end-to-end per [`05-providers-transport.md`](05-providers-transport.md), line by line. Do not skip acceptance criteria. Do not mark the phase complete until every §10 checkbox is true and CI gates pass.

**Outcome:** Split `src/lattice/providers/transport.py` (1539 LoC) into `providers/transport/` (7 modules); move adapters to `providers/adapters/`; merge duplicated streaming into one `_stream()`; consolidate tool sanitizers; add `RateLimitTracker` TTL eviction; export all 17 adapters from `lattice.providers`.

---

## Mandatory reading (in order)

1. [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) — §2 public surface, §4 ground rules (R1–R7), §5 dependency direction, §7 CI gates.
2. [`05-providers-transport.md`](05-providers-transport.md) — **entire file** (§1–§13).
3. [`FINAL_LAYOUT.md`](FINAL_LAYOUT.md) — `providers/` target tree.
4. [`PHASE_COMPLETION_TRACKER.md`](PHASE_COMPLETION_TRACKER.md) — Phases 0–5 closed; update Phase 6 when done.
5. [`AGENTS.md`](../../AGENTS.md) — conventions (`Result`, ruff, mypy, no deleted imports).
6. Skim callers: `rg "from lattice.providers" src/ tests/ benchmarks/ gateway/ proxy/`

---

## Ground rules (non-negotiable)

| Rule | Requirement |
|------|-------------|
| **R1 Public surface** | `from lattice.providers import DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager` unchanged. All **17** adapter classes importable from `lattice.providers` (§6.6, §9.3). |
| **R2 Tests gate merge** | `uv run ruff check src/ tests/`, `uv run mypy src/lattice/`, `uv run pytest tests/ -q`, `uv run pytest tests/contract/ -q` — all green. |
| **R3 File size** | No file under `src/lattice/providers/` > **850 LoC** after split. |
| **R4 No bridge layers** | No new `_compat` modules. Keep **both** public methods `completion_stream` and `completion_stream_with_stall_detect`; bodies delegate to unified `_stream(stall_detect=...)` (§6.10). |
| **R5 Imports downhill** | `providers/transport/*` may import `providers/adapters`, `providers/credentials`, `transport/congestion`, `core/*`. Adapters must not import `transport/completion.py`. Shared helpers live in `transport/helpers.py` only (§11 risk: no cycle between completion ↔ streaming). |
| **R6 Honest naming** | `stall_detector` lives under `providers/transport/`. |
| **R7 No TODO debt** | File gaps in tracker/docs, not `TODO` in production code. |

---

## PR strategy (three sequential PRs — §13)

Use branches and merge in order (each independently revertible):

### PR 6a — `refactor/phase-6a-providers-adapters`

**Scope:** §6.1–6.2, §6.5, §6.7, §6.9 (adapter imports only).

```bash
git checkout main && git pull && git checkout -b refactor/phase-6a-providers-adapters
mkdir -p src/lattice/providers/adapters
git mv src/lattice/providers/base.py src/lattice/providers/adapters/base.py
git mv src/lattice/providers/openai.py src/lattice/providers/adapters/openai.py
git mv src/lattice/providers/openai_compatible.py src/lattice/providers/adapters/openai_compatible.py
git mv src/lattice/providers/anthropic.py src/lattice/providers/adapters/anthropic.py
git mv src/lattice/providers/azure.py src/lattice/providers/adapters/azure.py
git mv src/lattice/providers/bedrock.py src/lattice/providers/adapters/bedrock.py
git mv src/lattice/providers/gemini.py src/lattice/providers/adapters/gemini.py
git mv src/lattice/providers/ollama.py src/lattice/providers/adapters/ollama.py
```

- Implement `providers/adapters/__init__.py` exactly per §6.5 (`__all__` lists all 17 adapters).
- Fix adapter imports to **relative** `.base`, `.openai` where appropriate (§6.7).
- Rewrite `from lattice.providers.{openai,anthropic,...}` across `src/ tests/ benchmarks/` per §6.9 cheatsheet.
- Update `providers/__init__.py` to re-export all 17 adapters (§6.6).
- **Do not** split `transport.py` yet.
- Add/update contract test `test_all_adapters_importable_at_top_level` (§9.3).
- CI gates green; open PR.

### PR 6b — `refactor/phase-6b-providers-sanitizer-stall`

**Scope:** §4, §5, §6.2 (stall only), §6.11–6.12.

```bash
git checkout main && git pull  # after 6a merged
git checkout -b refactor/phase-6b-providers-sanitizer-stall
mkdir -p src/lattice/providers/transport
git mv src/lattice/providers/stall_detector.py src/lattice/providers/transport/stall_detector.py
```

- Refactor `tool_sanitizer.py` per §4 (`ToolSanitizer` base + `AnthropicToolSanitizer` / `BedrockToolSanitizer` subclasses).
- Implement `transport/rate_limits.py` with `RateLimitState.last_seen_at`, `RATE_LIMIT_TTL_S`, `_maybe_cleanup` (§5).
- Extract `RateLimitTracker` from monolith `transport.py` into `rate_limits.py` **or** stub package re-export from monolith until 6c — prefer moving class now if imports stay stable.
- Rewrite `from lattice.providers.stall_detector` → `from lattice.providers.transport.stall_detector` (§6.9).
- Add `tests/unit/providers/test_tool_sanitizer_inheritance.py`, `tests/unit/providers/transport/test_rate_limits_ttl.py` (§9.2).
- CI green; open PR.

### PR 6c — `refactor/phase-6c-providers-transport-split`

**Scope:** §3, §6.3–6.4, §6.8, remaining §9.

```bash
git checkout main && git pull  # after 6b merged
git checkout -b refactor/phase-6c-providers-transport-split
```

1. Copy/split `transport.py` into package per §2.1 and §6.3 table:
   - `registry.py` — `ProviderRegistry`, `_resolve_provider_name`, aliases
   - `pool.py` — `ConnectionPoolManager`
   - `rate_limits.py` — (if not done in 6b) `RateLimitTracker`
   - `helpers.py` — URL/key resolution, SSE parse, `_build_request`, retry helpers, TACC helpers
   - `streaming.py` — unified `_stream(..., stall_detect=, stream_state=)`; thin `completion_stream` / `completion_stream_with_stall_detect` (<30 LoC each)
   - `completion.py` — `DirectHTTPProvider` with **Option A mixin** from `streaming.py` (§6.3)
   - `stall_detector.py` — already in package from 6b
2. `providers/transport/__init__.py` per §6.4.
3. **Delete** `providers/transport.py`.
4. `registry.py` imports adapters from `lattice.providers.adapters` (§6.8).
5. Run full import rewrite; verify `rg "from lattice.core.pipeline|CompressorPipeline"` unaffected.
6. Add `tests/unit/providers/transport/test_streaming_single_path.py`, `tests/unit/providers/test_registry_complete.py` (§9.2).
7. Move tests per §9.1 where files exist (ok to defer minor moves to Phase 10 if paths already work — but §9.1 is in spec; prefer doing moves that match `test_*` file names in repo).

---

## Implementation details you must not get wrong

### Streaming merge (§3)

- Single `_stream(self, request, *, provider_name, stall_detect=False, stream_state=None, ...)`.
- Branch only at: stall_detector hooks, SSE line processor (`_process_sse_line_with_state` vs `_parse_sse_line`), shared buffer/retry/TACC/rate-limit finally blocks.
- **Keep** `completion_stream` and `completion_stream_with_stall_detect` as public API (gateway/proxy call the latter).

### RateLimitTracker (§5)

- `record()` updates `last_seen_at`; calls `_maybe_cleanup()` (throttled every 5 min).
- Evict entries older than `RATE_LIMIT_TTL_S = 3600.0`.
- Safe dict rebuild — no mutate-while-iterate (§11).

### Tool sanitizer (§4)

- `ContextVar` for per-task ID maps.
- `sanitize_tool_ids` / `restore_tool_call_ids` helpers preserved.

### Import stability

- `from lattice.providers.transport import DirectHTTPProvider` must keep working via package `__init__.py`.
- `mcp_to_anthropic` stays at `providers/mcp_to_anthropic.py`; adapters import it with `from lattice.providers.mcp_to_anthropic import ...`.

---

## Verification checklist (run before each PR merge)

```bash
# Layout
test ! -f src/lattice/providers/transport.py          # after 6c only
test -d src/lattice/providers/transport
test -d src/lattice/providers/adapters
test ! -f src/lattice/providers/base.py             # after 6a
test ! -f src/lattice/providers/stall_detector.py     # after 6b

# LoC cap
find src/lattice/providers -name '*.py' -exec wc -l {} + | awk '$1 > 850 {print; exit 1}'

# Imports
uv run python -c "
from lattice.providers import (
    DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager,
    OpenAIAdapter, AnthropicAdapter, AzureAdapter, BedrockAdapter,
    GeminiAdapter, VertexAdapter, GroqAdapter, TogetherAdapter, DeepSeekAdapter,
    PerplexityAdapter, MistralAdapter, FireworksAdapter, OpenRouterAdapter,
    CohereAdapter, AI21Adapter, OllamaAdapter, OllamaCloudAdapter,
)
print('imports ok')
"

# CI
uv run ruff check src/ tests/
uv run ruff format --check src/
uv run mypy src/lattice/ --ignore-missing-imports
uv run pytest tests/ -q
uv run pytest tests/contract/ -q
```

### Benchmark gate (§10 last item)

After 6c on `main`:

```bash
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 1 --warmup 0 --provider-warmup 0 \
  --output-json benchmarks/results/phase-6.json

python scripts/compare_benchmarks.py \
  benchmarks/results/phase-0-baseline.json \
  benchmarks/results/phase-6.json \
  --tolerance-pct 2
```

Requires `OLLAMA_CLOUD_API_KEY` in environment. If unavailable, document in PR and `PHASE_COMPLETION_TRACKER.md` — do not claim §10 complete without artifact or explicit waiver.

---

## §10 Acceptance — tick every box in PR description

Copy §10 from `05-providers-transport.md` into PR body and mark each item with evidence (command output or file path).

---

## Documentation updates when done

1. [`PHASE_COMPLETION_TRACKER.md`](PHASE_COMPLETION_TRACKER.md) — add Phase 6 section, all ✅ except benchmark if waived.
2. [`STATUS.md`](STATUS.md) — Phase 6 → ✅ Done with merge SHA.
3. [`AGENTS.md`](../../AGENTS.md) — fix stale paths (`transforms/registry.py`, Phases 0–5 complete, `providers/adapters/`).
4. [`04-transforms.md`](04-transforms.md) — no changes unless import paths touched.

---

## Anti-patterns (will fail review)

- Leaving `transport.py` monolith while adding `transport/` package (two sources of truth).
- Import cycles between `completion.py` and `streaming.py` (use `helpers.py`).
- Removing `completion_stream_with_stall_detect` (gateway depends on it).
- Exporting only 10 adapters at top level (§1 issue #6).
- Files >850 LoC (especially `streaming.py` — split helpers if needed).
- Committing benchmark JSON with secrets or local-only paths as required CI artefacts without key.

---

## Suggested first commands

```bash
cd lattice && git checkout main && git pull
wc -l src/lattice/providers/transport.py
rg "completion_stream_with_stall_detect" src/
rg "from lattice.providers\.(openai|anthropic|base)" src/ tests/ | wc -l
uv run pytest tests/unit/providers/ tests/unit/test_stall_detector.py -q
```

Good luck. **One concern per file. One streaming path. Seventeen adapters visible. Tests green.**
