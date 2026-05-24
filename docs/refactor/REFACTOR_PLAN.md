# LATTICE v1.0.0 Refactor — Master Plan

> **Purpose.** Take the current LATTICE codebase (166 Python files, ~50 KLOC, multiple parallel paths) and converge it to **one canonical path per concern** at version **1.0.0**. Preserve every user-visible feature. Break internal imports freely (major version bump).
>
> **Audience.** Anyone — including someone new to the repo — who needs to execute a slice of the refactor. Every phase document is self-contained: it tells you exactly which files to read, which functions to move, which lines to delete, and what to verify before declaring the phase done.

---

## 0. Why this refactor exists

Today the repo carries **at least three parallel implementations** of each major concern:

| Concern | Parallel implementations today |
|---|---|
| IR construction | `core/ir.py`, `core/ir_builder.py`, `core/ir_normalizer.py`, `core/ir_serializer.py`, `core/ir_transform.py`, `core/primitives.py`, `core/compiler.py` (7 files) |
| Pipeline | `core/pipeline.py` (v1), `core/pipeline_v2.py`, `core/pipeline_v2_wrapper.py`, `core/pipeline_factory.py` (4 files) |
| Scheduler / planner | `core/scheduler.py`, `core/optimizer_scheduler.py`, `core/unified_planner.py`, `planner/execution_builder.py` (4 files) |
| Transport | `transport/types.py` (Request/Response), `providers/transport/` (HTTP dispatch), `transport/congestion.py` (TACC), `protocol/` (binary framing) |
| Optimizers vs transforms | `optimizer/structure_optimizer.py` vs `optimizer/ir_structure_optimizer.py`; `optimizer/*` orchestrating `transforms/*` |

The result is a "feature-rich" codebase whose internal flow is impossible to summarise on one page. README claims **18 transforms**; the registry exposes **25**; some are no-op wrappers (`prefix_opt`), some aren't transforms at all (`semantic_segmenter`), some are misnamed (`runtime/router.py` — the README explicitly says "LATTICE is not a router").

**This refactor's hard guarantee:** at v1.0.0, for every concern (request shape, IR, planning, transform execution, provider dispatch, transport, telemetry, state), **there is exactly one file that owns it**. All others are deleted or renamed.

---

## 1. What v1.0.0 keeps and what it does not

### 1.1 User-visible features — all preserved

The plan **does not** remove any of the following. They must continue to work end-to-end with the same external contract:

- **CLI** — every `lattice <command>` from §"Public Surface Lock" below works with identical syntax.
- **Proxy HTTP endpoints** — every `/v1/*`, `/lattice/*`, `/healthz`, `/metrics` endpoint preserves request/response shape and headers.
- **17 providers** — OpenAI, Anthropic, Groq, DeepSeek, Mistral, Cohere, Gemini, Vertex, Azure, Bedrock, Ollama, Ollama Cloud, OpenRouter, Fireworks, Together, Perplexity, AI21.
- **5 agent integrations** — Claude Code, Codex, Cursor, OpenCode, GitHub Copilot. `lattice lace <agent>` and `lattice init` behave identically.
- **All 18 documented transforms** — content_profiler, runtime_contract, cache_arbitrage, prefix_optimizer (folded into content_profiler internally — see §3.4), message_dedup, reference_sub, rate_distortion, path_prefix, format_conversion, tool_projection, tool_filter, output_cleanup, columnar_pack, json_shape, extractive_compress, diagnostic_rle, causal_chain, constraint_lifting. Execution transforms (batching, speculative, delta_encode, auto_continuation) also preserved.
- **TACC** — token-aware congestion control.
- **Binary framing protocol** — native LATTICE wire protocol.
- **Delta encoding** — session-based incremental message sending.
- **Semantic cache** — exact-hash + approximate-fingerprint, in-memory & Redis.
- **MILV** — multi-input loss validation.
- **HMAC resume tokens** — for streaming resume.

### 1.2 What v1.0.0 *removes* (internal only, no user impact)

These are duplicated or dead internal artefacts. None are part of any user-facing contract.

| Removed | Why | What survives |
|---|---|---|
| `core/pipeline.py` (1092 LoC, v1) + `core/pipeline_v2_wrapper.py` (118) | Two parallel pipelines collapse to one | `core/pipeline_v2.py` → renamed `pipeline/runner.py` |
| `core/scheduler.py` (443) + `core/optimizer_scheduler.py` (131) | Two parallel schedulers collapse into the planner | `core/unified_planner.py` → renamed `planner/unified_planner.py` |
| `core/compiler.py` (58) | Pure 3-line wrapper around `build_ir → normalize_ir → serialize_ir` | Calls inlined into `ir/builder.py` |
| `transforms/prefix_opt.py` (161) | Its own header marks it deprecated; logic already in content_profiler | content_profiler is canonical |
| `src/lattice/evals/` | Empty / dead | `benchmarks/evals/` is canonical |
| `src/lattice/proxy/compat_exports.py` | Already deleted (git status) | — |
| Every transform's legacy `process(Request) → Result[Request]` method, where an IR-native `optimize(ir, ...) → Result[ir]` also exists | Eliminates the dual-path bridge in 11 transforms | Only `optimize(ir, ...)` remains |
| `optimizer/structure_optimizer.py` (text-based) | Superseded by `optimizer/ir_structure_optimizer.py` (IR-native) | The IR-native one |
| `transforms/strategy_selector.py` (728 LoC, bandit-learning) | No benchmark evidence; content_profiler heuristics replace it | Gate: deleted **unless** the audit in Phase 4 finds measured wins |
| `context_selector` information-theoretic variant (~half of 373 LoC) | No benchmark evidence; submodular is canonical | submodular path only, configurable by flag |
| `transforms/semantic_segmenter.py` (267 LoC) | Not a transform, just a dataclass module | Moved to `core/segmentation.py` |
| `runtime/router.py` (358 LoC) — file *name* only | Contradicts README's "not a router" claim | Code preserved, renamed to `runtime/tier_classifier.py` |

### 1.3 What v1.0.0 *splits* (no behavior change, structural only)

Three files exceed 700 LoC and conflate orthogonal concerns. Each becomes a package:

| File | LoC | Concerns it conflates | New package |
|---|---|---|---|
| `transforms/content_profiler.py` | 985 | profile classification · risk scoring · task classification · IR build · prefix canonicalisation · manifest building · cache plan simulation · execution plan derivation · scheduler bridge · segmenter glue | `transforms/content_profiler/{__init__.py,classifier.py,risk_scorer.py,task_classifier_bridge.py,planner_bridge.py}` |
| `transforms/format_conv.py` | 794 | Markdown↔CSV · JSON↔YAML · format detection · IR-native optimize · reverse | `transforms/format_converter/{__init__.py,table_converter.py,json_converter.py}` |
| ~~`providers/transport.py`~~ | — | **Shipped Phase 6:** split into `providers/transport/{registry,pool,rate_limits,helpers,completion,streaming,stall_detector}.py` + `providers/adapters/` |

### 1.4 What v1.0.0 *moves* (no behavior change, layout only)

| From | To | Reason |
|---|---|---|
| `core/ir.py`, `core/ir_builder.py`, `core/ir_normalizer.py`, `core/ir_serializer.py`, `core/ir_transform.py`, `core/primitives.py`, `core/semantic_graph.py` | `ir/` package | Domain cohesion |
| `optimizer/ir_native_optimizer.py`, `optimizer/validation.py`, `optimizer/quality_estimator.py` | `ir/` | IR-adjacent, used only by IR optimisers |
| `core/scheduler.py` *(deleted)*, `core/optimizer_scheduler.py` *(deleted)*, `core/unified_planner.py`, `core/task_classifier.py`, `core/runtime_state.py` | `planner/` | Joins existing `planner/` decision layer |
| `core/pipeline_v2.py`, `core/pipeline_factory.py`, `core/policy.py`, `core/guardrails.py`, `core/milv.py`, `core/auto_continuation.py`, `core/batch_accumulator.py`, `optimizer/representation_optimizer.py` | `pipeline/` | Execution-time concerns live together |
| `core/transport.py` (types: Role, Message, Request, Response, Transform protocols) | `transport/types.py` | Resolves the three-way name collision |
| `core/serialization.py`, `core/delta_wire.py`, `core/session.py` | `transport/` | Transport-layer concerns |
| `core/store.py`, existing `state/segment_store.py` | `state/` | All stateful persistence together |
| `core/semantic_cache.py` | `cache/semantic.py` | New top-level domain |
| `core/metrics.py`, `core/telemetry.py`, `core/agent_stats.py`, `core/cost_estimator.py`, `core/maintenance.py`, `utils/streaming_sketches.py` | `telemetry/` (was empty `observability/`) | Single telemetry surface |
| `core/transform_registry.py`, `core/transform_reputation.py` | `transforms/` | Owns its own registry & reputation |
| `utils/validation.py` | `safety/risk_scoring.py` | New domain |
| `utils/patterns.py` | `transforms/patterns.py` | Only used by transforms |
| `core/credentials.py` | `providers/credentials.py` | Provider-scoped |
| `optimizer/structure_optimizer.py` *(deleted)*, `optimizer/reference_optimizer.py`, `optimizer/tool_optimizer.py`, `optimizer/diagnostic_optimizer.py`, `optimizer/context_optimizer.py` | `transforms/optimizers/` | These orchestrate transforms; live with transforms |
| `runtime/router.py` | `runtime/tier_classifier.py` (renamed) | Resolves README contradiction |
| `transforms/semantic_segmenter.py` | `core/segmentation.py` | Not a transform |

### 1.5 What v1.0.0 *renames* (semantic clarity)

| Old | New | Reason |
|---|---|---|
| `core/pipeline_v2.py` | `pipeline/runner.py` | v1 gone; "v2" is just "the" runner |
| `core/unified_planner.py` | `planner/unified_planner.py` | Same name, new location |
| `runtime/router.py` | `runtime/tier_classifier.py` | Honest naming |
| `transforms/format_conv.py` | `transforms/format_converter/__init__.py` | Full name + package split |
| `transforms/prefix_opt.py` | *deleted* | — |
| `core/compiler.py` | *deleted* | — |

---

## 2. The public surface lock (the user contract)

Anything in this section **must** still work, byte-for-byte, at v1.0.0. The CI gate for each phase verifies it. This is the boundary between "internal refactor" (anything goes) and "user impact" (forbidden).

### 2.1 CLI commands

```
lattice --help / -h / -v / --version

lattice proxy run     [--host H] [--port P] [--workers N] [--mode safe|balanced|aggressive] [--reload] [--no-ui]
lattice proxy start   [--host H] [--port P] [--workers N] [--mode M]
lattice proxy stop    [--grace SECONDS] [--force]
lattice proxy restart
lattice proxy status

lattice init          [agents...] [--port P] [--local|--global] [--start-proxy]
lattice lace <agent>  [--port P] [--no-start] [--no-patch] [--no-tunnel] [--dry-run] [-- agent_args...]
lattice unlace <agent>

lattice info
lattice config        [--json]
lattice health        [--host H] [--port P]
lattice status
lattice doctor        [agent]
lattice benchmark
```

Supported `<agent>` values: `claude`, `codex`, `cursor`, `opencode`, `copilot`, `generic`.

### 2.2 HTTP endpoints

```
POST   /v1/chat/completions          (OpenAI format, streaming + non-streaming)
POST   /v1/messages                  (Anthropic format, streaming + non-streaming)
GET    /v1/models
POST   /v1/responses                 (OpenAI responses API)
GET    /v1/responses/{id}
DELETE /v1/responses/{id}
WS     /v1/responses                 (WebSocket)
WS     /v1/chat/completions          (WebSocket, Codex)
POST   /lattice/session/start
POST   /lattice/session/append
GET    /lattice/session/{session_id}
POST   /lattice/session/invalidate
POST   /lattice/gateway              (native LATTICE binary/JSON entry)
GET    /healthz
GET    /readyz
GET    /startupz
GET    /metrics                      (Prometheus)
GET    /stats                        (JSON snapshot)
```

Plus the Codex response aliases: `/v1/codex/responses`, `/backend-api/responses`.

### 2.3 Response headers

```
x-lattice-compression           (% saved)
x-lattice-session-id            (session opaque ID)
x-lattice-delta                 (true/false — was delta-encoded)
x-lattice-cost-usd              (estimated cost)
x-lattice-provider              (chosen provider)
x-lattice-transforms-applied    (comma-separated transform names)
```

Plus all `x-ratelimit-*` headers passed through from the upstream provider.

### 2.4 Public Python API

```python
from lattice import LatticeClient, LatticeProxyClient, CompressResult, wrap_openai_client
from lattice import __version__
```

`LatticeClient.compress(messages, model, mode)` and `.compress_request()`, `.decompress_response()`, `.health()`, `.compression_stats()`, `.count_tokens()` all keep the v0.x signatures.

### 2.5 Configuration

`LatticeConfig` field names and `lattice.yaml` keys are stable. Environment variables — `LATTICE_PROVIDER_BASE_URL`, `LATTICE_PROVIDER_BASE_URLS`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` — unchanged.

---

## 3. The 12 phases

Each phase has its own file in this directory. The phases are sequenced so each can land as a self-contained PR. Phase N's tests must pass before Phase N+1 starts.

| # | Phase | File | Net file count change |
|---|---|---|---|
| 0 | Audit & baseline (file inventory, API lockdown, CI gates) | [00-audit-baseline.md](00-audit-baseline.md) | 0 (read-only) |
| 1 | IR & core primitives — collapse 7 IR files → `ir/` package | [01-ir-primitives.md](01-ir-primitives.md) | −1 (delete `compiler.py`) |
| 2 | Pipeline runner — v1 gone, one canonical `pipeline/runner.py` | [02-pipeline-runner.md](02-pipeline-runner.md) | −2 (delete `pipeline.py`, `pipeline_v2_wrapper.py`) |
| 3 | Planner — collapse 3 schedulers into `planner/unified_planner.py`; rename `runtime/router.py` | [03-planner-collapse.md](03-planner-collapse.md) | −2 (delete `scheduler.py`, `optimizer_scheduler.py`) |
| 4 | Transforms — audit, split 3 god-files, delete `prefix_opt`, gate `strategy_selector`/`context_selector` on benchmarks | [04-transforms.md](04-transforms.md) | varies (see file) |
| 5 | Providers & transport — split `providers/transport.py` (1539 LoC) into 7-file package, merge duplicated streaming methods | [05-providers-transport.md](05-providers-transport.md) | +5 (split) but −300 LoC of duplication |
| 6 | Proxy / Gateway / SDK / CLI — wire health routes, lock public surface | [06-proxy-sdk-cli.md](06-proxy-sdk-cli.md) — ✅ STATUS Phase 7 | −1 (delete `compat_exports.py` if not already) |
| 7 | Agent integrations — extract base, per-agent diffs only | [07-integrations.md](07-integrations.md) — ⏳ STATUS Phase 8 | 0 |
| 8 | Observability / state / protocol / utils — populate `telemetry/`, move state files, redistribute utils | [08-observability-state.md](08-observability-state.md) | −1 (`utils/` shrinks to 1 file or empty) |
| 9 | Benchmarks & evals — `src/lattice/evals/` deleted, `benchmarks/` canonical, every README claim has a measurable run | [09-benchmarks.md](09-benchmarks.md) | −1 dir (`src/lattice/evals/`) |
| 10 | Tests reshape — mirror `tests/unit/` to new `src/` layout; feature-parity matrix | [10-tests.md](10-tests.md) | 0 (only moves) |
| 11 | Docs & release — README rewrite, runtime_v2 → runtime, CHANGELOG, MIGRATION.md, tag v1.0.0 | [11-docs-release.md](11-docs-release.md) | varies |

**Other reference docs** in this directory:

- [FINAL_LAYOUT.md](FINAL_LAYOUT.md) — the complete target directory tree
- [MIGRATION.md](MIGRATION.md) — v0.x → v1.0.0 user-facing migration (mostly: "import paths changed; CLI unchanged")
- [FEATURE_PARITY.md](FEATURE_PARITY.md) — checklist proving every feature from v0.x still works

---

## 4. Ground rules every phase obeys

These rules are repeated in each phase document for self-containment.

1. **Public surface (§2) is sacred.** Any phase that breaks it must include an explicit, accepted exception in this master plan.
2. **Tests gate the phase.** A phase is "done" only when: `uv run pytest tests/ -q` passes, `uv run ruff check src/` is clean, `uv run mypy src/lattice/` is clean, and a fresh `uv run python benchmarks/evals/cli.py --suite all` produces non-regressing numbers vs. the Phase 0 baseline (Phase 0 captures the baseline).
3. **One concern per file.** No file in v1.0.0 may exceed 800 LoC. If it does, split it.
4. **No bridge layers.** The plan has no "legacy adapter" files, no `_compat` modules, no `process()`-and-`optimize()` duals. v1 is dead at v1.0.0.
5. **Imports point downhill.** The dependency direction is fixed (§5 below). Cycles are bugs.
6. **Naming is honest.** A file called `router.py` either routes between providers or is renamed. A file called `compiler.py` either compiles something non-trivial or is deleted.
7. **No "TODO: clean up later" markers.** Anything not done is filed as a separate task in the parent project tracker, not as code comments.

---

## 5. Final dependency direction

This is the import hierarchy v1.0.0 enforces. A module may import from any layer **below** it; never above; cycles must be broken by injection.

```
                        +-------------------+
                        |   cli/, ui.py     |   (top — user-facing entry)
                        +---------+---------+
                                  |
                  +---------------+----------------+
                  |                                |
            +-----v------+                  +------v-----+
            |   proxy/   |                  |    sdk/,   |
            |  gateway/  |                  |  client.py |
            +-----+------+                  +------+-----+
                  |                                |
                  +---------------+----------------+
                                  |
                          +-------v--------+
                          |   pipeline/    |   (runner.py, factory, policy, guardrails, milv,
                          |                |    representation_optimizer, auto_continuation, batch_accumulator)
                          +-------+--------+
                                  |
                  +---------------+----------------+
                  |                                |
            +-----v------+                  +------v-----+
            |  planner/  |                  | transforms/|
            |            |                  |  optimizers/|
            +-----+------+                  +------+-----+
                  |                                |
                  +---------------+----------------+
                                  |
                  +---------------+----------------+
                  |               |                |
            +-----v---+    +------v-----+   +------v-----+
            |   ir/   |    |  providers/|   |  transport/|
            +-----+---+    +------+-----+   +------+-----+
                  |               |                |
                  +---------------+----------------+
                                  |
                  +---------------+----------------+
                  |               |                |
            +-----v---+    +------v---+    +-------v----+
            |  cache/ |    |  state/  |    |  protocol/ |
            +---------+    +----------+    +------------+
                                  |
                          +-------v--------+
                          |  telemetry/    |
                          +-------+--------+
                                  |
                          +-------v--------+
                          | core/, safety/ |   (leaf primitives — config, context, errors, result, risk scoring)
                          +-------+--------+
                                  |
                          +-------v--------+
                          |    utils/      |   (token_count.py only)
                          +----------------+
```

**Reading rule.** Anything in `pipeline/` may import from `planner/`, `transforms/`, `ir/`, `providers/`, `transport/`, `cache/`, `state/`, `protocol/`, `telemetry/`, `core/`, `safety/`, `utils/`. It may **not** import from `cli/`, `proxy/`, `gateway/`, `sdk/`.

The current codebase has several cycles between `core/` and `transforms/` (transforms import `core/transform_registry`; the registry imports `core/config` which imports back). Phase 1 fixes these by moving the registry into `transforms/` (where it belongs) and config back into a leaf `core/`.

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| A transform's legacy `process()` path is still used by a code path we missed → silent feature regression | Medium | Phase 4 includes a static grep for every `process()` call site; integration tests cover every transform |
| `pipeline/runner.py` (was `pipeline_v2.py`) doesn't yet handle some edge case that `pipeline.py` (v1) handled | Medium-high | Phase 2 ports every v1 test to v2 before deletion. Diff coverage required before merge. |
| Provider streaming after the `providers/transport.py` split has a regression | Medium | Phase 5 keeps a behaviour-only test suite that records SSE bytes from a known prompt and replays them post-refactor |
| The `core/transport.py` → `transport/types.py` move breaks every import in the repo | High (mechanical) | Phase 1 has a scripted `ruff --fix` + `sed` migration; manual review limited to non-import call sites |
| Benchmark numbers regress after `strategy_selector` / `context_selector` cuts | Low | Phase 4 will not delete them until benchmarks show no regression in compression %, quality, or latency |
| MCP integration breaks (it touches Anthropic's adapter through `mcp_to_anthropic.py`) | Low-medium | Phase 5 keeps `mcp_to_anthropic.py` as-is, just moves it under `providers/` |
| Redis backend regresses after `core/session.py` + `core/store.py` move to `state/` | Low | Phase 8 runs the existing `tests/integration/test_redis_*.py` suite |
| Public Python API breaks because someone imported from a moved path | Mitigated | `src/lattice/__init__.py` re-exports the v1.0 names from their new locations; `import lattice; lattice.LatticeClient` still works |

---

## 7. CI gates (what blocks a phase from merging)

Every phase PR must pass — in this exact order — before merge:

```bash
# 1. Lint
uv run ruff check src/ tests/ benchmarks/

# 2. Type-check
uv run mypy src/lattice/

# 3. Full test suite (1600+ tests)
uv run pytest tests/ -q

# 4. Benchmarks — no regression in compression %, quality, latency p99
uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json benchmarks/results/phase-N.json

# 5. Diff vs Phase 0 baseline
uv run python scripts/compare_benchmarks.py \
    benchmarks/results/phase-0-baseline.json \
    benchmarks/results/phase-N.json \
    --tolerance-pct 2
```

Tolerance is **±2 %** on every measured metric. A regression beyond that blocks the merge.

---

## 8. How to execute this plan

Per phase, the routine is:

1. Read this master plan + the phase document end-to-end.
2. Read every file listed under "Files touched" in that phase.
3. Create a feature branch named `refactor/phase-<N>-<slug>`.
4. Follow the per-file "before → after" instructions in the phase doc. Each instruction is granular enough to apply without rereading the source unless a conflict surfaces.
5. Run all five CI gates locally. Fix until green.
6. Open PR with title `Phase <N>: <one-line summary>`. PR body links to the phase doc.
7. After merge, update Phase N's task in the project tracker to `completed` and Phase N+1 to `in_progress`.

**Estimated effort** (single experienced contributor):

| Phase | Estimated effort |
|---|---|
| 0 | 0.5 day (mostly script writing) |
| 1 | 1.5 days |
| 2 | 2 days |
| 3 | 2 days |
| 4 | 4 days (transforms audit + 3 splits + 1 deletion + 2 gated deletions) |
| 5 | 3 days (providers/transport.py split is the heaviest) |
| 6 | 1 day |
| 7 | 1 day |
| 8 | 1.5 days |
| 9 | 1 day |
| 10 | 2 days |
| 11 | 1 day |
| **Total** | **~20 days of focused work** |

---

## 9. Acceptance criteria for v1.0.0

The release tag `v1.0.0` ships when **all** of the following are true:

- [ ] Every phase has its CI gate green and is merged to `main`.
- [ ] `src/lattice/` contains no file >800 LoC.
- [ ] `find src/lattice -name "*.py" | wc -l` is ≤ 160.
- [ ] `grep -r "v1\|v2\|legacy\|_compat\|wrapper\|TODO" src/lattice/` returns no matches except in deliberately-named files (e.g. `_compat.py`).
- [ ] The public surface (§2) passes its dedicated `tests/contract/` suite, byte-for-byte.
- [ ] `benchmarks/results/v1.0.0.json` shows ≥ v0.x performance on every metric.
- [ ] README's claim count matches reality (transforms, providers, integrations).
- [ ] `docs/architecture/runtime.md` (renamed from `runtime_v2.md`) is the only architecture doc; the word "v2" appears nowhere outside the migration guide.
- [ ] `pyproject.toml` version is `1.0.0`.
- [ ] `CHANGELOG.md` and `docs/refactor/MIGRATION.md` are complete.
- [ ] PyPI publish dry-run passes.

When those boxes are ticked, tag and release.
