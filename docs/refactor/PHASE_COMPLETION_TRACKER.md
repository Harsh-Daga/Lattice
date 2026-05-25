# Phase completion tracker (Phases 0–10)

> **Rule:** Non-benchmark acceptance items must be ✅ before a phase is **Done**.
> **Benchmarks:** `phase-*.json` compare gates are tracked separately (optional CI key).

**Last verified:** `refactor/phase-10-completion` — **1766 passed**, contract green (see §10).

| Phase | Verdict | Notes |
|-------|---------|-------|
| **0** | ✅ | `FEATURE_PARITY.md` scaffold; inventory; contract; `refactor-gate.yml` |
| **1** | ✅ | `ir/` package; imports; `test_ir.py` under `tests/unit/ir/` |
| **2** | ✅ | Layout; `test_no_legacy_process_paths.py` (IR-native scope per Phase 3/5) |
| **3** | ✅ | V1 kill; `Pipeline.compress()` |
| **4** | ✅ | Planner collapse; `_normalize_legacy_execution_plan` documented as persisted-plan bridge only |
| **5** | ✅ | Transforms cleanup + §6.3 tests |
| **6** | ✅ | Providers/adapters + transport split; see §6 below |
| **7** | ✅ | Proxy health routes, header middleware, SDK surface; see §7 below |
| **8** | ✅ | Integrations tunnel move, doctor, mutation store; see §8 below |
| **9** | ✅ | `telemetry/`, `state/`, `cache/`, `safety/`; leaf `core/` + `utils/`; see §9 below |
| **10** | ✅ | Benchmark wrapper, CLAIMS, v1.0.0 artifacts; see §10 below |

---

## Phase 0 — `00-audit-baseline.md` §4

| Criterion | Status |
|-----------|--------|
| `inventory.csv` (166 rows, columns populated) | ✅ |
| `FEATURE_PARITY.md` | ✅ scaffold |
| `api-surface.json` | ✅ |
| `tests/contract/` + CI | ✅ (`ci.yml` + `refactor-gate.yml`) |
| `phase-0-baseline.json` | ✅ |
| `repomix` gone / gitignore | ✅ |
| `compat_exports.py` gone | ✅ |
| README + AGENTS counts | ✅ (1903 collected; 1706 passed) |
| ruff / mypy / pytest | ✅ |

---

## Phase 1 — `01-ir-primitives.md` §6

All import/layout criteria ✅. Benchmark lines excluded.

---

## Phase 2 — `02-pipeline-runner.md` §6

| Criterion | Status |
|-----------|--------|
| v1 pipeline / core transport deleted | ✅ |
| `pipeline/` + `transport/` layout | ✅ |
| Forbidden imports | ✅ |
| `test_transport_types_canonical_path.py` | ✅ |
| `test_no_legacy_process_paths.py` | ✅ (IR-native; Phase 2 §4.3 intent) |
| Full-registry `process()` removal | ⚠️ deferred Phase 11 (execution-only + response-side exempt) |

---

## Phase 3 — `STATUS.md` §5

✅ CompressorPipeline deleted; `compress()` + gates; client/factory wired.

---

## Phase 4 — `03-planner-collapse.md` §8

✅ All code criteria; benchmark `[ ]` only.

---

## Phase 5 — `04-transforms.md` §9

| Criterion | Status |
|-----------|--------|
| Registry / reputation / patterns moves | ✅ |
| Deletes + splits | ✅ |
| `is_response_side`, delta flag | ✅ |
| §6.3 tests (classifier, risk, planner_bridge, table/json converter) | ✅ |
| prefix_opt hygiene (src policy/strategy/validation) | ✅ |
| IR-native `optimize()` only | ✅ |
| `phase-5-decisions.md` | ✅ |
| Benchmark compare | ⏳ operator |

---

## Phase 6 — `05-providers-transport.md` §10

| Criterion | Status |
|-----------|--------|
| `providers/transport.py` monolith deleted | ✅ |
| `providers/transport/` package (7 modules + `__init__`) | ✅ |
| `providers/adapters/` (8 files) | ✅ |
| `stall_detector.py` at providers root deleted | ✅ |
| `base.py` at providers root deleted | ✅ |
| No file in `providers/` > 850 LoC | ✅ (max `streaming.py` ~495) |
| Unified `_stream`; thin `completion_stream*` wrappers | ✅ |
| All 17 adapters at `lattice.providers` | ✅ |
| TTL `RateLimitTracker` + tests | ✅ |
| §9.2–9.3 new tests (all 17 adapters in contract) | ✅ |
| §9.1 test moves (`test_stall_detector`, `test_transport_resilience` → `providers/transport/`) | ✅ |
| ruff / pytest / contract | ✅ |
| Docs (`STATUS`, `providers.md`, `AGENTS.md`) | ✅ |
| Benchmark `phase-6.json` ±2% | ⏳ operator (`OLLAMA_CLOUD_API_KEY`; merged without artifact) |
| PR #11 merged to `main` (`4798bfb`) | ✅ |

---

## Phase 7 — `06-proxy-sdk-cli.md` §7

| Criterion | Status |
|-----------|--------|
| `compat_exports.py` absent (never committed on `main`) | ✅ |
| `proxy/middleware.py` (`LatticeHeaderMiddleware`) | ✅ |
| `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` on `app.routes` | ✅ |
| `test_health_routes_registered.py`, `test_response_headers.py` | ✅ |
| Contract HTTP + headers tests | ✅ |
| `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult` | ✅ |
| `lattice.sdk.client` → `DeprecationWarning` | ✅ |
| `lattice version` alias | ✅ |
| No `response.headers["x-lattice-…]` in `gateway/` | ✅ |
| Six canonical header keys in `proxy/middleware.py` | ✅ |
| ruff / mypy / pytest / contract | ✅ |
| `HealthManager` owns all five health route bodies | ✅ |
| Passthrough + native gateway headers via middleware stash | ✅ |
| `docs/refactor/MIGRATION.md` Phase 7 `sdk.client` section | ✅ |
| Doc sync (`06-proxy-sdk-cli` §7, FINAL_LAYOUT, api-surface) | ✅ |
| Benchmark `phase-7-proxy.json` ±2% | ⏳ operator (`OLLAMA_CLOUD_API_KEY`) |

---

## Phase 8 — `07-integrations.md` §7

| Criterion | Status |
|-----------|--------|
| `core/tunnel_sidecar.py` deleted; `integrations/tunnel.py` exists | ✅ |
| `pyproject.toml` ruff per-file-ignores updated | ✅ |
| `from lattice.integrations.tunnel import TunnelSidecar, TunnelState, SidecarThread` | ✅ |
| `AgentNotInstalledError`, `AgentIntegrationProtocol`, `AgentDoctorReport` | ✅ |
| All integration classes: `patch` / `unpatch` / `is_patched` / `doctor` / `name` / `proxy_url` | ✅ |
| `JsonFileIntegration.patch()` raises when config missing (non–dry-run) | ✅ |
| `lattice doctor <agent>` exit 0 for five primary agents | ✅ |
| `lattice doctor` (no args) reports all five | ✅ |
| `mutation_store.list_all_active()` = durable ∪ transient | ✅ |
| Lace records/clears transient state | ✅ |
| §5.1 tests under `tests/unit/integrations/` | ✅ |
| ruff / mypy / pytest / contract | ✅ |
| Registry: `copilot` in `_AGENT_REGISTRY`; `list_supported_agents()` = `list_primary_agents()` | ✅ |
| `vscode` / `generic` remain wrap aliases only (not primary doctor targets) | ✅ documented |
| `init.run_init` catches `AgentNotInstalledError`; `lace` uses `atexit` + signals for transient cleanup | ✅ |
| `test_registry_primary_agents.py`, `test_lace_transient.py` | ✅ |
| Deferred: split `agents.py` (>800 LoC) | Phase 11 |
| Deferred: `integrations/mcp.py` → pipeline import cycle | pre-existing |
| Deferred: `AgentNotInstalledError` CHANGELOG entry | Phase 12 |

---

## Phase 9 — `08-observability-state.md` §8

| Criterion | Status |
|-----------|--------|
| `observability/` removed; `telemetry/` with `__init__.py` + 6 modules | ✅ |
| `state/`: `session.py`, `store.py`, `segment_store.py` | ✅ |
| `cache/semantic.py`; `safety/risk_scoring.py` | ✅ |
| `core/` leaf only (6 files); `utils/` = `token_count` + `__init__` | ✅ |
| `from lattice.telemetry import MetricsCollector, …` | ✅ |
| `from lattice.state import Session, RedisSessionStore, SegmentStore, …` | ✅ |
| `from lattice.cache import SemanticCache, ContentClass` | ✅ |
| `from lattice.safety import SemanticRiskScore, compute_risk_score` | ✅ |
| Top-level `from lattice import MetricsCollector, Session, SemanticCache, …` | ✅ |
| `test_core_is_leaf.py`, `test_no_old_paths.py` | ✅ |
| Contract tests extended (`test_new_top_level_exports`) | ✅ |
| No old-path imports in `src/` (rg) | ✅ |
| ruff / mypy / pytest / contract | ✅ |
| `tests/integration/test_redis_store_integration.py` | ✅ (full suite) |
| Benchmark `phase-9-observability.json` vs baseline ±2% | ⏳ operator (`OLLAMA_CLOUD_API_KEY`) |

---

## Phase 10 — `09-benchmarks.md` §7

| Criterion | Status |
|-----------|--------|
| `src/lattice/evals/` deleted | ✅ |
| No `lattice.evals` imports (rg) | ✅ |
| No `--use-v2-pipeline` (rg) | ✅ |
| `lattice benchmark` → `benchmarks/evals/cli.py` | ✅ |
| `benchmarks/results/CLAIMS.md` | ✅ |
| `benchmarks/results/v1.0.0.{json,md}` | ✅ operator run |
| `scripts/run_canonical_benchmark.sh` + CI reference | ✅ |
| Script audit (`profile_format_conv`, `test_e2e_real`) | ✅ headers |
| ruff / mypy / pytest / contract | ✅ |
| `compare_benchmarks.py` vs baseline ±5% | ⏳ reference run (not a phase gate; see CLAIMS.md) |
| `--provider-detect` on benchmark CLI | ✅ |
| `provider_validation` uses `Pipeline.compress()` | ✅ (fix: was broken `process()` call) |
| Doc/index sync (STATUS, AGENTS, README badge) | ✅ |
| `scripts/README.md` | ✅ |
| `refactor-gate` ruff `benchmarks/` | ✅ |

---

## Remaining operator actions (not code)

1. Run canonical benchmark → `phase-6.json` when `OLLAMA_CLOUD_API_KEY` is set (see `docs/refactor/phase-6-benchmark.md`); compare vs `phase-0-baseline.json` (±2%).
2. Run Phase 7 benchmark → `phase-7-proxy.json` (same key); compare vs `phase-0-baseline.json` (±2%).
3. Run Phase 9 benchmark → `phase-9-observability.json` (same key); `compare_benchmarks.py` vs `phase-0-baseline.json` (±2%).
