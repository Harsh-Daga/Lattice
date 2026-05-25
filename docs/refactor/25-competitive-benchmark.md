# Phase 25 — Competitive Benchmark

> **Status:** Skeleton — competitive matrix + quality-floor config land in this phase's implementation PR. + Quality SLOs

> **Footprint impact.** 0 `src/lattice/` LoC; +800 LoC `benchmarks/competitive/`; +600 LoC tests.
> **Algorithm location.** Matrix runner in `benchmarks/competitive/matrix.py`; quality floor gate in `core/config.py::quality_floor`.
> **External-service requirement.** Operator API keys for upstream provider and competitor gateways (CI nightly, non-blocking).
> **Transport role.** Compares end-to-end proxy latency and wire size using Phase 20 transport; competitors run with identical upstream provider+model.
> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
>
> **Goal.** Third-party-verifiable comparison vs Helicone, Portkey, OpenRouter, LiteLLM, Cloudflare AI Gateway.
> **Outcome.** Published matrix in `benchmarks/results/CLAIMS.md`; optional `quality_floor` runtime gate.
> **Estimated effort.** 5 days (1 PR).

---

## 1. Comparison matrix

| Dimension | Metric |
|---|---|
| Cost | Provider $ per 1k completions on fixed prompt set |
| Latency | p50 / p95 / p99 added latency vs direct provider |
| Cache | Hit rate on identical replayed traffic |
| Quality | Task-equivalence judge (v1.0.0 eval rubric) |
| Wire | Payload size reduction |

---

## 2. Methodology

Identical input set, identical upstream provider+model, three runs, median. Versions, prompts, judge model, and rubric pinned in `benchmarks/competitive/README.md`.

---

## 3. Quality SLO surface

```toml
# lattice.toml
quality_floor = 0.85  # auto-disables lossy transforms below measured quality
```

Gate consumes per-segment quality estimates from [Phase 28](23-segment-aware-planning.md). Documented in README quick-start.

---

## 4. CI

`.github/workflows/competitive-bench.yml` — nightly, operator-run, fails on >5% regression vs last publish (does not block merge).

---

## 5. Acceptance criteria

1. `benchmarks/competitive/matrix.py` produces reproducible artifact JSON.
2. `CLAIMS.md` gains "vs. competitors (run YYYY-MM-DD)" section.
3. `quality_floor` config validated in `tests/unit/core/test_config.py`.
