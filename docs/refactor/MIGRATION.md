# LATTICE migration guide

> Records user-visible changes as each refactor phase ships. Import-path mapping for v1.0.0 lands in Phase 11 (`11-docs-release.md`).

## v2.0 forward plan (not yet shipped)

Phases 12–27 are documented in [`FORWARD_PLAN.md`](FORWARD_PLAN.md). Product positioning:

- **LATTICE is the transport / network layer for LLM traffic** — compression is one policy on that layer.
- **Lightweight default install** — no required model downloads; runs on a 4 GB laptop.
- **No external LLM** beyond the provider you already use.
- **Self-hosted only** — no `lattice.cloud`, no SaaS.
- **Thin SDKs** — proxy mode = set `baseURL`; no algorithm reimplementation in SDK source.

Breaking changes for v2.0 will be listed here as phases ship. See also [`PHASE_GUIDELINES.md`](PHASE_GUIDELINES.md).

---

## v0.x → v1.0.0 migration (in progress)

## Phase 7 — Proxy / SDK / CLI (shipped on `refactor/phase-7-proxy-sdk-cli`)

### Python imports

| Old (deprecated v1.0.0) | New (canonical) |
|-------------------------|-----------------|
| `from lattice.sdk.client import LatticeClient` | `from lattice import LatticeClient` |
| `from lattice.sdk.client import CompressResult` | `from lattice import CompressResult` |
| `from lattice.proxy.compat_exports import *` | **Removed** — no replacement |

`import lattice.sdk.client` still works in v1.0.0 but emits:

```text
DeprecationWarning: lattice.sdk.client is deprecated; import from `lattice` or
`lattice.sdk` instead. This module will be removed in v1.1.
```

### HTTP

- `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` are registered on the proxy app.
- Response `x-lattice-*` headers on compat routes are emitted by `LatticeHeaderMiddleware`
  (`src/lattice/proxy/middleware.py`) from `request.state`, not per-handler assignment.

### CLI

- `lattice version` is an alias for `lattice --version` (unchanged output).

## Phase 8 — Agent integrations (shipped on `refactor/phase-8-integrations`)

### Python imports

| Old | New |
|-----|-----|
| `from lattice.core.tunnel_sidecar import TunnelSidecar` | `from lattice.integrations.tunnel import TunnelSidecar` |
| `from lattice.core.tunnel_sidecar import SidecarThread` | `from lattice.integrations.tunnel import SidecarThread` |
| `from lattice.core.tunnel_sidecar import TunnelState` | `from lattice.integrations.tunnel import TunnelState` |

### CLI behavior

- `lattice doctor` with no argument runs health checks for all five primary agents (`claude`, `codex`, `cursor`, `opencode`, `copilot`).
- `lattice doctor <agent>` uses per-integration `doctor()` (install / durable / transient lace / proxy `/healthz`).
- `JsonFileIntegration.patch()` (via `wrap_agent` / durable patch paths) raises `AgentNotInstalledError` when the agent config file is missing (non–dry-run).
- `lattice status` uses `mutation_store.list_all_active()` (durable init ∪ live transient lace).

## Phase 9 — Observability, state, cache, safety (shipped on `refactor/phase-9-observability-state`)

### Python imports

| Old | New |
|-----|-----|
| `lattice.core.metrics.MetricsCollector` | `lattice.telemetry.MetricsCollector` |
| `lattice.core.metrics.LatencyTracker` | `lattice.telemetry.LatencyTracker` |
| `lattice.core.telemetry.DowngradeCategory` | `lattice.telemetry.DowngradeCategory` |
| `lattice.core.telemetry.DowngradeTelemetry` | `lattice.telemetry.DowngradeTelemetry` |
| `lattice.core.telemetry.TransportOutcome` | `lattice.telemetry.TransportOutcome` |
| `lattice.core.agent_stats.AgentStatsCollector` | `lattice.telemetry.AgentStatsCollector` |
| `lattice.core.cost_estimator.CostEstimator` | `lattice.telemetry.CostEstimator` |
| `lattice.core.maintenance.MaintenanceCoordinator` | `lattice.telemetry.MaintenanceCoordinator` |
| `lattice.utils.streaming_sketches.CountMinSketch` | `lattice.telemetry.CountMinSketch` |
| `lattice.core.session.Session` | `lattice.state.Session` |
| `lattice.core.session.SessionManager` | `lattice.state.SessionManager` |
| `lattice.core.store.RedisSessionStore` | `lattice.state.RedisSessionStore` |
| `lattice.core.semantic_cache.SemanticCache` | `lattice.cache.SemanticCache` |
| `lattice.utils.validation.SemanticRiskScore` | `lattice.safety.SemanticRiskScore` |
| `lattice.utils.validation.compute_risk_score` | `lattice.safety.compute_risk_score` |

Note: `core.telemetry` module file is now `telemetry/downgrade.py` (package `telemetry`, module `downgrade`).

### Top-level convenience imports (v1.0.0)

```python
from lattice import (
    MetricsCollector,
    DowngradeCategory,
    Session,
    SessionManager,
    SegmentStore,
    SemanticCache,
    SemanticRiskScore,
    compute_risk_score,
)
```
