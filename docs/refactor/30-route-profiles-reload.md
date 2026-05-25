# Phase 30 — Per-Route Profiles + Hot Reload

> **Status:** Skeleton — profile registry + reload land in this phase's implementation PR.

> **Footprint impact.** Pydantic profiles; reload via SIGHUP or optional `watchdog` (`[hot-reload]` extra ~200 KB). Default: polling/SIGHUP only.
> **Algorithm location.** `src/lattice/policy/profiles.py`, `matchers.py`, `src/lattice/config/reload.py`. Registry: [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) §14.
> **External-service requirement.** None.
> **Transport role.** Profiles override cache/guardrail/agent settings on `request.state` before `transport.request()`; they do not add per-route retry code.
> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
>
> **Goal.** `/api/chat` and `/api/code` use different compression recipes; config changes without process restart.
> **Outcome.** YAML profile registry with URL/header/model matchers; atomic config swap on SIGHUP or file change.
> **Estimated effort.** 2 days (1 PR).

---

## 3. When this ships without the other v2 ops phases

Route profiles and hot reload work **without** [Phase 28](28-receipts.md) or [Phase 29](29-bandit-routing.md). Bandit can read `default_arm` from a resolved profile when both are enabled.

---

## 1. Why this phase exists

One global config cannot fit chat, code, and realtime endpoints simultaneously. HA deployments cannot tolerate restart for policy tweaks.

---

## 2. Files touched

### Created

```
src/lattice/policy/profiles.py
src/lattice/policy/matchers.py
src/lattice/config/reload.py
src/lattice/config/diff.py
tests/unit/policy/
tests/unit/config/test_reload.py
tests/integration/test_hot_reload.py
examples/profiles.yaml
docs/operations/route_profiles.md
```

### Modified

| File | Change |
|---|---|
| `src/lattice/proxy/middleware.py` | Resolve profile; stash on `request.state` |
| `src/lattice/proxy/server.py` | SIGHUP handler; optional admin reload route |
| `src/lattice/core/config.py` | `ProfilesConfig`, `HotReloadConfig` |

---

## 3. Profiles and reload

First matching profile wins (`url_prefix`, headers, model prefix, tenant). `ConfigReloader` validates new config before atomic swap; in-flight requests keep old config reference.

**Cannot hot reload:** provider adapter registration, OTel exporter init, listen port.

---

## 4. Test plan

| Check | Command |
|---|---|
| Unit | `uv run pytest tests/unit/policy tests/unit/config -q` |
| E2E | `tests/integration/test_hot_reload.py` |
| Unsafe reload | Removing signing key while receipts enabled → refused unless `LATTICE_ALLOW_UNSAFE_RELOAD=true` |

---

## 5. Acceptance criteria

1. Request to `/api/code` uses `code-internal` profile without affecting `/api/support`.
2. SIGHUP reload affects next request only; in-flight uses old config.
3. Unsafe reload refused with clear error by default.

---

## 6. Out of scope

| Topic | Phase |
|---|---|
| Per-user sub-tenant profiles | [Phase 32](32-cloud-multitenant.md) |
| Cloud receipt search UI | Future |
