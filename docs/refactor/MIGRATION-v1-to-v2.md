# Migration: v1.x → v2 forward plan

> Companion to [MIGRATION.md](MIGRATION.md) (v0.x → v1.0.0 import map). This doc covers **user-visible** changes from the v2 forward plan (Phases 13–34).
>
> Link from: [FORWARD_PLAN.md](FORWARD_PLAN.md), [README.md](../../README.md), [AGENTS.md](../../AGENTS.md), [CHANGELOG.md](../../CHANGELOG.md).

---

## 1. Header changes (`x-lattice-*`)

| Header | v1.x | v2 | Introduced in |
|---|---|---|---|
| `x-lattice-receipt` | absent | JWT audit receipt id | Phase 28 |
| `x-lattice-quality-signal` | absent | Bandit feedback (`good`/`bad`) | Phase 29 |
| `x-lattice-cache-hit` | present | unchanged | v1.0 |
| `x-lattice-cost-usd` | present | omitted on semantic cache hit (zero billed) | v1.0 / Phase 13 fix |

All response headers are emitted only from `src/lattice/proxy/middleware.py` (Phase 13 honesty pass).

---

## 2. Config shape

Dead flags removed in Phase 13 (were silent no-ops):

- `transform_prefix_opt`
- `transform_constraint_lifting`
- `transform_strategy_selector`

Migration helper:

```bash
uv run python scripts/migrate_v1_config_to_v2.py lattice.toml -o lattice.v2.toml
```

v2 adds (when phases land): `quality_floor` (Phase 25), `receipts.*` (Phase 28), `profiles.yaml` path (Phase 30).

---

## 3. Cache invalidation

| Change | Phase | Action |
|---|---|---|
| Fingerprint in `cache/fingerprint.py` | 13 (shipped) | Flush semantic cache after upgrade |
| Provider-invariant fingerprint | 27 | Re-warm after provider switch |
| Namespace includes auth principal | 17 | Per-tenant cache isolation when auth enabled |

---

## 4. SDK upgrade

- **Python:** `from lattice import LatticeClient, …` unchanged (Phase 7 surface).
- **CLI:** `lattice` package is `src/lattice/cli/` (Phase 13 split); entry point unchanged.
- **TypeScript:** `@lattice/sdk` (Phase 20) — install when published.
- **Native wheel:** optional `lattice-core-py` (Phase 31).

---

## 5. Internal renames (operators)

| Old | New | Phase |
|---|---|---|
| `pipeline/milv.py` | `pipeline/post_transform_guard.py` | 13 |
| `BatchAccumulator` | `RequestCoalescer` | 13 |
| `planner/execution_plan.py` | `ir/primitives.ExecutionPlan` + `planner/session_plan.SessionExecutionPlan` | 13 |
| `gateway/compat.py` monolith | `gateway/compat/` package | 13 |

---

## 6. Rollback

Pin PyPI: `lattice-transport==1.0.*` while v2.x stabilizes. Config and cache flush recommended when downgrading after v2 fingerprint changes.
