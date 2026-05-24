# LATTICE v0.x → v1.0.0 migration (in progress)

> Full import-path mapping lands in Phase 11 (`11-docs-release.md`). This file records
> user-visible changes as each refactor phase ships.

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
