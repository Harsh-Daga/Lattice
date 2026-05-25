# Phase 34 — Agent-Loop-Aware Compression, Cursor Visualizer, Minimal Release

> **Footprint impact.** Step classifier + per-step profiles: zero new deps. Cursor extension: client-side only — runs in the IDE, talks to the proxy over HTTP. Release artifacts (Docker image, PyPI wheel, npm package, sideload `.vsix`): no runtime impact.
>
> **Algorithm location.** Step classifier in `src/lattice/agent/step_classifier.py`; uses [Phase 34](26-agent-memory.md)'s relevance scorer infrastructure. Per-step profiles wire into [Phase 28](28-receipts.md)'s profile system. Cursor extension is a thin TypeScript client that reads receipts ([Phase 28](28-receipts.md)) and OTel spans ([Phase 22](19-otel-genai.md)) — no algorithm code.
>
> **External-service requirement.** None.
>

> **Transport role.** Per-step profiles adjust pipeline plan before transport dispatch; extension reads receipts (no transport code).
> **Registry.** §4 step classifier + step profiles.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Outcome.** Inside an agent loop, LATTICE applies different compression profiles per step type (tool-call planning vs tool-result digestion vs final-answer generation vs reasoning). Cursor users see a live decoration in the editor showing what LATTICE did per request (compression ratio, cache layer, transforms applied, cost saved). Final v2.0 release as: PyPI wheel + Docker image (multi-arch) + npm package + sideload Cursor extension `.vsix`. **No marketplace ceremony, no Helm chart, no signed-wheel sigstore ritual.**
>
> **Estimated effort.** 6 days (1 PR, ~+2300 LoC — smaller than prior because the release artifacts list got trimmed).

---

## 1. Why this phase exists, and what brutally changed from the prior draft

### 1.1 The agent-loop fact

Inside a single agent loop, consecutive requests differ in shape:

| Step type | Characteristics | Right compression |
|---|---|---|
| Planning | "What should I do next?" — needs full prior context, no tools yet committed | Aggressive `reference_sub`, no `tool_filter` |
| Tool-call dispatch | "Call `read_file(...)`" — JSON-only response | Strong `tool_filter`, `response_format` validation, JSON repair |
| Tool-result digestion | Long tool output → next assistant message | `format_conversion` + `path_prefix` + aggressive `rate_distortion` if long |
| Final answer | Generates user-visible text | Conservative — no quality risks |
| Reasoning (o1 / Claude thinking) | Internal thinking + final answer | Light transforms only; never touch reasoning content |

Today LATTICE treats every request identically. Per-step profiles change that.

### 1.2 The Cursor visualizer

Cursor users get a real-time decoration showing what LATTICE did. Built as a sideload `.vsix` extension (manual install via "Install from VSIX"), not published to the marketplace. Users who want it install it; everyone else ignores it.

### 1.3 The brutal change vs the prior draft

The prior plan included:

- VSCode marketplace publication (requires Microsoft publisher account, certificate, marketplace approval flow, ongoing maintenance)
- Helm chart with HorizontalPodAutoscaler, PodDisruptionBudget, NetworkPolicy, ServiceMonitor
- Signed wheels via sigstore with `cosign verify` published instructions
- Multi-region deployment guides

All ops-team theatre, not user value. Trimmed to:

- Docker image (multi-arch: amd64 + arm64; published to ghcr.io and Docker Hub)
- PyPI wheel (already in CI)
- npm package (already in CI from [Phase 24](20-typescript-sdk.md))
- Cursor extension as sideload `.vsix` (built in CI; downloadable from GitHub Releases)
- A simple `docker-compose.yml` example for self-hosted multi-user with Postgres ([Phase 32](32-cloud-multitenant.md))

Anyone running Kubernetes can write their own manifest from the Docker image in 30 minutes. We don't ship one.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/agent/step_classifier.py             # rule-based step type detection
src/lattice/agent/step_profiles.py               # mapping: step type → profile
src/lattice/agent/loop_state.py                  # per-session step counter, loop trajectory

tools/cursor-extension/
  package.json                                   # extension manifest
  tsconfig.json
  src/
    extension.ts                                 # entry
    receiptsClient.ts                            # fetch from /lattice/receipts (Phase 28)
    decorations.ts                               # editor decoration rendering
    statusBar.ts                                 # token + cost surface
    settings.ts
  README.md
  CHANGELOG.md
  .vscodeignore
  scripts/package.sh                             # builds .vsix; called from CI

deploy/
  docker/
    Dockerfile                                   # multi-stage, ~120 MB final
    docker-compose.example.yml                   # proxy + postgres example for Phase 32 multi-user
  README.md                                      # how to docker run / docker compose up

scripts/
  release.sh                                     # version-bump + tag + CI trigger
  build_cursor_vsix.sh
  build_docker_multiarch.sh

tests/unit/agent/test_step_classifier.py
tests/unit/agent/test_step_profiles.py
tests/integration/test_per_step_compression_loop.py
tests/integration/test_cursor_extension_receipts_e2e.py
tests/contract/test_release_artifacts.py         # CI checks artifact sizes and bundle contents
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/state/session.py](../../src/lattice/state/session.py) | `Session.loop_state: LoopState \| None` |
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | If `loop_state` set, override the profile to the step-classifier's choice |
| [src/lattice/proxy/middleware.py](../../src/lattice/proxy/middleware.py) | Stash `request.state.loop_state`; emit `x-lattice-step-type` header |
| `.github/workflows/release.yml` | Builds Docker, PyPI, npm, vsix; uploads to GitHub Release |
| `README.md` | Final v2.0 README with quickstart for all four artifacts |
| `docs/CHANGELOG.md` | v2.0 entry |

### 2.3 Deleted

Cleanup of any prior draft files referencing Helm / marketplace / sigstore that were never created — none exist in the codebase yet.

---

## 3. Step-by-step

### 3.1 Step classifier (rule-based, zero deps)

```python
# src/lattice/agent/step_classifier.py
class AgentStepType(StrEnum):
    PLANNING = "planning"
    TOOL_DISPATCH = "tool_dispatch"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    REASONING = "reasoning"
    UNKNOWN = "unknown"


class StepClassifier:
    """Pure-Python rule-based classifier. Zero deps, ~ 100 µs per request."""

    def classify(self, request: Request, loop_state: LoopState | None) -> StepClassification:
        last_msg = request.messages[-1] if request.messages else None
        is_tool_result = last_msg and last_msg.role == "tool"
        has_tools_defined = bool(request.tools)
        wants_json = request.response_format and request.response_format != "text"
        is_reasoning_model = request.model.startswith(("o1", "o3", "deepseek-r1"))

        if is_reasoning_model:
            return StepClassification(AgentStepType.REASONING, confidence=1.0)
        if is_tool_result:
            return StepClassification(AgentStepType.TOOL_RESULT, confidence=1.0)
        if has_tools_defined and wants_json:
            return StepClassification(AgentStepType.TOOL_DISPATCH, confidence=0.9)
        if has_tools_defined and not loop_state:
            return StepClassification(AgentStepType.PLANNING, confidence=0.8)
        if loop_state and loop_state.iteration_count >= loop_state.max_iterations - 1:
            return StepClassification(AgentStepType.FINAL_ANSWER, confidence=0.7)
        if not has_tools_defined and loop_state and loop_state.iteration_count > 0:
            return StepClassification(AgentStepType.FINAL_ANSWER, confidence=0.6)
        return StepClassification(AgentStepType.UNKNOWN, confidence=0.5)
```

### 3.2 Per-step profiles

```python
# src/lattice/agent/step_profiles.py
DEFAULT_STEP_PROFILES: dict[AgentStepType, ProfileOverride] = {
    AgentStepType.PLANNING: ProfileOverride(
        enable=("reference_sub", "path_prefix", "rate_distortion_conservative"),
        disable=("tool_filter",),
        cache_jaccard_threshold=0.88,
    ),
    AgentStepType.TOOL_DISPATCH: ProfileOverride(
        enable=("reference_sub", "tool_filter_strict", "format_conversion"),
        disable=("rate_distortion",),
        cache_jaccard_threshold=0.92,
        guardrails_output_action="repair",
    ),
    AgentStepType.TOOL_RESULT: ProfileOverride(
        enable=("reference_sub", "tool_filter", "path_prefix", "format_conversion",
                "rate_distortion_aggressive", "tool_diff"),
        cache_jaccard_threshold=0.85,
    ),
    AgentStepType.FINAL_ANSWER: ProfileOverride(
        enable=("reference_sub", "output_cleanup"),
        disable=("rate_distortion", "format_conversion"),
        cache_jaccard_threshold=0.95,
    ),
    AgentStepType.REASONING: ProfileOverride(
        enable=("reference_sub",),                                # very conservative
        disable=("rate_distortion", "tool_filter", "format_conversion"),
        max_loss_budget=0.05,
    ),
    AgentStepType.UNKNOWN: ProfileOverride.no_change(),
}
```

The override layers on top of the tenant's base profile (from [Phase 28](28-receipts.md)). Users can supply their own per-step profiles via config.

### 3.3 Loop state tracking

```python
# src/lattice/agent/loop_state.py
@dataclass(frozen=True, slots=True)
class LoopState:
    session_id: str
    iteration_count: int
    max_iterations: int                                # from request hint or default 50
    step_history: tuple[AgentStepType, ...]
    started_at: int                                    # unix seconds

    def with_step(self, step: AgentStepType) -> "LoopState":
        return replace(self,
                       iteration_count=self.iteration_count + 1,
                       step_history=self.step_history + (step,))
```

Stored on `Session.loop_state` (Phase 7 store). Sessions detect agent-loop usage via [src/lattice/integrations/agent_stats.py](../../src/lattice/integrations/agent_stats.py); first agent-detected request initializes `LoopState`.

### 3.4 Cursor extension

Minimal scope: read receipts from `/lattice/receipts/{id}` after every chat request, surface compression decisions as editor decorations next to the cursor.

```typescript
// tools/cursor-extension/src/extension.ts
export async function activate(context: vscode.ExtensionContext) {
  const config = vscode.workspace.getConfiguration("lattice");
  const proxyUrl = config.get<string>("proxyUrl", "http://localhost:8787");
  const enabled = config.get<boolean>("enabled", false);          // OFF by default — user enables
  if (!enabled) return;

  const client = new ReceiptsClient(proxyUrl, config.get("apiKey"));
  const decorations = new DecorationManager();
  const statusBar = new StatusBar();

  const subscription = client.subscribeToRecentReceipts(receipt => {
    decorations.showReceipt(receipt);
    statusBar.update(receipt);
  });

  context.subscriptions.push(decorations, statusBar, subscription);
}
```

`ReceiptsClient` polls `/lattice/receipts?since=...` every 2s when an active editor exists. No WebSocket required (keeps things simple). For higher-rate workloads users can install the optional SSE-based receipt stream — but that's a follow-up; this phase ships the polling client.

Decoration content: small inline annotation in the gutter — `LATTICE: 3.2× compression · cache: ir · $0.0012 saved`. Hover for the full receipt JSON.

Settings:

- `lattice.enabled` (default false; user opts in)
- `lattice.proxyUrl`
- `lattice.apiKey` (when proxy has [Phase 32](32-cloud-multitenant.md) auth enabled)
- `lattice.showStatusBar`
- `lattice.showInlineDecorations`
- `lattice.pollIntervalMs` (default 2000)

### 3.5 Packaging — what we actually ship

| Artifact | Where | Built by | Size |
|---|---|---|---|
| `lattice-transport-2.0.0.tar.gz` + wheels | PyPI | `.github/workflows/release.yml` | ~ 25 MB |
| `ghcr.io/harsh-daga/lattice:2.0.0` (amd64 + arm64) | GitHub Container Registry | release workflow with buildx | ~ 120 MB final image |
| `harshdaga/lattice:2.0.0` | Docker Hub | mirror of ghcr | same |
| `@lattice/sdk@2.0.0` | npm with provenance | release workflow | ≤ 60 KB |
| `@lattice/core-wasm@1.0.0` | npm with provenance | release workflow | ≤ 200 KB |
| `lattice-core-py-1.0.0-{platform}.whl` | PyPI | release workflow (maturin) | ~ 3 MB each |
| `lattice-cursor-2.0.0.vsix` | GitHub Release attachment (sideload) | release workflow | ~ 200 KB |

### 3.6 What we deliberately don't ship

- VSCode marketplace listing (requires publisher account + ongoing publisher relations)
- Helm chart (write your own from the Docker image — 30 minutes)
- Sigstore-signed wheels (overhead for an OSS project of this size; provenance via GitHub Actions OIDC already attaches attestations)
- Multi-region deployment Terraform (open-source self-hosted target is single-region)
- Hosted lattice.cloud or any managed service

A `deploy/README.md` documents these choices honestly so operators know what's there and what isn't.

### 3.7 Release process

```bash
# 1. Bump versions (single command, walks workspaces)
./scripts/release.sh prepare 2.0.0

# 2. Update CHANGELOG.md (manual review)

# 3. Tag
./scripts/release.sh tag 2.0.0

# 4. Push tag → CI builds and publishes all artifacts atomically
git push origin v2.0.0
```

CI workflow `.github/workflows/release.yml`:

1. Runs full test suite (parallel: unit, integration, contract, footprint, canonical benchmark)
2. Builds PyPI wheel + sdist
3. Builds multi-arch Docker image via buildx
4. Builds npm packages (`@lattice/sdk`, `@lattice/core-wasm`)
5. Builds Rust `lattice-core-py` wheels for 5 platforms via maturin
6. Builds Cursor `.vsix`
7. Creates GitHub Release with all artifacts attached
8. Publishes PyPI + Docker + npm
9. **Aborts and rolls back if any step fails** — no partial releases

---

## 4. Per-step compression observability

Each request emits:

- `x-lattice-step-type` header
- `lattice.step.type` OTel attribute
- Receipt field `step_type: "tool_dispatch"`
- Bandit reward attribution (Phase 28) is partitioned by step type so each profile self-tunes

Inside Cursor, the receipt panel groups by step type so users see "where" their tokens go.

---

## 5. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/agent -q` | All pass |
| Step classifier | property | Tool-result detection always correct; reasoning-model always classified as reasoning |
| Per-step compression E2E | `tests/integration/test_per_step_compression_loop.py` | 30-step agent loop shows step-specific transform application; compression ratio varies as expected per step type |
| Cursor extension build | `cd tools/cursor-extension && npm run build && ./scripts/package.sh` | Produces `.vsix` ≤ 250 KB |
| Cursor extension receipts integration | `tests/integration/test_cursor_extension_receipts_e2e.py` | Mock proxy serves receipts; extension polls and renders decorations |
| Release artifact contracts | `tests/contract/test_release_artifacts.py` | Wheel under 25 MB; Docker image under 130 MB; npm bundle under 60 KB; vsix under 250 KB |
| Docker image boot | `docker run --rm lattice:test --version` | succeeds < 1.5 s cold |
| Docker compose example | `docker compose up && curl healthz` | All services healthy in < 10 s |
| Canonical bench | usual | ±2% |
| Footprint | `tests/integration/footprint/test_4gb_laptop.py` | unchanged from Phase 32 |

---

## 6. Acceptance criteria

1. A 30-iteration agent loop produces different `x-lattice-step-type` headers across iterations; transforms applied per request match the per-step profile mapping.
2. Reasoning-model requests (`o1-mini`, `claude-opus-thinking`, etc.) never apply `rate_distortion`, `tool_filter`, or `format_conversion`. Property test enforces.
3. Cursor extension installed via "Install from VSIX": opens a Cursor window with the proxy URL configured; making a request from within Cursor surfaces a status-bar item showing compression ratio and a gutter annotation in the active file.
4. `docker pull ghcr.io/harsh-daga/lattice:2.0.0 && docker run --rm -p 8787:8787 ...` works on both amd64 and arm64; `curl http://localhost:8787/healthz` returns 200 within 1.5 s.
5. `pip install lattice-transport==2.0.0 && lattice proxy run` works on Python 3.11, 3.12, 3.13.
6. `npm install @lattice/sdk@2.0.0` works in a fresh Node 22 project; quickstart from [Phase 24](20-typescript-sdk.md) README runs without errors.
7. GitHub Release for `v2.0.0` includes all seven artifacts (Python wheel + sdist, Docker tag info, two npm packages, five Rust wheels, vsix) and a changelog excerpt.
8. Footprint test passes at v2.0.0: default install ≤ 28 MB, idle RSS ≤ 100 MB, 4 GB laptop boot test passes.
9. Canonical bench ±2% vs phase-0 baseline.

---

## 7. Out of scope (forever)

| Topic | Why not |
|---|---|
| VSCode marketplace publication | Publisher account overhead; sideload `.vsix` works for users who want it |
| Helm chart | 30 minutes of work for any operator; we don't maintain k8s expertise |
| Sigstore-signed wheels with `cosign verify` instructions | GitHub Actions OIDC attestation is sufficient for OSS at this size |
| Multi-region active-active deployment guide | Out of scope for open-source self-hosted single-region target |
| Hosted lattice.cloud, SaaS, managed offering | Open-source self-hosted only ([Phase 32](32-cloud-multitenant.md) §6) |
| Cursor extension on the VSCode marketplace | Same reason as #1 |
| Built-in IDE LSP for prompt linting | Future, post-2.0; not v2.0 critical path |
