# Phase 23 — Compression Receipts, Self-Tuning Bandit, Per-Route Profiles, Hot Reload

> **Footprint impact.** + `pyjwt[crypto]` (~ 100 KB) for receipt signing. The bandit is pure numpy on the Beta distribution — **no ML library, no model download, no training pipeline**. Per-route profiles are pydantic config; hot reload uses Python stdlib `watchdog` (optional `[hot-reload]` extra ~ 200 KB) or polling (default; zero deps).
>
> **Algorithm location.** New `src/lattice/audit/`, `src/lattice/planner/bandit/`, `src/lattice/policy/profiles.py`, `src/lattice/config/reload.py`. Receipt schema lives in `audit/receipts.py`; SDKs never construct receipts (the proxy does). Bandit lives entirely in the planner; the only "feedback" interface is HTTP headers (`x-lattice-quality-signal`) and HTTP 5xx detection — no SDK changes required.
>
> **External-service requirement.** None. Receipts persist to the same backend Phase 14 / Phase 25 use (memory / SQLite / Postgres / Redis). The bandit's Beta posteriors persist locally per tenant. No external ML service, no remote training, no third-party feedback collector.
>
> **The bandit is intentionally simple.** Per-tenant per-task-class Beta(α, β) posteriors over each transform's reward distribution. Thompson-sampled per request. Reward = 1 - normalized quality risk. Update on response. **No neural network. No PyTorch. No fine-tuning.** This is the lightweight alternative to the per-tenant LoRA distillation pipeline cut from the prior [Phase 25](25-cloud-multitenant.md) draft — captures ~ 80% of the value with zero infrastructure.
>

> **LoC delta (declared).** +2200 net (`audit/`, `planner/bandit/`, `policy/`).
> **Transport role.** Receipts record transport attempts/RTT/breaker state (Phase 27) + compression decisions.
> **Registry.** §11 audit + §4 bandit + profiles.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Goal.** Operationalize LATTICE for production at scale: every response carries an HMAC-signed compression receipt that proves what was done (compliance + audit unlock); the transform allowlist self-tunes per tenant via Thompson sampling against measured quality outcomes (no static config required); per-route profiles let `/api/chat` and `/api/code` use different compression recipes; configuration hot-reloads without restart.
>
> **Outcome.** A new `audit/receipts.py` issues `LatticeReceipt` JWTs returned in `x-lattice-receipt` header and queryable at `GET /lattice/receipts/{id}`. Receipts record transforms applied, tokens saved, cache layer hit, guardrails fired, model used — without any user content. A `planner/bandit.py` runs Thompson sampling per (tenant, task_class) over the transform allowlist, learning the reward function from feedback (HTTP 5xx rate, downstream LLM-judge if enabled, manual user feedback via header). A `policy/profiles.py` lets ops define route-specific overrides matched by URL path / header / model prefix. A `config/reload.py` watches the config file and reloads atomically (no in-flight request disruption).
>
> **Estimated effort.** 5 days (1 PR, ~+2200 LoC).

---

## 1. Why this phase exists

Once LATTICE is in production multi-tenant deployment, three operational gaps surface:

1. **Compliance can't trust opaque compression.** Healthcare/finance auditors need a verifiable trail: "what did your gateway do to my request?". Per-response receipts close this.
2. **Static transform allowlists waste money.** A coding tenant benefits from `reference_sub` more than `rate_distortion`; a summarization tenant the reverse. Hand-tuning per tenant doesn't scale. A bandit learns.
3. **One config per proxy** — a single hand-tuned policy can't fit `/v1/chat/completions` *and* `/api/internal/code` *and* `/api/realtime` simultaneously. Per-route profiles fix this.
4. **Config changes require restart** — high-availability deployments can't tolerate this; hot reload is table stakes.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/audit/__init__.py
src/lattice/audit/receipts.py                # LatticeReceipt + signing
src/lattice/audit/receipt_store.py           # in-memory / Redis / Postgres
src/lattice/audit/receipt_router.py          # GET /lattice/receipts/{id}
src/lattice/planner/bandit/__init__.py
src/lattice/planner/bandit/thompson.py       # Beta posterior per arm
src/lattice/planner/bandit/reward.py         # reward function from feedback signals
src/lattice/planner/bandit/store.py          # posterior persistence
src/lattice/policy/profiles.py               # per-route profile resolution
src/lattice/policy/matchers.py               # url / header / model prefix matchers
src/lattice/config/reload.py                 # file watcher + atomic swap
src/lattice/config/diff.py                   # safe-diff: warn on unsafe changes
tests/unit/audit/test_receipts.py
tests/unit/audit/test_receipt_store.py
tests/unit/planner/bandit/test_thompson.py
tests/unit/planner/bandit/test_reward.py
tests/unit/policy/test_profiles.py
tests/unit/policy/test_matchers.py
tests/unit/config/test_reload.py
tests/integration/test_receipt_e2e.py
tests/integration/test_bandit_e2e.py
tests/integration/test_hot_reload.py
docs/operations/receipts.md
docs/operations/bandit_tuning.md
docs/operations/route_profiles.md
examples/profiles.yaml
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | Emit receipt at end of compress(); bandit consults posterior for transform selection |
| [src/lattice/planner/unified_planner.py](../../src/lattice/planner/unified_planner.py) | Pass bandit-selected allowlist when enabled |
| [src/lattice/proxy/middleware.py](../../src/lattice/proxy/middleware.py) | Attach `x-lattice-receipt` header; route to profile resolver |
| [src/lattice/proxy/server.py](../../src/lattice/proxy/server.py) | Register receipt router + reload signal handler (SIGHUP) |
| [src/lattice/core/config.py](../../src/lattice/core/config.py) | `ReceiptsConfig`, `BanditConfig`, `ProfilesConfig`, `HotReloadConfig` |

---

## 3. Compression receipts

### 3.1 Shape

```python
# src/lattice/audit/receipts.py
@dataclass(frozen=True, slots=True)
class LatticeReceipt:
    """Per-response audit record. NEVER contains user content.

    Issued as a JWT (signed with HMAC-SHA256 by default; RS256 optional)
    so downstream systems can verify without contacting the proxy.
    """
    version: int                                       # current: 1
    id: str                                            # ulid
    issued_at: float
    tenant: str                                        # opaque tenant id, not API key
    request_id: str

    # What was sent vs what was received
    model_requested: str
    model_responded: str
    tokens_input_before: int
    tokens_input_after: int
    tokens_output: int
    cached_tokens: int

    # What we did
    transforms_applied: tuple[str, ...]
    transforms_rolled_back: tuple[str, ...]
    cache_layer_hit: str | None                        # exact | ir | jaccard | embed | portable | None
    guardrail_violations: tuple[str, ...]              # e.g. ("pii:tokenize:email",)
    guardrail_modifications: tuple[str, ...]
    agent_gc_dropped: int                              # count of pruned messages
    agent_retry_attempts: int

    # Compute envelope
    proxy_latency_ms: float
    pipeline_compute_ms: float
    bandit_arm_pulled: str | None                      # which compression recipe arm we selected

    # Signature
    sig: str                                           # base64(HMAC(key, canonical_form))
```

### 3.2 Issuance

At the end of `Pipeline.compress()` + dispatch + reverse:

```python
receipt = LatticeReceipt.build(
    tenant=ctx.tenant, request=request, response=response, ctx=ctx,
)
ctx.receipts_store.put(receipt)
ctx.response_headers["x-lattice-receipt"] = receipt.encode_jwt(self._signing_key)
```

`receipt.encode_jwt` uses `pyjwt` (light dep). HS256 by default; configurable to RS256/EdDSA with a public-key-published flow.

### 3.3 Retrieval

`GET /lattice/receipts/{id}`:

- Returns the receipt JSON if the caller has the right tenant or is admin
- Verifies signature server-side as a self-check before returning
- Returns 404 if receipt expired (default TTL: 30 days)

For compliance review: an auditor with the public key can verify any receipt JWT offline without contacting the proxy.

### 3.4 Storage backends

`audit/receipt_store.py`:

- `MemoryReceiptStore` (dev / single-node)
- `RedisReceiptStore` (multi-node)
- `PostgresReceiptStore` (audit-grade durability — required for compliance use cases)

Per-tenant TTL via `receipts.retention_days` in config.

### 3.5 What receipts deliberately don't include

- No raw user message content
- No raw tool args
- No raw model output text
- No raw guardrail violation values (only "kinds")

This is the design constraint that makes receipts safely shareable with auditors / cross-team consumers.

---

## 4. Self-tuning bandit

### 4.1 Arm definition

An "arm" is a candidate compression allowlist for a given (tenant, task_class) pair:

```python
@dataclass(frozen=True, slots=True)
class CompressionArm:
    id: str                                            # stable hash of transform set
    transforms: frozenset[str]                         # e.g. frozenset({"reference_sub", "tool_filter", "path_prefix"})
    description: str
```

The bandit maintains a Beta posterior per arm per (tenant, task_class). Default arms include:

| Arm | Transforms |
|---|---|
| `safe-minimal` | content_profiler, output_cleanup |
| `chat-default` | + reference_sub, tool_filter |
| `code-heavy` | + path_prefix, format_conversion, columnar_pack |
| `agent-heavy` | + tool_projection, tool_diff |
| `summarization` | + rate_distortion, llmlingua (when [llmlingua] installed) |

Operators can add custom arms via config.

### 4.2 Thompson sampling

```python
# src/lattice/planner/bandit/thompson.py
class ThompsonBandit:
    def __init__(self, store: BanditStore, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._store = store
        self._prior_a = prior_alpha
        self._prior_b = prior_beta

    def select_arm(self, tenant: str, task: TaskClass, arms: Sequence[CompressionArm]) -> CompressionArm:
        samples = []
        for arm in arms:
            post = self._store.posterior(tenant, task, arm.id) or (self._prior_a, self._prior_b)
            sample = np.random.beta(*post)
            samples.append((arm, sample))
        return max(samples, key=lambda x: x[1])[0]

    def update(self, tenant: str, task: TaskClass, arm_id: str, reward: float) -> None:
        """reward in [0, 1]. We treat the Beta posterior as Beta(α + reward, β + (1 - reward))."""
        a, b = self._store.posterior(tenant, task, arm_id) or (self._prior_a, self._prior_b)
        self._store.update(tenant, task, arm_id, a + reward, b + (1.0 - reward))
```

### 4.3 Reward signals

```python
# src/lattice/planner/bandit/reward.py
def compute_reward(receipt: LatticeReceipt, feedback: FeedbackSignals) -> float:
    """Composite reward in [0, 1]. Weights configurable per tenant.

    Inputs:
      - HTTP outcome (200 vs 4xx/5xx)
      - User feedback header (`x-lattice-feedback: good|bad|neutral`)
      - Optional LLM-judge score if benchmark-mode enabled
      - Cost reduction (tokens_before - tokens_after) / tokens_before
      - Latency (penalized only if exceeded budget)
      - Quality regression signal (if a downstream evaluator emitted one)
    """
    r = 0.0
    r += 0.4 if feedback.http_ok else 0.0
    r += 0.2 * (1.0 if feedback.user_thumbs == "good" else 0.0 if feedback.user_thumbs == "neutral" else -1.0).clip(0, 1)
    r += 0.25 * max(0.0, receipt.tokens_input_before - receipt.tokens_input_after) / max(1, receipt.tokens_input_before)
    r += 0.1 * (1.0 if not receipt.guardrail_violations else 0.0)
    r += 0.05 * (1.0 if receipt.proxy_latency_ms <= feedback.latency_budget_ms else 0.0)
    return max(0.0, min(1.0, r))
```

Reward updates happen async on a background worker that consumes the receipt + feedback stream.

### 4.4 Safety / cold-start

Cold-start uses the operator-configured default arm for at least 100 observations per (tenant, task) before bandit selection takes over (`bandit.min_samples_per_arm = 100`). This prevents new tenants from getting random arms in their first hour.

Exploration cap: arms with > 5% measured regression on quality signal get a permanent posterior penalty so the bandit doesn't oscillate back to them.

---

## 5. Per-route profiles

### 5.1 Profile schema

`examples/profiles.yaml`:

```yaml
version: 1
default:
  cache:
    layers: ["exact", "ir", "jaccard", "embed"]
  guardrails: { pii: { mode: warn }, injection: { mode: warn } }
  agent_memory: { enabled: true, gc_threshold_tokens: 16000 }

profiles:
  - name: customer-support
    match: { url_prefix: "/api/support", header: { x-app: support } }
    overrides:
      guardrails: { pii: { mode: tokenize } }
      bandit: { default_arm: chat-default }

  - name: code-internal
    match: { url_prefix: "/api/code" }
    overrides:
      bandit: { default_arm: code-heavy }
      agent_memory: { gc_threshold_tokens: 32000 }

  - name: realtime-voice
    match: { url_prefix: "/v1/realtime" }
    overrides:
      cache: { enabled: false }
      agent_memory: { enabled: false }
```

### 5.2 Matchers

```python
# src/lattice/policy/matchers.py
class ProfileMatcher(Protocol):
    def matches(self, request: ProxyRequest) -> bool: ...

class UrlPrefixMatcher: ...
class HeaderMatcher: ...
class ModelPrefixMatcher: ...
class TenantMatcher: ...
class AndMatcher: ...                                  # all sub-matchers
class OrMatcher: ...
```

First profile whose matcher returns True wins. Default profile is applied with no matcher.

### 5.3 Resolution

```python
# src/lattice/policy/profiles.py
class ProfileResolver:
    def resolve(self, request: ProxyRequest) -> ResolvedProfile:
        for profile in self._profiles:
            if profile.matcher.matches(request):
                return self._merge(self._default, profile)
        return self._default
```

Resolved profile is stashed on `request.state.profile` and consumed by:
- Cache layer (which layers are enabled)
- Guardrail policy
- Agent memory config
- Bandit (default arm + per-route exploration)

---

## 6. Hot reload

### 6.1 Mechanism

```python
# src/lattice/config/reload.py
class ConfigReloader:
    """Watches the config file; on change, performs an atomic swap.

    Reload triggers:
      - SIGHUP signal
      - File mtime change (when LATTICE_CONFIG_WATCH=true)
      - HTTP POST /lattice/admin/reload (gated by admin token)

    Reload safety:
      - New config is parsed and validated in full before swap
      - In-flight requests continue using old config (they hold a reference)
      - Diff is computed and unsafe changes logged at WARN
      - Unsafe changes (e.g. removing the signing key while receipts are issuing)
        require LATTICE_ALLOW_UNSAFE_RELOAD=true
    """

    def reload(self, path: Path) -> ReloadResult:
        try:
            new_cfg = LatticeConfig.from_file(path)
        except ValidationError as exc:
            return ReloadResult.failed(exc)
        diff = config_diff(self._current, new_cfg)
        if diff.has_unsafe() and not self._allow_unsafe:
            return ReloadResult.refused(diff)
        with self._swap_lock:
            old = self._current
            self._current = new_cfg
            self._notify_subscribers(old, new_cfg)
        return ReloadResult.success(diff)
```

### 6.2 Subscribers

Components register a callback at startup:

```python
config.subscribe(on_change=cache_layered.reconfigure)
config.subscribe(on_change=guardrails.reconfigure)
config.subscribe(on_change=bandit.reconfigure)
config.subscribe(on_change=profile_resolver.reconfigure)
```

Each subscriber implements `reconfigure(old, new)` atomically.

### 6.3 What can NOT hot reload

- Provider adapter registration (requires process restart due to credential lifecycle)
- OTel exporter endpoint (otel SDK init is one-shot)
- Listening port

The reload result tells the operator which fields had no effect.

---

## 7. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/audit tests/unit/planner/bandit tests/unit/policy tests/unit/config -q` | All pass |
| Receipt E2E | `tests/integration/test_receipt_e2e.py` | Header present; GET endpoint returns valid signature; expired returns 404 |
| Bandit convergence | `tests/integration/test_bandit_e2e.py` (simulator with known optimal arm) | Bandit selects optimal arm ≥ 90% of pulls after 500 observations |
| Profile precedence | `tests/unit/policy/test_profiles.py` | First-match-wins; default fallback works |
| Hot reload | `tests/integration/test_hot_reload.py` (SIGHUP + file change) | In-flight request completes with old config; subsequent requests use new |
| Unsafe reload refused | `tests/integration/test_hot_reload.py::test_remove_signing_key_refused` | Reload returns `refused`; signing key unchanged |
| Receipt content audit | `tests/unit/audit/test_receipts.py::test_no_user_content_in_receipt` | Property: receipt JSON contains no substring of message content |
| Lean install | base | Receipts + bandit + profiles + reload all work without optional deps |
| Canonical bench | usual | ±2% |

---

## 8. Acceptance criteria

1. Every successful response includes `x-lattice-receipt` header with a verifiable JWT.
2. `GET /lattice/receipts/{id}` returns the structured receipt; signature passes verification with the published key.
3. After 500 observations per arm, the Thompson bandit selects the highest-reward arm for ≥ 90% of subsequent requests in a controlled simulator.
4. A request matching `/api/code` uses the `code-internal` profile's bandit default and GC threshold without affecting `/api/support` requests in the same proxy.
5. Sending SIGHUP triggers a config reload that takes effect on the next request; in-flight requests are unaffected.
6. A reload that removes the receipt signing key while receipts are enabled is refused with a clear error unless `LATTICE_ALLOW_UNSAFE_RELOAD=true`.
7. Receipt content audit test passes — no user content leaks.
8. Canonical bench ±2%.

---

## 9. Out of scope

| Topic | Phase |
|---|---|
| Cloud receipt search UI | [Phase 25](25-cloud-multitenant.md) |
| Bandit across tenants (transfer learning) | Future — privacy considerations |
| Receipt encryption (currently only signed, not encrypted) | Future — receipts are already content-free |
| Per-user (sub-tenant) profiles | [Phase 25](25-cloud-multitenant.md) |
