# Phase 22 — Cross-Provider Cache Portability, Cold-Start Warmer, KV-Cache Compatibility Analyzer

> **Footprint impact.** Zero new runtime deps. Portability manifests are signed using the same HMAC machinery [Phase 23](23-receipts-bandit-profiles.md) uses for receipts (which uses `pyjwt`, already an optional `[auth]` dep — if `[auth]` isn't installed, the portability manifest falls back to unsigned). CLI utilities ship in the base wheel.
>
> **Algorithm location.** New `src/lattice/cache/portability/` package, `src/lattice/cache/warmer.py`, `src/lattice/cache/analyzer.py`. All reuse Phase 14's tier and tenant infrastructure. No SDK involvement — the portability CLI talks to the local proxy's `/lattice/cache/{warm,analyze}` JSON endpoints.
>
> **External-service requirement.** None. No cloud, no external storage, no third-party manifest server. Manifests live in the same backend Phase 14 is using (memory / Redis / Postgres). The user-initiated provider switch is entirely local.
>
> **Reaffirms the no-multi-provider-routing rule.** This phase optimizes for the case where a user *deliberately decides* to change provider (e.g. moves from `openai/gpt-4o` to `anthropic/claude-3-5-sonnet` via a config change). LATTICE does not pick the provider; the user does. The portability layer ensures the next provider's experience is instantly warm.
>

> **LoC delta (declared).** +2400 net (`cache/portability/`).
> **Transport role.** Portable manifests attach to cache entries; no multi-provider routing — user-initiated provider switch only.
> **Registry.** §5 portability + warmer + analyzer.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Goal.** Three closely-related capabilities that all exploit our IR-canonical layer to extract value no other gateway can: (1) a portable cache manifest that lets a **user-initiated** switch between providers inherit the warmth of LATTICE's hybrid cache (not multi-provider routing — the user decides the switch, we make their next-provider experience instantly warm); (2) a cold-start cache warmer that pre-populates the layered cache from known prompt templates the moment a tenant boots; (3) a developer tool that analyzes a tenant's prompts and reports a "KV-cache compatibility score" with concrete suggestions to restructure prompts for higher provider-native caching.
>
> **Outcome.** A new `cache/portability/` package emits a signed `PortableCacheManifest` per cached entry recording structural fingerprints invariant across providers, plus per-provider native cache keys (OpenAI `prompt_cache_key`, Anthropic `cache_control` shapes, Gemini `cachedContent` IDs). When a user moves from `openai/gpt-4o` to `anthropic/claude-3-5-sonnet` (their decision, e.g. via deployment config change), every cached entry that has both providers in its manifest stays a hit. A new `lattice cache warm` CLI ingests YAML/JSON of prompt templates and pre-populates all four cache layers. A new `lattice cache analyze` CLI parses an OpenTelemetry trace or a directory of recent requests and produces a report of "your prompt template has 87% provider-native cache compatibility — here's what to change to get to 100%".
>
> **Estimated effort.** 5 days (1 PR, ~+2400 LoC).

---

## 1. Why this phase exists, and why this is NOT multi-provider routing

The audit highlighted three independent user pains:

| Pain | Today |
|---|---|
| User decides to switch providers (cost, latency, capability) → loses all cache warmth | Cache hits drop to 0 for days |
| Brand new deployment / new tenant → all cache layers cold; first hour of traffic is uncached | Painful onboarding metrics |
| Users know provider-native caching exists (OpenAI cached_tokens, Anthropic prompt caching) but don't know how to structure prompts for it | Leave 50% off the table |

**Explicit non-goal:** routing. We never decide to switch providers for a request. The user decides (via their config / SDK call / proxy config). What we do is make their decision cheaper:

- The portable manifest is **read-only** — we record what we've seen, we don't act on it autonomously.
- Cache warming uses whichever provider the tenant has configured as their target — not "the cheapest".
- The analyzer is purely descriptive.

This phase makes LATTICE the only gateway with cache continuity across user-initiated provider changes.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/cache/portability/__init__.py
src/lattice/cache/portability/manifest.py        # PortableCacheManifest dataclass + signing
src/lattice/cache/portability/exporter.py        # serialize manifest for export/import
src/lattice/cache/portability/translator.py      # provider-specific cache key translators
src/lattice/cache/portability/store.py           # storage on top of state.store
src/lattice/cache/warmer.py                      # cold-start warm-up engine
src/lattice/cache/analyzer.py                    # KV-cache compatibility analyzer
src/lattice/cli/cache.py                         # lattice cache {warm, analyze, export, import, stats}
tests/unit/cache/portability/test_manifest.py
tests/unit/cache/portability/test_translator.py
tests/unit/cache/test_warmer.py
tests/unit/cache/test_analyzer.py
tests/integration/cache/test_portability_e2e.py
tests/integration/cache/test_warmer_e2e.py
docs/operations/cache_portability.md
docs/operations/prompt_caching_guide.md          # user-facing guide
examples/cache-warm-templates.yaml
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/cache/semantic.py](../../src/lattice/cache/semantic.py) (post-Phase-12 split) | On store, also persist manifest; on lookup, attempt manifest re-key |
| [src/lattice/cache/ir_fingerprint.py](../../src/lattice/cache/ir_fingerprint.py) (from Phase 14) | Add `provider_invariant_fingerprint()` that excludes provider-specific metadata |
| [src/lattice/providers/adapters/openai.py](../../src/lattice/providers/adapters/openai.py) | Surface `prompt_cache_key` extraction from response, `cached_tokens` from usage |
| [src/lattice/providers/adapters/anthropic/__init__.py](../../src/lattice/providers/adapters/anthropic/__init__.py) | Surface cache_creation/read tokens; auto-attach `cache_control` markers where appropriate |
| [src/lattice/cli/__init__.py](../../src/lattice/cli/__init__.py) | Register `cache` subcommand tree |
| [src/lattice/core/config.py](../../src/lattice/core/config.py) | Add `CachePortabilityConfig`, `CacheWarmConfig` |

### 2.3 Deleted

None.

---

## 3. PortableCacheManifest

### 3.1 Shape

```python
# src/lattice/cache/portability/manifest.py
@dataclass(frozen=True, slots=True)
class PortableCacheManifest:
    """One per cached entry; survives provider switches.

    Stored in state.store alongside the cached response under key
    "portable:{tenant}:{ir_fp}".
    """
    version: int                                 # current: 1
    tenant: str

    # Provider-invariant fingerprints
    ir_fingerprint: str                          # canonical IR hash (Phase 14)
    provider_invariant_fingerprint: str          # IR hash excluding provider-specific bits
    semantic_fingerprint: str                    # embedding-vector hash (rounded to ε)
    jaccard_signature: str                       # MinHash-like sketch for quick lookups

    # Provider-specific addressing (filled in as we observe each provider)
    provider_keys: Mapping[str, ProviderCacheBinding]    # "openai" -> binding, "anthropic" -> binding, ...

    # Bookkeeping
    created_at: float
    last_observed_at: float
    observation_count: int
    response_blob_hash: str                      # CAS pointer (Phase 14)

    # Signature
    signature: str                               # HMAC over the above; Phase 23 receipt-aligned key
```

```python
@dataclass(frozen=True, slots=True)
class ProviderCacheBinding:
    provider: str
    model: str
    # OpenAI: prompt_cache_key value returned in usage.prompt_tokens_details
    # Anthropic: cache_control block IDs / system block hashes
    # Gemini: cachedContent ID
    native_cache_key: str | None
    cached_tokens_observed: int                  # last seen cached_tokens count
    cache_creation_tokens: int                   # for Anthropic
    last_seen_at: float
```

### 3.2 Storage and lookup

When a response is cached (Phase 14), we also create or update the manifest:

```python
# in semantic cache on store path
manifest = self._portable.upsert_manifest(
    tenant=tenant,
    request=request,
    response=response,
    ir=ir,
    embedding=embedding,
    response_blob_hash=blob_hash,
)
self._store.set(f"portable:{tenant}:{manifest.ir_fingerprint}", manifest.serialize(), ttl=...)
```

On lookup, before checking the layered cache normally:

```python
# in semantic cache on lookup
ir = profiler.build_ir_only(request)
ir_fp = ir_canonical_fingerprint(ir, namespace=tenant, model=request.model)
manifest = self._portable.lookup_by_ir(ir_fp, tenant=tenant)

# Standard layered lookup (unchanged)
hit = standard_layered_lookup(request, tenant)
if hit is not None:
    return hit

# If standard lookup missed but we have a manifest from another provider, re-check
# with provider-invariant fingerprint — this catches the user-switched-provider case.
if manifest is not None:
    invariant_fp = ir_provider_invariant_fingerprint(ir)
    cross_hit = self._lookup_by_invariant_fp(invariant_fp, tenant=tenant)
    if cross_hit:
        # Record the cross-provider hit
        self._metrics.increment("cache.hits", tags={"layer": "portable", "tenant": tenant})
        return cross_hit
return None
```

This lookup path is bounded: it only fires when an IR-fingerprint miss happens AND a manifest exists. Total added latency: ≤ 2 ms on miss.

### 3.3 Provider-invariant fingerprint

`src/lattice/cache/ir_fingerprint.py` gets a new function:

```python
def ir_provider_invariant_fingerprint(ir: PromptIRV2) -> str:
    """Hash the canonical IR with all provider-specific metadata excluded.

    Removes:
    - model name (gpt-4o vs claude-3-5-sonnet)
    - reasoning/thinking blocks
    - cache_control markers
    - response_format provider-specific shapes
    - tool_choice provider-specific values

    Preserves:
    - message content
    - role structure
    - tool definitions (semantic shape)
    - structured outputs (schema, not provider implementation)
    """
    stripped = _strip_provider_specific(ir)
    return _hash_canonical(stripped)
```

This is the *provably* invariant signal: two requests with the same stripped IR are semantically equivalent regardless of provider.

### 3.4 Signing

Manifest signature uses the same HMAC key as the [Phase 23](23-receipts-bandit-profiles.md) receipts — single key rotation surface. Signature verifies integrity on load (prevents tampering); not required for correctness but required for cloud multi-tenant trust ([Phase 25](25-cloud-multitenant.md)).

---

## 4. Cold-start cache warmer

### 4.1 Template format

`examples/cache-warm-templates.yaml`:

```yaml
version: 1
tenant: my-tenant
templates:
  - name: support-system-prompt
    description: Customer support agent system prompt
    request:
      model: openai/gpt-4o
      messages:
        - role: system
          content: |
            You are a helpful customer support agent for Acme Corp...
        # Optionally include an example user turn for warm-up
        - role: user
          content: "How do I reset my password?"
      tools:
        - { type: function, function: { name: lookup_order, ... } }
    warm_strategy: full              # or "ir-fingerprint-only" (no response cached, just metadata)

  - name: code-review-system
    request:
      model: anthropic/claude-3-5-sonnet-latest
      messages:
        - role: system
          content: |
            <file:cache-warm-templates/code_review_system.md>
```

`<file:...>` refs read external files for long prompts.

### 4.2 Warming engine

`src/lattice/cache/warmer.py`:

```python
class CacheWarmer:
    """Pre-populates the layered cache from a list of templates.

    For each template:
      1. Build IR (no transforms)
      2. Compute all four cache keys (exact / IR fingerprint / Jaccard sketch / embedding)
      3. Optionally execute against the target provider once and cache the response
      4. Record a manifest

    If warm_strategy == "ir-fingerprint-only", no upstream call is made — we
    just populate metadata so subsequent matches accelerate the lookup path.
    """

    async def warm(self, templates: list[WarmTemplate], *, tenant: str) -> WarmReport:
        report = WarmReport(tenant=tenant)
        for template in templates:
            try:
                ir = self._profiler.build_ir_only(template.request)
                fps = self._compute_fingerprints(ir, template.request)
                report.fingerprints_recorded += 1
                if template.warm_strategy == "full":
                    response = await self._dispatcher.send(template.request, ctx=warming_ctx(tenant))
                    self._cache.store(template.request, response, namespace=tenant, ttl_seconds=template.ttl)
                    report.responses_cached += 1
                self._portable.record_warm_template(tenant, template.name, fps)
            except Exception as exc:
                report.failures.append((template.name, str(exc)))
        return report
```

The `warm` CLI exposes this:

```bash
lattice cache warm --config cache-warm-templates.yaml --tenant my-tenant [--strategy full|ir-fingerprint-only] [--dry-run]
```

A typical "warm at deploy" flow: tenant deploys, post-deploy hook runs `lattice cache warm --config ./prompts/templates.yaml --tenant prod`. First minute of traffic has hits instead of all misses.

### 4.3 Auto-warming from traces

`benchmarks/datasets/replay_traces.jsonl` (Phase 10) is a natural input. We expose a flag:

```bash
lattice cache warm --from-trace traces.jsonl --tenant prod --top-frequency 100
```

Picks the top-100 most frequent prompt templates from the trace, generates warm templates automatically, and warms them.

---

## 5. KV-cache compatibility analyzer

### 5.1 What it analyzes

OpenAI and Anthropic both reward prompts whose **prefix** is stable across requests. Cached tokens are billed at 50-90% discount. Many prompts inadvertently break the prefix:

- Putting a dynamic timestamp at the start of the system prompt
- Reordering tool definitions across requests
- Concatenating a per-user UUID into the system message
- Returning a different tool list order

`src/lattice/cache/analyzer.py`:

```python
class KVCacheAnalyzer:
    """Analyzes prompt traffic and produces a compatibility score + suggestions.

    Inputs:
      - A directory of OpenTelemetry trace files (otelcol JSON exports), or
      - A JSONL of recent requests, or
      - The session store of a running proxy

    Outputs a report:
      - Per template: score in [0, 1], cached_tokens_observed / total, prefix_stability
      - Per template: ordered list of suggestions (e.g. "Move the dynamic timestamp out of message[0]")
      - Top 10 cost wins ranked by potential $ saved if suggestion adopted
    """

    def analyze(self, source: AnalysisSource) -> CompatibilityReport:
        templates = self._extract_templates(source)
        report = CompatibilityReport()
        for tmpl_id, requests in templates.items():
            stability = self._compute_prefix_stability(requests)
            cached_ratio = mean(r.cached_tokens / max(1, r.prompt_tokens) for r in requests if r.cached_tokens is not None)
            suggestions = self._suggest(requests, stability)
            potential = self._estimate_savings(requests, current_ratio=cached_ratio, target_ratio=0.85)
            report.templates.append(TemplateReport(
                id=tmpl_id, sample_count=len(requests),
                prefix_stability=stability, observed_cached_ratio=cached_ratio,
                suggestions=suggestions, potential_monthly_savings_usd=potential,
            ))
        report.templates.sort(key=lambda t: t.potential_monthly_savings_usd, reverse=True)
        return report
```

### 5.2 Suggestion engine

Suggestions are pattern-matched against known anti-patterns:

| Anti-pattern | Detection | Suggestion |
|---|---|---|
| Timestamp/UUID/locale in first 200 chars of system message | Token diff across requests in known regex bands | "Move dynamic values into a separate trailing message or use prompt_cache_key" |
| Tool definition order varies | Set equality but sequence inequality | "Sort tool definitions by name for stable prefix" |
| System prompt content drifts per request | Levenshtein distance / message > threshold | "Externalize per-request variables into the user message" |
| Reasoning model with non-anchor system | Reasoning-model path + variable system | "Use anchor blocks (Anthropic) or prompt_cache_key (OpenAI)" |

### 5.3 Output

`lattice cache analyze --source ./traces/ --output report.md`:

```markdown
# KV-Cache Compatibility Report
Generated 2026-05-24 — 12,043 requests analyzed across 47 templates

## Top 10 Potential Wins

| Template | Cached % now | Cached % possible | Δ tokens/month | $ saved / month |
|---|---|---|---|---|
| customer_support_system | 12% | 88% | 4.2B | $4,200 |
| code_review_v3 | 0% | 75% | 980M | $980 |
...

## Template: customer_support_system
- Prefix stability: 0.34 (low)
- Detected anti-patterns:
  1. Timestamp at messages[0].content[12:32] — varies per request
  2. User UUID injected into system prompt at messages[0].content[120:156]
- Suggestions:
  1. Move "Today's date: ..." to a separate system message after the static portion
  2. Pass user_id via tool context or response_format extras, not the system prompt
  3. After changes, enable `prompt_cache_key` per session for 0.5× discount
```

The same report is available as JSON for programmatic consumption.

---

## 6. Step-by-step delivery

### 6.1 Step 1 — Portable manifest dataclass + signing
### 6.2 Step 2 — Cache integration (store + lookup wired through)
### 6.3 Step 3 — Provider-invariant fingerprint
### 6.4 Step 4 — Manifest exporter/importer (`lattice cache export/import`)
### 6.5 Step 5 — Warmer engine + template parser
### 6.6 Step 6 — `lattice cache warm` CLI
### 6.7 Step 7 — Analyzer engine + suggestion patterns
### 6.8 Step 8 — `lattice cache analyze` CLI
### 6.9 Step 9 — Provider-native cache key extraction (OpenAI / Anthropic / Gemini)
### 6.10 Step 10 — Documentation: `docs/operations/cache_portability.md`, `docs/operations/prompt_caching_guide.md`

Each step is implemented behind a feature flag; the full suite turns on with `cache.portability.enabled = true` in config.

---

## 7. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/cache/portability tests/unit/cache/test_warmer.py tests/unit/cache/test_analyzer.py -q` | All pass |
| E2E portability | `tests/integration/cache/test_portability_e2e.py` (user switches openai→anthropic mid-test) | Manifest re-key hit observed; metric `cache.hits{layer=portable}` increments |
| Warmer E2E | `tests/integration/cache/test_warmer_e2e.py` | Pre-warmed templates hit on first user request |
| Analyzer | `tests/unit/cache/test_analyzer.py` | Known-anti-pattern fixtures produce expected suggestions |
| Signature | `tests/unit/cache/portability/test_manifest.py` | Tampered manifest fails verification |
| Lean install | base | Portability + warmer + analyzer all work without optional deps |
| Canonical bench | usual | ±2% |

---

## 8. Acceptance criteria

1. After caching N requests with `openai/gpt-4o`, sending the same requests with the user-configured target switched to `anthropic/claude-3-5-sonnet-latest` results in cache hits with `x-lattice-cache-layer: portable` for ≥ 80% of cases where the request body is semantically identical.
2. `lattice cache warm --config templates.yaml --tenant t` pre-populates the cache and the first user request matching a template hits.
3. `lattice cache analyze --source traces.jsonl --output report.md` produces a Markdown report with a top-10 wins table and at least one concrete suggestion for each anti-pattern fixture.
4. Manifest signature is required before trusting cross-provider re-key lookups in cloud multi-tenant mode (Phase 25).
5. No code path performs an *automatic* multi-provider switch — every dispatcher call respects the model field on the incoming request verbatim. Verified by a contract test that fails CI on any new code path matching `provider != request.provider`.
6. Canonical bench ±2%.

---

## 9. Out of scope

| Topic | Phase |
|---|---|
| Automatic cheapest-provider routing | **Forever out** |
| Cross-tenant manifest sharing | Privacy — never |
| Compatibility analyzer recommending provider changes | Out — only restructure prompts on current provider |
| Real-time analyzer (live trace ingestion) | Future |
