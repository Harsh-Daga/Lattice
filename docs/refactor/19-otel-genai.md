# Phase 19 — OpenTelemetry GenAI Semantic Conventions

> **Footprint impact.** OTel export is **opt-in** via the `[otel]` extra (~ 5 MB: `opentelemetry-sdk` + exporters). Default install ships zero OTel dependencies — `LATTICE_TELEMETRY` defaults to `none` and the entire telemetry/otel module is unimported in that case. With the extra installed and the env var set, +10 MB RSS at idle, +20 MB under load.
>
> **Algorithm location.** New `src/lattice/telemetry/otel/` package. Span construction lives entirely in the proxy; SDKs do not construct spans themselves. The OTel hook in [Phase 19](16-python-sdk-quality.md) / [Phase 24](20-typescript-sdk.md) is a client-side observability hook that wraps user-side timing only; the authoritative spans come from the proxy.
>
> **External-service requirement.** None required. The OTel exporter targets the user's existing observability stack via OTLP (which can point at anything — local Jaeger, Datadog Agent, Honeycomb, Phoenix, OpenTelemetry Collector). No vendor lock-in; no LATTICE-owned telemetry service.
>

> **LoC delta (declared).** +1400 net (`telemetry/otel/`). Opt-in `[otel]` extra.
> **Transport role.** Spans cover transport (`lattice.transport.*`) + pipeline; transport metrics from Phase 20 feed OTel.
> **Registry.** §10 telemetry/otel.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Goal.** Ship native OpenTelemetry export using the `gen_ai.*` semantic conventions that became the industry standard in 2025-2026. A user with an existing observability stack (Datadog, Honeycomb, Phoenix/Arize, Grafana Tempo, Jaeger, AWS X-Ray) sees LATTICE traces and metrics natively alongside the rest of their app, without writing any glue code.
>
> **Outcome.** A new `telemetry/otel/` package emits CLIENT spans named `gen_ai.<operation>` with all spec-defined attributes plus a `lattice.*` namespace for our differentiated insights (compression ratio, transforms applied, cache layer hit, TACC window, guardrail violations). One environment variable picks the preset: `LATTICE_TELEMETRY=otlp|datadog|honeycomb|phoenix|none`. Upstream provider request IDs are correlated. Content capture is opt-in per spec. Existing OpenLLMetry-instrumented apps see LATTICE traces nested correctly inside their existing spans.
>
> **Estimated effort.** 4 days (1 PR, ~+1500/-100 LoC).

---

## 1. Why this phase exists

The OpenTelemetry GenAI semantic conventions (specs at `opentelemetry.io/docs/specs/semconv/gen-ai/`) define a stable contract: every LLM client emits spans named `gen_ai.<operation>` with `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.cache_read.input_tokens`, etc.

In 2026 every other proxy emits these. LATTICE emits none. Today's observability is Prometheus metrics + response headers — useful but unconnected to the user's trace graph. A user who runs LATTICE alongside their own app cannot pivot from a slow request in Datadog to the LATTICE span that explains *why*.

The strategic move: implement the standard properly **and** add a `lattice.*` extension namespace for the insights only we can produce.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/telemetry/otel/__init__.py
src/lattice/telemetry/otel/exporter.py          # OTLP exporter wiring
src/lattice/telemetry/otel/spans.py             # span helpers; gen_ai + lattice attrs
src/lattice/telemetry/otel/metrics.py           # OTel meter wrapping our MetricsCollector
src/lattice/telemetry/otel/presets.py           # datadog | honeycomb | phoenix | jaeger | otlp
src/lattice/telemetry/otel/content_capture.py   # opt-in per spec for input/output payloads
src/lattice/telemetry/otel/correlation.py       # upstream request-id parsing
tests/unit/telemetry/otel/test_spans.py
tests/unit/telemetry/otel/test_attribute_compliance.py
tests/unit/telemetry/otel/test_presets.py
tests/integration/telemetry/otel/test_otlp_roundtrip.py    # uses a local OTel collector container fixture
docs/operations/observability.md                # new operator-facing doc
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/telemetry/__init__.py](../../src/lattice/telemetry/__init__.py) | Re-export `configure_otel`, `OTelConfig`, `LatticeAttributes` |
| [src/lattice/proxy/middleware.py](../../src/lattice/proxy/middleware.py) | Start root request span; attach `x-trace-id` header on response |
| [src/lattice/providers/transport/completion.py](../../src/lattice/providers/transport/completion.py) | Capture upstream `x-request-id` / `request-id` headers into the active span |
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | Per-transform child span when OTel is enabled |
| [src/lattice/sdk/proxy_client.py](../../src/lattice/sdk/proxy_client.py) | Honour incoming W3C traceparent; create CLIENT span on the user side |
| [src/lattice/sdk/sync_proxy_client.py](../../src/lattice/sdk/sync_proxy_client.py) | Same |
| [src/lattice/core/config.py](../../src/lattice/core/config.py) | Add `TelemetryConfig` section |
| [pyproject.toml](../../pyproject.toml) | New optional group: `otel = ["opentelemetry-api>=1.27", "opentelemetry-sdk>=1.27", "opentelemetry-exporter-otlp>=1.27"]` |

### 2.3 Deleted

None.

---

## 3. Step-by-step

### 3.1 Step 1 — `TelemetryConfig`

```python
# src/lattice/core/config.py
class OTelConfig(BaseModel):
    enabled: bool = False
    preset: Literal["otlp", "datadog", "honeycomb", "phoenix", "jaeger", "none"] = "none"
    endpoint: str | None = None                              # overrides preset endpoint
    headers: dict[str, str] = Field(default_factory=dict)
    service_name: str = "lattice"
    service_version: str | None = None                       # defaults to __version__
    sampling_ratio: float = 1.0                              # head-based sampling
    capture_content: Literal["never", "user-opt-in", "always"] = "never"
    content_max_bytes: int = 8192                            # truncate captured payloads
    semconv_opt_in: Literal["stable", "experimental"] = "experimental"
```

ENV variable convenience:

```
LATTICE_TELEMETRY=otlp                           # or datadog | honeycomb | phoenix | jaeger
LATTICE_TELEMETRY_ENDPOINT=https://...           # optional override
LATTICE_TELEMETRY_HEADERS=k1=v1,k2=v2            # auth headers (e.g. Datadog API key)
LATTICE_TELEMETRY_CAPTURE_CONTENT=never          # default
LATTICE_TELEMETRY_SAMPLING_RATIO=1.0
LATTICE_TELEMETRY_SERVICE_NAME=lattice-prod
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental   # per OTel spec
```

If `LATTICE_TELEMETRY` is set, `OTelConfig.enabled = True` and `preset = <env value>`.

### 3.2 Step 2 — Presets

`src/lattice/telemetry/otel/presets.py`:

```python
@dataclass(frozen=True, slots=True)
class Preset:
    endpoint: str
    headers: Mapping[str, str]
    protocol: Literal["grpc", "http/protobuf"]


def resolve_preset(cfg: OTelConfig) -> Preset:
    match cfg.preset:
        case "otlp":
            return Preset(
                endpoint=cfg.endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                headers=cfg.headers,
                protocol="grpc",
            )
        case "datadog":
            return Preset(
                endpoint=cfg.endpoint or "https://trace.agent.datadoghq.com",
                headers={"DD-API-KEY": os.environ["DD_API_KEY"], **cfg.headers},
                protocol="http/protobuf",
            )
        case "honeycomb":
            return Preset(
                endpoint=cfg.endpoint or "https://api.honeycomb.io",
                headers={"x-honeycomb-team": os.environ["HONEYCOMB_API_KEY"], **cfg.headers},
                protocol="http/protobuf",
            )
        case "phoenix":
            return Preset(
                endpoint=cfg.endpoint or "http://localhost:6006/v1/traces",
                headers=cfg.headers,
                protocol="http/protobuf",
            )
        case "jaeger":
            return Preset(
                endpoint=cfg.endpoint or "http://localhost:14268/api/traces",
                headers=cfg.headers,
                protocol="http/protobuf",
            )
        case _:
            raise ValueError(f"Unknown preset {cfg.preset!r}")
```

### 3.3 Step 3 — One-shot configuration

`src/lattice/telemetry/otel/exporter.py`:

```python
def configure_otel(config: OTelConfig) -> None:
    """Called once at proxy startup (proxy/bootstrap.py) or SDK init.

    Idempotent: calling twice is a no-op. Tests rely on this.
    """
    if not config.enabled or config.preset == "none":
        return
    global _CONFIGURED
    if _CONFIGURED:
        return

    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    preset = resolve_preset(config)
    resource = Resource.create({
        "service.name": config.service_name,
        "service.version": config.service_version or __version__,
        "telemetry.sdk.name": "lattice",
    })

    exporter = _make_span_exporter(preset)
    provider = TracerProvider(resource=resource,
                              sampler=_make_sampler(config.sampling_ratio))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    metric_exporter = _make_metric_exporter(preset)
    meter_provider = MeterProvider(resource=resource, metric_readers=[
        PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30_000),
    ])
    otel_metrics.set_meter_provider(meter_provider)

    _CONFIGURED = True
```

### 3.4 Step 4 — Span attributes per spec

`src/lattice/telemetry/otel/spans.py`:

```python
class LatticeAttributes:
    """Stable attribute names. Don't inline magic strings elsewhere.

    `gen_ai.*` keys follow the OTel GenAI semantic conventions.
    `lattice.*` keys are our extensions and are documented in docs/operations/observability.md.
    """
    # gen_ai (OTel spec)
    GEN_AI_SYSTEM = "gen_ai.system"                          # "openai" | "anthropic" | ...
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"          # "chat" | "embeddings" | ...
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # lattice (extensions)
    L_COMPRESSION_RATIO = "lattice.compression.ratio"
    L_COMPRESSION_TOKENS_BEFORE = "lattice.compression.tokens_before"
    L_COMPRESSION_TOKENS_AFTER = "lattice.compression.tokens_after"
    L_TRANSFORMS_APPLIED = "lattice.transforms.applied"
    L_TRANSFORMS_ROLLED_BACK = "lattice.transforms.rolled_back"
    L_CACHE_LAYER = "lattice.cache.layer"                    # "exact" | "ir" | "jaccard" | "embed" | "miss"
    L_CACHE_KEY = "lattice.cache.key"                        # hashed; never raw content
    L_TACC_WINDOW = "lattice.tacc.window_size"
    L_TACC_ADMISSION_LATENCY_MS = "lattice.tacc.admission_latency_ms"
    L_EXECUTION_PLAN_TIER = "lattice.execution_plan.tier"
    L_EXECUTION_PLAN_QUALITY_FLOOR = "lattice.execution_plan.quality_floor"
    L_GUARDRAIL_VIOLATIONS = "lattice.guardrail.violations"      # list[str], e.g. ["pii:tokenize:email", "injection:warn"]
    L_GUARDRAIL_MODIFICATIONS = "lattice.guardrail.modifications"
    L_TENANT_ID = "lattice.tenant.id"                        # never the raw API key
    L_UPSTREAM_REQUEST_ID = "lattice.upstream.request_id"
    L_RECEIPT_ID = "lattice.receipt.id"
```

The span builder:

```python
@contextmanager
def gen_ai_span(operation: str, *, system: str, model: str, span_kind=SpanKind.CLIENT) -> Iterator[Span]:
    name = f"gen_ai.{operation}"
    tracer = trace.get_tracer("lattice")
    with tracer.start_as_current_span(name, kind=span_kind) as span:
        span.set_attribute(LatticeAttributes.GEN_AI_SYSTEM, system)
        span.set_attribute(LatticeAttributes.GEN_AI_OPERATION_NAME, operation)
        span.set_attribute(LatticeAttributes.GEN_AI_REQUEST_MODEL, model)
        yield span
```

### 3.5 Step 5 — Per-transform child spans

`pipeline/executor.py` (post-split from Phase 16) wraps each transform call:

```python
def _run_transform(self, transform, request, ctx, plan):
    if ctx.otel_enabled:
        with otel.transform_span(transform.name) as span:
            span.set_attribute("lattice.transform.priority", transform.priority)
            span.set_attribute("lattice.transform.safety_class", transform.safety_class)
            result = self._do_run(transform, request, ctx, plan)
            span.set_attribute(LatticeAttributes.L_COMPRESSION_TOKENS_BEFORE, ctx.tokens_before)
            span.set_attribute(LatticeAttributes.L_COMPRESSION_TOKENS_AFTER, ctx.tokens_after)
            if result.is_err():
                span.set_status(StatusCode.ERROR, str(result.unwrap_err()))
            return result
    return self._do_run(transform, request, ctx, plan)
```

Span hierarchy:

```
gen_ai.chat (CLIENT) – parent
├── lattice.pipeline.compress (INTERNAL)
│   ├── lattice.transform.content_profiler (INTERNAL)
│   ├── lattice.transform.cache_arbitrage (INTERNAL)
│   ├── lattice.transform.reference_sub (INTERNAL)
│   └── lattice.transform.tool_filter (INTERNAL)
├── lattice.cache.lookup (INTERNAL)        # only when not a hit-and-return
├── lattice.provider.dispatch (CLIENT)     # the upstream call
│   └── (provider-internal spans inherit via traceparent if upstream is instrumented)
└── lattice.pipeline.reverse (INTERNAL)
```

### 3.6 Step 6 — Upstream request-id correlation

`src/lattice/telemetry/otel/correlation.py`:

```python
_UPSTREAM_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "openai-request-id", "anthropic-request-id")


def attach_upstream_request_id(span: Span, headers: Mapping[str, str]) -> None:
    for h in _UPSTREAM_REQUEST_ID_HEADERS:
        if (val := headers.get(h) or headers.get(h.upper())) is not None:
            span.set_attribute(LatticeAttributes.L_UPSTREAM_REQUEST_ID, val)
            return
```

Called inside `providers/transport/completion.py` after the upstream response arrives.

### 3.7 Step 7 — Content capture (opt-in per spec)

Per the OTel spec, instrumentations SHOULD NOT capture instructions, inputs, or outputs by default. We honour this:

```python
def maybe_capture_content(span: Span, request: Request, response: Response | None, cfg: OTelConfig) -> None:
    if cfg.capture_content == "never":
        return
    if cfg.capture_content == "user-opt-in" and not request.headers.get("x-lattice-capture-content"):
        return
    # cfg.capture_content == "always" or user opt-in
    span.set_attribute("gen_ai.input.messages",
                        _truncate(json.dumps([m.model_dump() for m in request.messages]),
                                  cfg.content_max_bytes))
    if response:
        span.set_attribute("gen_ai.output.messages",
                            _truncate(json.dumps(response.serialize_choices()), cfg.content_max_bytes))
```

Content capture also redacts via the PII tokenizer ([Phase 21](18-native-guardrails.md)) before attribute attachment, so even with capture-on we never put raw PII into spans.

### 3.8 Step 8 — Metrics export

```python
# src/lattice/telemetry/otel/metrics.py
def install_metric_bridge(collector: MetricsCollector) -> None:
    """Mirror existing Prometheus metrics into OTel meter on every increment.

    We do not replace the Prometheus collector; we wrap it. This keeps the
    /metrics endpoint working and dual-emits for OTel-only consumers.
    """
    meter = otel_metrics.get_meter("lattice")
    counters = {}
    histograms = {}

    def on_increment(name: str, value: float, tags: Mapping[str, str]):
        c = counters.get(name) or counters.setdefault(name, meter.create_counter(name))
        c.add(value, attributes=dict(tags))

    def on_observe(name: str, value: float, tags: Mapping[str, str]):
        h = histograms.get(name) or histograms.setdefault(name, meter.create_histogram(name, unit="ms"))
        h.record(value, attributes=dict(tags))

    collector.subscribe(on_increment, on_observe)
```

### 3.9 Step 9 — W3C traceparent propagation

Both the SDK and the proxy honour the standard `traceparent` / `tracestate` headers:

- SDK client: if a `traceparent` is present in the user's environment (via OTel context), attach it on the outgoing request.
- Proxy: extract `traceparent` from the inbound request, set it as the parent of the root `gen_ai.chat` span. The upstream provider call inherits.

Result: a user instrumented with OpenLLMetry sees their LATTICE traces nested correctly under their app spans.

### 3.10 Step 10 — Documentation

`docs/operations/observability.md`:

- Quick start (one env var)
- Full attribute reference (every `gen_ai.*` we emit, every `lattice.*` extension)
- Sample queries for Datadog APM, Honeycomb, Grafana Tempo, Jaeger UI
- Cost-savings dashboard JSON (importable for Grafana)
- Privacy notes on content capture

---

## 4. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/telemetry/otel -q` | All pass |
| Attribute compliance | `tests/unit/telemetry/otel/test_attribute_compliance.py` | Every span we emit has every required gen_ai attribute |
| Preset wiring | `tests/unit/telemetry/otel/test_presets.py` | Each preset resolves to non-empty endpoint + headers |
| OTLP roundtrip | `tests/integration/telemetry/otel/test_otlp_roundtrip.py` (testcontainers OTel collector) | Spans emitted from proxy match expected shape in collector logs |
| Lean install | `pip install lattice-transport && lattice proxy run` (no `[otel]`) | Boots with telemetry disabled cleanly |
| Latency | OTel-enabled vs disabled proxy overhead | ≤ 0.5 ms median additional |
| Content opt-out | Default config sends no message content in spans | Captured fixture contains no user content |
| Canonical bench | usual | ±2% |

### 4.1 Compliance test

```python
def test_chat_completion_span_compliance():
    """Every required gen_ai attribute is present on the chat span."""
    span = run_request_and_capture_span("chat", model="openai/gpt-4o")
    required = {
        LatticeAttributes.GEN_AI_SYSTEM, LatticeAttributes.GEN_AI_OPERATION_NAME,
        LatticeAttributes.GEN_AI_REQUEST_MODEL, LatticeAttributes.GEN_AI_RESPONSE_MODEL,
        LatticeAttributes.GEN_AI_USAGE_INPUT_TOKENS, LatticeAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
    }
    missing = required - span.attributes.keys()
    assert not missing, f"missing required attributes: {missing}"
```

---

## 5. Acceptance criteria

1. `LATTICE_TELEMETRY=otlp lattice proxy run` starts, emits a `gen_ai.chat` span to `http://localhost:4317` for each chat request.
2. `LATTICE_TELEMETRY=datadog DD_API_KEY=… lattice proxy run` emits to Datadog; spans appear in APM within 30s.
3. Without `[otel]` installed, the proxy starts cleanly and `lattice info` reports `telemetry: disabled`.
4. With `LATTICE_TELEMETRY_CAPTURE_CONTENT=never` (default), no captured span attribute contains any message content.
5. Per-transform child spans appear with `lattice.transform.<name>` names and rollup correctly into the parent `gen_ai.chat` span (visible in Jaeger UI).
6. Upstream `x-request-id` is attached as `lattice.upstream.request_id` on the parent span.
7. OTel-enabled vs disabled proxy benchmark shows ≤ 0.5 ms median additional overhead.

---

## 6. Out of scope

| Topic | Phase |
|---|---|
| Per-tenant attribution labels (`tenant.id`) | [Phase 32](32-cloud-multitenant.md) |
| Distributed trace pivoting from TACC events | Could be added later if requested |
| Replacing Prometheus with OTel-only export | Out — Prometheus stays for ops who need scraped pull-based metrics |
