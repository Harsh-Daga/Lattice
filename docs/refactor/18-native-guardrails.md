# Phase 18 — Native Guardrails (lightweight defaults)

> **Footprint impact.** **Zero new runtime deps in the default install.** Rule-based PII detection (regex), heuristic injection detection (phrase + role-confusion), and JSON structural repair (pure Python) ship in the base wheel. Presidio (~1 GB with spaCy) and the ONNX DeBERTa-v3 injection classifier (~90 MB onnxruntime + 5 MB model) are **opt-in extras** clearly labeled with their footprint cost. The default config gives meaningful coverage for the common cases without any download.
>
> **Algorithm location.** New `safety/` module containing `pii/`, `injection/`, `output/` subpackages. Reversible PII tokenization reuses the alias-table machinery from `src/lattice/transforms/reference_sub.py` — single source of truth for reverse-pass. No SDK reimplements any of this; guardrail decisions return on response headers and OTel attributes.
>
> **External-service requirement.** None for default detectors. Optional: any of the heavy detectors can be bound to a cloud API instead (Bedrock Guardrails, Azure Content Safety) — for users who already have those subscriptions. We never require them.
>

> **LoC delta (declared).** +1700 net (`safety/`). Within cap 2000.
> **Transport role.** Input guardrails pre-dispatch; output guardrails post-response — hooks on transport path, not inside adapters.
> **Registry.** §4 pipeline + safety subpackages.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Outcome.** PII tokenization is reversible: the provider never sees raw PII (default rule detector covers email / SSN / phone / credit-card / API-key / IP). The placeholder reverses on the response — clients see originals. Injection detection runs as a pipeline gate (heuristic by default; ONNX classifier opt-in for higher recall). Output validation enforces JSON Schema with auto-repair before retry, saving the round-trip cost. Every guardrail is per-tenant configurable and surfaces in receipts ([Phase 28](28-receipts.md)) and OTel spans ([Phase 22](19-otel-genai.md)).
>
> **Estimated effort.** 4 days (1 PR, ~+1700 LoC — smaller than prior because the heavy detectors are deferred to opt-in adapters).

---

## 1. Why this phase exists, and what changed from the prior draft

### 1.1 The audit findings still stand

Every other gateway ships PII / injection / output guardrails out of the box. LATTICE ships none. For healthcare / finance / legal / government users this is a hard blocker.

### 1.2 The brutal change vs the prior draft

The prior plan named **Presidio as the default PII detector**. Presidio plus spaCy plus `en_core_web_lg` is ~ 1 GB. That blew the lightweight budget on day one. It also implied that the "default" guardrail experience required a multi-minute install on a laptop.

**New default:** the rule-based regex PII detector covers ~ 80% of real-world PII (email, SSN, phone, credit card, API key, IP) with zero deps and zero install footprint. Presidio is now a clearly-labeled `[pii]` extra for users who need higher recall — and the error message when they call Presidio without installing the extra tells them exactly what to do.

Same change for injection detection: heuristic phrase matcher is default, ONNX classifier is opt-in.

The novel architectural angle survives: **reversible PII via our existing alias-table machinery.** The provider never sees raw PII even with the lightweight rule detector. No other gateway can do this because no other gateway has a reversible compression pipeline.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/safety/__init__.py
src/lattice/safety/policy.py                       # composite per-tenant GuardrailPolicy
src/lattice/safety/violations.py                   # GuardrailViolation dataclass

src/lattice/safety/pii/__init__.py
src/lattice/safety/pii/detector_base.py            # PIIDetector protocol
src/lattice/safety/pii/rule_detector.py            # DEFAULT — zero deps
src/lattice/safety/pii/presidio_detector.py        # OPT-IN — clear ImportError when extra missing
src/lattice/safety/pii/tokenizer.py                # reversible <pii_kind_N> substitution
src/lattice/safety/pii/policy.py

src/lattice/safety/injection/__init__.py
src/lattice/safety/injection/detector_base.py
src/lattice/safety/injection/heuristic_detector.py # DEFAULT — zero deps
src/lattice/safety/injection/onnx_detector.py      # OPT-IN
src/lattice/safety/injection/policy.py

src/lattice/safety/output/__init__.py
src/lattice/safety/output/validator.py             # JSON Schema enforcement
src/lattice/safety/output/repair.py                # structural JSON repair (no LLM)

src/lattice/pipeline/guardrail_gates.py            # integrates with Pipeline.compress() gate stack

tests/unit/safety/pii/test_rule_detector.py
tests/unit/safety/pii/test_tokenizer_roundtrip.py
tests/unit/safety/pii/test_policy_modes.py
tests/unit/safety/injection/test_heuristic.py
tests/unit/safety/output/test_validator.py
tests/unit/safety/output/test_repair.py
tests/integration/safety/test_e2e_pii_reversal.py
tests/integration/safety/test_e2e_injection_block.py
tests/integration/safety/test_e2e_output_repair.py
tests/contract/test_default_install_no_guardrail_downloads.py
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | Insert guardrail gate (between policy and runtime budget); insert reverse gate post-response |
| [src/lattice/pipeline/gates.py](../../src/lattice/pipeline/gates.py) | Hook for guardrail check; no implementation inline (delegation only) |
| [src/lattice/core/config.py](../../src/lattice/core/config.py) | Add `GuardrailConfig` with conservative defaults |
| [src/lattice/proxy/middleware.py](../../src/lattice/proxy/middleware.py) | New headers: `x-lattice-guardrail-pii`, `x-lattice-guardrail-injection`, `x-lattice-guardrail-output` |
| [pyproject.toml](../../pyproject.toml) | Optional groups: `pii = ["presidio-analyzer>=2.2", "presidio-anonymizer>=2.2", "spacy>=3.7"]`, `injection = ["onnxruntime>=1.18"]` — both clearly **opt-in** |

### 2.3 Deleted

None.

---

## 3. Step-by-step

### 3.1 PII rule detector (default — zero deps)

```python
# src/lattice/safety/pii/rule_detector.py
class RulePIIDetector:
    """Zero-dep regex PII detector. Default backend.

    Coverage:
      - email (RFC-relaxed)
      - SSN (US-format dashed)
      - phone (E.164 + US-format)
      - credit card (Luhn-validated)
      - generic API key patterns (sk-…, pk-…, api_…, ghp_… for GitHub, etc.)
      - IPv4 + IPv6

    For higher recall on names, locations, dates of birth, organizations,
    install the `[pii]` extra to use the Presidio backend.
    """
    name = "rule"

    _RULES: Mapping[PIIKind, re.Pattern] = {
        PIIKind.EMAIL: re.compile(r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b"),
        PIIKind.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        PIIKind.PHONE: re.compile(r"\+?\d{1,3}?[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"),
        PIIKind.CREDIT_CARD: re.compile(r"\b(?:\d[ \-]*?){13,19}\b"),
        PIIKind.API_KEY: re.compile(r"\b(?:sk|pk|api|ghp|gho|ghu|ghr|ghs|gha)[-_][A-Za-z0-9]{20,}\b"),
        PIIKind.IPV4: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        PIIKind.IPV6: re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"),
    }

    def detect(self, text: str) -> Sequence[PIISpan]:
        if not text:
            return ()
        spans: list[PIISpan] = []
        for kind, pat in self._RULES.items():
            for m in pat.finditer(text):
                if kind == PIIKind.CREDIT_CARD and not _luhn_valid(m.group()):
                    continue
                spans.append(PIISpan(start=m.start(), end=m.end(), kind=kind, confidence=1.0, text=m.group()))
        return _resolve_overlaps(spans)
```

This is ~120 LoC of pure Python. Zero deps. Fast (< 1 ms on a 4 KB message).

### 3.2 PII detector — opt-in Presidio

```python
# src/lattice/safety/pii/presidio_detector.py
class PresidioPIIDetector:
    """Higher-recall PII detector via Microsoft Presidio.

    Adds ~1 GB to install (presidio-analyzer + spaCy + en_core_web_lg).
    Use only when:
      - Default rule detector misses critical PII kinds (names, locations, DOB, organizations)
      - Regulatory requirements mandate NER-grade detection

    Install: pip install "lattice-transport[pii]"
    """
    name = "presidio"

    def __init__(self, *, languages=("en",), score_threshold=0.5):
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise ImportError(
                "Presidio PII detector requires `pip install lattice-transport[pii]` "
                "(adds ~1 GB to install). For lightweight default coverage of email/SSN/phone/"
                "credit-card/API-key/IP, use detector='rule' instead."
            ) from exc
        self._engine = AnalyzerEngine(default_score_threshold=score_threshold,
                                       supported_languages=list(languages))
```

The error message guides users back to the lightweight path. Same pattern for every opt-in heavy detector.

### 3.3 Reversible tokenization

```python
# src/lattice/safety/pii/tokenizer.py
def tokenize(text: str, spans: Sequence[PIISpan]) -> tuple[str, PIIAliasTable]:
    """Replace every PII span with a unique <pii_kind_N> placeholder.

    Uses the SAME alias-table machinery as transforms/reference_sub.py.
    Reverse pass on the response side restores originals.
    """
    if not spans:
        return text, PIIAliasTable(())
    sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
    counters: dict[PIIKind, int] = defaultdict(int)
    entries: list[PIIAliasEntry] = []
    out = list(text)
    for span in sorted_spans:
        counters[span.kind] += 1
        placeholder = f"<pii_{span.kind.value.lower()}_{counters[span.kind]}>"
        out[span.start:span.end] = placeholder
        entries.append(PIIAliasEntry(placeholder=placeholder, kind=span.kind,
                                     original=span.text, span=(span.start, span.end)))
    return "".join(out), PIIAliasTable(tuple(entries))


def merge_with_existing(table: PIIAliasTable, existing: AliasTable) -> AliasTable:
    """Combine PII placeholders with reference_sub placeholders so the reverse pass walks both."""
    ...
```

This is the novel structural angle. The proxy holds the PII alias table in-memory for the request's duration; the response reverse pass restores originals; **the table is never persisted** (not in receipts, not in logs, not in OTel attributes). Receipts only record the **kinds** that were tokenized — auditable without leaking the data.

### 3.4 Policy modes

```python
# src/lattice/safety/pii/policy.py
class PIIMode(StrEnum):
    BLOCK = "block"          # PII present → request rejected
    MASK = "mask"            # PII replaced with [REDACTED_KIND]; irreversible
    TOKENIZE = "tokenize"    # default if PII detection enabled — reversible via alias table
    WARN = "warn"            # passthrough; metric only


class PIIPolicy:
    def apply(self, text: str, spans: Sequence[PIISpan], mode: PIIMode) -> PIIResult:
        match mode:
            case PIIMode.BLOCK:
                if spans:
                    raise GuardrailViolation(kind="pii", detail=f"{len(spans)} PII spans detected")
                return PIIResult.passthrough(text)
            case PIIMode.MASK:
                return PIIResult(text=mask_spans(text, spans), alias_table=None)
            case PIIMode.TOKENIZE:
                tokenized, table = tokenize(text, spans)
                return PIIResult(text=tokenized, alias_table=table)
            case PIIMode.WARN:
                metrics.increment("guardrail.pii.warn", tags={"kinds": [s.kind.name for s in spans]})
                return PIIResult.passthrough(text)
```

### 3.5 Injection — heuristic detector (default — zero deps)

```python
# src/lattice/safety/injection/heuristic_detector.py
_KNOWN_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore the above",
    "disregard your previous",
    "you are now",                       # role-takeover
    "system prompt:",
    "</s>",                              # special-token leak
    "<|im_start|>system",                # ChatML hijack
    "do anything now",
    "developer mode",
    "jailbreak",
    "[INST]", "[/INST]",                 # Llama-format hijack
)

class HeuristicInjectionDetector:
    """Zero-dep injection detector. Default backend.

    Signals:
      - Known injection phrases (case-insensitive)
      - Suspiciously long single-line user content (>4k chars, no newlines)
      - Special-token leak attempts (model control sequences in user text)
      - System-role redefinition attempts

    For higher recall (paraphrase attacks, novel patterns), opt into the ONNX
    classifier with `pip install lattice-transport[injection]`.
    """
    name = "heuristic"

    def detect(self, messages: Sequence[Message]) -> InjectionResult:
        score = 0.0
        signals: list[InjectionSignal] = []
        for msg in messages:
            if msg.role not in ("user", "tool"):
                continue
            content = (msg.content or "").lower()
            for phrase in _KNOWN_INJECTION_PHRASES:
                if phrase in content:
                    score += 0.4
                    signals.append(InjectionSignal(kind="phrase", detail=phrase, message_index=msg.index))
            if len(content) > 4000 and "\n" not in content:
                score += 0.1
                signals.append(InjectionSignal(kind="long_unbroken", message_index=msg.index))
            if _looks_like_special_token(content):
                score += 0.3
                signals.append(InjectionSignal(kind="special_token", message_index=msg.index))
        return InjectionResult(score=min(1.0, score), signals=tuple(signals))
```

~ 60 LoC of pure Python. Catches the common cases without an ML model. False-positive rate is acceptable for the default `warn` mode; users who want lower FP rate can switch to `block` mode + ONNX classifier.

### 3.6 Injection — opt-in ONNX classifier

```python
# src/lattice/safety/injection/onnx_detector.py
class ONNXInjectionDetector:
    """Higher-recall injection detector via 5 MB DeBERTa-v3 ONNX classifier.

    Adds ~90 MB to install (onnxruntime). Model is downloaded lazily on first
    use to assets/models/injection-classifier.onnx (~5 MB).

    Install: pip install "lattice-transport[injection]"
    """
    name = "onnx"

    def __init__(self, model_path: Path | None = None):
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "ONNX injection classifier requires `pip install lattice-transport[injection]` "
                "(adds ~90 MB to install). For lightweight default coverage of common injection "
                "phrases and role-takeover patterns, use detector='heuristic' instead."
            ) from exc
        ...
```

Composition: heuristic runs first (zero cost); if its score ≥ 0.6, short-circuit to block. Otherwise (when both enabled) the ONNX classifier runs for ambiguous cases. Median cost stays at < 1 ms; ONNX inference fires on ~ 5% of traffic.

### 3.7 Output validator (always available)

```python
# src/lattice/safety/output/validator.py
class OutputValidator:
    """Validates model output against a user-supplied JSON Schema; attempts auto-repair on malformed JSON.

    Triggered when:
      - response_format is "json_object" or "json_schema"
      - or x-lattice-output-validate header is set
    """

    def validate(self, response: Response, *, schema: dict | None) -> OutputValidationResult:
        text = response.choices[0].message.content
        if not text:
            return OutputValidationResult.passthrough()
        try:
            parsed = orjson.loads(text)
        except orjson.JSONDecodeError as exc:
            repaired = self._repair(text)
            if repaired is None:
                return OutputValidationResult(valid=False, repaired=None, error=str(exc))
            parsed = orjson.loads(repaired)
            response = response.with_text(repaired)
            metrics.increment("guardrail.output.repaired")
        if schema:
            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(parsed))
            if errors:
                return OutputValidationResult(valid=False, repaired=response, error=_format_errors(errors))
        return OutputValidationResult(valid=True, repaired=response)
```

`safety/output/repair.py` implements structural repair patterns (trailing comma, code-fence stripping, missing closing bracket recovery, comment removal, unquoted keys, schema-coerced field types, truncation recovery). All pure Python, no model calls. Each pattern is a `(repaired, applied) = pattern(text)` pure function; the repair pipeline applies until JSON parses or no pattern made progress.

We deliberately don't call the LLM for repair — that's the explicit "inference-aware retry with reduced context" path in [Phase 34](26-agent-memory.md).

### 3.8 Integration with `Pipeline.compress()`

Insert guardrail gate at the right point in the gate stack:

```python
# pipeline/runner.py — inside compress()
# Gate 0.5: input-side guardrails (between policy gate and runtime budget)
guardrail_result = self._guardrails.check_input(request, ctx)
if guardrail_result.blocked:
    return Err(GuardrailError(guardrail_result))
if guardrail_result.modified_request is not None:
    request = guardrail_result.modified_request
    ctx.guardrail_alias_table = guardrail_result.alias_table
```

Reverse pipeline gains a step:

```python
# pipeline/reverse.py
def reverse(self, response, plan, ctx):
    response = self._run_response_transforms(response, plan, ctx)
    if ctx.guardrail_alias_table:
        response = ctx.guardrail_alias_table.reverse_response(response)
    out = self._guardrails.check_output(response, ctx)
    if not out.valid:
        if ctx.config.guardrails.output.action == "retry":
            return Err(NeedsRetryError(out))
        ctx.headers["x-lattice-guardrail-output"] = "invalid"
    return response
```

`pipeline/guardrail_gates.py` is the new module orchestrating PII / injection / output sub-components.

### 3.9 Per-tenant policy

```python
# src/lattice/safety/policy.py
@dataclass(frozen=True, slots=True)
class GuardrailPolicy:
    pii: PIIConfig
    injection: InjectionConfig
    output: OutputConfig

    @classmethod
    def from_headers(cls, request: Request, default: "GuardrailPolicy") -> "GuardrailPolicy":
        """Per-request overrides via headers:
          x-lattice-guardrail-pii: block | mask | tokenize | warn | off
          x-lattice-guardrail-injection: block | warn | off
          x-lattice-guardrail-output: enforce | repair | warn | off
        """
        ...
```

Per-tenant policy via [Phase 32 self-hosted auth](32-cloud-multitenant.md) when enabled; otherwise single global policy from config.

### 3.10 Lightweight default config

```python
# src/lattice/core/config.py
class PIIConfig(BaseModel):
    enabled: bool = False                      # OFF by default — users opt in
    detector: Literal["rule", "presidio", "off"] = "rule"
    kinds: tuple[PIIKind, ...] = (PIIKind.EMAIL, PIIKind.SSN, PIIKind.PHONE, PIIKind.CREDIT_CARD, PIIKind.API_KEY, PIIKind.IPV4)
    mode: Literal["block", "mask", "tokenize", "warn"] = "tokenize"
    score_threshold: float = 0.5


class InjectionConfig(BaseModel):
    enabled: bool = True                       # ON by default — heuristic detector is free
    detector: Literal["heuristic", "onnx", "off"] = "heuristic"
    mode: Literal["block", "warn", "off"] = "warn"     # warn by default (low FP cost)
    score_threshold: float = 0.6


class OutputConfig(BaseModel):
    enabled: bool = True                       # ON by default — only fires when response_format set
    action: Literal["repair", "enforce", "warn"] = "repair"


class GuardrailConfig(BaseModel):
    pii: PIIConfig = PIIConfig()
    injection: InjectionConfig = InjectionConfig()
    output: OutputConfig = OutputConfig()
    log_violations: bool = True
```

Defaults explained:

- **PII detection off by default.** Adding PII tokenization changes upstream request bodies; users should explicitly opt in. When opted in, the rule detector (zero deps) covers common cases.
- **Heuristic injection on by default in `warn` mode.** Free; only emits metrics and headers. Users who want blocking flip to `mode: block`.
- **Output validator on by default in `repair` mode.** Only fires when the user requested a structured response; auto-repairs malformed JSON before retry; saves cost.

---

## 4. Observability

Each guardrail emits:

- Response headers (`x-lattice-guardrail-{pii,injection,output}`)
- Metrics (counters / histograms by kind / mode / action)
- OTel attribute on the parent span ([Phase 22](19-otel-genai.md)): `lattice.guardrail.violations[]`, `lattice.guardrail.modifications[]`. **Only the kinds — never the raw values.**
- Compression receipts ([Phase 28](28-receipts.md)) for compliance audit — kinds only.

---

## 5. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/safety -q` | All pass |
| Round-trip | `tests/unit/safety/pii/test_tokenizer_roundtrip.py` | Property: tokenize+reverse on every kind preserves text |
| E2E PII reversal | `tests/integration/safety/test_e2e_pii_reversal.py` | Provider HTTP fixture sees only placeholders; client sees originals |
| E2E injection block | `tests/integration/safety/test_e2e_injection_block.py` | Known injection prompt with mode=block → 400 with GuardrailViolation body |
| Output repair | `tests/unit/safety/output/test_repair.py` | 20 malformed JSON fixtures → ≥ 18 repaired |
| Lean install | `pip install lattice-transport` (no extras) | Rule detector + heuristic + repair work; opt-in detectors give clear ImportError |
| No download contract | `tests/contract/test_default_install_no_guardrail_downloads.py` | Booting proxy with defaults never writes to assets/models/ |
| Latency | heuristic injection median | < 1 ms |
| Latency | rule PII p99 on 4 KB text | < 5 ms |
| Footprint | `tests/integration/footprint/test_4gb_laptop.py` | Default install fits well under budget |
| Canonical bench | usual | ±2% |

### 5.1 Key property test

```python
@hypothesis.given(text=st.text(), pii=st.lists(pii_strategy(), max_size=10))
def test_tokenize_reverse_identity(text, pii):
    """For any text with any PII spans, tokenize + reverse == original."""
    annotated = inject_pii(text, pii)
    spans = detect_test_spans(annotated, pii)
    tokenized, table = tokenize(annotated, spans)
    for entry in table.entries:
        assert entry.original not in tokenized
    restored = table.reverse_substitute(tokenized)
    assert restored == annotated
```

---

## 6. Acceptance criteria

1. With `guardrails.pii.enabled = true, mode = "tokenize"` (rule detector), sending `"my email is alice@example.com"` to the proxy results in the upstream HTTP body containing `<pii_email_1>` and the client response containing the original `alice@example.com`. Verified by a `respx`-mocked end-to-end test inspecting captured upstream body.
2. Sending a known injection prompt with `guardrails.injection.mode = "block"` returns HTTP 400 with body `{"error": {"type": "guardrail_violation", "kind": "injection", "score": ...}}`.
3. Sending `response_format = "json_schema"` with a model that returns malformed JSON results in the client receiving a repaired, schema-valid JSON without a retry round-trip.
4. **Lean install** (`pip install lattice-transport` with no extras) starts cleanly. Rule PII + heuristic injection + JSON repair all work. Receipt headers indicate `detector=rule` and `detector=heuristic` so operators know they're not on the heavy detectors.
5. Calling `PresidioPIIDetector()` without `[pii]` extra raises a clear `ImportError` with install instructions and a pointer back to the rule detector.
6. Receipt and OTel span record violation kinds only — no raw PII values. Property test enforces.
7. Default install adds < 1 MB beyond Phase 16 baseline. Footprint test passes.
8. New guardrail gates add ≤ 3 ms median proxy overhead at default config.
9. Canonical bench ±2%.

---

## 7. Out of scope

| Topic | Phase |
|---|---|
| Bedrock Guardrails / Azure Content Safety adapters | Future (post-26). User provides API key; we route to their existing subscription. |
| LLM-as-judge hallucination detection | Future. |
| Encrypted receipt storage | [Phase 28](28-receipts.md) (receipts are already content-free). |
| Tenant-level policy in Postgres | [Phase 32](32-cloud-multitenant.md) self-hosted auth. |
