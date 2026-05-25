# Phase 19 — Compression Intelligence: Streaming, Tool-Diff, JSON Repair (lightweight) + LLMLingua-2 (opt-in heavy)

> **Footprint impact.** Three of the four features in this phase are **zero new deps**: streaming-native compression on response chunks, tool-result diffing across consecutive calls, and JSON output structure repair. The fourth — LLMLingua-2 — is a clearly-labeled `[llmlingua]` extra that adds ~600 MB (onnxruntime + ~500 MB ONNX model). It is opt-in. The default compression story stays with our deterministic transforms which already handle structural cases well; LLMLingua-2 fires only when the user opts in **and** the workload (RAG / summarization / long-context analysis) is worth the inference cost.
>
> **Algorithm location.** Streaming chunk buffer ports to [Phase 24](24-edge-wasm-core.md)'s shared core so both Python proxy and TS/WASM edge SDK use the same implementation. Tool-diff detector is a new transform in `src/lattice/transforms/`. JSON repair patterns live in `src/lattice/safety/output/repair.py` (added in [Phase 15](15-native-guardrails.md); extended here). LLMLingua-2 lives in `src/lattice/transforms/llmlingua/` as an isolated package, never imported unless enabled.
>
> **External-service requirement.** None. Optional: the user's own provider's cheap text model can be used as an LLM-judge for tuning if explicitly opted in (benchmark-only, never in hot path).
>

> **LoC delta (declared).** +900 net (streaming + tool_diff; LLMLingua opt-in isolated).
> **Transport role.** Streaming reverse-pass on the **response transport stream** (SSE); integrates with Phase 27 stream resumption.
> **Registry.** §4 streaming + tool_diff + optional llmlingua.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Outcome.** Streaming response chunks decode placeholders mid-stream (drops TTFT vs response-side reverse). Tool result diffing collapses repeated `read_file(path=X, offset=N)` patterns to a delta sentinel, saving 90%+ on the earlier result's tokens. JSON output structural repair fixes malformed JSON before counting as a retry, saving the round-trip cost. LLMLingua-2 (opt-in) plugs into the existing beam search for users who want 2-5× compression on RAG/summarization workloads.
>
> **Estimated effort.** 5 days (1 PR, ~+2400 LoC — smaller than prior because LLMLingua becomes a self-contained opt-in package rather than a "default ML feature").

---

## 1. Why this phase exists, and what changed from the prior draft

### 1.1 The four real user pains

| Pain | Today | After this phase |
|---|---|---|
| Long-output streaming: client waits until full response for `<ref_N>` reverse | Reverse pass is response-side only | Streaming-native compression applies mid-stream — **drops TTFT** |
| Agent re-reads same file with different offsets, re-sends prior result | Full re-emission every turn | Tool-result diffing collapses to a delta with a stable reference |
| LLM emits malformed JSON: forces a retry round-trip = wasted cost | Validator reports failure | Structural repair attempted in-process; retry only when repair fails |
| RAG / summarization could see 2-5× compression with LLMLingua | Conservative rate_distortion only | LLMLingua-2 fires when (a) opted in and (b) wins the beam search |

### 1.2 The brutal change vs the prior draft

The prior plan implied LLMLingua-2 as part of the "default compression story". 500 MB model download + onnxruntime + transformers install. That violates lightweight.

**New positioning:** LLMLingua-2 is a flagged opt-in extra. The default compression story is three things that are genuinely lightweight:

1. **Streaming-native compression on response chunks** — pure-Python; uses the same alias-table machinery the runtime already has.
2. **Tool-result diffing** — pure-Python heuristic on JSON args. Zero deps. High win on agent workloads.
3. **JSON structural repair** — pure-Python pattern matching (extends Phase 15's repair). Zero deps.

These three together cover ~ 80% of the practical compression-intelligence win for typical workloads. LLMLingua adds another ~30-50% on the specific RAG/summarization slice — and that slice is what users opt into the heavy extra for.

---

## 2. Files touched

### 2.1 Created

```
# Lightweight (default)
src/lattice/pipeline/streaming/__init__.py
src/lattice/pipeline/streaming/chunk_buffer.py        # delegates to Phase 24 native core when present
src/lattice/pipeline/streaming/reverse.py
src/lattice/pipeline/streaming/transforms.py          # streaming-capable forward transforms
src/lattice/transforms/tool_diff/__init__.py
src/lattice/transforms/tool_diff/detector.py
src/lattice/transforms/tool_diff/applier.py
src/lattice/safety/output/repair_v2.py                # extends Phase 15 repair with more patterns

# Opt-in heavy
src/lattice/transforms/llmlingua/__init__.py
src/lattice/transforms/llmlingua/transform.py
src/lattice/transforms/llmlingua/model.py             # lazy import — clear ImportError when extra missing
src/lattice/transforms/llmlingua/tokenizer.py
src/lattice/transforms/llmlingua/gate.py

tests/unit/transforms/tool_diff/test_detector.py
tests/unit/transforms/tool_diff/test_applier.py
tests/unit/pipeline/streaming/test_chunk_buffer.py
tests/unit/pipeline/streaming/test_reverse.py
tests/unit/safety/output/test_repair_v2.py
tests/unit/transforms/llmlingua/test_transform.py
tests/unit/transforms/llmlingua/test_gate.py
tests/integration/test_streaming_compression.py
tests/integration/test_tool_diff_e2e.py
tests/integration/test_llmlingua_end_to_end.py        # gated by [llmlingua] extra
tests/contract/test_default_install_no_llmlingua.py   # base install never loads onnxruntime

benchmarks/suites/public/longbench.py                 # LLMLingua-enabled comparison
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/transforms/registry.py](../../src/lattice/transforms/registry.py) | Register `tool_diff` (SAFE, default on); register `llmlingua` (CONDITIONAL, only when extra installed) |
| [src/lattice/planner/unified_planner.py](../../src/lattice/planner/unified_planner.py) | `tool_diff` in all default tier allowlists; `llmlingua` in `LONG_CONTEXT` / `RAG` allowlists when available |
| [src/lattice/pipeline/representation_optimizer.py](../../src/lattice/pipeline/representation_optimizer.py) | Beam search candidates compare LLMLingua vs `rate_distortion` for the same budget |
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | Stream-aware paths invoke streaming compression; reverse path threaded through chunk buffer |
| [src/lattice/safety/output/__init__.py](../../src/lattice/safety/output/__init__.py) | Wire `repair_v2` into the validator |
| [pyproject.toml](../../pyproject.toml) | New optional: `llmlingua = ["onnxruntime>=1.18", "transformers>=4.45"]` — **clearly heavy** |

### 2.3 Deleted

None.

---

## 3. Lightweight default — three features

### 3.1 Streaming-native compression

The current reverse pass is response-side only. For long outputs (multi-page summaries, code generation), clients wait hundreds of ms before the first byte appears. This phase moves the reverse pass into the proxy's SSE streaming path:

```python
# src/lattice/pipeline/streaming/chunk_buffer.py
@dataclass
class ChunkBuffer:
    """Sliding-window encoder/decoder for SSE chunks.

    Delegates to native Rust core (Phase 24) when lattice-core-py is installed —
    byte-identical behaviour. Pure-Python fallback otherwise.

    Holds a tail of bytes equal to the maximum placeholder length so that
    placeholders split across chunk boundaries are correctly resolved before
    a chunk is forwarded downstream.
    """
    max_tail_bytes: int
    alias_table: AliasTable | None
    structural_transforms: tuple[StreamingTransform, ...]
    _buffered: str = ""

    @classmethod
    def for_request(cls, ctx: TransformContext) -> "ChunkBuffer":
        # Use native core when available
        try:
            from lattice_core_py.streaming import ChunkBuffer as NativeBuffer
            return NativeBuffer.for_request_native(ctx)
        except ImportError:
            return cls(
                max_tail_bytes=ctx.alias_table.max_placeholder_length() if ctx.alias_table else 0,
                alias_table=ctx.alias_table,
                structural_transforms=ctx.streaming_structural_transforms,
            )

    def feed(self, delta: str) -> str:
        """Add a new text delta; return the substring that's safe to emit."""
        self._buffered += delta
        for tx in self.structural_transforms:
            self._buffered = tx.apply_incremental(self._buffered)
        cut = max(0, len(self._buffered) - self.max_tail_bytes)
        emit, self._buffered = self._buffered[:cut], self._buffered[cut:]
        if not emit:
            return ""
        if self.alias_table:
            emit = self.alias_table.reverse_substitute(emit)
        return emit

    def flush(self) -> str:
        tail = self._buffered
        self._buffered = ""
        if self.alias_table:
            tail = self.alias_table.reverse_substitute(tail)
        return tail
```

**Streaming-capable forward transforms** in `pipeline/streaming/transforms.py`:

| Transform | Streaming-capable? | How |
|---|---|---|
| `tool_filter` | Yes | Detects tool-call JSON boundaries; strips fields as they appear |
| `reference_sub` | Reverse only | Forward pass requires whole-message view |
| `output_cleanup` | Yes | Whitespace normalization, code-fence stripping per line |
| `format_conversion` | No | Whole-message dependency |
| `path_prefix` | No | Whole-message scan |

Only the streaming-capable ones plug into the buffer. Everything else stays response-side.

**Hooking into the proxy stream** in [src/lattice/gateway/compat/openai_chat.py](../../src/lattice/gateway/compat/openai_chat.py) (post-Phase-12 split):

```python
async def stream_chat(upstream: AsyncIterator[bytes], ctx: TransformContext) -> AsyncIterator[bytes]:
    buf = ChunkBuffer.for_request(ctx)
    async for raw in upstream:
        sse_event = parse_sse_event(raw)
        if sse_event.is_text_delta():
            decoded = buf.feed(sse_event.text)
            if decoded:
                yield sse_event.with_text(decoded).serialize()
        elif sse_event.is_done():
            tail = buf.flush()
            if tail:
                yield make_text_only_event(tail).serialize()
            yield raw
        else:
            yield raw
```

TTFT delta: ≤ 2 ms with no placeholders in flight, ≤ 10 ms otherwise. Validated by `tests/integration/test_streaming_compression.py::test_ttft_unchanged_without_placeholders`.

### 3.2 Tool result diffing (zero deps, pure-Python heuristic)

```python
# src/lattice/transforms/tool_diff/detector.py
@dataclass(frozen=True, slots=True)
class ToolSiblingRelation:
    """Two tool calls related by:
      - same tool name
      - args differ only in one numeric or pagination key
    """
    earlier_call_index: int
    later_call_index: int
    differing_keys: tuple[str, ...]


_PAGINATION_KEYS = re.compile(r"(?i)offset|page|cursor|after|before|since|until|skip|limit|range|line_start|line_end")


def detect_sibling_calls(messages: Sequence[Message]) -> list[ToolSiblingRelation]:
    """O(n) detector; no ML."""
    tool_calls_by_name = group_tool_calls(messages)
    siblings = []
    for name, calls in tool_calls_by_name.items():
        for earlier, later in pairwise(calls):
            differing = compare_args(earlier.args, later.args)
            if len(differing) == 1 and _PAGINATION_KEYS.search(differing[0]):
                siblings.append(ToolSiblingRelation(
                    earlier_call_index=earlier.message_index,
                    later_call_index=later.message_index,
                    differing_keys=tuple(differing),
                ))
    return siblings
```

**Applier** in `applier.py` replaces the earlier tool result with a stable sentinel:

```
[tool_result earlier]
<lattice:diff_base id="tool_result_3" tool="filesystem.read_file" args={"path": "/x", "offset": 0}>
<contents_elided_for_brevity: see continuation in tool_result_4 with offset=4096>
</lattice:diff_base>
```

90%+ token savings on the earlier result with zero quality loss (the agent has the up-to-date later result). The receipt machinery ([Phase 23](23-receipts-bandit-profiles.md)) records the diff so reconstruction is possible.

Tool-diff is `SAFE`, enabled by default for agent traffic (detected via [src/lattice/integrations/agent_stats.py](../../src/lattice/integrations/agent_stats.py)). No-op when no sibling relations exist.

### 3.3 JSON structural repair (extends Phase 15)

`src/lattice/safety/output/repair_v2.py` extends [Phase 15](15-native-guardrails.md)'s repair with:

1. **Code fence stripping** — `\`\`\`json\n{...}\n\`\`\`` → `{...}`
2. **Trailing comma removal** — `{"a": 1,}` → `{"a": 1}`
3. **Single-quote → double-quote** at structural positions only
4. **Missing closing brackets** — balanced via token-count append
5. **Comment removal** — `//` and `/* */`
6. **Unquoted keys** — `{a: 1}` → `{"a": 1}` when JSON parse fails on key position
7. **Truncation recovery** — `finish_reason: length` → close open structures + surface `repair.truncated: true` flag for [Phase 21](21-agent-memory.md)'s continuation logic
8. **Schema-coerced field types** — `"42"` → `42` when schema demands integer

Each pattern is a pure function `(repaired, applied: bool)`. Repair runs patterns until JSON parses or no pattern made progress. **Never calls the LLM** — that's the explicit retry path in [Phase 21](21-agent-memory.md).

Metrics: `guardrail.output.repair_success`, `guardrail.output.repair_attempts`, `guardrail.output.repair_unrecoverable`.

---

## 4. Opt-in heavy — LLMLingua-2

### 4.1 Positioning

```
pip install "lattice-transport[llmlingua]"
```

The first time the LLMLingua transform is loaded the proxy prints a clear notice:

```
[lattice] LLMLingua-2 transform enabled.
  - Downloads ~500 MB ONNX model to assets/models/llmlingua-2.onnx on first request.
  - Adds ~200 MB CPU RAM at idle, ~500 MB under load.
  - Designed for RAG / summarization / long-context analysis workloads only.
  - Will not fire on reasoning, code generation, structured output, tool-calling.
  - Skip download with LATTICE_LLMLINGUA_MODEL_PATH=/your/model.onnx
  - Disable with `transforms.llmlingua.enabled = false`.
```

Users know exactly what they're opting into.

### 4.2 IR-native transform

```python
# src/lattice/transforms/llmlingua/transform.py
class LLMLinguaTransform:
    name = "llmlingua"
    priority = 22                         # same band as rate_distortion
    safety_class = SafetyClass.CONDITIONAL

    def __init__(self, model: LLMLinguaModel, target_ratio: float = 0.4):
        self._model = model
        self._target_ratio = target_ratio
        self._gate = LLMLinguaGate()

    def can_process(self, request: Request, ctx: TransformContext) -> bool:
        return self._gate.is_allowed(request, ctx)

    def optimize(self, ir, request, ctx) -> Result[PromptIRV2, TransformError]:
        try:
            new_sections = []
            for section in ir.sections:
                if not section.is_compressible_with_llmlingua():
                    new_sections.append(section); continue
                compressed = self._compress_section_text(section)
                if compressed is None:
                    new_sections.append(section); continue
                new_sections.append(section.with_text(compressed).with_metadata(
                    llmlingua_ratio=len(compressed)/max(1, len(section.text))))
            return Ok(ir.with_sections(tuple(new_sections)))
        except Exception as exc:
            return Err(TransformError(name=self.name, message=str(exc), recoverable=True))

    def reverse(self, response, ctx) -> Response:
        return response                   # one-way lossy; no reverse
```

### 4.3 Lazy model loading

```python
# src/lattice/transforms/llmlingua/model.py
class LLMLinguaModel:
    MODEL_URL = "https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank/resolve/main/model.onnx"
    MODEL_SHA256 = "..."

    def __init__(self, model_path: Path | None = None, providers=("CPUExecutionProvider",)):
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "LLMLingua-2 transform requires `pip install lattice-transport[llmlingua]` "
                "(adds ~600 MB to install: onnxruntime + transformers + 500 MB model on first use). "
                "For lightweight default compression of structural content, "
                "leave the transform disabled and rely on reference_sub + rate_distortion + tool_filter."
            ) from exc
        path = model_path or _default_path()
        if not path.exists():
            _print_download_notice(path, size_mb=500)
            _download_with_checksum(self.MODEL_URL, self.MODEL_SHA256, path)
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() // 2)
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = onnxruntime.InferenceSession(str(path), opts, providers=list(providers))
        self._tokenizer = _LLMLinguaTokenizer()
```

The model is loaded once per process. If the user wants to avoid the download (e.g. in CI), they can pre-fetch and set `LATTICE_LLMLINGUA_MODEL_PATH`.

### 4.4 Gate

```python
# src/lattice/transforms/llmlingua/gate.py
class LLMLinguaGate:
    """Decides when LLMLinguaTransform may fire. Conservative."""

    _ALLOWED_TASKS = frozenset({TaskClass.SUMMARIZATION, TaskClass.RETRIEVAL, TaskClass.ANALYSIS})
    _MIN_INPUT_TOKENS = 1000              # not worth firing on short prompts (model inference cost)
    _MAX_INPUT_TOKENS = 32000             # avoid huge inference cost on very long prompts

    def is_allowed(self, request, ctx) -> bool:
        task = ctx.task_classification.task_class
        if task not in self._ALLOWED_TASKS: return False
        if request.has_tool_calls() or request.has_response_format(): return False
        tokens = ctx.token_count_estimate
        if tokens < self._MIN_INPUT_TOKENS or tokens > self._MAX_INPUT_TOKENS: return False
        if request.headers.get("x-lattice-llmlingua") == "disabled": return False
        return True
```

### 4.5 Beam-search selection

The beam search in [src/lattice/pipeline/representation_optimizer.py](../../src/lattice/pipeline/representation_optimizer.py) already enumerates candidates and picks by `composite_score`. When both `llmlingua` and `rate_distortion` are eligible, the beam expands both branches and picks the higher-scoring candidate. The pipeline never runs both — they target overlapping content.

Header override: `x-lattice-llmlingua: required` forces LLMLingua over rate_distortion (benchmarks / experiments).

---

## 5. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | `uv run pytest tests/unit/transforms/tool_diff tests/unit/pipeline/streaming tests/unit/safety/output -q` | All pass |
| LLMLingua unit (only when extra installed) | `uv run pytest tests/unit/transforms/llmlingua -q` | All pass |
| Default-install-no-onnxruntime contract | `tests/contract/test_default_install_no_llmlingua.py` | Booting proxy with defaults never imports onnxruntime, never touches assets/models/llmlingua-2.onnx |
| Streaming TTFT | `tests/integration/test_streaming_compression.py::test_ttft_unchanged_without_placeholders` | TTFT delta ≤ 2 ms vs no-LATTICE baseline |
| Streaming reverse correctness | property test over random chunk boundaries | Streamed concat == full reverse |
| Streaming Rust parity | with `lattice-core-py` installed | Native chunk buffer byte-identical to Python |
| Tool diff E2E | `tests/integration/test_tool_diff_e2e.py` | Sibling reads → 90%+ saving on earlier result tokens |
| JSON repair coverage | `tests/unit/safety/output/test_repair_v2.py` (40 fixtures) | ≥ 35/40 repaired without LLM round-trip |
| LLMLingua E2E (when extra) | `tests/integration/test_llmlingua_end_to_end.py` | LongBench summarization within 5% of paper |
| Footprint default | `tests/integration/footprint/test_4gb_laptop.py` | ≤ Phase 14 footprint (LLMLingua not loaded) |
| Footprint with LLMLingua | manual | adds ~600 MB; documented |
| Canonical bench | usual | ±2% |

### 5.1 LongBench targets (opt-in `[llmlingua]`)

| Subset | LLMLingua-2 paper (2k constraint, 5×) | Our target |
|---|---|---|
| Single-doc QA | 39.1 | ≥ 38 |
| Multi-doc QA | 33.4 | ≥ 32 |
| Summarization | 25.3 | ≥ 24 |
| FewShot | 66.4 | ≥ 64 |
| Code | 58.9 | ≥ 57 |

---

## 6. Acceptance criteria

1. **Default install:** `pip install lattice-transport` ships streaming-native compression + tool-diff + JSON repair. Footprint test passes. No mention of LLMLingua in startup logs.
2. **Default install:** Streaming TTFT delta ≤ 2 ms when no reference placeholders in flight; property test passes 1000 random chunk-boundary cases; tool_diff E2E shows 90%+ savings on sibling tool results; JSON repair fixes ≥ 35/40 malformed fixtures.
3. **With `[llmlingua]` extra:** A clear console notice prints once explaining the model download and footprint cost; LongBench summarization subset within 5% of paper's reported numbers.
4. **Without `[llmlingua]` extra:** importing `lattice.transforms.llmlingua` raises a clear `ImportError` pointing back to lightweight defaults.
5. Contract test confirms default install never imports onnxruntime nor downloads a model.
6. With `lattice-core-py` installed, streaming `ChunkBuffer` is delegated to native Rust core with byte-identical output.
7. Canonical bench ±2%.

---

## 7. Out of scope

| Topic | Phase / future |
|---|---|
| Per-tenant fine-tuned LLMLingua model | **Cut from plan** — required cloud + GPU + training infra. Bandit (Phase 23) covers the value with zero infra. |
| Streaming compression on the request side | Future — most request bodies are small enough to compress whole. |
| LLM-judge in-line validation | Existing post-transform-guard (Phase 12) handles structural validation; LLM-judge is benchmark-only. |
| Tool-diff across multiple agent turns (current scope: within a single request's tool_results) | Future — requires session-store integration. |
