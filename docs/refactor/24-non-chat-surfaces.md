# Phase 24 — Non-Chat Surfaces: Embeddings, Batch, Audio, Files, Realtime

> **Footprint impact.** Zero new runtime deps. Audio file handling uses Python stdlib (`wave`, `audioop` if needed); the proxy forwards audio bytes to the user's provider unchanged. Realtime WebSocket uses FastAPI's built-in `WebSocket`. The per-text embedding cache reuses [Phase 20](17-hybrid-semantic-cache.md)'s storage backends.
>
> **Algorithm location.** New gateway endpoints in `src/lattice/gateway/{embeddings,batch,audio,files,realtime}.py`. Embedding-dedup logic lives in `src/lattice/transforms/embedding_dedup.py` — a regular transform, registered in the existing transform registry, never duplicated in any SDK. The embeddings dispatcher is the same one [Phase 20](17-hybrid-semantic-cache.md)'s `UserProviderEmbeddingBackend` reuses — single source of truth.
>
> **External-service requirement.** None. The user's already-configured provider is the only external service. We never spin up our own Whisper / TTS / embedding model — passthrough only. Single-provider constraint upheld: every surface in this phase routes to the user's chosen provider, never selects between providers.
>

> **LoC delta (declared).** +4200 net (`gateway/` endpoints). Within cap 3200 — may require cap bump with justification.
> **Transport role.** New HTTP surfaces (embeddings, batch, audio, realtime) all dispatch through unified `TransportDispatcher` (Phase 20).
> **Registry.** §12 gateway endpoints + embedding_dedup transform.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Goal.** LATTICE handles every modality a production LLM stack actually uses — not just `/v1/chat/completions`. Embeddings get per-text dedup + a per-tenant text-to-embedding cache (huge RAG ingestion savings). Batch API integrations target the real OpenAI Batch endpoint (50% discount) and the real Anthropic Message Batch endpoint, replacing the in-process coalescer renamed in [Phase 16](13-honesty-pass.md). Audio (transcriptions / translations / TTS) and Realtime WebSocket are first-class. Files API supports the chosen provider's native file storage. All single-provider per the constraint — no routing.
>
> **Outcome.** Five new `/v1/*` endpoints fully implemented (`embeddings`, `batches`, `audio/transcriptions`, `audio/speech`, `files`) plus `/v1/realtime` WebSocket. Embeddings get per-input dedup (a 100-input batch with 60% duplicates costs 40% of the original $); per-tenant embedding cache stores `(model, input_hash) → vector` so identical text across requests is free **and** powers Phase 20's optional cache embedding tier when enabled. Batch dispatch uses the real provider APIs and surfaces status/results through LATTICE-typed responses. Realtime sessions stream through TACC with binary framing. Files survive a session and are referenced by IDs. SDK wrappers cover the new surfaces ([Phase 19](16-python-sdk-quality.md), [Phase 24](20-typescript-sdk.md)) — thin clients only; no algorithm code.
>
> **Estimated effort.** 7 days (1 PR, ~+4200 LoC).

---

## 1. Why this phase exists

The audit's non-chat coverage matrix had only one green cell — chat. Yet:

| Surface | Why it matters | Today |
|---|---|---|
| Embeddings | RAG / search / classification = bulk of token spend at many shops | No endpoint |
| Batch API | 50% discount (OpenAI) / 50% (Anthropic) for non-interactive workloads | Renamed scaffolding ([Phase 16](13-honesty-pass.md)); no real dispatch |
| Audio (Whisper / TTS) | Voice agents, transcription pipelines | No endpoint |
| Realtime | OpenAI Realtime API for voice agents | No endpoint |
| Files | Required for batch, fine-tuning, multimodal | No endpoint |

LiteLLM has all five. Portkey has four. Bifrost has three. We have one. Closing this gap turns LATTICE from "great if you only do chat" into "comprehensive". Compression on embeddings inputs and batch payloads stacks **on top of** the provider discount — the math is dramatic for ingestion workloads.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/gateway/embeddings.py
src/lattice/gateway/batch.py
src/lattice/gateway/audio.py
src/lattice/gateway/files.py
src/lattice/gateway/realtime.py
src/lattice/providers/embeddings_dispatch.py
src/lattice/providers/batch_dispatch.py
src/lattice/providers/audio_dispatch.py
src/lattice/providers/files_dispatch.py
src/lattice/providers/realtime_dispatch.py
src/lattice/pipeline/embeddings_pipeline.py            # dedup + cache + per-text-vector recombine
src/lattice/pipeline/batch_pipeline.py                 # compress every entry before submit
src/lattice/state/batch_jobs.py                        # tracking submitted batches
src/lattice/state/files.py                             # mapping file_id <-> provider file_id + metadata
src/lattice/cache/embeddings_cache.py                  # per-tenant text-to-vector cache
src/lattice/sdk/_resources/embeddings.py               # SDK resource
src/lattice/sdk/_resources/batches.py
src/lattice/sdk/_resources/audio.py
src/lattice/sdk/_resources/files.py
src/lattice/sdk/_resources/realtime.py                 # WebSocket client
tests/unit/gateway/test_embeddings.py
tests/unit/gateway/test_batch.py
tests/unit/gateway/test_audio.py
tests/unit/gateway/test_files.py
tests/unit/pipeline/test_embeddings_pipeline.py
tests/unit/cache/test_embeddings_cache.py
tests/integration/gateway/test_embeddings_dedup_e2e.py
tests/integration/gateway/test_batch_e2e.py            # requires fixture or LIVE
tests/integration/gateway/test_realtime_ws.py
benchmarks/suites/cost/embedding_dedup.py
benchmarks/suites/cost/batch_discount.py
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/proxy/server.py](../../src/lattice/proxy/server.py) | Register five new routers |
| [src/lattice/gateway/compat/__init__.py](../../src/lattice/gateway/compat/__init__.py) (post-Phase-12) | Add cross-format translators where applicable (e.g. /v1/embeddings → Cohere format when tenant pinned to Cohere) |
| [src/lattice/providers/adapters/openai.py](../../src/lattice/providers/adapters/openai.py) | Add `embeddings`, `batches`, `audio_transcriptions`, `audio_speech`, `files`, `realtime` methods |
| [src/lattice/providers/adapters/anthropic.py](../../src/lattice/providers/adapters/anthropic/__init__.py) (post-Phase-12 split) | Add `batches` (Message Batches) and `files` methods |
| [openapi/lattice-proxy.yaml](../../openapi/lattice-proxy.yaml) | Add the five new endpoints (covered by Phase 24 generator) |
| [pyproject.toml](../../pyproject.toml) | Add `websockets>=12` to base; `audio = ["soundfile>=0.12"]` optional for content-type sniffing |

### 2.3 Deleted

None (the fake `BatchAccumulator` was already removed in Phase 16).

---

## 3. Embeddings

### 3.1 Request shape

`POST /v1/embeddings` — OpenAI-compatible:

```json
{
  "model": "openai/text-embedding-3-small",
  "input": ["text 1", "text 2", "text 1"],
  "encoding_format": "float",
  "dimensions": 512
}
```

Response shape matches OpenAI verbatim.

### 3.2 Compression pipeline

`src/lattice/pipeline/embeddings_pipeline.py`:

```python
class EmbeddingsPipeline:
    """Pre-processes an embeddings request for cost minimization.

    Steps:
      1. Normalize inputs (strip + NFC unicode)
      2. Deduplicate identical inputs; track original index -> dedup index
      3. Look up each dedup input in the per-tenant text-to-vector cache
      4. Dispatch only cache-misses to provider in a single batched call
      5. Reassemble final vector list at original ordering
    """

    def prepare(self, req: EmbeddingsRequest, ctx: EmbeddingsContext) -> EmbeddingsPlan:
        normalized = [normalize(x) for x in req.input]
        unique_inputs, index_map = dedup_preserving_order(normalized)

        cached_vectors: dict[int, np.ndarray] = {}
        misses: list[tuple[int, str]] = []
        for i, text in enumerate(unique_inputs):
            cached = self._cache.lookup(model=req.model, text=text, namespace=ctx.tenant)
            if cached is not None:
                cached_vectors[i] = cached
            else:
                misses.append((i, text))

        return EmbeddingsPlan(
            original=req,
            unique_inputs=unique_inputs,
            index_map=index_map,
            cached_vectors=cached_vectors,
            misses=misses,
        )

    def finalize(self, plan: EmbeddingsPlan, miss_vectors: list[np.ndarray]) -> EmbeddingsResponse:
        # Store miss vectors in cache
        for (i, text), vec in zip(plan.misses, miss_vectors, strict=True):
            self._cache.store(model=plan.original.model, text=text, vector=vec, namespace=...)
            plan.cached_vectors[i] = vec
        # Reassemble in original order
        ordered = [plan.cached_vectors[plan.index_map[orig_i]] for orig_i in range(len(plan.original.input))]
        return EmbeddingsResponse(
            object="list",
            model=plan.original.model,
            data=[EmbeddingData(index=i, embedding=v.tolist(), object="embedding") for i, v in enumerate(ordered)],
            usage=EmbeddingsUsage(
                prompt_tokens=sum(approximate_tokens(t) for _, t in plan.misses),
                total_tokens=sum(approximate_tokens(t) for _, t in plan.misses),
            ),
        )
```

`src/lattice/cache/embeddings_cache.py` is a thin specialization of the [Phase 20](17-hybrid-semantic-cache.md) layered store using `(tenant, model, sha256(text))` keys and the same pluggable backends (memory / Redis / Postgres). TTL defaults to 30 days — embeddings are deterministic and don't expire functionally.

### 3.3 Realistic savings

`benchmarks/suites/cost/embedding_dedup.py` measures on:

- 10k synthetic RAG chunks with 30% duplicate rate
- 10k real Wikipedia paragraphs (low duplicate)
- 10k production trace (replay corpus from Phase 10)

Expected savings:

| Workload | Dedup savings | Cache hit savings (steady state) |
|---|---|---|
| RAG chunks (30% dup) | 30% | 70%+ after first ingestion |
| Wikipedia | 2% | 5-10% |
| Production replay | 12-25% | 40-60% |

---

## 4. Batch API

### 4.1 Surfaces

`POST /v1/batches` — OpenAI Batch API compatible:

```json
{
  "input_file_id": "file_abc",
  "endpoint": "/v1/chat/completions",
  "completion_window": "24h",
  "metadata": {"customer_id": "..."}
}
```

`GET /v1/batches/{id}`, `GET /v1/batches` (list), `POST /v1/batches/{id}/cancel`, retrieve results via `output_file_id`.

For Anthropic: `POST /v1/messages/batches` and equivalents.

### 4.2 Pre-submit compression

`src/lattice/pipeline/batch_pipeline.py`:

```python
class BatchPipeline:
    """Compress each line of a JSONL batch file before submission.

    Reads the staged file via state.files; runs each entry through Pipeline.compress();
    writes a new compressed file; returns the new file ID for submission.

    Maintains a (compressed_file_id -> alias_table_per_line) mapping so result lines
    can be reverse-pass restored when the batch completes.
    """
    async def precompress(self, input_file_id: str, *, ctx) -> str:
        async with self._files.open(input_file_id, mode="r") as fin, \
                    self._files.create(suffix=".compressed.jsonl") as fout:
            line_aliases: list[AliasTable | None] = []
            async for line in fin:
                entry = orjson.loads(line)
                compressed = await self._pipeline.compress_async(entry["body"], ctx)
                entry["body"] = compressed.compressed_request.model_dump()
                line_aliases.append(compressed.alias_table)
                await fout.write(orjson.dumps(entry) + b"\n")
            new_id = fout.id
        self._aliases.put(new_id, line_aliases)
        return new_id
```

Submission flow:

1. User uploads file via `POST /v1/files` (multipart).
2. `POST /v1/batches` with `input_file_id`.
3. LATTICE pre-compresses, creates a compressed-file alongside, submits **compressed** file to provider.
4. State stored in `state.batch_jobs`: `{lattice_batch_id, provider_batch_id, compressed_file_id, original_file_id, line_aliases_ref}`.
5. Poll worker (or webhook) tracks completion.
6. On completion, results file downloaded; each line's response reverse-passed using stored alias table; resulting file exposed via `output_file_id`.

The user sees the **decompressed** results — they never know the compression happened, except for the smaller `usage` numbers (and 50% discount applied on top).

### 4.3 Critical invariants

- Per-line alias tables are essential — different lines have different placeholders.
- Provider batch IDs are not exposed to users; only LATTICE batch IDs.
- Batch cancellation propagates: `cancel` calls upstream cancel and marks `state.batch_jobs.status = cancelled`.
- Webhooks ([Phase 32](32-cloud-multitenant.md)) emit on batch completion.

---

## 5. Audio

`POST /v1/audio/transcriptions` (Whisper):

- Accepts multipart form-data (file + model + language + response_format)
- Dispatches to `providers/adapters/openai.py::audio_transcriptions`
- No compression applied — audio binary is opaque
- Caching: per-tenant `(model, sha256(file_bytes), language)` → response. Realistic hit rate on dev/test workflows is high.

`POST /v1/audio/speech` (TTS):

- Accepts `{model, input, voice, response_format, speed}`
- Per-tenant `(model, input_hash, voice, format, speed)` → audio bytes cache
- Returns audio bytes with correct `content-type`

Streaming TTS (chunk-based) supported with `transfer-encoding: chunked`.

---

## 6. Realtime WebSocket

`/v1/realtime` (WebSocket; OpenAI Realtime API compatible):

- Bidirectional voice + tool calling
- LATTICE acts as a TACC-managed shim:
  - Incoming user audio frames forwarded to provider
  - Outgoing model audio frames forwarded to client
  - Tool call events compressed via `tool_filter` / `tool_projection` (text-only events)
  - Session config events normalized
- Per-session compression context retained for the duration of the WS
- Frame metrics emitted as OTel events ([Phase 22](19-otel-genai.md))

Concurrency: each WS session gets a dedicated TACC slot; back-pressure flows through congestion windowing.

---

## 7. Files

`POST /v1/files` (multipart): upload a file, returns `file_id`.
`GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `DELETE /v1/files/{id}`.

`src/lattice/state/files.py`:

```python
@dataclass
class StoredFile:
    lattice_file_id: str            # ulid
    provider: str
    provider_file_id: str | None    # set after upstream sync
    bytes: int
    purpose: Literal["batch", "fine-tune", "assistants", "vision", "user_data"]
    filename: str
    created_at: float
    tenant: str
```

Files are mirrored to the upstream provider's file store; the upstream `provider_file_id` is what's used in `batches.input_file_id` after substitution.

Local-only storage option for tenants who pin to a single provider that supports file IDs (OpenAI). For Anthropic, files are inlined in subsequent calls.

---

## 8. SDK exposure

Python ([Phase 19](16-python-sdk-quality.md) follow-up):

```python
async with LatticeProxyClient() as c:
    emb = await c.embeddings.create(model="openai/text-embedding-3-small", input=["x", "y"])
    file = await c.files.upload(open("input.jsonl", "rb"), purpose="batch")
    batch = await c.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
    while (b := await c.batches.retrieve(batch.id)).status not in ("completed", "failed", "cancelled"):
        await asyncio.sleep(30)
    output = await c.files.download(b.output_file_id)
```

TypeScript ([Phase 24](20-typescript-sdk.md) follow-up):

```typescript
const file = await client.files.upload({ file: blob, purpose: "batch" });
const batch = await client.batches.create({ inputFileId: file.id, endpoint: "/v1/chat/completions", completionWindow: "24h" });
```

Wrappers (`wrap_openai`, `wrapOpenAI`) auto-route `client.embeddings.create`, `client.batches.create`, `client.audio.*`, `client.files.*` through LATTICE.

---

## 9. Test plan

| Check | Command | Threshold |
|---|---|---|
| Unit | new gateway + pipeline + cache tests | All pass |
| Dedup property | random inputs with controlled dup rate | Provider sees only unique non-cached entries |
| Batch reverse-pass | E2E with fixture provider | Each output line reverse-passed with the right alias table |
| Realtime | `tests/integration/gateway/test_realtime_ws.py` (fixture WS upstream) | Bidirectional flow; tool events compressed; session cleanup on close |
| Files | `tests/unit/gateway/test_files.py` | Upload + retrieve + download roundtrips |
| Lean install | base install | Embeddings + batch + files work; audio missing optional dep prints clear instruction |
| Bench: embedding dedup | `benchmarks/suites/cost/embedding_dedup.py` | RAG: ≥ 25% dedup; production replay: ≥ 12% |
| Canonical bench | usual | ±2% |

---

## 10. Acceptance criteria

1. `POST /v1/embeddings` with a 100-element input where 60 are duplicates results in the upstream provider seeing exactly 40 inputs (verified via fixture upstream counting requests).
2. Second identical embeddings request within TTL returns from cache with `x-lattice-cache-layer: embed`.
3. `POST /v1/batches` with a 1000-line JSONL file results in:
   - A compressed file (≤ original size) submitted to OpenAI
   - On completion, the user's downloaded `output_file_id` content has all reference placeholders reverse-passed
   - `state.batch_jobs` row created and updated through the lifecycle
4. `POST /v1/audio/transcriptions` returns a Whisper-compatible response from a fixture upstream.
5. `/v1/realtime` WebSocket round-trips audio + tool events with TACC pacing observed in OTel metrics.
6. `POST /v1/files` accepts multipart upload; subsequent batch references the file ID correctly.
7. All five SDK resources work in both Python and TypeScript.
8. Canonical bench ±2%.

---

## 11. Out of scope

| Topic | Phase |
|---|---|
| Cross-provider batch (run a single batch across two providers) | **Forever out** — multi-provider routing constraint |
| Fine-tuning API | Future — complex per-provider divergence |
| Assistants v2 / Threads | Future — superseded by Responses API |
| Image generation | Future |
