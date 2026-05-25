# Phase 20 — TypeScript SDK (`@lattice/sdk`) as a Thin Client

> **Footprint impact.** npm bundle ≤ 25 KB gzipped (edge entry); ≤ 60 KB gzipped (node entry). Zero runtime dependencies — `openai`, `@anthropic-ai/sdk`, `ai` are peer-deps only. The optional `@lattice/core-wasm` ([Phase 31](31-edge-wasm-core.md)) adds ≤ 200 KB gzipped when the user opts into in-process compression.
>
> **Algorithm location.** Zero algorithm code in the TypeScript SDK. Reverse-pass, IR fingerprinting, streaming chunk buffering, alias substitution — all live in `crates/lattice-core/`, exposed via `@lattice/core-wasm`. The SDK in `packages/typescript-sdk/src/` orchestrates calls into the WASM core (in-process mode) or just redirects HTTP to the proxy (proxy mode). `scripts/check_sdk_no_algorithm_duplication.sh` enforces this.
>
> **External-service requirement.** None for proxy mode (just point at a running proxy). None for in-process mode beyond `@lattice/core-wasm`. The user's provider is the only external service.
>

> **Transport role.** Default: HTTP to proxy transport. In-process: `@lattice/core-wasm` only — no TS transport stack.
> **Registry.** §12 TypeScript SDK.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Outcome.** A `packages/typescript-sdk/` workspace publishes `@lattice/sdk` to npm. Types are generated from a single OpenAPI spec (`openapi/lattice-proxy.yaml`) shared with the Python SDK — drift is impossible. `wrapOpenAI`, `wrapAnthropic`, `wrapVercelAI` are URL redirectors (proxy mode) or WASM-core orchestrators (in-process mode). Edge runtimes (Cloudflare Workers, Vercel Edge, Deno Deploy, Bun) are first-class.
>
> **Estimated effort.** 5 days (1 PR, ~+2200 LoC TypeScript — much smaller than the prior 5500 LoC plan because the SDK contains no algorithm code).

---

## 1. Why this phase exists, and what changed from the prior draft

### 1.1 The audit findings still stand

TypeScript-and-Next.js is the single largest unaddressed user market. Today the only TS path is `process.env.OPENAI_BASE_URL = ...` — works but no typed client, no edge story, no native Vercel AI integration.

### 1.2 The brutal change vs the prior draft

The prior plan was ~5500 LoC and included:

- TS-native `reverseAliasStream` (port of the Python `iter_decoded_chunks`)
- TS-native SSE parser for OpenAI and Anthropic streams
- TS-native `LatticeHooks` type system mirroring Python
- TS-native `AliasTable` with `reverseSubstitute`
- Vendored chunk-buffer with sliding-window placeholder handling

That's drift waiting to happen. Algorithm code in TypeScript that the Python team will edit, the Rust core will edit, and the three impls will silently diverge until a cache key stops matching.

This rewrite makes the TypeScript SDK contain:

- HTTP request construction (typed from OpenAPI)
- Configuration plumbing (baseUrl, headers, hooks-as-orchestration)
- Thin `wrapXxx` functions that **redirect base URLs**
- Optional in-process mode that **delegates every algorithm call to `@lattice/core-wasm`**

That's it. No SSE parser. No reverse-pass. No alias table. The runtime owns those.

---

## 2. Files touched

### 2.1 New workspace

```
packages/typescript-sdk/
  package.json                          # @lattice/sdk
  tsconfig.json
  tsconfig.build.json
  README.md
  LICENSE
  vitest.config.ts
  src/
    index.ts                            # public exports
    client.ts                           # LatticeClient: typed HTTP client
    types/
      openapi.d.ts                      # GENERATED from openapi/lattice-proxy.yaml
      hand.d.ts                         # hand-written supplements (Hook types)
    wrappers/
      openai.ts                         # wrapOpenAI: URL redirect + optional in-process delegate
      anthropic.ts                      # same for Anthropic
      vercel-ai.ts                      # wrapVercelAI
    in_process.ts                       # OPTIONAL: lazy-loads @lattice/core-wasm
    hooks/
      types.ts                          # orchestration hook type defs only
      builtin.ts                        # logToConsole, redactKeys, attachRequestId
    edge.ts                             # edge runtime entry (same exports; tagged build)
  tests/
    unit/
      wrap-openai.spec.ts               # asserts wrap = URL redirect (no logic)
      wrap-anthropic.spec.ts
      wrap-vercel-ai.spec.ts
      hooks-composition.spec.ts
      no-algorithm-duplication.spec.ts  # companion to scripts/check_sdk_no_algorithm_duplication.sh
    contract/
      api-surface.spec.ts
      readme-examples.spec.ts
      python-parity.spec.ts             # diff vs Python proxy on same fixture requests
    integration/
      proxy.spec.ts                     # spawns lattice proxy, runs requests
      edge-cloudflare.spec.ts           # miniflare; covers in-process mode with WASM
      edge-without-wasm.spec.ts         # confirms passthrough-mode fallback with warning
    fixtures/
      fake-openai.ts                    # SSE server fixture

openapi/
  lattice-proxy.yaml                    # single source of truth for HTTP surface
  generate.sh                           # runs openapi-typescript + datamodel-code-generator

scripts/
  generate_openapi_types.py             # Python TypedDict regeneration from openapi
  verify_openapi_parity.py              # CI check: openapi.yaml has no drift vs handlers
```

### 2.2 Modified

| File | Change |
|---|---|
| Top-level `package.json` (new) | npm workspace declaration including `packages/*` |
| `.github/workflows/typescript.yml` | Runs `pnpm --filter @lattice/sdk build`, `vitest run`, drift gate |
| `.github/workflows/openapi-parity.yml` | Verifies generated types are committed and current |
| [src/lattice/proxy/server.py](../../src/lattice/proxy/server.py) | Serves `openapi/lattice-proxy.yaml` at `/openapi.yaml` |
| [docs/getting-started/quickstart.md](../../docs/getting-started/quickstart.md) | Add TS quickstart section |
| [pyproject.toml](../../pyproject.toml) | Add `datamodel-code-generator` to dev deps |

---

## 3. The two operating modes (mirrors the doctrine from [Phase 31](31-edge-wasm-core.md))

### 3.1 Proxy mode (default, recommended)

```typescript
import OpenAI from "openai";
import { wrapOpenAI } from "@lattice/sdk";

const client = wrapOpenAI(new OpenAI(), { baseUrl: "http://localhost:8787/v1" });
const r = await client.chat.completions.create({ model: "openai/gpt-4o", messages: [...] });
```

The implementation:

```typescript
// packages/typescript-sdk/src/wrappers/openai.ts
export function wrapOpenAI<C extends { baseURL?: string }>(
  client: C,
  opts: WrapOpts = {},
): C {
  client.baseURL = opts.baseUrl ?? "http://localhost:8787/v1";
  if (opts.hooks) installObservabilityHooks(client, opts.hooks);
  return client;
}
```

Three lines. The proxy handles compression, reverse-pass, cache, guardrails, everything. Streaming chunks arrive at the SDK **already decoded** (proxy did the work in [Phase 27](22-compression-intelligence.md)). The SDK has nothing to do.

### 3.2 In-process mode (advanced, edge-only)

For users on Cloudflare Workers / Vercel Edge / Deno who explicitly opt out of a central proxy:

```typescript
import OpenAI from "openai";
import { wrapOpenAIInProcess } from "@lattice/sdk/in-process";

// Requires @lattice/core-wasm to be installed.
const client = await wrapOpenAIInProcess(new OpenAI({ apiKey: env.OPENAI_API_KEY }));
const r = await client.chat.completions.create({ ... });
```

Implementation:

```typescript
// packages/typescript-sdk/src/in_process.ts
let coreInit: Promise<typeof import("@lattice/core-wasm") | null> | null = null;

async function loadCore(): Promise<typeof import("@lattice/core-wasm") | null> {
  if (coreInit) return coreInit;
  coreInit = (async () => {
    try {
      const mod = await import("@lattice/core-wasm");
      await mod.default();   // initialize WASM module
      return mod;
    } catch (e) {
      console.warn(
        "[@lattice/sdk] @lattice/core-wasm not installed; in-process compression unavailable. " +
        "Either install it or use proxy mode (recommended).",
      );
      return null;
    }
  })();
  return coreInit;
}

export async function wrapOpenAIInProcess<C extends { chat: any }>(client: C): Promise<C> {
  const core = await loadCore();
  if (!core) {
    // Graceful degradation: forward requests unchanged. No silent re-implementation.
    return client;
  }
  const original = client.chat.completions.create.bind(client.chat.completions);
  client.chat.completions.create = async (body: any) => {
    const irJson = core.build_canonical_ir(JSON.stringify(body));
    const { ir: compressedIr, aliasHandle } = core.apply_pipeline(irJson);
    const newBody = core.ir_to_chat_request(compressedIr);

    if (body.stream) {
      const upstream = await original(newBody);
      return streamThroughCore(core, upstream, aliasHandle);
    }
    const response = await original(newBody);
    response.choices.forEach((c: any) => {
      if (c.message?.content) {
        c.message.content = core.alias_table_reverse(aliasHandle, c.message.content);
      }
    });
    return response;
  };
  return client;
}

async function* streamThroughCore(core: any, upstream: AsyncIterable<any>, aliasHandle: any) {
  const bufHandle = core.chunk_buffer_new(64, aliasHandle);
  for await (const chunk of upstream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (typeof delta === "string") {
      const decoded = core.chunk_buffer_feed(bufHandle, delta);
      if (decoded) {
        chunk.choices[0].delta.content = decoded;
        yield chunk;
      }
    } else {
      yield chunk;
    }
  }
  const tail = core.chunk_buffer_flush(bufHandle);
  if (tail) yield makeFinalTextChunk(tail);
}
```

Notice what's **not** in this file:

- No regex for placeholder detection
- No manual byte slicing for sliding window
- No reverse-substitute implementation
- No SSE parsing beyond what the upstream OpenAI SDK already does
- No alias table class

Every algorithm primitive is a call into the WASM core. The SDK is glue.

### 3.3 The graceful degradation path

If a user installs `@lattice/sdk` on Cloudflare but doesn't install `@lattice/core-wasm` and doesn't have a proxy to point at, `wrapOpenAIInProcess` returns the client **unchanged** with a one-time console warning. The request goes directly to the provider. No compression, no cache, no telemetry — but also no broken silent reimplementation.

This is critical: **the SDK never re-invents the algorithm to "be helpful"**. Either the runtime is reachable (proxy or WASM) or the SDK is a passthrough.

---

## 4. OpenAPI as the single source of truth

[Phase 19](16-python-sdk-quality.md) typed the Python SDK. To keep TS and Python types in lockstep, both generate from a single OpenAPI spec.

`openapi/lattice-proxy.yaml` covers every endpoint:

- `POST /v1/chat/completions` (request + streaming response)
- `POST /v1/embeddings` ([Phase 31](24-non-chat-surfaces.md))
- `POST /v1/messages` (Anthropic)
- `POST /v1/responses` + `GET/DELETE /v1/responses/{id}`
- `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`
- `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{id}/content`
- `POST /v1/batches`, `GET /v1/batches/{id}`
- `POST /lattice/session/{start,get,append,invalidate}`
- `GET /healthz`, `GET /readyz`, `GET /stats`, `GET /metrics`, `GET /openapi.yaml`
- `POST /lattice/cache/warm`, `POST /lattice/cache/analyze` ([Phase 17](27-cache-portability.md))
- `GET /lattice/receipts/{id}` ([Phase 28](28-receipts.md))

`scripts/verify_openapi_parity.py` introspects FastAPI's runtime-generated `/openapi.json` and diffs against the committed YAML. CI fails on drift. New endpoints must land in the YAML first or break the build.

`openapi/generate.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# TS types
npx openapi-typescript openapi/lattice-proxy.yaml \
  --output packages/typescript-sdk/src/types/openapi.d.ts --root-types

# Python TypedDicts (used by Phase 19 SDK)
uv run python scripts/generate_openapi_types.py \
  --input openapi/lattice-proxy.yaml \
  --output src/lattice/sdk/_types/openapi_generated.py
```

CI runs this and fails if either output differs from the committed copy.

---

## 5. `LatticeClient` — typed thin HTTP client

```typescript
// packages/typescript-sdk/src/client.ts
import type { paths } from "./types/openapi.d.ts";

type ChatBody = paths["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"];
type ChatResp = paths["/v1/chat/completions"]["post"]["responses"]["200"]["content"]["application/json"];

export class LatticeClient {
  readonly chat: ChatResource;
  readonly embeddings: EmbeddingsResource;
  readonly sessions: SessionsResource;

  constructor(private readonly opts: LatticeClientOptions = {}) {
    this.chat = new ChatResource(this);
    this.embeddings = new EmbeddingsResource(this);
    this.sessions = new SessionsResource(this);
  }

  private get baseUrl() { return this.opts.baseUrl ?? "http://localhost:8787"; }
  private get fetch() { return this.opts.fetch ?? globalThis.fetch; }

  async _post<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
    const res = await this.fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...this.authHeaders(), ...init?.headers },
      body: JSON.stringify(body),
      signal: init?.signal,
    });
    if (!res.ok) throw await LatticeError.fromResponse(res);
    return res.json() as Promise<T>;
  }

  async *_stream<T>(path: string, body: unknown, init?: RequestInit): AsyncGenerator<T> {
    const res = await this.fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream", ...this.authHeaders(), ...init?.headers },
      body: JSON.stringify({ ...body, stream: true }),
      signal: init?.signal,
    });
    if (!res.ok || !res.body) throw await LatticeError.fromResponse(res);
    // The PROXY has already done reverse-pass; we just parse SSE structurally.
    yield* parseStandardSSE<T>(res.body);
  }

  private authHeaders(): Record<string, string> {
    return this.opts.apiKey ? { authorization: `Bearer ${this.opts.apiKey}` } : {};
  }
}

class ChatResource {
  constructor(private readonly client: LatticeClient) {}

  create(body: ChatBody & { stream?: false }): Promise<ChatResp>;
  create(body: ChatBody & { stream: true }): AsyncGenerator<ChatCompletionChunk>;
  create(body: ChatBody) {
    if (body.stream) return this.client._stream("/v1/chat/completions", body);
    return this.client._post<ChatResp>("/v1/chat/completions", body);
  }
}
```

`parseStandardSSE` is a 30-line standard SSE parser (data-event splitting, JSON.parse per event). It is **not** a LATTICE-specific algorithm — it's the W3C SSE spec parsed structurally. It's allowed in the SDK because every fetch-based SSE consumer needs it. Confirmed by the CI gate: the forbidden pattern list does not include `parseSSE`.

---

## 6. `wrapVercelAI` — the differentiated win

Vercel AI SDK v5's `LanguageModelV2` interface:

```typescript
import { openai } from "@ai-sdk/openai";
import { wrapVercelAI } from "@lattice/sdk/vercel";
import { Agent, stepCountIs } from "ai";

const model = wrapVercelAI(openai("gpt-4o"));
const agent = new Agent({ model, tools: {...}, stopWhen: stepCountIs(20) });
```

Implementation: proxy-mode by default (intercept `doGenerate` / `doStream` and route through the LATTICE proxy by rewriting the model's `baseURL`); in-process mode delegates to WASM core. The agent loop benefits on every iteration because every tool-call and tool-result message passes through compression.

This is the integration no other gateway has. Vercel AI Gateway is bundled with Next.js but does no compression. We're the only one with both: TypeScript-first SDK + compression-aware AI SDK integration.

---

## 7. Edge entrypoint

`packages/typescript-sdk/src/edge.ts` re-exports the same surface as `index.ts`; the difference is the build target:

```json
{
  "exports": {
    ".": {
      "edge-light": "./dist/edge/index.js",
      "workerd": "./dist/edge/index.js",
      "browser": "./dist/edge/index.js",
      "default": "./dist/node/index.js"
    },
    "./in-process": "./dist/in_process.js",
    "./vercel": "./dist/wrappers/vercel-ai.js"
  }
}
```

The edge build:

- Tree-shakes any Node-only imports
- Uses Web Streams API (no `eventsource` lib)
- Lazy-imports `@lattice/core-wasm` only in `in-process.ts`

Result: the default edge bundle (proxy mode) is ~ 8 KB gzipped. With in-process WASM enabled, ~ 200 KB gzipped (mostly the WASM itself).

---

## 8. Hooks as orchestration only

`packages/typescript-sdk/src/hooks/types.ts`:

```typescript
export interface LatticeHooks {
  onRequest?: ((body: unknown) => unknown | void | Promise<unknown | void>)[];
  onResponse?: ((response: unknown) => void | Promise<void>)[];
  onError?: ((err: unknown) => void | Promise<void>)[];
  strict?: boolean;
}
```

Three callback lists. Same shape as Python ([Phase 19](16-python-sdk-quality.md) §4.6) — generated from the OpenAPI spec where possible. Pre-built helpers (`hooks/builtin.ts`): `logToConsole()`, `redactKeys(set)`, `attachRequestId()`, `emitOTel(tracer)` (when `@opentelemetry/api` is present).

In proxy mode, `onRequest` is observational — modifying the body before sending to the proxy is allowed but discouraged (the proxy is the source of truth). In in-process mode, `onRequest` runs before the WASM compression call and may modify the body.

---

## 9. Publishing

```json
{
  "name": "@lattice/sdk",
  "version": "1.0.0",
  "license": "MIT",
  "type": "module",
  "exports": { ... },
  "peerDependencies": {
    "openai": ">=4.55.0",
    "@anthropic-ai/sdk": ">=0.27.0",
    "ai": ">=5.0.0",
    "@lattice/core-wasm": ">=0.1.0"
  },
  "peerDependenciesMeta": {
    "openai": { "optional": true },
    "@anthropic-ai/sdk": { "optional": true },
    "ai": { "optional": true },
    "@lattice/core-wasm": { "optional": true }
  },
  "publishConfig": { "access": "public" }
}
```

Every peer dep is optional. The SDK works with none of them installed (proxy mode + LatticeClient + fetch).

Release flow: `pnpm --filter @lattice/sdk build && pnpm --filter @lattice/sdk publish`. No marketplace ceremony, no signed-package ritual — just `npm publish` with provenance.

---

## 10. README — five working paths

```typescript
// Path 1: drop-in proxy (zero install, just env var)
process.env.OPENAI_BASE_URL = "http://localhost:8787/v1";

// Path 2: wrap OpenAI sync (proxy mode; one-liner)
import OpenAI from "openai";
import { wrapOpenAI } from "@lattice/sdk";
const openai = wrapOpenAI(new OpenAI());

// Path 3: typed native LATTICE client
import { LatticeClient } from "@lattice/sdk";
const lattice = new LatticeClient({ baseUrl: "http://localhost:8787" });
const r = await lattice.chat.create({ model: "openai/gpt-4o", messages: [...] });

// Path 4: Anthropic wrap
import Anthropic from "@anthropic-ai/sdk";
import { wrapAnthropic } from "@lattice/sdk";
const anth = wrapAnthropic(new Anthropic());

// Path 5: Vercel AI SDK (differentiated; agent loops benefit on every step)
import { openai } from "@ai-sdk/openai";
import { wrapVercelAI } from "@lattice/sdk/vercel";
import { Agent, stepCountIs } from "ai";
const agent = new Agent({ model: wrapVercelAI(openai("gpt-4o")), tools, stopWhen: stepCountIs(20) });

// Path 6 (edge, advanced): in-process compression via WASM core
// npm install @lattice/core-wasm
import { wrapOpenAIInProcess } from "@lattice/sdk/in-process";
const client = await wrapOpenAIInProcess(new OpenAI({ apiKey: env.OPENAI_API_KEY }));
```

Each runs in `tests/contract/readme-examples.spec.ts`.

---

## 11. Test plan

| Check | Command | Threshold |
|---|---|---|
| TS lint | `pnpm --filter @lattice/sdk lint` | 0 errors |
| TS types | `pnpm --filter @lattice/sdk tsc --noEmit` | 0 errors strict |
| Unit | `pnpm --filter @lattice/sdk test` | All pass |
| Algorithm-duplication gate | `bash scripts/check_sdk_no_algorithm_duplication.sh` | exit 0 |
| Edge (with WASM) | `pnpm --filter @lattice/sdk test:edge` (miniflare) | In-process mode compresses; output byte-identical to Python proxy |
| Edge (without WASM) | `tests/integration/edge-without-wasm.spec.ts` | Passthrough mode; one-time warning; requests succeed |
| Browser | `pnpm --filter @lattice/sdk test:browser` (jsdom) | Proxy mode works |
| OpenAPI parity | `python scripts/verify_openapi_parity.py` | YAML matches FastAPI runtime spec |
| Type parity | `bash openapi/generate.sh && git diff --exit-code` | No drift |
| Bundle size | `pnpm --filter @lattice/sdk size` | edge entry ≤ 25 KB gzipped; node entry ≤ 60 KB gzipped |
| Python-vs-TS parity | `tests/contract/python-parity.spec.ts` | Same fixture requests produce byte-identical compressed bodies through Python proxy and TS in-process |
| Contract: readme | `tests/contract/readme-examples.spec.ts` | All six paths execute |
| Integration: proxy | `tests/integration/proxy.spec.ts` (spawns `lattice proxy run`) | All paths succeed |

---

## 12. Acceptance criteria

1. `npm install @lattice/sdk` in a fresh Node project; `wrapOpenAI(new OpenAI())` works against a local proxy.
2. `wrapVercelAI(openai("gpt-4o"))` works with `Agent.stream({...})` end-to-end; tool-calling agent runs through LATTICE compression with no SDK-side decoding.
3. Edge bundle imports cleanly into Cloudflare Workers (miniflare); proxy mode works.
4. Edge bundle with `@lattice/core-wasm` installed runs in-process compression with byte-identical output to Python proxy (parity test).
5. Edge bundle **without** `@lattice/core-wasm` and **without** a proxy works in passthrough mode with one console warning.
6. `bash scripts/check_sdk_no_algorithm_duplication.sh` exits 0 — TypeScript source contains zero algorithm implementations.
7. Generated `openapi.d.ts` matches the YAML; generated Python `_types/openapi_generated.py` matches the same source.
8. Bundle sizes: edge entry ≤ 25 KB gzipped; node entry ≤ 60 KB gzipped; WASM ≤ 200 KB gzipped.
9. `npm publish` with provenance succeeds.

---

## 13. Out of scope

| Topic | Phase / future |
|---|---|
| In-edge compression in pure TypeScript (no WASM) | Forever out — would require reimplementing algorithms in TS; violates the doctrine. Edge compression goes via WASM core or not at all. |
| Go / Rust SDKs | The crate is on crates.io; third parties can FFI. We don't ship maintained SDKs in those languages. |
| Browser-side OAuth / API key vault | Future. |
| TS port of the Pipeline runner | Forever out — the runner is server-side. |
