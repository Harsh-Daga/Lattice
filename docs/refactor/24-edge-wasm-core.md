# Phase 24 — Shared Core: Rust crate + PyO3 wheel + WASM bindings

> **This is the architectural keystone of v2.0.** It exists to enforce two constraints in the [FORWARD_PLAN](FORWARD_PLAN.md):
> - Every algorithm has exactly one implementation. SDKs and runtimes consume it; they never reimplement.
> - The TypeScript / edge story is real without bloating the SDK.
>
> **Footprint impact.** Native Python wheel `lattice-core-py` adds ~3 MB. WASM bundle `@lattice/core-wasm` is ≤ 200 KB gzipped. Both are **optional**. The base Python install still works with pure-Python implementations of every primitive in the crate. The base npm install still works in "proxy-only mode" (no WASM, no algorithm code, no compression in the SDK).
>
> **Algorithm location.** This phase establishes the single source of truth in `crates/lattice-core/` for every primitive shared across surfaces. After this phase ships, the drift-prevention CI gate (`scripts/check_sdk_no_algorithm_duplication.sh`) starts enforcing that no SDK source file contains a reimplementation.
>
> **External-service requirement.** None.
>
> **Estimated effort.** 10 days (1 PR for the core + bindings; ~+6000 LoC Rust, ~+400 LoC integration glue across Python and TS).

> **LoC delta (declared).** +4500 net (`crates/` + bindings; not counted in `src/lattice/` cap).
> **Transport role.** Accelerates framing, fingerprint, streaming buffer used **on** the transport path; does not replace Phase 27 dispatcher.
> **Registry.** §2–§4 Rust mirrors + SDK FFI surfaces.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.

---

## 1. Why this phase is the keystone

The previous SDK plans (Python in [Phase 13](13-python-sdk-quality.md), TypeScript in [Phase 17](17-typescript-sdk.md)) drifted toward reimplementing the same primitives in each language. That's the disease this phase cures:

| Drift surface | Today's risk | After this phase |
|---|---|---|
| Reverse-pass alias substitution | Python in `sdk/wrappers.py`, TS in `sdk/streaming/reverse.ts` | One Rust impl, exposed via PyO3 + WASM. SDKs call it. |
| Streaming chunk buffer with sliding-window placeholder handling | Same duplication | Same fix |
| Canonical PromptIR builder + fingerprint | Python only today; would be re-coded for TS edge | One Rust impl, used everywhere |
| `reference_sub`, `path_prefix`, `format_conversion`, `output_cleanup` | Pure-Python today; would need TS port for edge | Rust impls become canonical; Python keeps pure-Python fallback for users who don't install the native wheel |
| Binary framing (LATT protocol) | Python in `protocol/framing.py` | Rust impl shared |
| xxhash3-128 fingerprinting | Python via `xxhash` package | Rust crate `xxhash-rust` (smaller, deterministic) |

**The hard rule established here:** algorithm code in any SDK source file is a CI failure. SDKs are URL redirectors (proxy mode) or shared-core consumers (in-process mode). Nothing else.

---

## 2. The two-mode SDK doctrine

Every SDK runs in one of two modes:

### Mode A: Proxy mode (default for everyone)

```
User code → LATTICE SDK (thin HTTP client) → LATTICE proxy (full pipeline) → Provider
```

- SDK does ONE thing: set `baseURL` to the proxy, pass requests through.
- Streaming reverse-pass happens **server-side** in the proxy (Phase 19's streaming-native compression already does this). Chunks arrive at the SDK already-decoded.
- SDK source contains zero algorithm code. Confirmed by CI.
- Works in any runtime. Cloudflare Workers, Vercel Edge, Deno, Bun, Node, browser, Python — all just need `fetch`.

### Mode B: In-process mode (advanced; only when there is no proxy)

```
User code → LATTICE SDK → shared core (PyO3 wheel or WASM module) → Provider
```

- Used by edge runtimes that can't reach a central proxy with acceptable latency.
- The SDK loads `@lattice/core-wasm` (TypeScript) or imports `lattice-core-py` (Python).
- SDK orchestrates: "build IR", "apply transforms", "send request", "decode chunks". Each call delegates to the shared core.
- **The SDK still contains zero algorithm code.** Orchestration is allowed; reimplementation is not.

If neither a proxy nor the shared core is available (e.g. someone runs `@lattice/sdk` on Cloudflare without installing `@lattice/core-wasm`), the SDK **forwards the request unchanged** and adds `x-lattice-passthrough: true` to the response. No silent quality regression, no silent reimplementation.

---

## 3. Workspace layout

```
crates/
  lattice-core/
    Cargo.toml
    src/
      lib.rs                                # public API surface
      ir/
        mod.rs
        canonical.rs                        # PromptIR canonical builder
        serialize.rs                        # canonical JSON (simd-json optional via feature)
        fingerprint.rs                      # xxh3-128
      transforms/
        mod.rs
        reference_sub.rs                    # UUID/URL/path detection + substitution
        path_prefix.rs                      # common-prefix detection
        format_conv.rs                      # markdown <-> JSON tables
        output_cleanup.rs                   # whitespace, code-fence
      framing/
        mod.rs                              # LATT binary framing
        delta.rs
      alias/
        mod.rs                              # AliasTable type
        reverse.rs                          # reverse-substitute (response side)
      streaming/
        mod.rs
        chunk_buffer.rs                     # sliding-window decoder/encoder
      util/
        mod.rs
        intern.rs                           # &'static str interning for hot strings
        rope.rs                             # rope structure for safe in-place edits
    tests/                                  # cargo test
      ir_canonical.rs
      reference_sub.rs
      path_prefix.rs
      format_conv.rs
      framing.rs
      streaming.rs
    benches/                                # criterion benches
      reference_sub.rs
      path_prefix.rs
      ir_canonical.rs

bindings/
  python/                                   # PyO3 wheel: lattice-core-py
    Cargo.toml
    pyproject.toml
    src/lib.rs
    python/lattice_core_py/__init__.py
    tests/test_parity_with_python.py        # parity tests vs the Python fallback
  wasm/                                     # @lattice/core-wasm
    Cargo.toml
    src/lib.rs
    package.json
    pkg/                                    # generated by wasm-pack (committed for diff visibility)
    tests/parity.test.ts

scripts/
  check_sdk_no_algorithm_duplication.sh     # the CI gate that protects the doctrine
```

The Cargo workspace also publishes `lattice-core` as a standalone Rust crate to crates.io. Third-party tooling (a Go SDK, a Java SDK, anyone who wants to write a new language binding) can FFI into the same `.so/.dylib/.dll`.

---

## 4. Crate design constraints

| Constraint | Reason |
|---|---|
| Core is `no_std`-compatible (uses `alloc` only) | Enables `wasm32-unknown-unknown` without `wasm-bindgen-rayon` or `wasi`. WASM bundle stays tiny. |
| No tokio, no async in the core | Runtime-agnostic. Async is the binding's problem, not the algorithm's. |
| All public API takes byte slices and writes to caller-provided buffers where possible | Zero-copy across the WASM/PyO3 boundary; avoids costly bytes->String conversions. |
| Public API is `#[non_exhaustive]` on structs | Semver discipline; we can add fields without breaking. |
| Deterministic byte-for-byte equivalence with Python fallback impl | Parity tests in both bindings. Cache keys must match across runtimes. |
| No `unsafe` outside `util/intern.rs` and `util/rope.rs` | Every `unsafe` block has a safety comment proving the invariant. |
| MSRV: stable Rust (no nightly features) | CI build everywhere. |
| WASM binary ≤ 200 KB gzipped after `wasm-opt -Oz` | Edge runtime cap. |
| PyO3 wheel ≤ 5 MB per platform | PyPI manageable. |
| Cargo workspace, not single crate | Independent dep trees for the three bindings. |

---

## 5. Public API (Rust)

```rust
// crates/lattice-core/src/lib.rs
pub mod ir {
    pub use canonical::{CanonicalIR, CanonicalInput, build_canonical};
    pub use fingerprint::{Fingerprint, fingerprint, provider_invariant_fingerprint};
    pub use serialize::{serialize_canonical, deserialize_canonical};
}
pub mod transforms {
    pub use reference_sub::{ReferenceSub, ReferenceSubResult};
    pub use path_prefix::{PathPrefix, PathPrefixResult};
    pub use format_conv::{FormatConv, FormatConvResult};
    pub use output_cleanup::{OutputCleanup, OutputCleanupResult};
}
pub mod alias {
    pub use AliasTable;
    pub use reverse_substitute;
}
pub mod streaming {
    pub use ChunkBuffer;
}
pub mod framing {
    pub use FramingEncoder, FramingDecoder;
}
```

Every function in this surface is the *single* implementation in the workspace. Python fallback (in `src/lattice/`) and PyO3/WASM bindings both call into these symbols.

---

## 6. PyO3 binding — drop-in acceleration for Python

The Python proxy ships with **pure-Python implementations** of every primitive (existing code, no change). The native wheel is purely additive: when present, the Python transform classes detect it and route to native.

```python
# src/lattice/transforms/reference_sub.py
try:
    from lattice_core_py import reference_sub as _native
    _USE_NATIVE = True
except ImportError:
    _native = None
    _USE_NATIVE = False


class ReferenceSubstitution:
    def optimize(self, ir, request, ctx):
        if _USE_NATIVE and not ctx.disable_native:
            return self._optimize_native(ir, request, ctx)
        return self._optimize_python(ir, request, ctx)

    def _optimize_native(self, ir, request, ctx):
        ir_bytes = serialize_canonical_python(ir)
        result_bytes, alias_obj = _native.apply(ir_bytes, alias_prefix=ctx.alias_prefix)
        return Ok(deserialize_canonical_python(result_bytes).with_alias_table(AliasTable.from_native(alias_obj)))

    def _optimize_python(self, ir, request, ctx):
        # existing pure-Python implementation, unchanged
        ...
```

**Parity test** (mandatory):

```python
# bindings/python/tests/test_parity_with_python.py
@hypothesis.given(text=st.text(), num_uuids=st.integers(min_value=0, max_value=20))
def test_reference_sub_parity(text, num_uuids):
    ir = build_test_ir(text, num_uuids)
    py_ir, py_alias = python_reference_sub(ir, alias_prefix="ref")
    rs_ir, rs_alias = native_reference_sub(ir, alias_prefix="ref")
    assert canonical_dump(py_ir) == canonical_dump(rs_ir)
    assert py_alias.to_dict() == rs_alias.to_dict()
```

Same property test for every primitive in the crate. CI runs them on every PR touching either the Rust crate or the Python fallback.

---

## 7. WASM binding — the TypeScript SDK's only algorithm source

```rust
// bindings/wasm/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct AliasTableHandle(lattice_core::alias::AliasTable);

#[wasm_bindgen]
pub struct ChunkBufferHandle(lattice_core::streaming::ChunkBuffer);

#[wasm_bindgen]
pub fn build_canonical_ir(input_json: &str) -> Result<JsValue, JsValue> {
    let input: CanonicalInput = serde_json::from_str(input_json).map_err(js_err)?;
    let ir = lattice_core::ir::build_canonical(&input).map_err(js_err)?;
    serde_wasm_bindgen::to_value(&ir).map_err(js_err)
}

#[wasm_bindgen]
pub fn apply_reference_sub(ir_json: &str, alias_prefix: &str) -> Result<JsValue, JsValue> {
    // ... single call into lattice_core::transforms::reference_sub
}

#[wasm_bindgen]
pub fn alias_table_new() -> AliasTableHandle { ... }

#[wasm_bindgen]
pub fn alias_table_reverse(handle: &AliasTableHandle, text: &str) -> String { ... }

#[wasm_bindgen]
pub fn chunk_buffer_new(max_tail_bytes: usize, alias_handle: &AliasTableHandle) -> ChunkBufferHandle { ... }

#[wasm_bindgen]
pub fn chunk_buffer_feed(handle: &mut ChunkBufferHandle, delta: &str) -> String { ... }

#[wasm_bindgen]
pub fn chunk_buffer_flush(handle: &mut ChunkBufferHandle) -> String { ... }
```

Build pipeline: `wasm-pack build --target web --release` → `wasm-opt -Oz` → ≤ 200 KB gzipped.

The TS SDK ([Phase 17](17-typescript-sdk.md)) consumes only this binding:

```typescript
// packages/typescript-sdk/src/streaming.ts (in-process mode)
import init, * as core from "@lattice/core-wasm";

let initialized: Promise<void> | null = null;
function ensureCore(): Promise<void> {
  return initialized ??= init();
}

export async function* streamWithReverse(
  upstream: AsyncIterable<ChunkPayload>,
  aliasHandle: AliasTableHandle,
): AsyncGenerator<ChunkPayload> {
  await ensureCore();
  const buf = core.chunk_buffer_new(64, aliasHandle);
  for await (const chunk of upstream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (typeof delta !== "string") { yield chunk; continue; }
    const decoded = core.chunk_buffer_feed(buf, delta);
    if (decoded) { chunk.choices[0].delta.content = decoded; yield chunk; }
  }
  const tail = core.chunk_buffer_flush(buf);
  if (tail) yield makeFinalTextChunk(tail);
}
```

No regex, no manual sliding-window logic, no per-language reimplementation. The TS SDK *orchestrates*; the shared core *implements*.

---

## 8. The drift-prevention CI gate

`scripts/check_sdk_no_algorithm_duplication.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Forbidden patterns in SDK source: algorithm implementations that must live in lattice-core only.
FORBIDDEN=(
  'function\s+reverseSubstitute'                # must come from WASM core
  'function\s+buildCanonicalIR'
  'function\s+applyReferenceSub'
  'function\s+applyPathPrefix'
  'class\s+ChunkBuffer'
  'function\s+xxh3'
  'function\s+canonicalSerialize'
  'function\s+computeFingerprint'
  'const\s+UUID_REGEX'                          # detector regexes are core's job
  'const\s+URL_REGEX'
)

SDK_DIRS=("packages/typescript-sdk/src" "bindings/wasm/pkg")

failed=0
for dir in "${SDK_DIRS[@]}"; do
  for pat in "${FORBIDDEN[@]}"; do
    if rg -n "$pat" "$dir" 2>/dev/null; then
      echo "FAIL: forbidden algorithm implementation in $dir matches pattern: $pat"
      failed=1
    fi
  done
done

# Same check on Python SDK: lattice/sdk/ must not contain transform implementations.
# (Acceptable: thin client + adapter glue.)
PY_SDK_DIR="src/lattice/sdk"
FORBIDDEN_PY=(
  'class\s+ReferenceSubstitution'               # belongs in src/lattice/transforms/
  'def\s+reverse_substitute'                    # belongs in core or src/lattice/transforms/alias
  'def\s+build_canonical'
  'def\s+xxh3'
)
for pat in "${FORBIDDEN_PY[@]}"; do
  if rg -n "$pat" "$PY_SDK_DIR" 2>/dev/null; then
    echo "FAIL: forbidden algorithm implementation in $PY_SDK_DIR matches pattern: $pat"
    failed=1
  fi
done

exit $failed
```

Wired into `.github/workflows/refactor-gate.yml`. Blocks merge on hit.

---

## 9. Step-by-step delivery

### Step 1 — Port `reference_sub` to Rust, build PyO3 binding, ship parity test

Pick the highest-leverage transform first. After step 1, the Python proxy with `lattice-core-py` installed shows ≥ 3× speedup on the existing reference-sub benchmark, with byte-identical output to the Python fallback.

### Step 2 — Port `path_prefix`, `format_conv`, `output_cleanup`

Three more transforms. Same pattern.

### Step 3 — Port canonical IR + fingerprint + alias table + reverse-substitution

These are the data structures shared across all bindings. Critical for the cache-key compatibility invariant (Phase 14).

### Step 4 — Port streaming `ChunkBuffer`

Required for both Phase 19 (server-side streaming compression) and Phase 17 (TS in-process mode).

### Step 5 — Port binary framing

Smaller scope. Used by transport and the experimental binary path.

### Step 6 — Build WASM binding, ship `@lattice/core-wasm` to npm

`wasm-pack build` + `wasm-opt -Oz`. Verify ≤ 200 KB gzipped. Publish.

### Step 7 — Wire Phase 17 TS SDK to consume `@lattice/core-wasm` for in-process mode

`packages/typescript-sdk/src/edge.ts` is the only file that imports WASM. The rest of the SDK is proxy-mode-only.

### Step 8 — Land the drift-prevention CI gate

Once Phase 17's TS SDK exists, turn on `check_sdk_no_algorithm_duplication.sh`. From this point, drift = build break.

### Step 9 — Publish to package registries

- `lattice-core` to crates.io
- `lattice-core-py` to PyPI (wheels for manylinux2014 x86_64 + aarch64, macOS x86_64 + aarch64, Windows x86_64; Python 3.11, 3.12, 3.13)
- `@lattice/core-wasm` to npm with provenance attestation

### Step 10 — Documentation

`docs/architecture/shared_core.md` explains the two-mode SDK doctrine for contributors. Cross-link from every SDK doc.

---

## 10. Test plan

| Check | Command | Threshold |
|---|---|---|
| Rust unit | `cargo test --workspace` | All pass on x86_64-linux, aarch64-darwin, wasm32 |
| Clippy | `cargo clippy --workspace -- -D warnings` | 0 warnings |
| Rust bench | `cargo bench --bench reference_sub` | ≥ 3× faster than Python fallback on the 32 KB / 50-UUID input |
| Python parity | `pytest bindings/python/tests/test_parity_with_python.py -q` | Property tests pass 500 random cases each |
| TS parity | `vitest run bindings/wasm/tests/parity.test.ts` | Same |
| WASM bundle size | `wc -c bindings/wasm/pkg/lattice_core_wasm_bg.wasm.gz` | ≤ 200_000 bytes |
| Drift gate | `bash scripts/check_sdk_no_algorithm_duplication.sh` | exit 0 |
| Lean install (no native) | `pip install lattice-transport && pytest tests/unit/transforms -q` | Pure-Python fallback passes; all tests green |
| Lean install (no WASM) | `npm install @lattice/sdk && pnpm test` | Proxy-mode SDK passes; in-process mode skipped |
| Memory leak | 1M-call stress test on native wheel | RSS growth ≤ 10 MB |
| Cold start | `tests/integration/edge/test_cold_start.spec.ts` (miniflare) | p99 ≤ 5 ms after first request; first request ≤ 5 ms cold |
| Footprint | `tests/integration/footprint/*` | Default install still under budget |

---

## 11. Acceptance criteria

1. `cargo test --workspace` passes on x86_64-linux, aarch64-darwin, wasm32-unknown-unknown.
2. `pip install lattice-core-py` installs successfully on all five published platforms.
3. With `lattice-core-py` installed, `bash scripts/run_canonical_benchmark.sh` shows ≥ 3× faster median compression latency vs the pure-Python baseline. Output is byte-identical.
4. **Without** `lattice-core-py`, the same benchmark runs and produces the same output (slower). Property tests confirm parity.
5. `bindings/wasm/pkg/lattice_core_wasm_bg.wasm.gz` is ≤ 200 KB.
6. `npm install @lattice/sdk` followed by `@lattice/core-wasm` works in a fresh Cloudflare Worker (miniflare test); 100 random fixture requests produce byte-identical compressed IR to the Python proxy's output.
7. `npm install @lattice/sdk` **without** `@lattice/core-wasm` works in proxy mode against a running proxy; SDK source contains zero algorithm implementations (CI gate enforces).
8. `bash scripts/check_sdk_no_algorithm_duplication.sh` exits 0 on the head commit.
9. Footprint tests (`test_4gb_laptop.py`, `test_2gb_vps.py`) still pass — the optional native deps are not loaded by default.

---

## 12. Out of scope

| Topic | Reason |
|---|---|
| Porting LLMLingua-2 (ONNX) to Rust | ONNX is already cross-platform; not worth a port. |
| Porting Presidio / spaCy to Rust | Same. |
| Porting the Pipeline runner / planner to Rust | The runner is mostly orchestration + Pydantic models; Rust port would dwarf the gains. |
| GPU-accelerated paths | Future. |
| Native semantic cache | Redis/pgvector clients are already fast enough; Rust cache impl would duplicate them. |
| Go / Java SDKs | The crate is on crates.io and exposes a C ABI via `cbindgen`. Third parties can FFI in. We don't ship maintained SDKs. |
