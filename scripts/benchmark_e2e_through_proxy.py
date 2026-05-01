#!/usr/bin/env python3
"""End-to-end benchmark through LATTICE proxy with real model calls."""
import json
import sys
import time

sys.path.insert(0, "src")

import httpx

from lattice.providers.transport import DirectHTTPProvider
from lattice.core.pipeline import CompressorPipeline
from lattice.core.transport import Message, Request, Response
from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import unwrap
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.tool_filter import ToolOutputFilter
from lattice.transforms.prefix_opt import PrefixOptimizer
from lattice.transforms.format_conv import FormatConverter
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.core.session import MemorySessionStore, SessionManager
from lattice.transforms.delta_encode import DeltaEncoder
from lattice.utils.token_count import TiktokenCounter

PROXY = "http://127.0.0.1:8787"
OLLAMA = "http://127.0.0.1:11434"
MODEL = "ollama/glm-5.1:cloud"


def count_tokens(text):
    try:
        return TiktokenCounter("gpt-4").count(text)
    except Exception:
        return len(text) // 4


def call_proxy(messages, compress=True):
    """Call proxy with x-lattice-disable-transforms header if needed."""
    headers = {"Content-Type": "application/json"}
    if not compress:
        headers["x-lattice-disable-transforms"] = "1"
    body = {"model": MODEL, "messages": messages, "max_tokens": 100}
    r = httpx.post(PROXY + "/v1/chat/completions", json=body, headers=headers, timeout=60)
    d = r.json()
    if "error" in d:
        return "ERROR: " + str(d["error"])
    return d["choices"][0]["message"]["content"]


def call_direct_ollama(messages):
    """Call Ollama directly without proxy."""
    r = httpx.post(
        OLLAMA + "/api/chat",
        json={"model": "glm-5.1:cloud", "messages": messages, "stream": False, "options": {"num_predict": 100}},
        timeout=60,
    )
    d = r.json()
    return d.get("message", {}).get("content", "")


def benchmark_prompt(name, messages, expected_keyword=""):
    print("")
    print("=" * 60)
    print(f"[{name}] Prompt: {messages[0].get('content', '')[:60]}...")
    print("=" * 60)

    # Token counts
    content = messages[0].get("content", "")
    tokens_before = count_tokens(content)
    print(f"Tokens (input): {tokens_before}")

    # 1. Direct Ollama (baseline)
    t0 = time.perf_counter()
    resp_direct = call_direct_ollama(messages)
    lat_direct = (time.perf_counter() - t0) * 1000
    print(f"Direct Ollama: {lat_direct:.0f}ms | {resp_direct[:80]}")

    # 2. Proxy WITHOUT compression
    t0 = time.perf_counter()
    resp_baseline = call_proxy(messages, compress=False)
    lat_baseline = (time.perf_counter() - t0) * 1000
    print(f"Proxy (raw):   {lat_baseline:.0f}ms | {resp_baseline[:80]}")

    # 3. Proxy WITH compression
    t0 = time.perf_counter()
    resp_compressed = call_proxy(messages, compress=True)
    lat_compressed = (time.perf_counter() - t0) * 1000
    print(f"Proxy (comp):  {lat_compressed:.0f}ms | {resp_compressed[:80]}")

    # Quality check
    has_keyword_direct = expected_keyword.lower() in resp_direct.lower()
    has_keyword_comp = expected_keyword.lower() in resp_compressed.lower()
    quality = "PASS" if (has_keyword_direct == has_keyword_comp) else "CHECK"

    # Compress the message ourselves to show savings
    config = LatticeConfig(graceful_degradation=True)
    pipeline = CompressorPipeline(config=config)
    pipeline.register(PrefixOptimizer())
    pipeline.register(ReferenceSubstitution())
    pipeline.register(FormatConverter())
    pipeline.register(ToolOutputFilter())
    pipeline.register(OutputCleanup())

    import asyncio
    req = Request(messages=[Message(role=m["role"], content=m.get("content", "")) for m in messages])
    ctx = TransformContext(request_id="test", provider="ollama", model=MODEL)
    result = asyncio.run(pipeline.process(req, ctx))
    compressed = unwrap(result)
    tokens_after = sum(count_tokens(m.content) for m in compressed.messages)
    savings = tokens_before - tokens_after
    ratio = savings / max(tokens_before, 1)

    print(f"")
    print(f"Compression:   {tokens_before} -> {tokens_after} tokens ({savings:+d}, {ratio:.1%})")
    print(f"Quality:       {quality}")
    print(f"Proxy overhead vs direct: {lat_baseline - lat_direct:.0f}ms")
    print(f"Compression overhead:     {lat_compressed - lat_baseline:.0f}ms")


def main():
    # Verify proxy is up
    try:
        r = httpx.get(PROXY + "/healthz", timeout=5)
        data = r.json()
        print(f"Proxy: {data['status']}, provider={data.get('provider','?')}")
        if data.get("provider") != "direct_http":
            print("WARNING: proxy may be running old version!")
    except Exception as exc:
        print(f"ERROR: Proxy not reachable: {exc}")
        return

    # Prompt 1: UUID-heavy
    content = "What is 2+2? Also analyze these IDs: 550e8400-e29b-41d4-a716-446655440000, 6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    benchmark_prompt("uuid", [{"role": "user", "content": content}], expected_keyword="4")

    # Prompt 2: simple
    benchmark_prompt("simple", [{"role": "user", "content": "Hello, how are you? Hi!"}], expected_keyword="hello")

    # Prompt 3: JSON-heavy tool output
    employees = [{"id": i, "name": f"Emp{i}", "salary": 100000 + i * 1000,
                  "metadata": {"created_at": "2024-01-01", "internal": True}}
                 for i in range(15)]
    content = f"Summarize:\n```{json.dumps({'employees': employees}, indent=2)}\n```"
    benchmark_prompt("json_tool", [{"role": "tool", "content": content, "tool_call_id": "call_1"},
                                   {"role": "user", "content": "Top 3 earners?"}], expected_keyword="earn")

    print("")
    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
