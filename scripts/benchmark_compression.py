#!/usr/bin/env python3
"""Real compression vs baseline test with actual model calls."""
import asyncio
import sys
sys.path.insert(0, "src")

from lattice.providers.transport import DirectHTTPProvider
from lattice.core.pipeline import CompressorPipeline
from lattice.core.transport import Message, Request
from lattice.core.context import TransformContext
from lattice.core.config import LatticeConfig
from lattice.core.result import unwrap
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.tool_filter import ToolOutputFilter
from lattice.transforms.prefix_opt import PrefixOptimizer
from lattice.transforms.format_conv import FormatConverter
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.utils.token_count import TiktokenCounter

MODEL = "ollama/glm-5.1:cloud"
BASE_URL = "http://127.0.0.1:11434"
MAX_TOKENS = 120


def make_uuid_prompt():
    uuids = "550e8400-e29b-41d4-a716-446655440000, 6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    return "What is 2+2? Also analyze these IDs: " + uuids, "uuid"


def make_simple_prompt():
    return "Hello, how are you?", "simple"


def count_tokens(text):
    try:
        return TiktokenCounter("gpt-4").count(text)
    except Exception:
        return len(text) // 4


async def run_single(content, label):
    provider = DirectHTTPProvider(default_api_base=BASE_URL)

    print("")
    print("=" * 60)
    print("[" + label + "] Baseline call")
    print("=" * 60)
    baseline_text = content
    baseline_tokens = count_tokens(baseline_text)
    print("Content: " + baseline_text[:100])
    print("Tokens: " + str(baseline_tokens))

    import time
    t0 = time.perf_counter()
    r1 = await provider.completion(model=MODEL, messages=[{"role": "user", "content": baseline_text}], max_tokens=MAX_TOKENS)
    baseline_latency = (time.perf_counter() - t0) * 1000
    print("Response: " + r1.content[:100])
    print("Latency: " + str(int(baseline_latency)) + "ms")

    print("")
    print("[" + label + "] LATTICE call")
    config = LatticeConfig(graceful_degradation=True)
    pipeline = CompressorPipeline(config=config)
    pipeline.register(PrefixOptimizer())
    pipeline.register(ReferenceSubstitution())
    pipeline.register(FormatConverter())
    pipeline.register(ToolOutputFilter())
    pipeline.register(OutputCleanup())

    req = Request(messages=[Message(role="user", content=content)])
    ctx = TransformContext(request_id="bench-" + label, provider="ollama", model=MODEL)

    result = await pipeline.process(req, ctx)
    compressed = unwrap(result)
    compressed_text = compressed.messages[0].content
    compressed_tokens = count_tokens(compressed_text)

    from lattice.core.transport import Response

    t0 = time.perf_counter()
    r2 = await provider.completion(model=MODEL, messages=[{"role": "user", "content": compressed_text}], max_tokens=MAX_TOKENS)
    lattice_latency = (time.perf_counter() - t0) * 1000

    # Reverse transforms on response
    resp = Response(content=r2.content)
    reversed_resp = await pipeline.reverse(resp, ctx)

    savings = baseline_tokens - compressed_tokens
    ratio = savings / max(baseline_tokens, 1)

    print("Compressed content: " + compressed_text[:100])
    print("Tokens: " + str(compressed_tokens))
    print("Response (reversed): " + reversed_resp.content[:100])
    print("Latency: " + str(int(lattice_latency)) + "ms")
    print("Savings: " + str(savings) + " tokens (" + str(round(ratio * 100, 1)) + "%)")

    has_4_base = "4" in r1.content or "four" in r1.content.lower()
    has_4_lat = "4" in r2.content or "four" in r2.content.lower()
    quality = "PASS" if (has_4_base == has_4_lat) else "CHECK"
    print("Quality: " + quality)

    # Cleanup pool
    await provider.pool.close()


async def main():
    prompts = [
        make_uuid_prompt(),
        make_simple_prompt(),
    ]
    for content, label in prompts:
        await run_single(content, label)

    print("")
    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
