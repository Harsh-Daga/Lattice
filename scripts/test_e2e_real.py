#!/usr/bin/env python3
"""Real end-to-end test: Proxy → DirectHTTPProvider → Ollama."""
from __future__ import annotations

import json
import time

import httpx

OLLAMA_BASE = "http://127.0.0.1:11434"
PROXY_BASE = "http://127.0.0.1:8787"

MODEL = "ollama/glm-5.1:cloud"


def test_direct_ollama():
    """Baseline: direct Ollama call."""
    print("=" * 60)
    print("Test 1: Direct Ollama HTTP call")
    print("=" * 60)
    t0 = time.perf_counter()
    r = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": "glm-5.1:cloud",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Status: {r.status_code}")
    print(f"Latency: {elapsed:.1f}ms")
    data = r.json()
    print(f"Content: {data.get('message',{}).get('content','')[:80]}")
    print()


def test_proxy_pipeline():
    """Proxy path: OpenAI API → compression → DirectHTTPProvider → Ollama."""
    print("=" * 60)
    print("Test 2: Proxy via DirectHTTPProvider")
    print("=" * 60)
    t0 = time.perf_counter()
    r = httpx.post(
        f"{PROXY_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Status: {r.status_code}")
    print(f"Latency: {elapsed:.1f}ms")
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:80]
    print(f"Content: {content}")

    # Verify adapter normalization worked
    assert data["object"] == "chat.completion"
    assert "id" in data
    print("Proxy normalization: PASS")
    print()


def test_proxy_compression():
    """Verify compression triggers on UUID-heavy prompt."""
    print("=" * 60)
    print("Test 3: Compression benchmark (UUID-heavy)")
    print("=" * 60)
    uuids = [
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    ]
    content = f"Analyze these IDs: {', '.join(uuids)} and explain what 2+2 is."
    t0 = time.perf_counter()
    r = httpx.post(
        f"{PROXY_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Status: {r.status_code}")
    print(f"Latency: {elapsed:.1f}ms")
    data = r.json()
    content_out = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:80]
    print(f"Content: {content_out}")
    # Response should NOT contain <ref_> aliases (they're reversed)
    assert "ref_" not in str(data), "ReferenceSubstitution was NOT reversed!"
    print("Reversible transform: PASS")
    print()


def test_health():
    print("=" * 60)
    print("Test 4: Proxy health & stats")
    print("=" * 60)
    r = httpx.get(f"{PROXY_BASE}/healthz")
    print(f"healthz: {r.json()}")
    r = httpx.get(f"{PROXY_BASE}/stats")
    data = r.json()
    print(f"stats: {json.dumps(data, indent=2)}")
    assert data["provider"] == "direct_http"
    assert "pools" in data
    print()


if __name__ == "__main__":
    print("LATTICE Transport Layer E2E Test")
    print("Model:", MODEL)
    print()
    test_health()
    test_direct_ollama()
    test_proxy_pipeline()
    test_proxy_compression()
    print("=" * 60)
    print("All E2E tests PASSED")
    print("=" * 60)
