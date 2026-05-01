"""Profile FormatConverter to isolate latency bottleneck."""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import time

from lattice.core.context import TransformContext
from lattice.core.transport import Message, Request
from lattice.transforms.format_conv import FormatConverter

def gen_table(n=500) -> str:
    return json.dumps([{"id": i, "name": f"u{i}", "status": "active"} for i in range(n)])

def profile_with_cprofile():
    fc = FormatConverter()
    payload = gen_table(500)
    req = Request(messages=[Message(role="user", content=payload)])
    ctx = TransformContext()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(50):
        fc.process(req.copy(), ctx)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats(30)
    print(s.getvalue())

def profile_steps():
    fc = FormatConverter()
    payload = gen_table(500)
    parsed = json.loads(payload)

    t0 = time.perf_counter()
    for _ in range(200):
        fc._detect_shape(parsed)
    t1 = time.perf_counter()
    print(f"_detect_shape: {(t1-t0)/200*1000:.4f} ms")

    t0 = time.perf_counter()
    for _ in range(200):
        fc._to_csv(parsed)
    t1 = time.perf_counter()
    print(f"_to_csv: {(t1-t0)/200*1000:.4f} ms")

    csv_text = fc._to_csv(parsed)
    t0 = time.perf_counter()
    for _ in range(200):
        fc._validate_roundtrip(payload, csv_text)
    t1 = time.perf_counter()
    print(f"_validate_roundtrip (csv): {(t1-t0)/200*1000:.4f} ms")

    # Also try YAML path with a config dict
    config = {"database": {"host": "localhost", "port": 5432, "credentials": {"user": "admin", "pass": "secret"}}, "features": {"logging": True, "caching": {"enabled": True, "ttl": 3600}}}
    config_json = json.dumps(config)
    config_parsed = json.loads(config_json)

    t0 = time.perf_counter()
    for _ in range(200):
        fc._to_yaml(config_parsed)
    t1 = time.perf_counter()
    print(f"_to_yaml: {(t1-t0)/200*1000:.4f} ms")

    yaml_text = fc._to_yaml(config_parsed)
    if yaml_text:
        t0 = time.perf_counter()
        for _ in range(200):
            fc._validate_roundtrip(config_json, yaml_text)
        t1 = time.perf_counter()
        print(f"_validate_roundtrip (yaml): {(t1-t0)/200*1000:.4f} ms")

if __name__ == "__main__":
    print("=== Step profiling ===")
    profile_steps()
    print("\n=== cProfile ===")
    profile_with_cprofile()
