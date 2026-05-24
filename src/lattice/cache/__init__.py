"""LATTICE semantic cache: exact-hash + approximate fingerprint.

Backends:
    InMemoryCacheBackend (default)
    RedisCacheBackend    (optional, requires ``lattice-transport[redis]``)
"""

from lattice.cache.semantic import (
    CacheBackend,
    CachedResponse,
    ContentClass,
    ExactResponseCache,
    InMemoryCacheBackend,
    RedisCacheBackend,
    SemanticCache,
    assemble_cached_response,
    compute_cache_key,
    generate_sse_chunks,
)

__all__ = [
    "SemanticCache",
    "ExactResponseCache",
    "ContentClass",
    "CachedResponse",
    "CacheBackend",
    "InMemoryCacheBackend",
    "RedisCacheBackend",
    "compute_cache_key",
    "assemble_cached_response",
    "generate_sse_chunks",
]
