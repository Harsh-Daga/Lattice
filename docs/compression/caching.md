# Caching

LATTICE provides two complementary caching layers: a response cache (Semantic Cache) and provider cache alignment (Cache Arbitrage).

## Semantic Cache

A hybrid exact + approximate response cache that sits after the transform pipeline and before the provider call. On cache hit, returns the stored response directly — bypassing the provider entirely.

### How It Works

```
Request → Pipeline → Compute cache key (SHA-256 of canonical JSON)
                         │
                ┌────────┴────────┐
                ▼                 ▼
           Exact hit?       Approximate hit?
           (same key)       (semantic fingerprint > threshold)
                │                 │
                ▼                 ▼
         Return cached      Return cached if similar enough
                │
                ▼ (miss)
          Call provider → Cache response
```

### Features

- **Exact matching**: SHA-256 hash of canonical request (model, messages, temperature, tools, stop)
- **Approximate matching**: Jaccard token similarity (40%), role pattern (20%), tool schema hash (20%), text similarity (20%)
- **TTL-based expiry**: Per-entry, not global sweep
- **LRU eviction**: Oldest entries evicted when max size reached
- **Max entry size**: Rejects responses > 512KB
- **Configurable threshold**: Semantic similarity threshold (default 0.86)

### Backends

| Backend | When to Use |
|---------|-------------|
| **InMemory** (default) | Single-process deployments |
| **Redis** | Multi-process/shared deployments |

```python
# Redis backend
cache = SemanticCache(
    backend=RedisCacheBackend(url="redis://localhost:6379/0"),
    ttl_seconds=300,
    max_entries=10000,
)
```

### API

```python
from lattice import LatticeClient

client = LatticeClient()

# Disable cache per-request
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[...],
    extra_headers={"x-lattice-disable-cache": "true"},
)
```

### Stats

```
GET /cache/stats

{
  "enabled": true,
  "entries": 142,
  "max_entries": 1000,
  "hits": 89,
  "misses": 53,
  "hit_rate": 0.627,
  "evictions": 12,
  "rejects": 3,
  "exact_hits": 74,
  "approximate_hits": 15,
  "maintenance_runs": 4,
  "stale_removed": 8
}
```

---

## Cache Arbitrage

Provider-side KV-cache optimization that reorders prompt messages to maximize cache hit probability at the LLM provider level.

### How It Works

1. **Message classification**: Each message is classified into stability buckets:
   - System messages (most stable)
   - Tool definitions
   - Static documentation
   - Variable user content (least stable)

2. **Reordering**: Messages are reordered into the canonical stability order

3. **Provider hints**: Provider-specific cache controls are applied:
   - **OpenAI**: `prompt_cache_key` and `prompt_cache_retention`
   - **Anthropic**: `cache_control` breakpoints on stable prefixes
   - **Gemini**: `cachedContent` references
   - **Bedrock**: `bedrock_prompt_caching` flag

4. **Prefix hash tracking**: A SHA-256 hash of the stable prefix is tracked across turns to identify cache repeats

### Provider Cache Modes

| Provider | Mode | How |
|----------|------|-----|
| OpenAI | AUTO_PREFIX | Stable prefix gets `prompt_cache_key` |
| Anthropic | EXPLICIT_BREAKPOINT | `cache_control` on system/tool blocks |
| Gemini, Vertex | EXPLICIT_CONTEXT | `cachedContent` with TTL |
| Bedrock | EXPLICIT_BREAKPOINT | `bedrock_prompt_caching` |
| Others | NONE | No provider-side caching |

### Cache Maintenance

```python
# Expire stale entries
removed = await cache.expire_stale()

# Clear all
count = await cache.clear()

# Invalidate by pattern
removed = await cache.invalidate_by_pattern(
    lambda resp: resp.model == "gpt-3.5-turbo"
)

# Invalidate by key
await cache.invalidate(key)
```
