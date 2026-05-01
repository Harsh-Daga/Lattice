# SDK

The LatticeClient provides programmatic access to LATTICE's compression, caching, and session features from Python.

## Basic Usage

```python
from lattice import LatticeClient

client = LatticeClient()
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Explain compression algorithms"}],
)
print(response.choices[0].message.content)
```

## Constructor

```python
client = LatticeClient(
    # Transport optimization
    enable_transport=True,       # Enable compression pipeline
    enable_batching=False,       # Enable request batching
    enable_speculative=False,    # Enable speculative execution

    # Session management
    session_id=None,             # Reuse an existing session

    # Caching
    enable_cache=True,           # Enable semantic cache
    cache_ttl_seconds=300,       # Cache TTL (5 min default)

    # Provider
    provider="openai",           # Default provider
    base_url=None,               # Upstream base URL override
    api_key=None,                # API key override
)
```

## Request Template Reference

The SDK compresses using the template-reference pattern. Your actual request is compared against a template to extract only the delta (changed content).

```python
# Template: what the model expects for this task
# Request: what the user actually asked
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": "Review the following code:\n```python\n...```"},
    ],
    template={
        "messages": [
            {"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": "Review the following code:\n{code}"},
        ]
    }
)
```

## Direct Compression

For raw text compression without an LLM call:

```python
from lattice import LatticeClient

client = LatticeClient()
result = client.compress(
    text="The session manifest should remain stable " * 10,
    model="openai/gpt-4o",
)
print(f"Original: {result.original_tokens} tokens")
print(f"Compressed: {result.compressed_tokens} tokens")
print(f"Ratio: {result.compression_ratio:.1%}")
print(f"Compressed text: {result.text}")
```

## Session Management

```python
# New session (auto-generated ID)
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "First question"}],
)
session_id = response.lattice_session_id

# Reuse session (delta encoding applies)
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Follow-up question"}],
    session_id=session_id,
)
```

## Request Headers

Custom headers for fine-grained control:

```python
headers = {
    "x-lattice-session-id": "my-session",
    "x-lattice-disable-transforms": "false",
    "x-lattice-disable-cache": "false",
    "x-lattice-client-profile": "my-app",
}
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[...],
    extra_headers=headers,
)
```

## Response Metadata

```python
response = client.chat.completions.create(...)

# Standard OpenAI fields
print(response.choices[0].message.content)
print(response.usage.total_tokens)

# LATTICE metadata
print(response.lattice_session_id)
print(response.lattice_compression_ratio)
print(response.lattice_tokens_saved)
```

## CompressResult

```python
from lattice import CompressResult

@dataclass
class CompressResult:
    text: str              # Compressed text
    original_tokens: int   # Tokens before compression
    compressed_tokens: int # Tokens after compression
    compression_ratio: float # 0.0 to 1.0
    transforms_applied: list[str]  # Which transforms ran
    metrics: dict          # Per-transform metrics
```

## Configuration

The SDK reads from the same `LatticeConfig` as the proxy:

```python
from lattice.core.config import LatticeConfig

config = LatticeConfig(
    compression_mode="aggressive",
    graceful_degradation=True,
    semantic_cache_enabled=True,
    semantic_cache_ttl_seconds=600,
)
client = LatticeClient(config=config)
```
