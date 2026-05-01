# Providers

LATTICE supports 17 LLM providers through direct adapters. Each adapter handles request serialization, response deserialization, and streaming chunk normalization. No routing — every request hits exactly one provider.

## Supported Providers

| Provider | Prefix | Adapter | HTTP/2 | Streaming | Cache | Notable |
|----------|--------|---------|--------|-----------|-------|---------|
| **OpenAI** | `openai/` | `OpenAIAdapter` | ✅ | SSE delta | AUTO_PREFIX | prompt_cache_key |
| **Anthropic** | `anthropic/`, `claude-` | `AnthropicAdapter` | ✅ | SSE | EXPLICIT_BREAKPOINT | cache_control, thinking |
| **Ollama** | `ollama/` | `OllamaAdapter` | — | SSE | — | Local models |
| **Ollama Cloud** | `ollama-cloud/` | `OllamaCloudAdapter` | ✅ | SSE | — | OpenAI-compatible |
| **Groq** | `groq/` | `GroqAdapter` | ✅ | SSE | — | LPU inference |
| **DeepSeek** | `deepseek/` | `DeepSeekAdapter` | ✅ | SSE | — | OpenAI-compatible |
| **Mistral** | `mistral/` | `MistralAdapter` | ✅ | SSE | — | La Plateforme API |
| **Cohere** | `cohere/` | `CohereAdapter` | ✅ | SSE | — | Command models |
| **Gemini** | `gemini/`, `google/` | `GeminiAdapter` | ✅ | SSE | EXPLICIT_CONTEXT | cachedContent |
| **Vertex** | `vertex/` | `VertexAdapter` | ✅ | SSE | EXPLICIT_CONTEXT | GCP IAM auth |
| **Azure** | `azure/` | `AzureAdapter` | ✅ | SSE | AUTO_PREFIX | Azure OpenAI |
| **Bedrock** | `bedrock/` | `BedrockAdapter` | ✅ | SSE | EXPLICIT_BREAKPOINT | AWS SigV4 |
| **OpenRouter** | `openrouter/` | `OpenRouterAdapter` | ✅ | SSE | — | Multi-model gateway |
| **Fireworks** | `fireworks/` | `FireworksAdapter` | ✅ | SSE | — | High-speed inference |
| **Together** | `together/` | `TogetherAdapter` | ✅ | SSE | — | Open-source models |
| **Perplexity** | `perplexity/` | `PerplexityAdapter` | ✅ | SSE | — | Search-augmented |
| **AI21** | `ai21/` | `AI21Adapter` | ✅ | SSE | — | Jamba models |

## Model Specification

Models are specified with provider prefixes:

```python
# Explicit prefix (always works)
"openai/gpt-4o"
"anthropic/claude-3-5-sonnet"
"ollama/llama3.2"

# Claude prefix shorthands
"claude-3-5-sonnet"  # maps to anthropic/

# Bare model names (openai by default)
"gpt-4o"             # maps to openai/gpt-4o
```

## Provider Detection

When using the proxy, providers can be specified via HTTP headers:

```bash
# Option 1: Model prefix
curl ... -d '{"model": "groq/llama-3.1-70b"}'

# Option 2: Explicit header
curl ... -H "x-lattice-provider: groq" \
  -d '{"model": "llama-3.1-70b"}'
```

## Connection Pooling

Each provider base URL gets a persistent `httpx.AsyncClient(http2=True)` with keep-alive. HTTP/2 is attempted first; falls back to HTTP/1.1 if unavailable (logged in `/stats` under `transport.pools`).

## Streaming

All providers support streaming via SSE. LATTICE normalizes provider-specific chunk formats (OpenAI delta, Anthropic content_block_delta, etc.) into a unified OpenAI-compatible SSE stream.

## Credential Resolution

API keys are resolved in order:
1. Provider-specific env var (e.g., `ANTHROPIC_API_KEY`)
2. Generic `OPENAI_API_KEY` (for OpenAI-compatible providers)
3. `LATTICE_PROVIDER_API_KEY` for forwarding
4. Credential file (`.env`, `lattice.yaml`)

## Adding a Provider

1. Extend `ProviderAdapter` in `src/lattice/providers/base.py`
2. Implement `supports()`, `serialize_request()`, `deserialize_response()`
3. Implement `deserialize_stream_chunk()` for streaming
4. Register in `ProviderRegistry` in `src/lattice/providers/transport.py`
5. Add capability entry in `CapabilityRegistry`
