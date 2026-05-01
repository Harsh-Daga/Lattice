# Quick Start

Get LATTICE running in 5 minutes.

## 1. Install

```bash
pip install lattice-transport
```

## 2. Start the Proxy

```bash
lattice proxy --port 8787
```

This starts a local proxy server on port 8787 that accepts standard OpenAI API requests.

## 3. Point Your App at the Proxy

```bash
export OPENAI_BASE_URL=http://localhost:8787/v1
```

Any OpenAI-compatible client now routes through LATTICE automatically.

Test it:

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 4. Check What's Happening

```bash
# See real-time stats
curl http://localhost:8787/stats | jq

# Check health
curl http://localhost:8787/healthz
```

## 5. Using the SDK (Optional)

```python
from lattice import LatticeClient

client = LatticeClient()
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Explain transformers"}],
)
print(response.choices[0].message.content)
```

## Next Steps

- [Full CLI Reference](cli.md)
- [Understanding Transforms](transforms.md)
- [Running Benchmarks](benchmarks.md)
