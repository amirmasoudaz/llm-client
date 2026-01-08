# llm-client

Asynchronous OpenAI client with built‑in caching, rate limiting, and streaming helpers.

This package wraps the official `openai` Python SDK and adds:

- Model profiles for GPT‑5 family and `text-embedding-3-*`
- Automatic token counting, usage and cost estimation
- Pluggable response caching (filesystem, Qdrant, PostgreSQL + Redis)
- Token‑aware rate limiting using a token bucket implementation
- Event-based streaming helpers (SSE + pub/sub sinks)
- Opinionated async interface suitable for high‑throughput backends

The package auto‑loads environment variables from the nearest `.env` file on import for convenient local development.

---
## Installation

From the `llm-client` repo on your machine:

```bash
pip install -e .
```

Or, from another project, pointing at your local checkout:

```text
-e /absolute/path/to/llm-client
```

### Requirements

- Python `>=3.10`
- An OpenAI API key (`OPENAI_API_KEY`)
- Optional backing services depending on the cache backend:
  - Filesystem: none
  - Qdrant: running Qdrant instance (`QDRANT_URL`, `QDRANT_API_KEY`)
  - PostgreSQL + Redis: `PG_DSN`, `REDIS_URL`

The package itself depends on `openai>=1.59`, `aiohttp`, `asyncpg`, `redis`, `tiktoken`, `numpy`, and a few utility libraries (see `pyproject.toml`).

---

## Quickstart

Minimal chat completion using the bundled GPT‑5 Nano profile and no caching:

```python
import asyncio
from llm_client import OpenAIClient, GPT5Nano


async def main() -> None:
    client = OpenAIClient(GPT5Nano, cache_backend=None)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    response = await client.get_response(messages=messages)
    print(response["output"])


if __name__ == "__main__":
    asyncio.run(main())
```

The repository includes a runnable example in `main.py` that demonstrates the packaged client with PostgreSQL + Redis caching.

---

## Core Concepts

### Model profiles

Models are described by `ModelProfile` subclasses in `src/llm_client/models.py`. Profiles encapsulate:

- `key` – short identifier (e.g. `"gpt-5-nano"`)
- `model_name` – actual API model name
- `category` – `"completions"` or `"embeddings"`
- `context_window`, `max_output`, `output_dimensions`
- `rate_limits` – tokens and requests per minute
- `usage_costs` – per‑token cost for input/output/cached input

Available profiles include:

- Completions: `GPT5`, `GPT5Mini`, `GPT5Nano`, `GPT5Point1`, `GPT5Point2`
- Embeddings: `TextEmbedding3Large`, `TextEmbedding3Small`

You can pass either a profile class or a registered key when constructing the client:

```python
from llm_client import OpenAIClient, GPT5Point1
from llm_client import ModelProfile

client = OpenAIClient(GPT5Point1)
# or
client = OpenAIClient("gpt-5.1")  # uses ModelProfile.get(...)
```

The profiles expose helpers like `count_tokens(context)` and `parse_usage(usage_dict)` for consistent accounting.

### OpenAIClient

The main entrypoint lives in `src/llm_client/client.py`:

```python
from llm_client import OpenAIClient, GPT5Nano

client = OpenAIClient(
    GPT5Nano,
    cache_backend="fs",          # "fs", "qdrant", "pg_redis", or None
    cache_dir="cache/",          # required for "fs"
    cache_collection="my_cache", # optional; varies by backend
)
```

Key responsibilities:

- Delegates to the appropriate model API (`chat.completions` vs `embeddings`)
- Normalizes parameters and response formats
- Applies token/request rate limits via `Limiter`
- Integrates with the configured cache backend
- Provides high‑level helpers for single and batched calls

#### `get_response(...)`

```python
response = await client.get_response(
    messages=[{"role": "user", "content": "Tell me a joke."}],
    cache_response=True,
    timeout=30.0,
)
print(response["status"], response["error"])
print(response["output"])  # model completion
print(response["usage"])   # normalized usage and cost
```

Important parameters:

- `identifier`: explicit cache key; if omitted, a deterministic hash of `messages` / `input` is used when `hash_as_identifier=True`.
- `cache_response`: enable caching for successful (and optionally failed) responses.
- `rewrite_cache`: force a new cache entry while still allowing reads from older entries.
- `regen_cache`: bypass existing cache entries and force recomputation.
- `attempts`, `backoff`: simple retry with exponential backoff for 5xx / connection errors.
- `timeout`: wraps the underlying API call in `asyncio.wait_for`.

The returned dictionary always includes at least:

- `params`: the parameters used for the underlying OpenAI call
- `output`: the model output (string for completions; vectors for embeddings)
- `usage`: normalized usage and cost information
- `status`: HTTP‑like status code
- `error`: `"OK"` on success, or an error description
- `identifier`: effective cache identifier

Advanced APIs:

- `await client.invoke(...)` returns an `LLMResult` object (same data, structured).
- `client.stream(...)` yields `LLMEvent` objects for event-driven streaming.
- `client.stream_sse(...)` yields SSE-formatted strings for web UIs.

#### Embeddings

When the selected profile is an embeddings model, `get_response` routes to the embeddings API:

```python
from llm_client import OpenAIClient, TextEmbedding3Small

client = OpenAIClient(TextEmbedding3Small)
resp = await client.get_response(input="Some text to embed")
emb = resp["output"]  # list[float] or list[list[float]]
```

By default, embeddings are returned as numeric vectors (the client decodes base64 when needed).

#### Batching helpers

`OpenAIClient` provides simple concurrency helpers:

- `await client.run_batch(coros: list)` – execute a list of coroutines with bounded concurrency; errors are wrapped as dicts with `"error"` and `"status"`.
- `async for result in client.iter_batch(coros: list)` – iterate results as they complete.

All individual coroutines should be calls to `client.get_response(...)`.

---

## Caching

Caching is abstracted behind `CacheCore` in `src/llm_client/cache.py` and supports multiple backends:

- `"fs"` – JSON files on disk
- `"qdrant"` – vector database used as a generic key/value store
- `"pg_redis"` – hybrid PostgreSQL + Redis cache
- `None` / `"none"` – disable caching

### Filesystem cache

```python
from pathlib import Path
from llm_client import OpenAIClient, GPT5Nano

client = OpenAIClient(
    GPT5Nano,
    cache_backend="fs",
    cache_dir=Path("cache/completions"),
)

resp = await client.get_response(
    messages=[{"role": "user", "content": "Cached?"}],
    cache_response=True,
)
```

The effective cache key is written as `<identifier>.json` under `cache_dir`.

### Qdrant cache

Requires a running Qdrant instance:

```python
client = OpenAIClient(
    GPT5Nano,
    cache_backend="qdrant",
    cache_collection="completions_cache",
    qdrant_url="http://localhost:6333",
    qdrant_api_key="your-key",  # optional
)
```

Responses are stored as payloads in a 1‑dimensional vector collection keyed by a stable hashed ID.

### PostgreSQL + Redis cache

Hybrid cache suitable for production:

```python
client = OpenAIClient(
    GPT5Nano,
    cache_backend="pg_redis",
    cache_collection="llm_cache",
    pg_dsn="postgresql://user:pass@host:5432/dbname",
    redis_url="redis://localhost:6379/0",
)
```

The cache layer:

- Keeps a durable copy in PostgreSQL (optionally compressed)
- Mirrors hot entries into Redis with a TTL (`redis_ttl_seconds`)
- Supports `rewrite_cache`/`regen_cache` semantics for controlled invalidation

For all backends, you can manually manage lifecycle:

```python
await client.warm_cache()  # backend-specific warmup
await client.close()       # close DB / Redis connections
```

---

## Streaming

Streaming is event-based and decoupled from transport. You can:

- Stream raw events with `client.stream(...)` (yields `LLMEvent`)
- Stream SSE strings with `client.stream_sse(...)`
- Push events to a pub/sub sink (e.g., Pusher) via `client.stream_to_sink(...)`

SSE example:

```python
stream = client.stream_sse(
    messages=[{"role": "user", "content": "Stream this."}],
    cache_response=False,
)

async for event in stream:
    print(event)
```

Common event types: `meta`, `text_delta`, `tool_call_delta`, `usage`, `error`, `done`.

Pusher example (event sink):

```python
from llm_client import PusherSink

async with PusherSink(channel="my-channel-name") as sink:
    await client.stream_to_sink(
        sink,
        messages=[{"role": "user", "content": "Stream this."}],
    )
```

Configure Pusher credentials via environment variables:

- `PUSHER_AUTH_KEY`
- `PUSHER_AUTH_SECRET`
- `PUSHER_AUTH_VERSION`
- `PUSHER_APP_CLUSTER`
- `PUSHER_APP_ID`

---

## Rate limiting

Rate limiting is handled by `Limiter` and `TokenBucket` in `src/llm_client/rate_limit.py`.

- Each `ModelProfile` defines `rate_limits` with `tkn_per_min` and `req_per_min`.
- `Limiter.limit(tokens, requests)` returns an async context manager that:
  - waits until enough token and request budget is available;
  - tracks output tokens and charges them back into the bucket on exit.

The client uses this internally to guard all OpenAI calls; you usually do not need to use `Limiter` directly.

---

## Environment and configuration

On import, `llm_client` automatically loads environment variables from the nearest `.env` file using `python-dotenv`. Relevant variables include:

- `OPENAI_API_KEY` – required by the `openai` SDK.
- `PG_DSN` – default PostgreSQL DSN for the hybrid cache.
- `REDIS_URL` – default Redis URL for the hybrid cache.
- `QDRANT_URL`, `QDRANT_API_KEY` – default Qdrant configuration.
- Pusher variables as listed in the Streaming section.

---

## Agents (tool loop)

The agent module provides a minimal tool-calling loop:

```python
from llm_client import OpenAIClient, Tool, AgentRunner

async def get_time(zone: str) -> str:
    return f"time in {zone}"

tools = [
    Tool(
        name="get_time",
        description="Get time in a specific zone.",
        parameters={
            "type": "object",
            "properties": {"zone": {"type": "string"}},
            "required": ["zone"],
        },
        func=get_time,
    )
]

client = OpenAIClient("gpt-5-nano")
runner = AgentRunner(client, tools)
result = await runner.run(messages=[{"role": "user", "content": "What time is it in UTC?"}])
print(result.output)
```

You can override most of these by passing explicit arguments to `OpenAIClient(...)`.

---

## Development

- Main entrypoint: `src/llm_client/client.py`
- Supporting modules:
  - `src/llm_client/models.py` – model profiles
  - `src/llm_client/cache.py` – cache backends and core
  - `src/llm_client/rate_limit.py` – rate limiting primitives
  - `src/llm_client/streaming.py` – streaming helpers (SSE + sinks)
  - `src/llm_client/exceptions.py` – custom exceptions

To run the included example from the repo root:

```bash
OPENAI_API_KEY=... python -m main
```

Make sure any backing services required by your chosen cache backend (PostgreSQL, Redis, Qdrant) are reachable before running tests or examples.
