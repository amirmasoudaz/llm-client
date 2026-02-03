# LLM-Client Architecture

This document describes the microkernel architecture of the llm-client package.

## Core Kernel Components

```
┌────────────────────────────────────────────────────────────────┐
│                         Application                            │
├────────────────────────────────────────────────────────────────┤
│  OpenAIClient  │  Agent  │  Conversation  │  User Code         │
├────────────────────────────────────────────────────────────────┤
│                    ExecutionEngine (Kernel)                    │
│  ┌──────────┬───────────┬──────────┬──────────┬─────────────┐  │
│  │ Routing  │ Caching   │ Retries  │ Hooks    │ Validation  │  │
│  └──────────┴───────────┴──────────┴──────────┴─────────────┘  │
├────────────────────────────────────────────────────────────────┤
│                       Provider Layer                           │
│  ┌──────────────┬────────────────┬─────────────────────────┐   │
│  │ OpenAI       │ Anthropic      │ Google                  │   │
│  └──────────────┴────────────────┴─────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Inside the Kernel

These components route through `ExecutionEngine` and benefit from:
- Caching
- Retry logic with circuit breakers
- Hook system (telemetry, logging)
- Request/response validation
- Concurrency limits

| Component | Flow |
|-----------|------|
| `ExecutionEngine.complete()` | ✅ Full kernel |
| `ExecutionEngine.stream()` | ✅ Full kernel |
| `ExecutionEngine.embed()` | ✅ Full kernel |
| `Agent.run()` | ✅ Uses ExecutionEngine |
| `OpenAIClient.get_response()` | ✅ Uses ExecutionEngine |
| `OpenAIClient.batch()` | ✅ Uses ExecutionEngine |

## Outside the Kernel (Legacy)

These methods bypass `ExecutionEngine` and are deprecated:

| Method | Issue | Alternative |
|--------|-------|-------------|
| `OpenAIClient.transcribe_pdf()` | Creates own AsyncOpenAI client | Use provider.complete() with file content |
| `OpenAIClient.transcribe_image()` | Bypasses some features | Use provider.complete() directly |
| `OpenAIClient._call_model()` | Deprecated | Use get_response() |
| `OpenAIClient._call_completions()` | Deprecated | Use get_response() |
| `OpenAIClient._call_responses()` | Deprecated | Use get_response() |
| `OpenAIClient._call_embeddings()` | Deprecated | Use get_response() |

## Key Abstractions

### RequestContext
Propagates request metadata through the system:
- `request_id` - Unique request identifier
- `tenant_id` - Multi-tenant isolation
- `trace_id`, `span_id` - Distributed tracing
- `cancellation_token` - Cooperative cancellation

### CancellationToken
Cooperative cancellation via `asyncio.Event`:
```python
token = CancellationToken()
ctx = RequestContext(cancellation_token=token)

# In another task:
token.cancel()  # Signals cancellation
```

### MiddlewareChain
Tool execution middleware with production defaults:
```python
chain = MiddlewareChain.production_defaults()
# Includes: telemetry, logging, policy, budget, 
# concurrency, timeout, retry, circuit breaker,
# result size, redaction
```

### Summarizer
Pluggable conversation summarization:
```python
summarizer = LLMSummarizer(engine=engine, config=config)
conversation = Conversation(summarizer=summarizer)
```

## Provider Interface

All providers implement:
```python
class Provider(Protocol):
    async def complete(messages, tools=None, ...) -> CompletionResult
    def stream(messages, ...) -> AsyncIterator[StreamEvent]
    async def embed(inputs, ...) -> EmbeddingResult
    async def complete_structured(messages, schema, ...) -> StructuredResult
```

## Caching

Cache key generation includes:
- Message content hash
- Model name
- Provider identifier
- Tenant ID (for isolation)

Embedding caching uses the same pattern:
```python
result = await engine.embed(
    ["text"],
    cache_response=True,
    cache_collection="embeddings",
)
```

## Migration Guide

### From Legacy Methods

```python
# OLD (deprecated)
client = OpenAIClient()
result = await client.transcribe_pdf("file.pdf")

# NEW (recommended)
provider = OpenAIProvider(model="gpt-4o")
engine = ExecutionEngine(provider=provider)
# Use engine.complete() with file content in messages
```

### Using Cancellation

```python
from llm_client import CancellationToken, RequestContext

token = CancellationToken()
ctx = RequestContext(cancellation_token=token)

# Start request
task = asyncio.create_task(engine.complete(spec, context=ctx))

# Cancel after timeout
await asyncio.sleep(5)
token.cancel()
```
