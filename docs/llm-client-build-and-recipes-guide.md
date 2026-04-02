# llm-client Build and Recipes Guide

This is the practical "how do I build with this package?" guide.

Use it together with:

- [llm-client-package-api-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-package-api-guide.md)
- [llm-client-usage-and-capabilities-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-usage-and-capabilities-guide.md)
- [examples/README.md](/home/namiral/Projects/Packages/llm-client-v1/examples/README.md)

The package README is the broad reference. This guide is the "show me how to
do the real thing" layer.

## 1. Install The Package For The Capabilities You Need

Minimal package install:

```bash
pip install llm-client
```

Useful extras:

```bash
pip install llm-client[anthropic,google]
pip install llm-client[postgres]
pip install llm-client[mysql]
pip install llm-client[redis]
pip install llm-client[qdrant]
pip install llm-client[telemetry]
pip install llm-client[server]
pip install llm-client[all]
```

If you are developing against the source tree:

```bash
pip install -e .[all]
```

For the `pg_redis` cache backend, install `llm-client[postgres]` for the
durable PostgreSQL layer and add `llm-client[redis]` if you also want the
Redis hot-cache path.

## 2. Choose The Right Entry Point First

Use `llm_client.providers` when:

- you want one provider directly
- you do not need engine-level retry, failover, cache, or hooks

Use `llm_client.engine` when:

- the code is part of a real backend path
- you want stable behavior around retries, failover, cache, diagnostics, or
  idempotency

Use `llm_client.agent` when:

- you need multi-turn conversation
- you need tool calling
- you need an assistant/copilot runtime rather than one-shot inference

Use `llm_client.adapters` when:

- the model needs controlled access to SQL, Redis, or vector stores
- you want generic connectivity and safety rules inside the package

## 3. Make A Direct LLM Call

```python
import asyncio

from llm_client.providers import OpenAIProvider


async def main() -> None:
    provider = OpenAIProvider(model="gpt-5-mini")
    result = await provider.complete("Reply with the word ok only.", max_tokens=64)

    print(result.content)
    print(result.usage.total_tokens if result.usage else None)
    print(result.usage.total_cost if result.usage else None)


asyncio.run(main())
```

Use this path for small integrations or provider-specific work.

## 4. Use The Engine For Production Paths

```python
import asyncio

from llm_client.engine import ExecutionEngine
from llm_client.providers import OpenAIProvider
from llm_client.types import Message, RequestSpec


async def main() -> None:
    engine = ExecutionEngine(provider=OpenAIProvider(model="gpt-5-mini"))
    spec = RequestSpec(
        provider="openai",
        model="gpt-5-mini",
        messages=[Message.user("Summarize this in one sentence: llm_client is now stable.")],
    )
    result = await engine.complete(spec)
    print(result.content)


asyncio.run(main())
```

This is the preferred default for backend code.

## 5. Create A Tool

The easiest tool path is the decorator:

```python
from llm_client.tools import tool


@tool
async def get_weather(city: str) -> dict[str, str]:
    return {"city": city, "forecast": "sunny", "temperature_c": "21"}
```

If you need stricter execution metadata, build a `Tool` explicitly or attach
middleware/execution metadata after creation.

## 6. Make An Agent Use The Tool

```python
import asyncio

from llm_client.agent import Agent
from llm_client.engine import ExecutionEngine
from llm_client.providers import OpenAIProvider
from llm_client.tools import tool


@tool
async def get_weather(city: str) -> dict[str, str]:
    return {"city": city, "forecast": "sunny", "temperature_c": "21"}


async def main() -> None:
    engine = ExecutionEngine(provider=OpenAIProvider(model="gpt-5-mini"))
    agent = Agent(
        engine=engine,
        system_message="Use tools when they help answer the user accurately.",
        tools=[get_weather],
    )

    result = await agent.run("What is the weather in Tokyo?")
    print(result.content)
    print(result.total_usage.total_cost if result.total_usage else None)


asyncio.run(main())
```

The agent owns the conversation loop. The engine owns request execution.

## 7. Control Tool Execution Mode

Use an `AgentDefinition` when you want explicit tool execution policy:

```python
from llm_client.agent import (
    Agent,
    AgentDefinition,
    AgentExecutionPolicy,
    ToolExecutionMode,
)

definition = AgentDefinition(
    name="ops-assistant",
    system_message="Prefer structured tool usage over guessing.",
    execution_policy=AgentExecutionPolicy(
        tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        max_tool_calls_per_turn=5,
        tool_timeout=15.0,
    ),
)

agent = Agent(
    engine=engine,
    definition=definition,
    tools=[get_weather],
)
```

Use:

- `SINGLE` when you want only one tool execution
- `SEQUENTIAL` when ordering matters
- `PARALLEL` when tools are independent
- `PLANNER` only for specialized tool execution planning paths

## 8. Get Structured Output

```python
import asyncio
from pydantic import BaseModel

from llm_client.providers import OpenAIProvider
from llm_client.structured import extract_structured


class Ticket(BaseModel):
    severity: str
    team: str
    summary: str


async def main() -> None:
    provider = OpenAIProvider(model="gpt-5-mini")
    result = await extract_structured(
        provider=provider,
        model="gpt-5-mini",
        schema=Ticket,
        prompt="Classify this incident: API 500s are spiking in prod for checkout.",
    )
    print(result.output)
    print(result.diagnostics.mode if result.diagnostics else None)


asyncio.run(main())
```

Use this for extraction, classification, and schema-shaped generation.

## 9. Send Multimodal And File Inputs

Use the content model when plain text is not enough:

```python
from llm_client.content import ContentMessage, FileBlock, TextBlock
from llm_client.providers.types import Role

message = ContentMessage(
    role=Role.USER,
    blocks=(
        TextBlock("Read this file and summarize the key risks."),
        FileBlock(file_path="/absolute/path/to/report.pdf"),
    ),
)
```

Important `FileBlock` rules:

- use one of `file_path`, `data`, `file_id`, or `file_url`
- native provider transport is used where supported
- non-native providers either degrade to `extracted_text` in lossy mode or
  fail in strict mode

See:

- [35_file_block_transport.py](/home/namiral/Projects/Packages/llm-client-v1/examples/35_file_block_transport.py)

## 10. Turn On Caching

The stable engine path supports both the legacy booleans and the newer
`CachePolicy` object.

```python
import asyncio
from pathlib import Path

from llm_client.cache import CachePolicy, CacheSettings, build_cache_core
from llm_client.engine import ExecutionEngine
from llm_client.providers import OpenAIProvider
from llm_client.types import Message, RequestSpec


async def main() -> None:
    cache = build_cache_core(
        CacheSettings(
            backend="fs",
            client_type="completions",
            cache_dir=Path(".llm-cache"),
            default_collection="demo",
        )
    )
    engine = ExecutionEngine(
        provider=OpenAIProvider(model="gpt-5-mini"),
        cache=cache,
    )

    spec = RequestSpec(
        provider="openai",
        model="gpt-5-mini",
        messages=[Message.user("Reply with the word ok only.")],
    )

    first = await engine.complete(
        spec,
        cache_policy=CachePolicy.default_response(collection="demo"),
    )
    second = await engine.complete(
        spec,
        cache_policy=CachePolicy.default_response(collection="demo"),
    )

    print(first.content)
    print(second.content)
    print(engine.cache.get_stats().to_dict())


asyncio.run(main())
```

Cache invalidation modes:

- normal read/write behavior:
  `CachePolicy.default_response(...)`
- force a new write without reading the existing record:
  `CachePolicy(enabled=True, collection=\"demo\", invalidation=CacheInvalidationMode.REWRITE)`
- ignore any existing entry and regenerate:
  `CachePolicy(enabled=True, collection=\"demo\", invalidation=CacheInvalidationMode.REGENERATE)`

Use the same pattern for embeddings with `CachePolicy.embeddings(...)`.

## 11. Inspect A Cached Request And Its Usage

Cached responses preserve the normalized response payload, including the usage
shape when the original response had usage data.

```python
from llm_client.cache import CacheInvalidationMode, CachePolicy, request_cache_key

cache_key = request_cache_key(spec, provider="openai")
lookup = await engine.cache.lookup(
    cache_key,
    rewrite_cache=False,
    regen_cache=False,
    collection="demo",
)

print(lookup.hit)
print(lookup.effective_key)
print(lookup.response.get("usage") if lookup.response else None)
```

Use this when you need raw cache inspection rather than just a returned
`CompletionResult`.

## 12. See LLM Cost And Aggregate Usage

Per-request cost is on the normalized result:

```python
print(result.usage.total_cost if result.usage else None)
```

Use the budget/ledger layer when you need aggregated usage across requests:

```python
import asyncio
from decimal import Decimal

from llm_client.budgets import Budget, Ledger
from llm_client.context import ExecutionContext


async def main() -> None:
    ledger = Ledger()
    ctx = ExecutionContext(scope_id="tenant-1", principal_id="user-42")

    ledger.set_budget(
        Budget(
            scope_id="tenant-1",
            principal_id="user-42",
            max_tokens_total=100_000,
            max_cost_total=Decimal("10.00"),
        )
    )

    await ledger.record_usage(ctx, result.usage, provider="openai", model="gpt-5-mini")
    usage = await ledger.get_usage(scope_id="tenant-1", principal_id="user-42")
    print(usage.total_tokens)
    print(usage.total_cost)


asyncio.run(main())
```

Useful ledger methods:

- `set_budget(...)`
- `require_budget(...)`
- `record_usage(...)`
- `record_tool_usage(...)`
- `record_connector_usage(...)`
- `get_usage(...)`
- `list_events(...)`

## 13. Add Observability And Diagnostics

```python
from llm_client.observability import EngineDiagnosticsRecorder, LifecycleRecorder

diagnostics = EngineDiagnosticsRecorder()
lifecycle = LifecycleRecorder()

engine.hooks.add(diagnostics)
engine.hooks.add(lifecycle)
```

Use this when you need:

- request diagnostics
- cache-hit visibility
- fallback visibility
- replay/lifecycle reports

If you need redaction policy, use the redaction helpers under
`llm_client.observability`.

## 14. Use SQL Adaptors Directly

```python
from llm_client.adapters import PostgresSQLAdaptor, SQLMutationRequest, SQLQueryRequest

adaptor = PostgresSQLAdaptor(pool=pg_pool, read_only=True)

query_result = await adaptor.query(
    SQLQueryRequest(
        statement="select id, title from incidents where severity = :severity limit 10",
        parameters={"severity": "critical"},
    )
)

print(query_result.rows)
```

Write operations are explicit:

```python
write_adaptor = PostgresSQLAdaptor(pool=pg_pool, read_only=False)
write_result = await write_adaptor.execute(
    SQLMutationRequest(
        statement="update incidents set owner = :owner where id = :id",
        parameters={"owner": "ops", "id": "inc-1"},
        allow_write=True,
    )
)
```

Safety rules:

- SQL is read-only by default unless the adaptor and request both allow writes
- named parameters are normalized by the adaptor
- timeouts flow through `AdaptorExecutionOptions`

## 15. Wrap Adaptors As Tools

```python
from llm_client.adapters import PostgresSQLAdaptor
from llm_client.adapters.tools import build_sql_query_tool

sql_adaptor = PostgresSQLAdaptor(pool=pg_pool, read_only=True)
sql_tool = build_sql_query_tool(
    sql_adaptor,
    name="incident_query",
    description="Run read-only incident analytics queries.",
)

agent = Agent(
    engine=engine,
    system_message="Use the SQL tool for incident facts. Do not invent DB results.",
    tools=[sql_tool],
)
```

The same pattern exists for:

- Redis: `build_redis_get_tool`, `build_redis_set_tool`, `build_redis_hash_get_tool`, ...
- Vector search: `build_vector_search_tool`, `build_vector_upsert_tool`, `build_vector_delete_tool`

## 16. Use Redis And Vector Adaptors

Redis:

```python
from llm_client.adapters import RedisKVAdaptor, RedisGetRequest, RedisSetRequest

redis_adaptor = RedisKVAdaptor(redis_client, key_prefix="copilot", read_only=False)
await redis_adaptor.set(RedisSetRequest(key="session:123", value="warm", ttl_seconds=300))
value = await redis_adaptor.get(RedisGetRequest(key="session:123"))
```

Qdrant/vector:

```python
from llm_client.adapters import QdrantVectorAdaptor, VectorPoint, VectorSearchRequest, VectorUpsertRequest

vector_adaptor = QdrantVectorAdaptor(qdrant_client)
await vector_adaptor.upsert(
    VectorUpsertRequest(
        collection="knowledge",
        points=[
            VectorPoint(point_id="doc-1", vector=[0.1, 0.2, 0.3], payload={"title": "Runbook"}),
        ],
        create_if_missing=True,
    )
)
matches = await vector_adaptor.search(
    VectorSearchRequest(collection="knowledge", query_vector=[0.1, 0.2, 0.3], limit=5)
)
```

## 17. Recommended Application Patterns

For a small API service:

- `ExecutionEngine`
- one provider
- optional cache
- optional lifecycle recorder

For a production agent backend:

- `ExecutionEngine`
- `Agent`
- `tool` and `ToolRegistry`
- `Ledger`
- observability hooks

For a retrieval-backed copilot:

- `ExecutionEngine`
- `Agent`
- vector adaptor or app-owned retrieval service
- context assembly and memory primitives

For a controlled database agent:

- SQL adaptor
- SQL adaptor tool wrapper
- read-only default
- explicit write opt-in only where necessary

## 18. Real Integration Patterns

The package is designed to support a few recurring integration styles:

- thin provider or engine-backed application services
- tool-calling agents with explicit tool registries and budgets
- retrieval-backed assistants built from adaptors, memory, and context
  assembly
- controlled service access through SQL, Redis, or vector adaptors

The important design point is that the package contract is the reusable kernel.
Applications compose their own product logic around these primitives instead of
depending on repo-local wrapper layers.

## 19. Best Reading Order

If you are new to the package:

1. [llm-client-package-api-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-package-api-guide.md)
2. this guide
3. [llm-client-usage-and-capabilities-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-usage-and-capabilities-guide.md)
4. [llm-client-examples-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-examples-guide.md)
5. the cookbook scripts in `examples/`
