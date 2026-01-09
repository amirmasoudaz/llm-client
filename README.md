# llm-client

Production-ready Python library for LLM interactions with **agent framework**, **tool calling**, **streaming**, and **caching**.

## Features

- **Agent Framework**: Autonomous agents with ReAct loops and multi-turn conversations
- **Tool Calling**: Easy tool definition with `@tool` decorator
- **Provider Abstraction**: Unified interface for LLM providers (OpenAI, with more coming)
- **Conversation Management**: Context window handling with multiple truncation strategies
- **Streaming**: Unified event-based streaming with SSE, WebSocket, and Pusher adapters
- **Caching**: Multiple backends (filesystem, PostgreSQL+Redis, Qdrant)
- **Rate Limiting**: Token bucket implementation respecting API limits
- **Batch Processing**: Concurrent request handling with checkpointing

Auto-loads environment variables from `.env` files on import.

---

## Installation

```bash
pip install -e .
```

**Requirements:**
- Python `>=3.10`
- `OPENAI_API_KEY` environment variable
- Optional backing services for caching (PostgreSQL, Redis, Qdrant)

---

## Quick Start

### Simple Completion

```python
import asyncio
from llm_client import OpenAIProvider

async def main():
    provider = OpenAIProvider(model="gpt-5-nano")
    result = await provider.complete("Hello, world!")
    print(result.content)
    await provider.close()

asyncio.run(main())
```

### Agent with Tools

```python
import asyncio
from llm_client import Agent, OpenAIProvider, tool

@tool
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
async def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

async def main():
    agent = Agent(
        provider=OpenAIProvider(model="gpt-5-nano"),
        tools=[get_weather, search_web],
        system_message="You are a helpful assistant with access to tools.",
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result.content)
    print(f"Tool calls made: {[tc.name for tc in result.all_tool_calls]}")

asyncio.run(main())
```

### Streaming

```python
import asyncio
from llm_client import OpenAIProvider, StreamEventType

async def main():
    provider = OpenAIProvider(model="gpt-5-nano")
    
    async for event in provider.stream("Write a haiku about coding"):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.DONE:
            print(f"\nTokens used: {event.data.usage.total_tokens}")
    
    await provider.close()

asyncio.run(main())
```

### Backward Compatible API

The original `OpenAIClient` still works:

```python
from llm_client import OpenAIClient

client = OpenAIClient(model="gpt-5-nano", cache_backend=None)
response = await client.get_response(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["output"])
```

---

## Core Concepts

### Providers

The `Provider` protocol defines a unified interface for LLM backends:

```python
from llm_client import OpenAIProvider, CompletionResult

provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="fs",
    cache_dir="./cache",
)

# Completion
result: CompletionResult = await provider.complete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
)

# Streaming
async for event in provider.stream("Tell me a story"):
    ...

# Embeddings (with embedding model)
embed_provider = OpenAIProvider(model="text-embedding-3-small")
result = await embed_provider.embed("Text to embed")
```

### Tools

Define tools using the `@tool` decorator:

```python
from llm_client import tool, sync_tool, ToolRegistry

@tool
async def calculate(expression: str) -> str:
    """Evaluate a math expression.
    
    Args:
        expression: Math expression like "2 + 2"
    """
    return str(eval(expression))

@sync_tool  # For synchronous functions
def get_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().isoformat()

# Register tools
registry = ToolRegistry([calculate, get_time])

# Execute a tool
result = await registry.execute("calculate", '{"expression": "2 + 2"}')
print(result.content)  # "4"
```

### Conversations

Manage multi-turn conversations with context window handling:

```python
from llm_client import Conversation, GPT5Nano

conv = Conversation(
    system_message="You are helpful.",
    max_tokens=4000,
    truncation_strategy="sliding",  # or "drop_oldest", "drop_middle"
)

conv.add_user("What is Python?")
conv.add_assistant("Python is a programming language...")
conv.add_user("Show me an example")

# Get messages for API (auto-truncated if needed)
messages = conv.get_messages(model=GPT5Nano)

# Fork conversation for branching
forked = conv.fork()

# Save/load
conv.save("conversation.json")
loaded = Conversation.load("conversation.json")
```

### Agents

Agents orchestrate providers, tools, and conversations:

```python
from llm_client import Agent, OpenAIProvider, AgentConfig

agent = Agent(
    provider=OpenAIProvider(model="gpt-5"),
    tools=[...],
    system_message="You are a research assistant.",
    config=AgentConfig(
        max_turns=10,
        parallel_tool_execution=True,
        tool_timeout=30.0,
    ),
)

# Run to completion
result = await agent.run("Research quantum computing")
print(result.content)
print(f"Turns: {result.num_turns}")
print(f"Status: {result.status}")

# Stream with tool execution
async for event in agent.stream("Search for Python tutorials"):
    if event.type == StreamEventType.TOKEN:
        print(event.data, end="")
    elif event.type == StreamEventType.META:
        if event.data.get("event") == "tool_result":
            print(f"\n[Tool: {event.data['tool_name']}]")

# Continue conversation
result2 = await agent.run("Now search for JavaScript")

# Reset for new conversation
agent.reset()
```

### Streaming Adapters

Transform streams for different output formats:

```python
from llm_client import (
    SSEAdapter,
    CallbackAdapter, 
    BufferingAdapter,
    collect_stream,
)

# SSE for web servers
adapter = SSEAdapter()
async for sse_string in adapter.transform(provider.stream(prompt)):
    yield sse_string  # Send to HTTP response

# Callbacks for custom handling
adapter = CallbackAdapter(
    on_token=lambda t: print(t, end=""),
    on_done=lambda r: print(f"\nDone: {r.content}"),
)
await adapter.consume(provider.stream(prompt))

# Buffering for full response
buffer = BufferingAdapter()
async for event in buffer.wrap(provider.stream(prompt)):
    pass  # Process events
print(buffer.content)  # Full accumulated text

# Utility: collect to CompletionResult
result = await collect_stream(provider.stream(prompt))
```

---

## Model Profiles

Available models:

| Key | Model | Category |
|-----|-------|----------|
| `gpt-5` | GPT-5 | completions |
| `gpt-5-mini` | GPT-5 Mini | completions |
| `gpt-5-nano` | GPT-5 Nano | completions |
| `gpt-5.1` | GPT-5.1 | completions |
| `gpt-5.2` | GPT-5.2 | completions |
| `text-embedding-3-large` | Text Embedding 3 Large | embeddings |
| `text-embedding-3-small` | Text Embedding 3 Small | embeddings |

All completion models support reasoning with configurable effort levels.

---

## Caching

```python
# Filesystem
provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="fs",
    cache_dir="./cache",
)

# PostgreSQL + Redis (production)
provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="pg_redis",
    pg_dsn="postgresql://...",
    redis_url="redis://...",
)

# Qdrant
provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="qdrant",
    qdrant_url="http://localhost:6333",
)

# With caching enabled
result = await provider.complete(
    "Hello",
    cache_response=True,
    cache_collection="my_cache",
)
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `PG_DSN` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection URL |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `PUSHER_*` | Pusher streaming credentials |

---

## Examples

See the `examples/` directory:

- `agent_with_tools.py` - Agent with multiple tools and streaming
- `provider_streaming.py` - Streaming with different adapters
- `conversation_management.py` - Conversation handling and persistence
- `basic_completions_fs_cache.py` - Simple completions with caching
- `streaming_sse.py` - SSE streaming for web apps

---

## Architecture

```
llm_client/
├── providers/          # Provider abstraction
│   ├── base.py        # Protocol and base class
│   ├── openai.py      # OpenAI implementation
│   └── types.py       # Message, CompletionResult, StreamEvent
├── tools/             # Tool system
│   ├── base.py        # Tool, ToolRegistry, ToolResult
│   └── decorators.py  # @tool, @sync_tool
├── agent.py           # Agent orchestrator
├── conversation.py    # Conversation management
├── streaming.py       # Stream adapters
├── models.py          # ModelProfile registry
├── cache.py           # Cache backends
├── rate_limit.py      # Rate limiting
├── batch_req.py       # Batch processing
└── client.py          # Backward-compatible OpenAIClient
```

---

## Migration from v1

The original `OpenAIClient` API is fully preserved. New code should use:

```python
# Old (still works)
from llm_client import OpenAIClient
client = OpenAIClient(model="gpt-5")
result = await client.get_response(messages=[...])

# New (recommended for agents)
from llm_client import Agent, OpenAIProvider
agent = Agent(provider=OpenAIProvider(model="gpt-5"), tools=[...])
result = await agent.run("prompt")

# New (recommended for simple completions)
from llm_client import OpenAIProvider
provider = OpenAIProvider(model="gpt-5")
result = await provider.complete("prompt")
```
