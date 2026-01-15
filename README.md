# llm-client

Production-ready Python library for LLM interactions with **agent framework**, **tool calling**, **streaming**, and **multi-provider support**.

## Features

- **Provider Abstraction**: Unified interface for OpenAI, Anthropic (Claude), with extensible design
- **Agent Framework**: Autonomous agents with ReAct loops, multi-turn conversations, and parallel tool execution
- **Tool Calling**: Easy tool definition with `@tool` decorator and automatic schema inference
- **Conversation Management**: Context window handling with multiple truncation strategies
- **Streaming**: Unified event-based streaming with SSE, callback, and buffering adapters
- **Session Persistence**: Save and restore agent sessions with conversation history
- **Caching**: Multiple backends (filesystem, PostgreSQL+Redis, Qdrant)
- **Rate Limiting**: Token bucket implementation respecting API limits
- **Retry Logic**: Automatic retry with exponential backoff for transient errors
- **Batch Processing**: Concurrent request handling with checkpointing

Auto-loads environment variables from `.env` files on import.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Providers](#providers)
  - [OpenAI Provider](#openai-provider)
  - [Anthropic Provider](#anthropic-provider)
  - [Provider Protocol](#provider-protocol)
- [Agent Framework](#agent-framework)
  - [Creating Agents](#creating-agents)
  - [Agent Configuration](#agent-configuration)
  - [Running Agents](#running-agents)
  - [Streaming Agents](#streaming-agents)
  - [Session Persistence](#session-persistence)
- [Tool System](#tool-system)
  - [Tool Decorator](#tool-decorator)
  - [Tool Registry](#tool-registry)
  - [Manual Tool Definition](#manual-tool-definition)
- [Conversation Management](#conversation-management)
  - [Basic Usage](#basic-conversation-usage)
  - [Context Window Management](#context-window-management)
  - [Serialization](#conversation-serialization)
- [Streaming](#streaming)
  - [Stream Events](#stream-events)
  - [SSE Adapter](#sse-adapter)
  - [Callback Adapter](#callback-adapter)
  - [Buffering Adapter](#buffering-adapter)
- [Caching](#caching)
- [Rate Limiting](#rate-limiting)
- [Batch Processing](#batch-processing)
- [Model Profiles](#model-profiles)
- [Types Reference](#types-reference)
- [Complete API Reference](#complete-api-reference)
- [Examples](#examples)
- [Migration from v1](#migration-from-v1)

---

## Installation

```bash
# Basic installation
pip install -e .

# With Anthropic support
pip install -e ".[anthropic]"

# All optional dependencies
pip install -e ".[all]"
```

**Requirements:**
- Python `>=3.10`
- `OPENAI_API_KEY` environment variable (for OpenAI)
- `ANTHROPIC_API_KEY` environment variable (for Anthropic)
- Optional: PostgreSQL, Redis, Qdrant for caching

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
    print(f"Tokens: {result.usage.total_tokens}")
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
    """Search the web for information."""
    return f"Results for: {query}"

async def main():
    agent = Agent(
        provider=OpenAIProvider(model="gpt-5-nano"),
        tools=[get_weather, search_web],
        system_message="You are a helpful assistant with access to tools.",
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result.content)
    print(f"Tool calls: {[tc.name for tc in result.all_tool_calls]}")

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
        elif event.type == StreamEventType.REASONING:
            print(f"[Thinking: {event.data}]", end="")
        elif event.type == StreamEventType.DONE:
            print(f"\n\nTokens: {event.data.usage.total_tokens}")
    
    await provider.close()

asyncio.run(main())
```

### Using Anthropic (Claude)

```python
import asyncio
from llm_client import AnthropicProvider, ANTHROPIC_AVAILABLE

async def main():
    if not ANTHROPIC_AVAILABLE:
        print("Install with: pip install llm-client[anthropic]")
        return
    
    provider = AnthropicProvider(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
    )
    
    result = await provider.complete("Explain quantum entanglement briefly.")
    print(result.content)

asyncio.run(main())
```

---

## Providers

### OpenAI Provider

The `OpenAIProvider` implements the full Provider protocol for OpenAI's API.

```python
from llm_client import OpenAIProvider

provider = OpenAIProvider(
    # Model selection
    model="gpt-5",              # Model name or ModelProfile class
    
    # API configuration
    api_key=None,               # Defaults to OPENAI_API_KEY env var
    base_url=None,              # Custom API endpoint
    organization=None,          # OpenAI organization ID
    
    # Cache configuration
    cache_backend="fs",         # "fs", "pg_redis", "qdrant", or None
    cache_dir="./cache",        # For filesystem cache
    cache_collection=None,      # Cache collection name
    
    # PostgreSQL + Redis cache
    pg_dsn=None,                # PostgreSQL connection string
    redis_url=None,             # Redis URL
    redis_ttl_seconds=86400,    # Cache TTL (24 hours)
    compress_pg=True,           # Compress PostgreSQL entries
    
    # Qdrant cache
    qdrant_url=None,            # Qdrant server URL
    qdrant_api_key=None,        # Qdrant API key
    
    # Advanced
    use_responses_api=False,    # Use OpenAI Responses API
)
```

#### Completion

```python
result = await provider.complete(
    messages="Hello!",                    # str, dict, Message, or list
    tools=None,                           # List of Tool objects
    tool_choice=None,                     # "auto", "none", "required", or tool name
    temperature=0.7,                      # Sampling temperature
    max_tokens=1000,                      # Maximum output tokens
    response_format="text",               # "text", "json_object", or JSON schema
    
    # Reasoning models (GPT-5, o1)
    reasoning_effort="medium",            # "low", "medium", "high"
    
    # Caching
    cache_response=True,                  # Enable response caching
    cache_collection="my_cache",          # Cache collection name
    rewrite_cache=False,                  # Overwrite existing cache
    regen_cache=False,                    # Ignore cache, regenerate
    
    # Retry logic
    attempts=3,                           # Retry attempts
    backoff=1.0,                          # Initial backoff (seconds)
)

# Result properties
result.content          # str: Response text
result.tool_calls       # List[ToolCall]: Tool calls made
result.usage            # Usage: Token counts and costs
result.reasoning        # str: Reasoning content (if model supports)
result.model            # str: Model name
result.finish_reason    # str: Why generation stopped
result.status           # int: HTTP status code
result.error            # str: Error message if failed
result.ok               # bool: True if successful
result.has_tool_calls   # bool: True if tool calls present
```

#### Streaming

```python
async for event in provider.stream(
    messages="Tell me a story",
    tools=None,
    tool_choice=None,
    temperature=0.7,
    max_tokens=1000,
    reasoning_effort=None,
):
    match event.type:
        case StreamEventType.TOKEN:
            print(event.data, end="")           # str: Token text
        case StreamEventType.REASONING:
            print(f"[Think: {event.data}]")     # str: Reasoning token
        case StreamEventType.TOOL_CALL_START:
            print(f"Calling: {event.data.name}") # ToolCallDelta
        case StreamEventType.TOOL_CALL_DELTA:
            pass                                 # ToolCallDelta with args
        case StreamEventType.TOOL_CALL_END:
            print(f"Tool: {event.data}")        # ToolCall (complete)
        case StreamEventType.USAGE:
            print(f"Tokens: {event.data}")      # Usage
        case StreamEventType.META:
            print(f"Meta: {event.data}")        # dict
        case StreamEventType.DONE:
            print(f"Result: {event.data}")      # CompletionResult
        case StreamEventType.ERROR:
            print(f"Error: {event.data}")       # dict with status, error
```

#### Embeddings

```python
# Use an embedding model
embed_provider = OpenAIProvider(model="text-embedding-3-large")

result = await embed_provider.embed(
    inputs="Text to embed",           # str or List[str]
    encoding_format="base64",         # "float" or "base64"
    dimensions=1536,                  # Output dimensions (if supported)
)

result.embeddings      # List[List[float]]: Embedding vectors
result.embedding       # List[float]: First embedding (convenience)
result.usage           # Usage: Token counts
result.ok              # bool: Success status
```

### Anthropic Provider

The `AnthropicProvider` implements the Provider protocol for Anthropic's Claude API.

```python
from llm_client import AnthropicProvider, ANTHROPIC_AVAILABLE

# Check if anthropic package is installed
if not ANTHROPIC_AVAILABLE:
    raise ImportError("Install with: pip install llm-client[anthropic]")

provider = AnthropicProvider(
    model="claude-3-5-sonnet-20241022",  # Claude model name
    api_key=None,                         # Defaults to ANTHROPIC_API_KEY
    base_url=None,                        # Custom API endpoint
    max_tokens=4096,                      # Default max tokens (required by Anthropic)
    default_temperature=None,             # Default temperature
)
```

#### Completion

```python
result = await provider.complete(
    messages="Explain recursion",
    tools=None,                    # Tools use Anthropic's tool_use format
    tool_choice=None,              # "auto", "any", or specific tool name
    temperature=0.7,
    max_tokens=1000,
)
```

#### Streaming

```python
async for event in provider.stream("Write a poem"):
    if event.type == StreamEventType.TOKEN:
        print(event.data, end="")
    elif event.type == StreamEventType.REASONING:
        # Extended thinking (for supported models)
        print(f"[Thinking: {event.data}]")
```

**Note:** Anthropic does not support embeddings. Use OpenAI or a dedicated embedding service.

### Provider Protocol

Create custom providers by implementing the `Provider` protocol:

```python
from llm_client.providers import Provider, BaseProvider, CompletionResult, StreamEvent

class MyProvider(BaseProvider):
    async def complete(self, messages, **kwargs) -> CompletionResult:
        # Implement completion logic
        ...
    
    async def stream(self, messages, **kwargs) -> AsyncIterator[StreamEvent]:
        # Implement streaming logic
        yield StreamEvent(type=StreamEventType.TOKEN, data="Hello")
        ...
    
    async def embed(self, inputs, **kwargs) -> EmbeddingResult:
        # Optional: implement embeddings
        raise NotImplementedError("Embeddings not supported")
```

---

## Agent Framework

### Creating Agents

```python
from llm_client import Agent, OpenAIProvider, AgentConfig

agent = Agent(
    # Required
    provider=OpenAIProvider(model="gpt-5"),
    
    # Optional
    tools=[tool1, tool2],            # List[Tool] or ToolRegistry
    system_message="You are helpful.", # System prompt
    conversation=None,                # Existing Conversation to continue
    config=None,                      # AgentConfig for advanced options
    
    # Shortcuts (override config)
    max_turns=10,                     # Maximum reasoning turns
    max_tokens=None,                  # Context window limit
)
```

### Agent Configuration

```python
from llm_client import AgentConfig

config = AgentConfig(
    # Turn limits
    max_turns=10,                          # Max reasoning turns before stopping
    max_tool_calls_per_turn=10,            # Max tools per turn
    
    # Tool execution
    parallel_tool_execution=True,          # Execute tools in parallel
    tool_timeout=30.0,                     # Timeout per tool (seconds)
    
    # Context management
    max_tokens=None,                       # Max context tokens
    reserve_tokens=2000,                   # Reserve for response
    
    # Behavior
    stop_on_tool_error=False,              # Stop on tool failure
    include_tool_errors_in_context=True,   # Include errors in conversation
    stream_tool_calls=True,                # Stream tool call events
)
```

### Running Agents

```python
# Run to completion
result = await agent.run(
    prompt="Research quantum computing",
    max_turns=5,                    # Override config for this run
)

# Result properties
result.content          # str: Final response
result.turns            # List[TurnResult]: All turns
result.num_turns        # int: Number of turns taken
result.all_tool_calls   # List[ToolCall]: All tool calls across turns
result.conversation     # Conversation: Full conversation history
result.total_usage      # Usage: Aggregated token usage
result.status           # "success", "max_turns", or "error"
result.error            # str: Error message if failed

# Simple chat interface
response = await agent.chat("Hello!")  # Returns just the text

# Continue conversation
result2 = await agent.run("Tell me more")

# Reset conversation (keeps system message)
agent.reset()

# Fork agent (copy with new conversation)
forked_agent = agent.fork()
```

### Streaming Agents

```python
async for event in agent.stream("Search for Python tutorials"):
    match event.type:
        case StreamEventType.META:
            if event.data.get("event") == "turn_start":
                print(f"\n--- Turn {event.data['turn']} ---")
            elif event.data.get("event") == "tool_result":
                print(f"\n[Tool: {event.data['tool_name']}]")
                print(f"Result: {event.data['content'][:100]}...")
        
        case StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        
        case StreamEventType.TOOL_CALL_START:
            print(f"\n> Calling {event.data.name}...")
        
        case StreamEventType.DONE:
            agent_result = event.data
            print(f"\n\nStatus: {agent_result.status}")
            print(f"Turns: {agent_result.num_turns}")
```

### Session Persistence

Save and restore agent sessions:

```python
# Save session (conversation + config + tool names)
agent.save_session("session.json")

# Load session
loaded_agent = Agent.load_session(
    "session.json",
    provider=OpenAIProvider(model="gpt-5"),
    tools=[my_tool],  # Must provide tools again
)

# Continue where you left off
result = await loaded_agent.run("Continue our conversation")
```

---

## Tool System

### Tool Decorator

The `@tool` decorator automatically infers JSON schema from type hints and docstrings:

```python
from llm_client import tool, sync_tool

@tool
async def search_database(
    query: str,
    limit: int = 10,
    include_metadata: bool = False,
) -> str:
    """Search the database for matching records.
    
    Args:
        query: Search query string
        limit: Maximum results to return
        include_metadata: Whether to include metadata in results
    
    Returns:
        JSON string of matching records
    """
    # Implementation
    results = await db.search(query, limit=limit)
    return json.dumps(results)

# For synchronous functions
@sync_tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a timezone.
    
    Args:
        timezone: IANA timezone name (e.g., "America/New_York")
    """
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now(tz).isoformat()
```

**Generated Schema:**
```json
{
  "name": "search_database",
  "description": "Search the database for matching records.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query string"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum results to return",
        "default": 10
      },
      "include_metadata": {
        "type": "boolean",
        "description": "Whether to include metadata in results",
        "default": false
      }
    },
    "required": ["query"]
  }
}
```

### Tool Registry

```python
from llm_client import ToolRegistry, Tool

# Create registry with tools
registry = ToolRegistry([tool1, tool2, tool3])

# Or add tools individually
registry = ToolRegistry()
registry.register(tool1)
registry.register(tool2)

# Access tools
registry.tools           # List[Tool]: All registered tools
registry.names           # List[str]: Tool names
registry.get("search")   # Tool: Get by name

# Execute a tool
result = await registry.execute(
    name="search_database",
    arguments='{"query": "python", "limit": 5}',  # JSON string
)

# Result properties
result.content     # Any: Tool return value
result.success     # bool: Whether execution succeeded
result.error       # str: Error message if failed
result.to_string() # str: String representation for LLM

# Remove a tool
registry.unregister("tool_name")
```

### Manual Tool Definition

```python
from llm_client import Tool

# Create tool manually
tool = Tool(
    name="calculate",
    description="Evaluate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate"
            }
        },
        "required": ["expression"]
    },
    handler=my_async_function,  # async callable
)

# Or from a function (supports both sync and async)
from llm_client import tool_from_function

# Async function
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    return await httpx.get(url)

# Sync function (automatically wrapped to run in executor)
def calculate(x: float, y: float) -> float:
    """Calculate x + y."""
    return x + y

fetch_tool = tool_from_function(fetch_data)
calc_tool = tool_from_function(calculate)
```

---

## Conversation Management

### Basic Conversation Usage

```python
from llm_client import Conversation, Message, Role

# Create conversation
conv = Conversation(
    system_message="You are a helpful assistant.",
    max_tokens=8000,           # Context limit (optional)
    truncation_strategy="sliding",
)

# Add messages
conv.add_user("What is Python?")
conv.add_assistant("Python is a programming language...")
conv.add_user("Show me an example")

# Add message with tool calls
from llm_client import ToolCall
conv.add_assistant_with_tools(
    content="Let me search for that.",
    tool_calls=[
        ToolCall(id="call_1", name="search", arguments='{"q": "python"}')
    ]
)

# Add tool result
conv.add_tool_result(
    tool_call_id="call_1",
    content="Found 10 results...",
    name="search"
)

# Access messages
len(conv)                      # Number of messages
conv[0]                        # First message
conv.get_last_message()        # Most recent
conv.get_last_user_message()   # Most recent user message
conv.get_last_assistant_message()  # Most recent assistant

# Get messages for API call
messages = conv.get_messages()           # List[Message]
messages_dict = conv.get_messages_dict() # List[dict] for API
```

### Context Window Management

```python
from llm_client import Conversation, GPT5

conv = Conversation(
    system_message="You are helpful.",
    max_tokens=8000,
    truncation_strategy="sliding",  # Keep most recent
    reserve_tokens=2000,            # Reserve for response
)

# Truncation strategies:
# - "sliding": Keep most recent messages that fit
# - "drop_oldest": Remove oldest messages first
# - "drop_middle": Keep first and last, remove middle

# Get messages with automatic truncation
messages = conv.get_messages(model=GPT5)  # Truncated to fit

# Count tokens
token_count = conv.count_tokens(GPT5)
```

### Conversation Serialization

```python
# To/from dictionary
data = conv.to_dict()
conv = Conversation.from_dict(data)

# To/from JSON string
json_str = conv.to_json()
conv = Conversation.from_json(json_str)

# Save/load from file
conv.save("conversation.json")
conv = Conversation.load("conversation.json")

# Fork (copy with new session ID)
forked = conv.fork()

# Branch from specific point
branched = conv.branch(from_index=3)  # Keep first 3 messages

# Clear messages (keeps system message)
conv.clear()

# Format as readable string
print(conv.format_history(max_messages=10))
```

---

## Streaming

### Stream Events

```python
from llm_client import StreamEvent, StreamEventType

# Event types
StreamEventType.TOKEN           # Content token (data: str)
StreamEventType.REASONING       # Reasoning token (data: str)
StreamEventType.TOOL_CALL_START # Tool call starting (data: ToolCallDelta)
StreamEventType.TOOL_CALL_DELTA # Tool arguments chunk (data: ToolCallDelta)
StreamEventType.TOOL_CALL_END   # Tool call complete (data: ToolCall)
StreamEventType.META            # Metadata (data: dict)
StreamEventType.USAGE           # Token usage (data: Usage)
StreamEventType.DONE            # Stream complete (data: CompletionResult)
StreamEventType.ERROR           # Error occurred (data: dict)

# Event properties
event.type       # StreamEventType
event.data       # Any: Event-specific data
event.timestamp  # float: Unix timestamp

# Convert to SSE format
sse_string = event.to_sse()  # "event: token\ndata: Hello\n\n"
```

### SSE Adapter

Transform stream events to Server-Sent Events format:

```python
from llm_client import SSEAdapter

adapter = SSEAdapter()

# For web frameworks (FastAPI, Starlette, etc.)
async def stream_response():
    async for sse_line in adapter.transform(provider.stream(prompt)):
        yield sse_line

# FastAPI example
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_endpoint(prompt: str):
    return StreamingResponse(
        adapter.transform(provider.stream(prompt)),
        media_type="text/event-stream"
    )
```

### Callback Adapter

Process stream events with callbacks:

```python
from llm_client import CallbackAdapter

adapter = CallbackAdapter(
    on_token=lambda t: print(t, end=""),
    on_reasoning=lambda r: print(f"[Think: {r}]"),
    on_tool_start=lambda tc: print(f"\nCalling: {tc.name}"),
    on_tool_end=lambda tc: print(f"Tool done: {tc.name}"),
    on_done=lambda result: print(f"\n\nDone! {result.usage}"),
    on_error=lambda err: print(f"Error: {err}"),
)

# Consume stream (blocks until complete)
await adapter.consume(provider.stream(prompt))
```

### Buffering Adapter

Accumulate stream into final result:

```python
from llm_client import BufferingAdapter

buffer = BufferingAdapter()

# Wrap stream (yields events while buffering)
async for event in buffer.wrap(provider.stream(prompt)):
    if event.type == StreamEventType.TOKEN:
        print(event.data, end="")

# Access accumulated data
print(buffer.content)      # Full text
print(buffer.reasoning)    # Full reasoning (if any)
print(buffer.tool_calls)   # List[ToolCall]
print(buffer.usage)        # Usage

# Get as CompletionResult
result = buffer.get_result()
```

### Utility Functions

```python
from llm_client import collect_stream, stream_to_string, format_sse_event

# Collect stream to CompletionResult
result = await collect_stream(provider.stream(prompt))

# Collect stream to string
text = await stream_to_string(provider.stream(prompt))

# Format single event as SSE
sse = format_sse_event("token", "Hello")  # "event: token\ndata: Hello\n\n"
```

---

## Caching

### Filesystem Cache

```python
provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="fs",
    cache_dir="./cache",
)

result = await provider.complete(
    "Hello",
    cache_response=True,
    cache_collection="my_collection",
)
```

### PostgreSQL + Redis Cache

Production-ready distributed cache:

```python
provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="pg_redis",
    pg_dsn="postgresql://user:pass@localhost/db",
    redis_url="redis://localhost:6379",
    redis_ttl_seconds=86400,  # 24 hours
    compress_pg=True,
)
```

### Qdrant Cache

Vector-based semantic cache:

```python
provider = OpenAIProvider(
    model="gpt-5",
    cache_backend="qdrant",
    qdrant_url="http://localhost:6333",
    qdrant_api_key="your-api-key",
    cache_collection="llm_cache",
)
```

### Cache Options

```python
result = await provider.complete(
    "Hello",
    cache_response=True,       # Enable caching
    cache_collection="custom", # Collection/table name
    rewrite_cache=False,       # Overwrite existing entry
    regen_cache=False,         # Ignore cache, regenerate
)

# Warm cache (for backends that support it)
await provider.warm_cache()
```

---

## Rate Limiting

Built-in token bucket rate limiting:

```python
from llm_client import Limiter, TokenBucket

# Automatic with providers (uses model limits)
provider = OpenAIProvider(model="gpt-5")
# Limiter is automatically configured based on model profile

# Manual rate limiting
limiter = Limiter(model_profile=GPT5)

async with limiter.limit(tokens=1000, requests=1) as ctx:
    # Make API call
    result = await client.completions.create(...)
    ctx.output_tokens = result.usage.completion_tokens

# Direct token bucket usage
bucket = TokenBucket(
    capacity=10000,      # Max tokens
    refill_rate=1000,    # Tokens per second
)

await bucket.acquire(500)  # Acquire 500 tokens
```

---

## Batch Processing

Process many requests concurrently with checkpointing using a worker pool pattern:

```python
from llm_client import BatchManager, OpenAIClient

client = OpenAIClient(model="gpt-5-nano")

# Initialize manager
manager = BatchManager(
    max_workers=10,
    checkpoint_file="batch_results.jsonl",
)

# Define requests
requests = [
    {"messages": [{"role": "user", "content": f"Question {i}"}]}
    for i in range(1000)
]

# Define processor function
async def processor(req):
    return await client.get_response(**req)

# Process batch (auto-resumes if checkpoint_file exists)
results = await manager.process_batch(
    items=requests,
    processor=processor,
    desc="Processing Batch"
)
```

---

## Model Profiles

Available models and their configurations:

| Key | Class | Category | Context | Reasoning |
|-----|-------|----------|---------|-----------|
| `gpt-5` | `GPT5` | completions | 128K | ✓ |
| `gpt-5-mini` | `GPT5Mini` | completions | 128K | ✓ |
| `gpt-5-nano` | `GPT5Nano` | completions | 128K | ✓ |
| `gpt-5.1` | `GPT5Point1` | completions | 128K | ✓ |
| `gpt-5.2` | `GPT5Point2` | completions | 128K | ✓ |
| `text-embedding-3-large` | `TextEmbedding3Large` | embeddings | 8K | ✗ |
| `text-embedding-3-small` | `TextEmbedding3Small` | embeddings | 8K | ✗ |

```python
from llm_client import ModelProfile, GPT5, GPT5Mini

# Get model by key
model = ModelProfile.get("gpt-5")

# Use model class directly
provider = OpenAIProvider(model=GPT5)

# Model properties
GPT5.model_name           # "gpt-5"
GPT5.category             # "completions"
GPT5.context_window       # 128000
GPT5.reasoning_model      # True
GPT5.reasoning_efforts    # ["low", "medium", "high"]

# Token counting
count = GPT5.count_tokens("Hello, world!")

# Usage parsing
usage = GPT5.parse_usage(api_response.usage)
```

---

## Types Reference

### Message

```python
from llm_client import Message, Role

# Create messages
msg = Message.user("Hello")
msg = Message.assistant("Hi there!")
msg = Message.system("You are helpful.")
msg = Message.tool_result(tool_call_id="123", content="Result", name="search")

# From dictionary
msg = Message.from_dict({"role": "user", "content": "Hello"})

# To dictionary
d = msg.to_dict()

# Properties
msg.role           # Role enum
msg.content        # str
msg.tool_calls     # List[ToolCall] or None
msg.tool_call_id   # str (for tool results)
msg.name           # str (optional name)
```

### ToolCall

```python
from llm_client import ToolCall

tc = ToolCall(
    id="call_abc123",
    name="search",
    arguments='{"query": "python"}',  # JSON string
)

# Parse arguments
args = tc.parse_arguments()  # {"query": "python"}
```

### Usage

```python
from llm_client import Usage

usage = Usage(
    input_tokens=100,
    output_tokens=50,
    total_tokens=150,
    input_tokens_cached=20,
    input_cost=0.001,
    output_cost=0.002,
    total_cost=0.003,
)

# To/from dict
d = usage.to_dict()
usage = Usage.from_dict(d)
```

### CompletionResult

```python
from llm_client import CompletionResult

result = CompletionResult(
    content="Hello!",
    tool_calls=None,
    usage=Usage(...),
    reasoning=None,
    model="gpt-5",
    finish_reason="stop",
    status=200,
    error=None,
)

result.ok              # True if status == 200 and no error
result.has_tool_calls  # True if tool_calls is not empty
result.to_message()    # Convert to Message
result.to_dict()       # Serialize to dict
```

### StreamEvent

```python
from llm_client import StreamEvent, StreamEventType

event = StreamEvent(
    type=StreamEventType.TOKEN,
    data="Hello",
    timestamp=time.time(),
)

event.to_sse()  # "event: token\ndata: Hello\n\n"
```

---

## Complete API Reference

### Top-Level Exports

```python
from llm_client import (
    # Providers
    Provider,              # Protocol
    BaseProvider,          # Abstract base class
    OpenAIProvider,        # OpenAI implementation
    AnthropicProvider,     # Anthropic implementation
    ANTHROPIC_AVAILABLE,   # bool: True if anthropic installed
    
    # Agent
    Agent,
    AgentConfig,
    AgentResult,
    TurnResult,
    quick_agent,           # One-shot agent function
    
    # Conversation
    Conversation,
    ConversationConfig,
    
    # Tools
    Tool,
    ToolResult,
    ToolRegistry,
    tool,                  # Async tool decorator
    sync_tool,             # Sync tool decorator
    tool_from_function,    # Create Tool from function
    
    # Types
    Message,
    Role,
    ToolCall,
    ToolCallDelta,
    Usage,
    CompletionResult,
    EmbeddingResult,
    StreamEvent,
    StreamEventType,
    MessageInput,          # Type alias
    normalize_messages,    # Utility function
    
    # Streaming
    SSEAdapter,
    CallbackAdapter,
    BufferingAdapter,
    PusherStreamer,
    format_sse_event,
    collect_stream,
    stream_to_string,
    
    # Models
    ModelProfile,
    GPT5,
    GPT5Mini,
    GPT5Nano,
    GPT5Point1,
    GPT5Point2,
    TextEmbedding3Large,
    TextEmbedding3Small,
    
    # Caching
    FSCache,
    QdrantCache,
    HybridRedisPostgreSQLCache,
    
    # Rate Limiting
    Limiter,
    TokenBucket,
    
    # Batch Processing
    BatchManager,
    RequestManager,        # Alias for BatchManager
    
    # Exceptions
    ResponseTimeoutError,
    
    # Backward Compatible
    OpenAIClient,
)
```

---

## Examples

See the `examples/` directory for comprehensive examples:

| File | Description |
|------|-------------|
| **Provider & Completion** | |
| `simple_generation.py` | Basic completion, input formats, caching, retry logic |
| `provider_streaming.py` | Streaming with SSE, callback, and buffering adapters |
| `sse_streaming.py` | Server-Sent Events for web servers |
| `embeddings_provider.py` | Embeddings, similarity search, batch embedding |
| **Agent & Tools** | |
| `agent_with_tools.py` | Agent with tools, multi-turn, streaming agent |
| `anthropic_agent.py` | Anthropic Claude provider, session persistence |
| `tool_registry.py` | Tool creation, registry management, execution, and provider integration |
| **Conversation & Output** | |
| `conversation_management.py` | Conversation handling, truncation, persistence |
| `json_structured_output.py` | JSON mode, Pydantic structured output |
| `reasoning_models.py` | Reasoning models with effort levels |
| **Infrastructure** | |
| `batch_processing.py` | BatchManager, checkpointing, concurrent requests |
| `rate_limiting.py` | Token bucket, Limiter, rate control |
| `basic_completions_fs_cache.py` | Filesystem caching (legacy client) |
| `embeddings_with_cache.py` | Embeddings with caching (legacy client) |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Anthropic |
| `PG_DSN` | PostgreSQL connection string | For pg_redis cache |
| `REDIS_URL` | Redis connection URL | For pg_redis cache |
| `QDRANT_URL` | Qdrant server URL | For qdrant cache |
| `QDRANT_API_KEY` | Qdrant API key | For qdrant cache |
| `PUSHER_APP_ID` | Pusher app ID | For Pusher streaming |
| `PUSHER_KEY` | Pusher key | For Pusher streaming |
| `PUSHER_SECRET` | Pusher secret | For Pusher streaming |
| `PUSHER_CLUSTER` | Pusher cluster | For Pusher streaming |

---

## Architecture

```
llm_client/
├── providers/              # Provider abstraction layer
│   ├── base.py            # Provider protocol & BaseProvider
│   ├── openai.py          # OpenAI implementation
│   ├── anthropic.py       # Anthropic (Claude) implementation
│   └── types.py           # Message, CompletionResult, StreamEvent, etc.
├── tools/                  # Tool system
│   ├── base.py            # Tool, ToolRegistry, ToolResult
│   └── decorators.py      # @tool, @sync_tool decorators
├── agent.py               # Agent orchestrator
├── conversation.py        # Conversation management
├── streaming.py           # Stream adapters (SSE, Callback, Buffer)
├── models.py              # ModelProfile registry
├── cache.py               # Cache backends (FS, PostgreSQL, Qdrant)
├── rate_limit.py          # Rate limiting (TokenBucket, Limiter)
├── batch_req.py           # Batch processing with checkpointing
├── exceptions.py          # Custom exceptions
└── client.py              # Backward-compatible OpenAIClient
```

---

## Migration from v1

The original `OpenAIClient` API is fully preserved for backward compatibility:

```python
# Old API (still works)
from llm_client import OpenAIClient

client = OpenAIClient(model="gpt-5")
result = await client.get_response(
    messages=[{"role": "user", "content": "Hello"}]
)
print(result["output"])

# New API (recommended for new code)
from llm_client import OpenAIProvider

provider = OpenAIProvider(model="gpt-5")
result = await provider.complete("Hello")
print(result.content)

# New API with Agent (recommended for tool calling)
from llm_client import Agent, OpenAIProvider, tool

@tool
async def my_tool(arg: str) -> str:
    """Tool description."""
    return f"Result: {arg}"

agent = Agent(
    provider=OpenAIProvider(model="gpt-5"),
    tools=[my_tool],
)
result = await agent.run("Use the tool")
print(result.content)
```

**Key differences:**
- `OpenAIClient.get_response()` returns `dict` → `OpenAIProvider.complete()` returns `CompletionResult`
- Tool calling requires manual handling in old API → Automatic with `Agent` in new API
- Streaming returns raw events → Unified `StreamEvent` objects with adapters

---

## License

MIT License - see LICENSE file for details.
