"""
LLM Client - A production-ready library for LLM interactions.

This package provides:
- **Provider abstraction**: Unified interface for LLM providers (OpenAI, etc.)
- **Agent framework**: Autonomous agents with tool calling and ReAct loops
- **Conversation management**: Context window handling and message history
- **Tool system**: Easy tool definition with decorators
- **Streaming**: Unified event-based streaming with multiple output adapters
- **Caching**: Multiple backend support (filesystem, PostgreSQL, Redis, Qdrant)
- **Rate limiting**: Token bucket implementation for API limits
- **Batch processing**: Concurrent request handling with checkpointing

Quick Start:
    ```python
    # Simple completion
    from llm_client import OpenAIProvider
    
    provider = OpenAIProvider(model="gpt-5")
    result = await provider.complete("Hello, world!")
    print(result.content)
    
    # Agent with tools
    from llm_client import Agent, tool
    
    @tool
    async def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"
    
    agent = Agent(
        provider=OpenAIProvider(model="gpt-5"),
        tools=[search],
        system_message="You are a helpful assistant."
    )
    result = await agent.run("Search for Python tutorials")
    print(result.content)
    ```

For backward compatibility, the original `OpenAIClient` is still available:
    ```python
    from llm_client import OpenAIClient
    
    client = OpenAIClient(model="gpt-5")
    result = await client.get_response(messages=[...])
    ```

Environment variables are automatically loaded from `.env` files.
"""
from dotenv import find_dotenv, load_dotenv

# Load environment variables on import
_ = load_dotenv(find_dotenv(), override=True)

# === Provider Layer ===
from .providers import (
    # Protocols and base classes
    BaseProvider,
    Provider,
    # Provider implementations
    OpenAIProvider,
    # Types
    CompletionResult,
    EmbeddingResult,
    Message,
    MessageInput,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    Usage,
    normalize_messages,
)

# === Agent Layer ===
from .agent import (
    Agent,
    AgentConfig,
    AgentResult,
    TurnResult,
    quick_agent,
)

# === Conversation ===
from .conversation import (
    Conversation,
    ConversationConfig,
)

# === Tool System ===
from .tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    sync_tool,
    tool,
    tool_from_function,
)

# === Streaming ===
from .streaming import (
    BufferingAdapter,
    CallbackAdapter,
    PusherStreamer,
    SSEAdapter,
    collect_stream,
    format_sse_event,
    stream_to_string,
)

# === Model Profiles ===
from .models import (
    GPT5,
    GPT5Mini,
    GPT5Nano,
    GPT5Point1,
    GPT5Point2,
    ModelProfile,
    TextEmbedding3Large,
    TextEmbedding3Small,
)

# === Caching ===
from .cache import (
    FSCache,
    HybridRedisPostgreSQLCache,
    QdrantCache,
)

# === Rate Limiting ===
from .rate_limit import Limiter, TokenBucket

# === Batch Processing ===
from .batch_req import BatchManager, RequestManager

# === Exceptions ===
from .exceptions import ResponseTimeoutError

# === Backward Compatible Client ===
from .client import OpenAIClient


__all__ = [
    # === Primary API (New) ===
    # Provider layer
    "Provider",
    "BaseProvider",
    "OpenAIProvider",
    # Agent framework
    "Agent",
    "AgentConfig",
    "AgentResult",
    "TurnResult",
    "quick_agent",
    # Conversation
    "Conversation",
    "ConversationConfig",
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "sync_tool",
    "tool_from_function",
    # Types
    "Message",
    "Role",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "CompletionResult",
    "EmbeddingResult",
    "StreamEvent",
    "StreamEventType",
    "MessageInput",
    "normalize_messages",
    # Streaming
    "SSEAdapter",
    "CallbackAdapter",
    "BufferingAdapter",
    "PusherStreamer",
    "format_sse_event",
    "collect_stream",
    "stream_to_string",
    # === Infrastructure ===
    # Models
    "ModelProfile",
    "GPT5",
    "GPT5Mini",
    "GPT5Nano",
    "GPT5Point1",
    "GPT5Point2",
    "TextEmbedding3Large",
    "TextEmbedding3Small",
    # Caching
    "QdrantCache",
    "FSCache",
    "HybridRedisPostgreSQLCache",
    # Rate limiting
    "Limiter",
    "TokenBucket",
    # Batch processing
    "BatchManager",
    "RequestManager",
    # Exceptions
    "ResponseTimeoutError",
    # === Backward Compatible ===
    "OpenAIClient",
]
