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

Environment loading is explicit via `llm_client.load_env`.
"""

# === Agent Layer ===
from .agent import (
    Agent,
    AgentConfig,
    AgentResult,
    TurnResult,
    quick_agent,
)

# === Batch Processing ===
from .batch_req import BatchManager, RequestManager

# === Caching ===
from .cache import (
    CacheStats,
    FSCache,
    HybridRedisPostgreSQLCache,
    QdrantCache,
)

# === Backward Compatible Client ===
from .client import OpenAIClient
from .config import load_env

# === Conversation ===
from .conversation import (
    Conversation,
    ConversationConfig,
)

# === Summarization ===
from .summarization import Summarizer, NoOpSummarizer, LLMSummarizer, LLMSummarizerConfig
from .sync import get_messages_sync, summarize_sync

# === Structured Output ===
from .structured import (
    StructuredOutputConfig,
    StructuredResult,
    extract_structured,
    validate_and_parse,
)

# === Execution Engine ===
from .engine import ExecutionEngine, RetryConfig

# === Cancellation ===
from .cancellation import CancellationToken, CancelledError

# === Exceptions ===
from .exceptions import ResponseTimeoutError
from .hooks import Hook, HookManager, InMemoryMetricsHook, OpenTelemetryHook, PrometheusHook

# === Model Profiles ===
from .models import (
    GPT5,
    Gemini15Pro,
    Gemini20Flash,
    GPT5Mini,
    GPT5Nano,
    GPT5Point1,
    GPT5Point2,
    ModelProfile,
    TextEmbedding3Large,
    TextEmbedding3Small,
)
from .perf import (
    FingerprintCache,
    clear_fingerprint_cache,
    fingerprint,
    fingerprint_messages,
    get_fingerprint,
)

# === Provider Layer ===
from .providers import (
    ANTHROPIC_AVAILABLE,
    GOOGLE_AVAILABLE,
    AnthropicProvider,
    # Protocols and base classes
    BaseProvider,
    # Types
    CompletionResult,
    EmbeddingResult,
    GoogleProvider,
    Message,
    MessageInput,
    # Provider implementations
    OpenAIProvider,
    Provider,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    Usage,
    normalize_messages,
)

# === Rate Limiting ===
from .rate_limit import Limiter, TokenBucket
from .resilience import CircuitBreaker, CircuitBreakerConfig
from .routing import ProviderRouter, StaticRouter

# === Hashing Utilities ===
from .hashing import (
    cache_key,
    compute_hash,
    content_hash,
    int_hash,
)

# === Serialization & Performance ===
from .serialization import (
    canonicalize,
    fast_json_dumps,
    fast_json_loads,
    stable_json_dumps,
)
from .spec import RequestContext, RequestSpec

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
from .telemetry import (
    CacheStats as TelemetryCacheStats,
)
from .telemetry import (
    Counter as TelemetryCounter,
)
from .telemetry import (
    Gauge as TelemetryGauge,
)
from .telemetry import (
    Histogram as TelemetryHistogram,
)

# === Telemetry ===
from .telemetry import (
    LatencyRecorder,
    MetricRegistry,
    RequestUsage,
    SessionUsage,
    TelemetryConfig,
    UsageTracker,
    get_registry,
    get_usage_tracker,
    set_registry,
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

__all__ = [
    # === Primary API (New) ===
    # Provider layer
    "Provider",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "ANTHROPIC_AVAILABLE",
    "GoogleProvider",
    "GOOGLE_AVAILABLE",
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
    "Gemini20Flash",
    "Gemini15Pro",
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
    # Config
    "load_env",
    # Execution engine
    "ExecutionEngine",
    "RetryConfig",
    "RequestContext",
    "RequestSpec",
    "Hook",
    "HookManager",
    "InMemoryMetricsHook",
    "OpenTelemetryHook",
    "PrometheusHook",
    # Telemetry
    "TelemetryConfig",
    "MetricRegistry",
    "TelemetryCounter",
    "TelemetryGauge",
    "TelemetryHistogram",
    "UsageTracker",
    "RequestUsage",
    "SessionUsage",
    "TelemetryCacheStats",
    "LatencyRecorder",
    "get_registry",
    "set_registry",
    "get_usage_tracker",
    "CacheStats",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    # Cancellation
    "CancellationToken",
    "CancelledError",
    "ProviderRouter",
    "StaticRouter",
    # Serialization & Performance
    "stable_json_dumps",
    "fast_json_dumps",
    "fast_json_loads",
    "canonicalize",
    "fingerprint",
    "fingerprint_messages",
    "FingerprintCache",
    "get_fingerprint",
    "clear_fingerprint_cache",
    # Hashing utilities
    "cache_key",
    "compute_hash",
    "content_hash",
    "int_hash",
]
