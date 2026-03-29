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
    from llm_client.providers import OpenAIProvider

    provider = OpenAIProvider(model="gpt-5")
    result = await provider.complete("Hello, world!")
    print(result.content)

    # Agent with tools
    from llm_client.agent import Agent
    from llm_client.tools import tool

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
    from llm_client.compat import OpenAIClient

    client = OpenAIClient(model="gpt-5")
    result = await client.get_response(messages=[...])
    ```

Environment loading is explicit via `llm_client.config.load_env`.

For new projects, prefer the stable module namespaces:
    - `llm_client.providers`
    - `llm_client.agent`
    - `llm_client.tools`
    - `llm_client.types`
    - `llm_client.content`
    - `llm_client.context`
    - `llm_client.observability`
    - `llm_client.compat`
"""

from __future__ import annotations

import importlib
import warnings

# === Agent Layer ===
from .agent import (
    Agent,
    AgentConfig,
    AgentResult,
    TurnResult,
    quick_agent,
)

# === Backward Compatible Client ===
from .config import load_env

# === Conversation ===
from .conversation import (
    Conversation,
    ConversationConfig,
)

# === Summarization ===
from .summarization import (
    Summarizer,
    SummarizationRequest,
    SummarizationResult,
    SummarizationStrategy,
    NoOpSummarizer,
    LLMSummarizer,
    LLMSummarizerConfig,
)

# === Context Planning / Memory ===
from .context_assembly import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    ContextSourceLoader,
    ContextSourcePayload,
    ContextSourceRequest,
    MultiSourceContextAssembler,
)
from .context_planning import (
    ContextPlan,
    ContextPlanner,
    ContextPlanningRequest,
    DefaultMemoryRetrievalStrategy,
    HeuristicContextPlanner,
    MemoryRetrievalStrategy,
    RelevanceSelectionStrategy,
    SemanticRelevanceSelector,
    SlidingWindowTrimmingStrategy,
    TieredTrimmingStrategy,
    TrimmingStrategy,
)
from .memory import (
    InMemorySummaryStore,
    MemoryQuery,
    MemoryReader,
    MemoryRecord,
    MemoryStore,
    MemoryWrite,
    MemoryWriter,
    ShortTermMemoryStore,
    SummaryRecord,
    SummaryStore,
)

# === Structured Output ===
from .structured import (
    StructuredExecutionMode,
    StructuredModeSelection,
    StructuredAttemptTrace,
    StructuredDiagnostics,
    StructuredExecutionFailure,
    StructuredExecutionOutcome,
    StructuredOutputConfig,
    StructuredResponseMode,
    StructuredResult,
    StructuredStreamEvent,
    StructuredStreamEventType,
    build_structured_response_format,
    extract_structured,
    finalize_structured_completion_loop,
    normalize_structured_schema,
    select_structured_mode,
    stream_structured,
    structured,
    validate_and_parse,
)

# === Benchmarking ===
from .benchmarks import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkComparisonRecord,
    BenchmarkComparisonReport,
    BenchmarkRecord,
    BenchmarkReport,
    BenchmarkRunMetadata,
    BenchmarkRunMode,
    build_cache_benchmark_case,
    build_completion_benchmark_case,
    build_context_planning_benchmark_case,
    build_embeddings_benchmark_case,
    build_failover_benchmark_case,
    build_stream_benchmark_case,
    build_structured_quality_benchmark_case,
    build_tool_execution_benchmark_case,
    compare_benchmark_reports,
    load_benchmark_report,
    run_benchmarks,
    save_benchmark_report,
)

# === Execution Engine ===
from .engine import ExecutionEngine, FailoverPolicy, RetryConfig

# === Cancellation ===
from .cancellation import CancellationToken, CancelledError

# === Exceptions ===
from .exceptions import ResponseTimeoutError
from .hooks import (
    BenchmarkRecorder,
    BenchmarkSnapshot,
    ContextPlanningRecorder,
    ContextPlanningSnapshot,
    EngineDiagnosticsRecorder,
    EngineDiagnosticsSnapshot,
    Hook,
    HookManager,
    InMemoryMetricsHook,
    LifecycleLoggingHook,
    LifecycleRecorder,
    LifecycleTelemetryHook,
    OpenTelemetryHook,
    PrometheusHook,
)
from .lifecycle import (
    LifecycleEvent,
    LifecycleEventType,
    RequestReport,
    SessionReport,
    UsageBreakdown,
    accumulate_session_report,
    build_request_report,
    normalize_lifecycle_event,
)

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
from .model_catalog import (
    ModelCatalog,
    ModelMetadata,
    get_default_model_catalog,
    infer_provider_for_model,
    metadata_from_profile,
)
from .provider_registry import (
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderRegistry,
    get_default_provider_registry,
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

from .routing import (
    ProviderRouter,
    RegistryRouter,
    RoutingRequirements,
    StaticRouter,
)
from .context import BudgetSpec, ExecutionContext, PolicyRef, RunVersions
from .budgets import (
    Budget,
    BudgetDecision,
    BudgetExceededError,
    InMemoryLedgerWriter,
    Ledger,
    LedgerEvent,
    LedgerEventType,
    LedgerWriter,
    UsageRecord,
)
from .replay import (
    EventFingerprint,
    RecordedEvent,
    Recording,
    ReplayMode,
    ReplayPlayer,
    ReplayRecorder,
    ReplayResult,
    ReplayValidationError,
    RunMetadata,
)
from .runtime_events import (
    ActionEvent,
    ArtifactEvent,
    EventBus,
    EventSubscription,
    FinalEvent,
    InMemoryEventBus,
    JobEvent,
    ModelEvent,
    ProgressEvent,
    RuntimeEvent,
    RuntimeEventType,
    ToolEvent,
)
from .redaction import (
    LogFieldClass,
    PayloadPreviewMode,
    ProviderPayloadCaptureMode,
    RedactionPolicy,
    ToolOutputPolicy,
    capture_provider_payload,
    preview_payload,
    sanitize_log_data,
    sanitize_payload,
    sanitize_tool_output,
)
from .spec import RequestContext, RequestSpec

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
    ToolExecutionBatch,
    ToolExecutionEngine,
    ToolExecutionEnvelope,
    ToolExecutionMetadata,
    ToolExecutionStatus,
    ToolRegistry,
    ToolResult,
    sync_tool,
    tool,
    ToolOutputPolicyMiddleware,
    tool_from_function,
)

_STABLE_EXPORTS = [
    # Providers
    "Provider",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "ANTHROPIC_AVAILABLE",
    "GoogleProvider",
    "GOOGLE_AVAILABLE",
    # Agent
    "Agent",
    "AgentConfig",
    "AgentResult",
    "TurnResult",
    "quick_agent",
    # Tools
    "Tool",
    "ToolExecutionBatch",
    "ToolExecutionEngine",
    "ToolExecutionEnvelope",
    "ToolExecutionMetadata",
    "ToolExecutionStatus",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "sync_tool",
    "ToolOutputPolicyMiddleware",
    "tool_from_function",
    # Content / types
    "Message",
    "MessageInput",
    "Role",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "CompletionResult",
    "EmbeddingResult",
    "StreamEvent",
    "StreamEventType",
    "normalize_messages",
    # Requests / execution
    "RequestContext",
    "RequestSpec",
    "ExecutionEngine",
    "FailoverPolicy",
    "RetryConfig",
    "BudgetSpec",
    "PolicyRef",
    "RunVersions",
    "ExecutionContext",
    "Budget",
    "BudgetDecision",
    "BudgetExceededError",
    "InMemoryLedgerWriter",
    "Ledger",
    "LedgerEvent",
    "LedgerEventType",
    "LedgerWriter",
    "UsageRecord",
    "CancellationToken",
    "CancelledError",
    # Conversation / structured / summarization
    "Conversation",
    "ConversationConfig",
    "StructuredOutputConfig",
    "StructuredExecutionMode",
    "StructuredResponseMode",
    "StructuredModeSelection",
    "StructuredAttemptTrace",
    "StructuredDiagnostics",
    "StructuredExecutionFailure",
    "StructuredExecutionOutcome",
    "StructuredResult",
    "StructuredStreamEventType",
    "StructuredStreamEvent",
    "build_structured_response_format",
    "extract_structured",
    "finalize_structured_completion_loop",
    "normalize_structured_schema",
    "select_structured_mode",
    "stream_structured",
    "structured",
    "validate_and_parse",
    "BenchmarkRunMode",
    "BenchmarkCategory",
    "BenchmarkCase",
    "BenchmarkRecord",
    "BenchmarkReport",
    "BenchmarkRunMetadata",
    "BenchmarkComparisonRecord",
    "BenchmarkComparisonReport",
    "run_benchmarks",
    "save_benchmark_report",
    "load_benchmark_report",
    "compare_benchmark_reports",
    "build_completion_benchmark_case",
    "build_stream_benchmark_case",
    "build_embeddings_benchmark_case",
    "build_tool_execution_benchmark_case",
    "build_cache_benchmark_case",
    "build_failover_benchmark_case",
    "build_context_planning_benchmark_case",
    "build_structured_quality_benchmark_case",
    "ContextPlan",
    "ContextPlanner",
    "ContextPlanningRequest",
    "ContextAssemblyRequest",
    "ContextAssemblyResult",
    "ContextSourceLoader",
    "ContextSourcePayload",
    "ContextSourceRequest",
    "MultiSourceContextAssembler",
    "DefaultMemoryRetrievalStrategy",
    "HeuristicContextPlanner",
    "MemoryRetrievalStrategy",
    "RelevanceSelectionStrategy",
    "SemanticRelevanceSelector",
    "SlidingWindowTrimmingStrategy",
    "TieredTrimmingStrategy",
    "TrimmingStrategy",
    "Summarizer",
    "SummarizationRequest",
    "SummarizationResult",
    "SummarizationStrategy",
    "NoOpSummarizer",
    "LLMSummarizer",
    "LLMSummarizerConfig",
    "MemoryQuery",
    "MemoryReader",
    "MemoryRecord",
    "MemoryStore",
    "MemoryWrite",
    "MemoryWriter",
    "ShortTermMemoryStore",
    "SummaryRecord",
    "SummaryStore",
    "InMemorySummaryStore",
    # Models
    "ModelProfile",
    "ModelMetadata",
    "ModelCatalog",
    "GPT5",
    "GPT5Mini",
    "GPT5Nano",
    "GPT5Point1",
    "GPT5Point2",
    "TextEmbedding3Large",
    "TextEmbedding3Small",
    "Gemini20Flash",
    "Gemini15Pro",
    "get_default_model_catalog",
    "infer_provider_for_model",
    "metadata_from_profile",
    # Core observability and routing
    "Hook",
    "HookManager",
    "ContextPlanningSnapshot",
    "ContextPlanningRecorder",
    "EngineDiagnosticsSnapshot",
    "EngineDiagnosticsRecorder",
    "BenchmarkSnapshot",
    "BenchmarkRecorder",
    "LifecycleEventType",
    "LifecycleEvent",
    "UsageBreakdown",
    "RequestReport",
    "SessionReport",
    "normalize_lifecycle_event",
    "build_request_report",
    "accumulate_session_report",
    "LifecycleRecorder",
    "LifecycleLoggingHook",
    "LifecycleTelemetryHook",
    "InMemoryMetricsHook",
    "OpenTelemetryHook",
    "PrometheusHook",
    "LogFieldClass",
    "PayloadPreviewMode",
    "ProviderPayloadCaptureMode",
    "RedactionPolicy",
    "ToolOutputPolicy",
    "sanitize_payload",
    "capture_provider_payload",
    "preview_payload",
    "sanitize_log_data",
    "sanitize_tool_output",
    "RuntimeEvent",
    "RuntimeEventType",
    "ProgressEvent",
    "ModelEvent",
    "ToolEvent",
    "ActionEvent",
    "ArtifactEvent",
    "JobEvent",
    "FinalEvent",
    "EventBus",
    "InMemoryEventBus",
    "EventSubscription",
    "RunMetadata",
    "EventFingerprint",
    "ReplayValidationError",
    "RecordedEvent",
    "Recording",
    "ReplayRecorder",
    "ReplayMode",
    "ReplayResult",
    "ReplayPlayer",
    "TelemetryConfig",
    "MetricRegistry",
    "UsageTracker",
    "RequestUsage",
    "SessionUsage",
    "LatencyRecorder",
    "get_registry",
    "set_registry",
    "get_usage_tracker",
    "ProviderRouter",
    "RegistryRouter",
    "RoutingRequirements",
    "StaticRouter",
    "ProviderCapabilities",
    "ProviderDescriptor",
    "ProviderRegistry",
    "get_default_provider_registry",
    # Config
    "load_env",
]

_COMPAT_EXPORTS = [
    "OpenAIClient",
    "ResponseTimeoutError",
]

__all__ = [*_STABLE_EXPORTS, *_COMPAT_EXPORTS]

_COMPAT_ALIASES: dict[str, tuple[str, str]] = {
    "OpenAIClient": (".compat", "OpenAIClient"),
    "ResponseTimeoutError": (".compat", "ResponseTimeoutError"),
}


def __getattr__(name: str):
    alias = _COMPAT_ALIASES.get(name)
    if alias is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = alias
    warnings.warn(
        f"`llm_client.{name}` is compatibility-only. Import it from `llm_client.compat` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
