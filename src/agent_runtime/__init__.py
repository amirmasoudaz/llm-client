"""
Agent Runtime - Orchestration layer for agent executions.

This package provides the "operating system" for agent executions:
- Job lifecycle management (queued, running, waiting_action, succeeded, failed)
- Action protocol for human-in-the-loop interactions
- Policy engine for tool/model gating and constraints
- Ledger for usage tracking, budgets, and quotas
- Unified event bus for observability
- Plugin system for extensible capabilities

The runtime builds on top of llm-client (Layer 0) and provides infrastructure
for business logic (Layer 2) to build upon.

Example:
    ```python
    from agent_runtime import RuntimeKernel, ExecutionRequest
    from llm_client import ExecutionEngine, OpenAIProvider

    # Set up the runtime
    kernel = RuntimeKernel.create(
        engine=ExecutionEngine(provider=OpenAIProvider(model="gpt-5")),
    )

    # Execute a job
    handle = await kernel.execute(ExecutionRequest(
        prompt="Hello!",
        scope_id="tenant-123",
        principal_id="user-456",
    ))

    # Stream events
    async for event in handle.events():
        print(event.type, event.data)

    # Get final result
    result = await handle.result()
    ```
"""

from .context import ExecutionContext, BudgetSpec, PolicyRef, RunVersions
from .events import (
    RuntimeEvent,
    RuntimeEventType,
    EventBus,
    InMemoryEventBus,
    EventSubscription,
    SSEEventAdapter,
    WebhookEventAdapter,
)
from .jobs import (
    JobStatus,
    JobRecord,
    JobManager,
    JobSpec,
    JobStore,
    InMemoryJobStore,
    JobFilter,
)
from .actions import (
    ActionStatus,
    ActionRecord,
    ActionManager,
    ActionSpec,
    ActionStore,
    InMemoryActionStore,
    ActionFilter,
)
from .policy import (
    PolicyEngine,
    PolicyDecision,
    PolicyDenied,
    PolicyRule,
    ToolPolicy,
    ModelPolicy,
    ConstraintPolicy,
    PolicyContext,
)
from .ledger import (
    Ledger,
    LedgerEvent,
    LedgerEventType,
    LedgerWriter,
    InMemoryLedgerWriter,
    Budget,
    BudgetDecision,
    UsageRecord,
    BudgetExceededError,
)
from .plugins import (
    Plugin,
    PluginManifest,
    PluginRegistry,
    PluginType,
    PluginCapability,
)
from .runtime import (
    RuntimeKernel,
    ExecutionRequest,
    ExecutionResult,
    ExecutionHandle,
)

# Optional storage adapters (require asyncpg)
try:
    from .storage import (
        PostgresJobStore,
        PostgresActionStore,
        PostgresLedgerWriter,
    )
    _STORAGE_EXPORTS = [
        "PostgresJobStore",
        "PostgresActionStore",
        "PostgresLedgerWriter",
    ]
except ImportError:
    _STORAGE_EXPORTS = []

# Optional Redis adapters
try:
    from .storage.redis import (
        SignalMessage,
        RedisSignalChannel,
        RedisJobCache,
        RedisActionStore,
    )
    _REDIS_EXPORTS = [
        "SignalMessage",
        "RedisSignalChannel",
        "RedisJobCache",
        "RedisActionStore",
    ]
except ImportError:
    _REDIS_EXPORTS = []

# Optional observability adapters (require opentelemetry)
try:
    from .observability import (
        OpenTelemetryAdapter,
        OTelConfig,
    )
    _OTEL_EXPORTS = [
        "OpenTelemetryAdapter",
        "OTelConfig",
    ]
except ImportError:
    _OTEL_EXPORTS = []

# Orchestration (multi-agent workflows)
from .orchestration import (
    Operator,
    OperatorResult,
    OperatorContext,
    AgentRole,
    RoutingDecision,
    OrchestrationConfig,
    Router,
    RoutingRule,
    DefaultRouter,
    GraphExecutor,
    GraphNode,
    GraphEdge,
    ExecutionGraph,
    NodeResult,
    MapReduceOperator,
    PlannerExecutorOperator,
    DebateOperator,
    ChainOperator,
    ParallelOperator,
)

# Replay module
from .replay import (
    RunMetadata,
    EventFingerprint,
    ReplayValidationError,
    ReplayRecorder,
    RecordedEvent,
    Recording,
    ReplayPlayer,
    ReplayMode,
    ReplayResult,
)

# Optional queue adapters
try:
    from .adapters import KafkaEventAdapter, KafkaConfig
    _KAFKA_EXPORTS = ["KafkaEventAdapter", "KafkaConfig"]
except ImportError:
    _KAFKA_EXPORTS = []

try:
    from .adapters import RedisStreamsAdapter, RedisStreamsConfig
    _REDIS_STREAMS_EXPORTS = ["RedisStreamsAdapter", "RedisStreamsConfig"]
except ImportError:
    _REDIS_STREAMS_EXPORTS = []

__version__ = "0.1.0"

__all__ = [
    # Context
    "ExecutionContext",
    "BudgetSpec",
    "PolicyRef",
    "RunVersions",
    # Events
    "RuntimeEvent",
    "RuntimeEventType",
    "EventBus",
    "InMemoryEventBus",
    "EventSubscription",
    "SSEEventAdapter",
    "WebhookEventAdapter",
    # Jobs
    "JobStatus",
    "JobRecord",
    "JobManager",
    "JobSpec",
    "JobStore",
    "InMemoryJobStore",
    "JobFilter",
    # Actions
    "ActionStatus",
    "ActionRecord",
    "ActionManager",
    "ActionSpec",
    "ActionStore",
    "InMemoryActionStore",
    "ActionFilter",
    # Policy
    "PolicyEngine",
    "PolicyDecision",
    "PolicyDenied",
    "PolicyRule",
    "ToolPolicy",
    "ModelPolicy",
    "ConstraintPolicy",
    "PolicyContext",
    # Ledger
    "Ledger",
    "LedgerEvent",
    "LedgerEventType",
    "LedgerWriter",
    "InMemoryLedgerWriter",
    "Budget",
    "BudgetDecision",
    "UsageRecord",
    "BudgetExceededError",
    # Plugins
    "Plugin",
    "PluginManifest",
    "PluginRegistry",
    "PluginType",
    "PluginCapability",
    # Runtime
    "RuntimeKernel",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionHandle",
    # Storage (optional - requires asyncpg)
    *_STORAGE_EXPORTS,
    # Redis (optional - requires redis)
    *_REDIS_EXPORTS,
    # Observability (optional - requires opentelemetry)
    *_OTEL_EXPORTS,
    # Orchestration (multi-agent workflows)
    "Operator",
    "OperatorResult",
    "OperatorContext",
    "AgentRole",
    "RoutingDecision",
    "OrchestrationConfig",
    "Router",
    "RoutingRule",
    "DefaultRouter",
    "GraphExecutor",
    "GraphNode",
    "GraphEdge",
    "ExecutionGraph",
    "NodeResult",
    "MapReduceOperator",
    "PlannerExecutorOperator",
    "DebateOperator",
    "ChainOperator",
    "ParallelOperator",
    # Replay
    "RunMetadata",
    "EventFingerprint",
    "ReplayValidationError",
    "ReplayRecorder",
    "RecordedEvent",
    "Recording",
    "ReplayPlayer",
    "ReplayMode",
    "ReplayResult",
    # Queue adapters (optional)
    *_KAFKA_EXPORTS,
    *_REDIS_STREAMS_EXPORTS,
]
