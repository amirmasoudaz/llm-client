# Implementation Documentation

This document provides a comprehensive overview of the implementation spanning both `llm-client` and `agent-runtime` packages.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [llm-client Changes](#llm-client-changes)
3. [agent-runtime Module Reference](#agent-runtime-module-reference)
4. [Integration Guide](#integration-guide)
5. [Configuration Options](#configuration-options)
6. [Storage Adapters](#storage-adapters)
7. [Observability Setup](#observability-setup)
8. [Multi-Agent Orchestration](#multi-agent-orchestration)
9. [Replay System](#replay-system)
10. [Queue Adapters](#queue-adapters)
11. [Examples and Usage Patterns](#examples-and-usage-patterns)

---

## Architecture Overview

The implementation follows a layered architecture:

```
┌────────────────────────────────────────────────────────────────────┐
│                        Business Logic (Layer 2)                    │
│                     Your application code, operators               │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                       agent-runtime (Layer 1)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   Jobs   │ │ Actions  │ │  Policy  │ │  Ledger  │ │  Events  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ Plugins  │ │Orchestr. │ │  Replay  │ │ Storage  │               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                        llm-client (Layer 0)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Providers │ │  Agent   │ │  Cache   │ │  Tools   │ │ Streaming│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ Engine   │ │Resilience│ │ Context  │ │Idempotent│               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
└────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Each layer has distinct responsibilities
2. **Optional Dependencies**: Production features (Postgres, Redis, Kafka) are optional
3. **Context Propagation**: Request context flows through all layers
4. **Event-Driven**: Unified event bus for observability and integration
5. **Deterministic Replay**: All operations can be recorded and replayed

---

## llm-client Changes

### 1. IdempotencyTracker Integration

**File**: `src/llm_client/engine.py`

The `ExecutionEngine` now supports idempotency tracking to prevent duplicate requests:

```python
from llm_client import ExecutionEngine, IdempotencyTracker

tracker = IdempotencyTracker(request_timeout=60.0)
engine = ExecutionEngine(
    provider=provider,
    idempotency_tracker=tracker,
)

# Requests with same idempotency key return cached result
result = await engine.complete(
    spec,
    idempotency_key="unique-request-123",
)
```

**Features**:
- Automatic deduplication of in-flight requests
- Cached result return for completed requests
- Hooks for tracking: `idempotency.start`, `idempotency.hit`, `idempotency.conflict`, `idempotency.complete`, `idempotency.fail`
- Works with both `complete()` and `stream()` methods

### 2. RequestContext Extensions

**File**: `src/llm_client/spec.py`

Extended `RequestContext` with new fields for agent-runtime integration:

```python
from llm_client import RequestContext

ctx = RequestContext(
    request_id="req-123",
    trace_id="trace-456",
    tenant_id="tenant-789",
    user_id="user-abc",
    session_id="session-def",  # NEW
    job_id="job-ghi",          # NEW
)

# Create child context for nested operations
child_ctx = ctx.child()

# Add job correlation
job_ctx = ctx.with_job("job-xyz")
```

### 3. Agent Middleware Integration

**File**: `src/llm_client/agent/core.py`

The `Agent` class now supports middleware chain for tool execution:

```python
from llm_client import Agent

agent = Agent(
    engine=engine,
    tools=[tool1, tool2],
    use_middleware=True,  # Enable default middleware chain
)

# Run with context
result = await agent.run(
    "Hello!",
    context=request_context,
)
```

The default middleware chain includes:
- Logging middleware
- Timeout middleware
- Retry middleware
- Budget middleware
- Telemetry middleware

---

## agent-runtime Module Reference

### Core Components

#### ExecutionContext

The universal context that flows through all operations:

```python
from agent_runtime import ExecutionContext, BudgetSpec, PolicyRef

ctx = ExecutionContext(
    scope_id="tenant-123",       # Multi-tenancy
    principal_id="user-456",     # Actor identity
    session_id="session-789",    # Conversation bucket
    budgets=BudgetSpec(
        max_tokens=100000,
        max_cost=10.0,
        max_turns=20,
    ),
    policy_ref=PolicyRef(name="production"),
)
```

#### Jobs

Job lifecycle management:

```python
from agent_runtime import JobManager, JobSpec, InMemoryJobStore

store = InMemoryJobStore()
manager = JobManager(store)

# Create a job
job = await manager.create(JobSpec(
    scope_id="tenant-123",
    principal_id="user-456",
    idempotency_key="unique-key",
))

# Update status
job = await manager.start(job.job_id)
job = await manager.complete(job.job_id, result_ref="result-123")
```

#### Actions

Human-in-the-loop protocol:

```python
from agent_runtime import ActionManager, ActionSpec, InMemoryActionStore

store = InMemoryActionStore()
manager = ActionManager(store)

# Request user action
action = await manager.require_action(ActionSpec(
    job_id="job-123",
    type="confirm",
    payload={"message": "Proceed with deployment?"},
    timeout_seconds=300,
))

# Later, resolve the action
await manager.resolve(
    action.resume_token,
    resolution={"confirmed": True},
)
```

#### Policy Engine

Centralized policy evaluation:

```python
from agent_runtime import PolicyEngine, ToolPolicy, ModelPolicy

engine = PolicyEngine()

# Add policies
engine.add_rule(ToolPolicy(
    tool_patterns=["file_*"],
    allowed_scopes=["internal"],
))

engine.add_rule(ModelPolicy(
    model_patterns=["gpt-4*"],
    require_approval=True,
))

# Check policy
result = await engine.check_tool("file_read", context)
if not result.allowed:
    raise PolicyDenied(result.reason)
```

#### Ledger

Usage tracking and budgets:

```python
from agent_runtime import Ledger, InMemoryLedgerWriter, Budget

writer = InMemoryLedgerWriter()
ledger = Ledger(writer)

# Set budget
ledger.set_budget(Budget(
    scope_id="tenant-123",
    max_tokens_daily=1000000,
    max_cost_daily=Decimal("100.00"),
))

# Check budget before operation
decision = await ledger.check_budget(ctx, pending_tokens=1000)
if decision == BudgetDecision.DENY:
    raise BudgetExceededError("Daily limit exceeded")

# Record usage
await ledger.record_model_usage(ctx, usage)
```

#### Event Bus

Unified event system:

```python
from agent_runtime import InMemoryEventBus, RuntimeEventType

bus = InMemoryEventBus()

# Subscribe to events
async def handler(event):
    print(f"Event: {event.type}")

subscription = await bus.subscribe(
    handler,
    event_types={RuntimeEventType.JOB_STARTED},
)

# Publish events
await bus.publish(RuntimeEvent(
    type=RuntimeEventType.JOB_STARTED,
    job_id="job-123",
))
```

#### RuntimeKernel

The main orchestrator:

```python
from agent_runtime import RuntimeKernel, ExecutionRequest

kernel = RuntimeKernel.create(
    engine=execution_engine,
    agent=agent,
)

# Execute a job
handle = await kernel.execute(ExecutionRequest(
    prompt="Analyze this data...",
    scope_id="tenant-123",
    principal_id="user-456",
))

# Stream events
async for event in handle.events():
    print(event.type, event.data)

# Get final result
result = await handle.result()
```

---

## Integration Guide

### Basic Setup

```python
from llm_client import ExecutionEngine, OpenAIProvider, Agent
from agent_runtime import RuntimeKernel, ExecutionRequest

# Layer 0: Set up LLM client
provider = OpenAIProvider(model="gpt-4")
engine = ExecutionEngine(provider=provider)
agent = Agent(engine=engine, use_middleware=True)

# Layer 1: Set up runtime
kernel = RuntimeKernel.create(
    engine=engine,
    agent=agent,
)

# Execute
handle = await kernel.execute(ExecutionRequest(
    prompt="Hello, world!",
))
result = await handle.result()
```

### With Persistent Storage

```python
import asyncpg
from agent_runtime import (
    RuntimeKernel,
    PostgresJobStore,
    PostgresActionStore,
    PostgresLedgerWriter,
)

# Create connection pool
pool = await asyncpg.create_pool("postgresql://...")

# Create stores
job_store = PostgresJobStore(pool)
action_store = PostgresActionStore(pool)
ledger_writer = PostgresLedgerWriter(pool)

# Create kernel with persistence
kernel = RuntimeKernel(
    engine=engine,
    agent=agent,
    job_store=job_store,
    action_store=action_store,
    ledger_writer=ledger_writer,
)
```

### With Redis Signaling

```python
import redis.asyncio as redis
from agent_runtime import RedisSignalChannel

client = await redis.from_url("redis://localhost")
signal_channel = RedisSignalChannel(client)
await signal_channel.start()

# Use in action manager for fast wait/resume
action_manager = ActionManager(
    action_store,
    signal_channel=signal_channel,
)
```

---

## Configuration Options

### BudgetSpec

```python
BudgetSpec(
    max_tokens=100000,          # Total tokens
    max_cost=10.0,              # Maximum cost in dollars
    max_tool_calls=100,         # Maximum tool invocations
    max_turns=20,               # Maximum conversation turns
    max_runtime_seconds=300.0,  # Maximum execution time
)
```

### OrchestrationConfig

```python
OrchestrationConfig(
    max_child_jobs=10,          # Max child jobs per parent
    max_depth=5,                # Max nesting depth
    timeout_seconds=300.0,      # Default timeout
    inherit_budgets=True,       # Child inherits parent budgets
    propagate_cancellation=True, # Cancel children on parent cancel
    parallel_execution=True,    # Run children in parallel
    max_parallel=5,             # Max concurrent children
)
```

### OTelConfig

```python
OTelConfig(
    tracer_name="agent_runtime",
    service_name="my-service",
    capture_input=False,        # PII concern
    capture_output=False,       # PII concern
    capture_tool_args=False,    # PII concern
    record_exceptions=True,
    sampling_rate=1.0,
)
```

---

## Storage Adapters

### PostgreSQL

Tables created:

**runtime_jobs**:
- `job_id` (TEXT PRIMARY KEY)
- `scope_id`, `principal_id`, `session_id`, `run_id` (TEXT)
- `status` (TEXT)
- `created_at`, `updated_at`, `started_at`, `completed_at` (TIMESTAMPTZ)
- `deadline`, `progress` (FLOAT)
- `budgets`, `policy_ref`, `versions`, `metadata`, `tags` (JSONB)

**runtime_actions**:
- `action_id` (TEXT PRIMARY KEY)
- `job_id`, `type`, `status` (TEXT)
- `payload`, `resolution` (JSONB)
- `resume_token` (TEXT UNIQUE)
- `created_at`, `expires_at`, `resolved_at` (TIMESTAMPTZ)

**runtime_ledger**:
- `event_id` (TEXT PRIMARY KEY)
- `type`, `job_id`, `scope_id`, `principal_id` (TEXT)
- `input_tokens`, `output_tokens`, `total_tokens` (INTEGER)
- `cost` (NUMERIC)
- `timestamp` (TIMESTAMPTZ)

### Redis

Components:
- `RedisSignalChannel`: Pub/sub for action wait/resume
- `RedisJobCache`: Read-through cache for jobs
- `RedisActionStore`: Redis-backed action persistence

---

## Observability Setup

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from agent_runtime import OpenTelemetryAdapter, OTelConfig

# Set up OTel
trace.set_tracer_provider(TracerProvider())

# Create adapter
adapter = OpenTelemetryAdapter(
    event_bus,
    config=OTelConfig(service_name="my-agent"),
)

await adapter.start()
```

Span hierarchy:
- `agent.job` (parent): Full job execution
- `agent.run` (child): Single LLM interaction
- `agent.tool` (child): Tool execution
- `agent.action` (child): Human-in-the-loop action

---

## Multi-Agent Orchestration

### Operator Pattern

```python
from agent_runtime import Operator, OperatorResult, OperatorContext

class SummarizerOperator(Operator):
    @property
    def name(self) -> str:
        return "summarizer"
    
    async def execute(
        self,
        input_data: dict,
        context: OperatorContext,
    ) -> OperatorResult:
        result = await self.agent.run(
            f"Summarize: {input_data['text']}"
        )
        return OperatorResult(
            content=result.content,
            success=True,
        )
```

### Built-in Patterns

**ChainOperator**: Sequential execution
```python
chain = ChainOperator([op1, op2, op3])
result = await chain.execute(input_data, context)
```

**ParallelOperator**: Parallel execution with aggregation
```python
parallel = ParallelOperator([op1, op2, op3])
result = await parallel.execute(input_data, context)
```

**MapReduceOperator**: Map-reduce pattern
```python
map_reduce = MapReduceOperator(
    map_operator=analyzer,
    reduce_operator=synthesizer,
)
```

**PlannerExecutorOperator**: Plan then execute
```python
planner_executor = PlannerExecutorOperator(
    planner=planner_agent,
    executor=executor_agent,
)
```

**DebateOperator**: Multi-agent debate
```python
debate = DebateOperator(
    debaters=[agent1, agent2],
    moderator=moderator,
    max_rounds=3,
)
```

### Graph Execution

```python
from agent_runtime import ExecutionGraph, GraphNode, GraphEdge, GraphExecutor

graph = ExecutionGraph()

# Add nodes
graph.add_node(GraphNode("planner", planner_op))
graph.add_node(GraphNode("executor", executor_op))
graph.add_node(GraphNode("reviewer", reviewer_op))

# Add edges (data flow)
graph.add_edge(GraphEdge("planner", "executor"))
graph.add_edge(GraphEdge("executor", "reviewer"))

# Execute
executor = GraphExecutor(graph)
results = await executor.execute(input_data, context)
```

---

## Replay System

### Recording

```python
from agent_runtime import ReplayRecorder, RunMetadata

metadata = RunMetadata.create(
    runtime_version="1.0.0",
    model_version="gpt-4-0125",
)

recorder = ReplayRecorder(event_bus, metadata)
await recorder.start(initial_input={"prompt": "Hello"})

# ... execution happens ...

recording = await recorder.stop()
recording.save("execution.replay.json")
```

### Replaying

```python
from agent_runtime import Recording, ReplayPlayer, ReplayMode

recording = Recording.load("execution.replay.json")
player = ReplayPlayer(recording, event_bus)

# Fast replay
result = await player.replay(mode=ReplayMode.FAST)

# Timed replay (original timing)
result = await player.replay(mode=ReplayMode.TIMED)

# Step through
async for event in player.step():
    print(event)
```

---

## Queue Adapters

### Kafka

```python
from agent_runtime import KafkaEventAdapter, KafkaConfig

config = KafkaConfig(
    bootstrap_servers="kafka:9092",
    topic_prefix="myapp.agent",
)

adapter = KafkaEventAdapter(event_bus, config)
await adapter.start()
```

### Redis Streams

```python
from agent_runtime import RedisStreamsAdapter, RedisStreamsConfig

config = RedisStreamsConfig(
    stream_prefix="myapp:events",
    max_len=100000,
)

adapter = RedisStreamsAdapter(redis_client, event_bus, config)
await adapter.start()

# Read events
async for event in adapter.read_events("myapp:events:job"):
    print(event)
```

---

## Examples and Usage Patterns

### Simple Agent Execution

```python
from llm_client import ExecutionEngine, OpenAIProvider, Agent
from agent_runtime import RuntimeKernel, ExecutionRequest

provider = OpenAIProvider(model="gpt-4")
engine = ExecutionEngine(provider=provider)
agent = Agent(engine=engine)

kernel = RuntimeKernel.create(engine=engine, agent=agent)

handle = await kernel.execute(ExecutionRequest(
    prompt="Write a poem about Python",
))

result = await handle.result()
print(result.content)
```

### With Budget Enforcement

```python
from agent_runtime import ExecutionRequest, BudgetSpec

handle = await kernel.execute(ExecutionRequest(
    prompt="Analyze this document...",
    scope_id="tenant-123",
    budgets=BudgetSpec(
        max_tokens=10000,
        max_cost=1.0,
    ),
))
```

### With Human-in-the-Loop

```python
# In the agent's tool
async def deploy_changes():
    # Request approval
    action = await action_manager.require_action(ActionSpec(
        job_id=context.job_id,
        type="confirm",
        payload={"message": "Deploy to production?"},
    ))
    
    # Wait for resolution
    await action_manager.wait_for_resolution(action.action_id)
    
    # Proceed if approved
    resolution = action.resolution
    if resolution.get("approved"):
        await do_deploy()
```

### Multi-Agent Research

```python
from agent_runtime import MapReduceOperator, PlannerExecutorOperator

# Create operators
researcher = ResearcherOperator(agent)
summarizer = SummarizerOperator(agent)
synthesizer = SynthesizerOperator(agent)

# Build workflow
workflow = PlannerExecutorOperator(
    planner=PlannerOperator(agent),
    executor=MapReduceOperator(
        map_operator=researcher,
        reduce_operator=summarizer,
    ),
    synthesizer=synthesizer,
)

result = await workflow.execute(
    {"topic": "Recent advances in AI"},
    context,
)
```

---

## Summary of Changes

### llm-client (Layer 0)

1. **IdempotencyTracker Integration**: `ExecutionEngine` now supports idempotency tracking
2. **RequestContext Extensions**: Added `session_id` and `job_id` fields
3. **Agent Middleware**: Added middleware chain support to `Agent`

### agent-runtime (Layer 1)

1. **Core Module**: `ExecutionContext`, `BudgetSpec`, `PolicyRef`, `RunVersions`
2. **Jobs Module**: `JobRecord`, `JobStatus`, `JobManager`, `JobStore`
3. **Actions Module**: `ActionRecord`, `ActionStatus`, `ActionManager`, `ActionStore`
4. **Policy Module**: `PolicyEngine`, policy rules (`ToolPolicy`, `ModelPolicy`, etc.)
5. **Ledger Module**: `Ledger`, `LedgerEvent`, `Budget`, `BudgetDecision`
6. **Events Module**: `RuntimeEvent`, `EventBus`, adapters
7. **Plugins Module**: `Plugin`, `PluginManifest`, `PluginRegistry`
8. **Runtime Module**: `RuntimeKernel`, `ExecutionRequest`, `ExecutionHandle`
9. **Storage Module**: `PostgresJobStore`, `PostgresActionStore`, `PostgresLedgerWriter`
10. **Redis Module**: `RedisSignalChannel`, `RedisJobCache`, `RedisActionStore`
11. **Observability Module**: `OpenTelemetryAdapter`
12. **Orchestration Module**: `Operator`, `Router`, `GraphExecutor`, patterns
13. **Replay Module**: `RunMetadata`, `ReplayRecorder`, `ReplayPlayer`
14. **Adapters Module**: `KafkaEventAdapter`, `RedisStreamsAdapter`
