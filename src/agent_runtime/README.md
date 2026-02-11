# agent-runtime Package Reference (v0.1.0)

This document is the canonical, user-facing reference for the `agent-runtime` Python package (import path: `agent_runtime`).
It describes capabilities, API surface, configuration, expected inputs/outputs, and integration patterns.

> Scope: This is primarily an API/usage reference. It builds on top of `llm-client` (Layer 0) to provide job lifecycle,
> policy enforcement, human-in-the-loop actions, usage tracking, and multi-agent orchestration.

---

## Table of Contents

- [What this package is](#what-this-package-is)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Concepts](#core-concepts)
- [Execution Context](#execution-context)
- [Jobs Module](#jobs-module)
- [Actions Module](#actions-module)
- [Policy Engine](#policy-engine)
- [Ledger and Budgets](#ledger-and-budgets)
- [Event Bus](#event-bus)
- [Plugin System](#plugin-system)
- [Runtime Kernel](#runtime-kernel)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
- [Replay System](#replay-system)
- [Storage Adapters](#storage-adapters)
- [Observability](#observability)
- [Queue Adapters](#queue-adapters)
- [Configuration Reference](#configuration-reference)
- [Integration with llm-client](#integration-with-llm-client)
- [Public API Index](#public-api-index)

---

## What this package is

`agent-runtime` is the orchestration layer (Layer 1) that sits between raw LLM interactions (`llm-client`, Layer 0) and
business logic (Layer 2). It provides:

- **Job lifecycle management**: Queue, run, pause, resume, cancel jobs with persistent state
- **Action protocol**: Human-in-the-loop interactions with timeout, resume tokens, and resolution
- **Policy engine**: Centralized policy evaluation for tools, models, and constraints
- **Ledger**: Usage tracking, budget enforcement, and audit trails
- **Event bus**: Unified event system for observability and integrations
- **Plugin system**: Extensible capabilities via tool plugins
- **Multi-agent orchestration**: Operators, routers, and graph-based execution
- **Replay system**: Deterministic replay for debugging and testing
- **Storage adapters**: PostgreSQL and Redis backends for production deployments

---

## Installation

### Requirements

- Python `>= 3.10`
- `llm-client` package (peer dependency)
- Async runtime (asyncio)

### Install

```bash
# Install with llm-client
pip install -e .

# Optional: PostgreSQL storage
pip install asyncpg

# Optional: Redis storage/signaling
pip install redis

# Optional: OpenTelemetry observability
pip install opentelemetry-api opentelemetry-sdk

# Optional: Kafka event streaming
pip install aiokafka
```

---

## Quick Start

### Basic execution

```python
import asyncio
from llm_client import ExecutionEngine, OpenAIProvider, Agent
from agent_runtime import RuntimeKernel, ExecutionRequest

async def main():
    # Set up llm-client (Layer 0)
    provider = OpenAIProvider(model="gpt-5-nano")
    engine = ExecutionEngine(provider=provider)
    agent = Agent(engine=engine)

    # Set up agent-runtime (Layer 1)
    kernel = RuntimeKernel.create(engine=engine, agent=agent)

    # Execute a job
    handle = await kernel.execute(ExecutionRequest(
        prompt="Write a haiku about Python",
        scope_id="tenant-123",
        principal_id="user-456",
    ))

    # Stream events
    async for event in handle.events():
        print(f"{event.type}: {event.data}")

    # Get final result
    result = await handle.result()
    print(result.content)

asyncio.run(main())
```

### With budget enforcement

```python
from agent_runtime import ExecutionRequest, BudgetSpec

handle = await kernel.execute(ExecutionRequest(
    prompt="Analyze this document...",
    scope_id="tenant-123",
    budgets=BudgetSpec(
        max_tokens=10000,
        max_cost=1.0,
        max_turns=5,
    ),
))
```

### With human-in-the-loop

```python
from agent_runtime import ActionManager, ActionSpec

# In your tool implementation
async def deploy_changes(context):
    action = await action_manager.require_action(ActionSpec(
        job_id=context.job_id,
        type="confirm",
        payload={"message": "Deploy to production?"},
        timeout_seconds=300,
    ))
    
    # Wait for user resolution
    await action_manager.wait_for_resolution(action.action_id)
    
    if action.resolution.get("approved"):
        await do_deploy()
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Business Logic (Layer 2)                     │
│                     Your application code, operators                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       agent-runtime (Layer 1)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │   Jobs   │ │ Actions  │ │  Policy  │ │  Ledger  │ │  Events  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │ Plugins  │ │Orchestr. │ │  Replay  │ │ Storage  │                │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        llm-client (Layer 0)                         │
│                  Providers, Agent, Tools, Streaming                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### ExecutionContext

The universal context that flows through all runtime operations. It carries:

- **Identity**: scope_id (tenant), principal_id (user), session_id, run_id, job_id
- **Tracing**: trace_id, span_id, parent_span_id
- **Policy**: PolicyRef for centralized policy
- **Budgets**: BudgetSpec for resource limits
- **Versions**: RunVersions for replay compatibility
- **Cancellation**: CancellationToken for cooperative cancellation

### Job

A `JobRecord` represents a single execution lifecycle:

- **States**: QUEUED → RUNNING → WAITING_ACTION → SUCCEEDED/FAILED/CANCELLED/TIMED_OUT
- **Tracking**: Progress, turn count, timestamps, error handling
- **Correlation**: parent_job_id for child jobs, idempotency_key for deduplication

### Action

An `ActionRecord` represents a human-in-the-loop request:

- **States**: PENDING → RESOLVED/EXPIRED/CANCELLED
- **Types**: confirm, choose, input, upload, reauth, apply_changes, approval
- **Security**: resume_token for secure resolution

### Policy

Centralized rules for controlling access to tools, models, and resources:

- **ToolPolicy**: Allow/deny tools by pattern
- **ModelPolicy**: Require approval for certain models
- **ConstraintPolicy**: Enforce limits (max tokens, cost, etc.)

### Ledger

Audit trail and budget enforcement:

- **Events**: Track all model usage, tool calls, and costs
- **Budgets**: Per-scope, per-principal, per-session limits
- **Decisions**: ALLOW, WARN, DENY, DEGRADE

---

## Execution Context

### `ExecutionContext`

```python
from agent_runtime import ExecutionContext, BudgetSpec, PolicyRef, RunVersions

ctx = ExecutionContext(
    # Identity
    scope_id="tenant-123",       # Multi-tenant isolation
    principal_id="user-456",     # Actor identity
    session_id="session-789",    # Conversation bucket
    run_id="run-abc",            # Single execution ID
    job_id="job-def",            # Job lifecycle ID
    
    # Tracing
    trace_id="trace-ghi",
    span_id="span-jkl",
    
    # Budgets
    budgets=BudgetSpec(
        max_tokens=100000,
        max_cost=10.0,
        max_tool_calls=100,
        max_turns=20,
        max_runtime_seconds=300.0,
    ),
    
    # Policy
    policy_ref=PolicyRef(name="production", version="v1"),
    
    # Versions (for replay)
    versions=RunVersions(
        runtime_version="1.0.0",
        llm_client_version="0.2.0",
        model_version="gpt-4-0125",
    ),
    
    # Metadata
    metadata={"env": "production"},
)
```

### Key Methods

```python
# Create child context for nested operations
child_ctx = ctx.child(new_span=True)

# Add job correlation
job_ctx = ctx.with_job("job-xyz")

# Convert to llm-client RequestContext
request_ctx = ctx.to_request_context()

# Serialization
data = ctx.to_dict()
ctx2 = ExecutionContext.from_dict(data)
```

### `BudgetSpec`

```python
BudgetSpec(
    max_tokens=100000,          # Total tokens (input + output)
    max_cost=10.0,              # Maximum cost in dollars
    max_tool_calls=100,         # Maximum tool invocations
    max_turns=20,               # Maximum conversation turns
    max_runtime_seconds=300.0,  # Maximum execution time
)
```

### `PolicyRef`

```python
PolicyRef(
    name="production",          # Policy name
    version="v1",               # Optional version
    overrides={"allow_web": True},  # Local overrides
)
```

### `RunVersions`

```python
RunVersions(
    runtime_version="1.0.0",
    llm_client_version="0.2.0",
    operator_version="1.0.0",
    model_version="gpt-4-0125",
    schema_version=1,
)
```

---

## Jobs Module

### `JobStatus` Enum

```python
from agent_runtime import JobStatus

JobStatus.QUEUED         # Initial state
JobStatus.RUNNING        # Actively executing
JobStatus.WAITING_ACTION # Paused for human input
JobStatus.SUCCEEDED      # Completed successfully
JobStatus.FAILED         # Error occurred
JobStatus.CANCELLED      # Explicitly cancelled
JobStatus.TIMED_OUT      # Deadline exceeded

# Properties
status.is_terminal  # True for SUCCEEDED, FAILED, CANCELLED, TIMED_OUT
status.is_active    # True for QUEUED, RUNNING, WAITING_ACTION
```

### `JobRecord`

```python
from agent_runtime import JobRecord

job = JobRecord(
    job_id="job-123",
    scope_id="tenant-123",
    principal_id="user-456",
    session_id="session-789",
    run_id="run-abc",
    parent_job_id=None,           # For child jobs
    idempotency_key="unique-key", # For deduplication
    status=JobStatus.QUEUED,
    deadline=time.time() + 300,   # Absolute timeout
    budgets=BudgetSpec(...),
    policy_ref=PolicyRef(...),
    metadata={"source": "api"},
    tags={"priority": "high"},
)

# State transitions
job = job.transition_to(JobStatus.RUNNING)
job = job.with_progress(0.5, turn=3)
job = job.with_error("Something failed", error_code="E001")
job = job.with_result("result-ref-123")

# Validation
job.can_transition_to(JobStatus.SUCCEEDED)  # True
```

### `JobManager`

```python
from agent_runtime import JobManager, JobSpec, InMemoryJobStore

store = InMemoryJobStore()
manager = JobManager(store)

# Create a job
job = await manager.create(JobSpec(
    scope_id="tenant-123",
    principal_id="user-456",
    idempotency_key="unique-key",
    budgets=BudgetSpec(max_tokens=10000),
))

# Lifecycle operations
job = await manager.start(job.job_id)
job = await manager.update_progress(job.job_id, 0.5, turn=3)
job = await manager.wait_for_action(job.job_id)
job = await manager.resume(job.job_id)
job = await manager.complete(job.job_id, result_ref="result-123")
job = await manager.fail(job.job_id, "Error message", error_code="E001")
job = await manager.cancel(job.job_id, reason="User cancelled")

# Query
job = await manager.get(job.job_id)
jobs = await manager.list(JobFilter(scope_id="tenant-123", status=JobStatus.RUNNING))
job = await manager.get_by_idempotency_key("unique-key", scope_id="tenant-123")
```

### `JobStore` Interface

```python
from agent_runtime import JobStore, JobFilter

class JobStore(ABC):
    async def create(self, job: JobRecord) -> JobRecord: ...
    async def get(self, job_id: str) -> JobRecord | None: ...
    async def update(self, job: JobRecord) -> JobRecord: ...
    async def delete(self, job_id: str) -> bool: ...
    async def list(self, filter: JobFilter | None = None) -> list[JobRecord]: ...
    async def get_by_idempotency_key(self, key: str, scope_id: str | None = None) -> JobRecord | None: ...
    async def count(self, filter: JobFilter | None = None) -> int: ...
```

Available implementations:
- `InMemoryJobStore`: Testing and single-process deployments
- `PostgresJobStore`: Production persistent storage (requires `asyncpg`)

---

## Actions Module

### `ActionStatus` and `ActionType`

```python
from agent_runtime import ActionStatus, ActionType

# Status
ActionStatus.PENDING    # Waiting for resolution
ActionStatus.RESOLVED   # Successfully resolved
ActionStatus.EXPIRED    # Deadline passed
ActionStatus.CANCELLED  # Explicitly cancelled

# Types
ActionType.CONFIRM       # User must confirm
ActionType.CHOOSE        # User must select from options
ActionType.INPUT         # User must provide text
ActionType.UPLOAD        # User must upload file
ActionType.REAUTH        # User must re-authenticate
ActionType.APPLY_CHANGES # User must approve changes
ActionType.APPROVAL      # General approval
ActionType.CUSTOM        # Custom type
```

### `ActionRecord`

```python
from agent_runtime import ActionRecord

action = ActionRecord(
    action_id="action-123",
    job_id="job-456",
    type="confirm",
    payload={"message": "Deploy to production?", "details": "..."},
    status=ActionStatus.PENDING,
    expires_at=time.time() + 300,  # 5 minute timeout
    resume_token="secure-token-xyz",
    metadata={"source": "deploy_tool"},
)

# Resolution
resolved = action.resolve({"approved": True, "comment": "LGTM"})
expired = action.expire()
cancelled = action.cancel("User cancelled")

# Properties
action.is_expired  # Check if past deadline
```

### `ActionManager`

```python
from agent_runtime import ActionManager, ActionSpec, InMemoryActionStore

store = InMemoryActionStore()
manager = ActionManager(store)

# Request an action
action = await manager.require_action(ActionSpec(
    job_id="job-123",
    type="confirm",
    payload={"message": "Proceed?"},
    timeout_seconds=300,
    metadata={"tool": "deploy"},
))

# Resolve by resume token (secure API endpoint)
resolved = await manager.resolve(
    resume_token=action.resume_token,
    resolution={"approved": True},
)

# Cancel action
await manager.cancel(action.action_id, reason="User cancelled")

# Wait for resolution (in agent code)
await manager.wait_for_resolution(
    action.action_id,
    timeout=300,
    poll_interval=1.0,
)

# List pending actions
pending = await manager.list_pending(job_id="job-123")
```

### `ActionStore` Interface

```python
from agent_runtime import ActionStore, ActionFilter

class ActionStore(ABC):
    async def create(self, action: ActionRecord) -> ActionRecord: ...
    async def get(self, action_id: str) -> ActionRecord | None: ...
    async def get_by_resume_token(self, token: str) -> ActionRecord | None: ...
    async def update(self, action: ActionRecord) -> ActionRecord: ...
    async def delete(self, action_id: str) -> bool: ...
    async def list(self, filter: ActionFilter | None = None) -> list[ActionRecord]: ...
    async def list_pending_for_job(self, job_id: str) -> list[ActionRecord]: ...
    async def list_expired(self) -> list[ActionRecord]: ...
```

Available implementations:
- `InMemoryActionStore`: Testing and single-process deployments
- `PostgresActionStore`: Production persistent storage (requires `asyncpg`)
- `RedisActionStore`: High-throughput scenarios (requires `redis`)

---

## Policy Engine

### Policy Rules

```python
from agent_runtime import PolicyEngine, ToolPolicy, ModelPolicy, ConstraintPolicy

engine = PolicyEngine()

# Tool policies
engine.add_rule(ToolPolicy(
    tool_patterns=["file_*", "shell_*"],  # Glob patterns
    decision="deny",
    reason="Dangerous tools not allowed",
))

engine.add_rule(ToolPolicy(
    tool_patterns=["search_*"],
    allowed_scopes=["internal", "admin"],  # Scope-based access
    decision="allow",
))

# Model policies
engine.add_rule(ModelPolicy(
    model_patterns=["gpt-4*", "claude-3*"],
    require_approval=True,
    max_tokens_per_request=10000,
))

# Constraint policies
engine.add_rule(ConstraintPolicy(
    max_cost_per_request=1.0,
    max_tokens_per_request=50000,
    max_tool_calls_per_turn=5,
))
```

### Policy Evaluation

```python
from agent_runtime import PolicyContext, PolicyDecision

# Create policy context
policy_ctx = PolicyContext(
    scope_id="tenant-123",
    principal_id="user-456",
    tool_name="file_read",
    model="gpt-4",
    requested_tokens=5000,
)

# Check specific policies
tool_result = await engine.check_tool("file_read", policy_ctx)
model_result = await engine.check_model("gpt-4", policy_ctx)
constraint_result = await engine.check_constraints(policy_ctx)

# Full evaluation
result = await engine.evaluate(policy_ctx)

if result.decision == PolicyDecision.DENY:
    raise PolicyDenied(result.reason, violations=result.violations)
elif result.decision == PolicyDecision.REQUIRE_APPROVAL:
    # Request human approval
    pass
```

### `PolicyResult`

```python
PolicyResult(
    decision=PolicyDecision.ALLOW,  # ALLOW, DENY, REQUIRE_APPROVAL, WARN
    reason="All policies passed",
    violations=[],                   # List of violated rules
    applied_rules=["rule1", "rule2"],
    metadata={},
)
```

---

## Ledger and Budgets

### `LedgerEvent`

```python
from agent_runtime import LedgerEvent, LedgerEventType

event = LedgerEvent(
    event_id="evt-123",
    type=LedgerEventType.MODEL_USAGE,
    timestamp=time.time(),
    job_id="job-456",
    run_id="run-789",
    scope_id="tenant-123",
    principal_id="user-456",
    provider="openai",
    model="gpt-4",
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
    cost="0.045",  # Decimal as string
    duration_ms=1234.5,
    metadata={},
)

# From llm-client Usage
event = LedgerEvent.from_llm_usage(
    usage,
    job_id="job-123",
    scope_id="tenant-123",
    provider="openai",
    model="gpt-4",
)
```

### `LedgerWriter` Interface

```python
from agent_runtime import LedgerWriter, InMemoryLedgerWriter

writer = InMemoryLedgerWriter()

# Write event
await writer.write(event)

# Query usage
usage = await writer.get_usage(
    scope_id="tenant-123",
    principal_id="user-456",
)

# List events
events = await writer.list_events(
    scope_id="tenant-123",
    event_type=LedgerEventType.MODEL_USAGE,
    limit=100,
)
```

### `Ledger`

```python
from agent_runtime import Ledger, Budget, BudgetDecision

ledger = Ledger(writer)

# Set budgets
ledger.set_budget(Budget(
    scope_id="tenant-123",
    max_tokens_daily=1000000,
    max_cost_daily=Decimal("100.00"),
    max_requests_daily=10000,
    warning_threshold=0.8,  # Warn at 80%
    exceed_strategy="deny",  # deny, degrade, warn
))

# Check budget before operation
decision, reason = await ledger.check_budget(
    ctx,
    pending_tokens=5000,
    pending_cost=Decimal("0.15"),
)

if decision == BudgetDecision.DENY:
    raise BudgetExceededError(reason)

# Record usage
await ledger.record_model_usage(ctx, usage)
await ledger.record_tool_usage(ctx, tool_name="search", duration_ms=123.4)
```

### `Budget`

```python
Budget(
    scope_id="tenant-123",
    principal_id="user-456",
    
    # Token limits
    max_tokens_per_request=10000,
    max_tokens_daily=1000000,
    max_tokens_monthly=10000000,
    max_tokens_total=100000000,
    
    # Cost limits
    max_cost_per_request=Decimal("1.00"),
    max_cost_daily=Decimal("100.00"),
    max_cost_monthly=Decimal("1000.00"),
    
    # Request limits
    max_requests_per_minute=60,
    max_requests_daily=10000,
    
    # Tool limits
    max_tool_calls_per_request=10,
    max_tool_calls_daily=1000,
    
    # Behavior
    warning_threshold=0.8,
    exceed_strategy="deny",  # deny, degrade, warn
)
```

---

## Event Bus

### `RuntimeEvent`

```python
from agent_runtime import RuntimeEvent, RuntimeEventType

event = RuntimeEvent(
    event_id="evt-123",
    type=RuntimeEventType.JOB_STARTED,
    timestamp=time.time(),
    job_id="job-456",
    run_id="run-789",
    trace_id="trace-abc",
    scope_id="tenant-123",
    principal_id="user-456",
    data={"model": "gpt-4", "prompt_tokens": 100},
)

# SSE formatting
sse_string = event.to_sse()

# From context
event = RuntimeEvent.from_context(ctx, RuntimeEventType.PROGRESS, {"progress": 0.5})
```

### Event Types

```python
RuntimeEventType.PROGRESS           # Progress update
RuntimeEventType.MODEL_TOKEN        # Token from model
RuntimeEventType.MODEL_REASONING    # Reasoning content
RuntimeEventType.MODEL_DONE         # Model response complete
RuntimeEventType.TOOL_START         # Tool execution started
RuntimeEventType.TOOL_END           # Tool execution completed
RuntimeEventType.TOOL_ERROR         # Tool execution failed
RuntimeEventType.ACTION_REQUIRED    # Human action needed
RuntimeEventType.ACTION_RESOLVED    # Action resolved
RuntimeEventType.ACTION_EXPIRED     # Action expired
RuntimeEventType.ACTION_CANCELLED   # Action cancelled
RuntimeEventType.ARTIFACT_CREATED   # Artifact produced
RuntimeEventType.ARTIFACT_UPDATED   # Artifact modified
RuntimeEventType.JOB_STARTED        # Job started
RuntimeEventType.JOB_STATUS_CHANGED # Job status changed
RuntimeEventType.JOB_COMPLETED      # Job succeeded
RuntimeEventType.JOB_FAILED         # Job failed
RuntimeEventType.JOB_CANCELLED      # Job cancelled
RuntimeEventType.FINAL_RESULT       # Final output
RuntimeEventType.FINAL_ERROR        # Final error
```

### `EventBus` Interface

```python
from agent_runtime import EventBus, InMemoryEventBus

bus = InMemoryEventBus()

# Subscribe
async def handler(event: RuntimeEvent):
    print(f"Event: {event.type}")

subscription = await bus.subscribe(
    handler,
    event_types={RuntimeEventType.JOB_STARTED, RuntimeEventType.JOB_COMPLETED},
)

# Publish
await bus.publish(RuntimeEvent(
    type=RuntimeEventType.JOB_STARTED,
    job_id="job-123",
))

# Unsubscribe
await subscription.unsubscribe()
```

### Event Adapters

```python
from agent_runtime import SSEEventAdapter, WebhookEventAdapter

# SSE streaming
sse_adapter = SSEEventAdapter()
await sse_adapter.start()

async for sse_string in sse_adapter.events():
    yield sse_string  # Send to client

# Webhook delivery (requires aiohttp)
webhook_adapter = WebhookEventAdapter(
    webhook_url="https://example.com/webhook",
    headers={"Authorization": "Bearer token"},
)
await webhook_adapter.adapt(event)
```

---

## Plugin System

### `Plugin` Interface

```python
from agent_runtime import Plugin, PluginManifest, PluginType

class MyToolsPlugin(Plugin):
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name="my-tools",
            version="1.0.0",
            plugin_type=PluginType.TOOLS,
            description="Custom tools for my application",
            capabilities=["search", "analyze"],
        )
    
    async def initialize(self, context: ExecutionContext) -> None:
        # Set up plugin resources
        pass
    
    async def shutdown(self) -> None:
        # Clean up resources
        pass
    
    def get_tools(self) -> list:
        return [search_tool, analyze_tool]
```

### `PluginRegistry`

```python
from agent_runtime import PluginRegistry

registry = PluginRegistry()

# Register plugin
registry.register(MyToolsPlugin())

# Get plugins by type
tool_plugins = registry.get_by_type(PluginType.TOOLS)

# Initialize all plugins
await registry.initialize_all(ctx)

# Get all tools from plugins
all_tools = registry.get_all_tools()

# Shutdown
await registry.shutdown_all()
```

---

## Runtime Kernel

### `RuntimeKernel`

The main orchestrator that ties all components together:

```python
from agent_runtime import RuntimeKernel, ExecutionRequest

# Create with defaults
kernel = RuntimeKernel.create(
    engine=execution_engine,
    agent=agent,
)

# Or with custom components
kernel = RuntimeKernel(
    engine=engine,
    agent=agent,
    job_store=PostgresJobStore(pool),
    action_store=PostgresActionStore(pool),
    ledger_writer=PostgresLedgerWriter(pool),
    policy_engine=PolicyEngine(),
    event_bus=InMemoryEventBus(),
    plugin_registry=PluginRegistry(),
)
```

### `ExecutionRequest`

```python
ExecutionRequest(
    prompt="Hello, world!",
    scope_id="tenant-123",
    principal_id="user-456",
    session_id="session-789",
    idempotency_key="unique-request-123",
    budgets=BudgetSpec(max_tokens=10000),
    policy_ref=PolicyRef(name="production"),
    metadata={"source": "api"},
    timeout_seconds=300.0,
)
```

### `ExecutionHandle`

```python
handle = await kernel.execute(request)

# Get job info
job = await handle.job()

# Stream events
async for event in handle.events():
    if event.type == RuntimeEventType.MODEL_TOKEN:
        print(event.data, end="")

# Get final result
result = await handle.result()
print(result.content)
print(result.usage)

# Cancel
await handle.cancel()
```

### `ExecutionResult`

```python
ExecutionResult(
    job_id="job-123",
    status="succeeded",
    content="Hello! How can I help you?",
    usage={
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "total_cost": "0.001",
    },
    turns=1,
    artifacts=[],
    error=None,
)
```

---

## Multi-Agent Orchestration

### Operators

```python
from agent_runtime import Operator, OperatorResult, OperatorContext, AgentRole

class SummarizerOperator(Operator):
    def __init__(self, agent):
        self._agent = agent
    
    @property
    def name(self) -> str:
        return "summarizer"
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.SPECIALIST
    
    @property
    def description(self) -> str:
        return "Summarizes text content"
    
    async def execute(
        self,
        input_data: dict,
        context: OperatorContext,
    ) -> OperatorResult:
        result = await self._agent.run(
            f"Summarize the following:\n\n{input_data['text']}"
        )
        return OperatorResult(
            content=result.content,
            success=True,
            operator_name=self.name,
            role=self.role,
        )
```

### Built-in Patterns

#### ChainOperator (Sequential)

```python
from agent_runtime import ChainOperator

chain = ChainOperator([
    preprocessor_op,
    analyzer_op,
    formatter_op,
])

result = await chain.execute({"text": "..."}, context)
```

#### ParallelOperator

```python
from agent_runtime import ParallelOperator

parallel = ParallelOperator(
    operators=[analyst1, analyst2, analyst3],
    aggregator=lambda results: {"combined": [r.content for r in results]},
    require_all_success=False,
)

result = await parallel.execute({"data": "..."}, context)
```

#### MapReduceOperator

```python
from agent_runtime import MapReduceOperator

map_reduce = MapReduceOperator(
    map_operator=document_summarizer,
    reduce_operator=summary_synthesizer,
    input_key="documents",
    max_parallel=5,
)

result = await map_reduce.execute({
    "documents": [doc1, doc2, doc3]
}, context)
```

#### PlannerExecutorOperator

```python
from agent_runtime import PlannerExecutorOperator

planner_executor = PlannerExecutorOperator(
    planner=planner_agent,
    executor=executor_agent,
    synthesizer=synthesizer_agent,
    max_subtasks=10,
    parallel_execution=True,
)

result = await planner_executor.execute({
    "task": "Research AI trends and write a report"
}, context)
```

#### DebateOperator

```python
from agent_runtime import DebateOperator

debate = DebateOperator(
    debaters=[agent1, agent2, agent3],
    moderator=moderator_agent,
    max_rounds=3,
    stop_on_consensus=True,
)

result = await debate.execute({
    "topic": "Best approach for solving X?"
}, context)
```

### Router

```python
from agent_runtime import DefaultRouter, RoutingRule

router = DefaultRouter(
    rules=[
        RoutingRule(
            operator_name="summarizer",
            keywords=["summarize", "summary", "brief"],
            priority=10,
        ),
        RoutingRule(
            operator_name="analyzer",
            keywords=["analyze", "analysis"],
            priority=5,
        ),
    ],
    default_operator="general",
)

decision = await router.route(
    {"prompt": "Please summarize this..."},
    available_operators,
)
```

### Graph Execution

```python
from agent_runtime import ExecutionGraph, GraphNode, GraphEdge, GraphExecutor

graph = ExecutionGraph()

# Add nodes
graph.add_node(GraphNode("planner", planner_op))
graph.add_node(GraphNode("exec1", executor_op))
graph.add_node(GraphNode("exec2", executor_op))
graph.add_node(GraphNode("synth", synthesizer_op))

# Define edges (data flow)
graph.add_edge(GraphEdge("planner", "exec1"))
graph.add_edge(GraphEdge("planner", "exec2"))
graph.add_edge(GraphEdge("exec1", "synth"))
graph.add_edge(GraphEdge("exec2", "synth"))

# Execute
executor = GraphExecutor(graph, max_parallel=5)
results = await executor.execute(input_data, context)

# Get output
final_content = executor.get_final_content()
```

---

## Replay System

### Recording

```python
from agent_runtime import ReplayRecorder, RunMetadata

metadata = RunMetadata.create(
    runtime_version="1.0.0",
    llm_client_version="0.2.0",
    model_version="gpt-4-0125",
    config={"max_turns": 10},
    tools=["search", "calculate"],
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

# Validate recording
is_valid, issues = await player.validate_recording()

# Fast replay (as fast as possible)
result = await player.replay(mode=ReplayMode.FAST)

# Timed replay (original timing)
result = await player.replay(mode=ReplayMode.TIMED)

# Deterministic replay (use recorded responses)
result = await player.replay(mode=ReplayMode.DETERMINISTIC)

# Step through manually
async for event in player.step():
    print(event)
    input("Press enter for next event...")
```

### Recording Format

```python
Recording(
    format_version=1,
    metadata=RunMetadata(...),
    initial_input={"prompt": "..."},
    events=[
        RecordedEvent(
            event=RuntimeEvent(...),
            fingerprint=EventFingerprint(...),
            relative_timestamp_ms=0.0,
            model_response="...",  # For deterministic replay
        ),
        ...
    ],
)
```

---

## Storage Adapters

### PostgreSQL

```python
import asyncpg
from agent_runtime import PostgresJobStore, PostgresActionStore, PostgresLedgerWriter

pool = await asyncpg.create_pool("postgresql://...")

job_store = PostgresJobStore(pool, table_name="runtime_jobs")
action_store = PostgresActionStore(pool, table_name="runtime_actions")
ledger_writer = PostgresLedgerWriter(pool, table_name="runtime_ledger")
```

**Table schemas** (auto-created):

- `runtime_jobs`: Job records with indexes on scope_id, status, idempotency_key
- `runtime_actions`: Action records with unique resume_token index
- `runtime_ledger`: Ledger events with indexes for querying

### Redis

```python
import redis.asyncio as redis
from agent_runtime import RedisSignalChannel, RedisJobCache, RedisActionStore

client = await redis.from_url("redis://localhost")

# Fast action signaling via pub/sub
signal_channel = RedisSignalChannel(client)
await signal_channel.start()

# Wait for action resolution (non-polling)
signal = await signal_channel.wait(action_id, timeout=300)

# Signal action resolution
await signal_channel.signal(action_id, SignalMessage(
    action_id=action_id,
    resume_token="...",
    status="resolved",
    resolution={"approved": True},
))

# Job cache (read-through)
job_cache = RedisJobCache(client, primary_store=postgres_job_store)
job = await job_cache.get(job_id)  # Hits Redis first

# Redis action store
action_store = RedisActionStore(client)
```

---

## Observability

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
    config=OTelConfig(
        tracer_name="agent_runtime",
        service_name="my-service",
        capture_tool_args=False,  # PII concern
        record_exceptions=True,
    ),
)

await adapter.start()

# Span hierarchy:
# - agent.job (parent): Full job execution
# - agent.run (child): Single LLM interaction
# - agent.tool (child): Tool execution
# - agent.action (child): Human-in-the-loop action

await adapter.stop()
```

---

## Queue Adapters

### Kafka

```python
from agent_runtime import KafkaEventAdapter, KafkaConfig

config = KafkaConfig(
    bootstrap_servers="kafka:9092",
    topic_prefix="myapp.agent",
    partition_by="job_id",  # or scope_id, session_id
    compression_type="gzip",
)

adapter = KafkaEventAdapter(event_bus, config)
await adapter.start()

# Events published to topics:
# - myapp.agent.job (job lifecycle)
# - myapp.agent.tool (tool events)
# - myapp.agent.action (action events)
# - myapp.agent.progress (progress updates)

await adapter.stop()
```

### Redis Streams

```python
from agent_runtime import RedisStreamsAdapter, RedisStreamsConfig

config = RedisStreamsConfig(
    stream_prefix="myapp:events",
    max_len=100000,
    consumer_group="workers",
    consumer_name="worker-1",
)

adapter = RedisStreamsAdapter(redis_client, event_bus, config)
await adapter.start()

# Read events
async for event in adapter.read_events("myapp:events:job"):
    print(event)

# Read range
events = await adapter.read_range(
    "myapp:events:job",
    start="-",  # Oldest
    end="+",    # Newest
    count=100,
)

await adapter.stop()
```

---

## Configuration Reference

### OrchestrationConfig

```python
OrchestrationConfig(
    max_child_jobs=10,           # Max child jobs per parent
    max_depth=5,                 # Max nesting depth
    timeout_seconds=300.0,       # Default timeout
    inherit_budgets=True,        # Child inherits parent budgets
    propagate_cancellation=True, # Cancel children on parent cancel
    parallel_execution=True,     # Run children in parallel
    max_parallel=5,              # Max concurrent children
)
```

### OTelConfig

```python
OTelConfig(
    tracer_name="agent_runtime",
    service_name="my-service",
    capture_input=False,         # PII concern
    capture_output=False,        # PII concern
    capture_tool_args=False,     # PII concern
    record_exceptions=True,
    sampling_rate=1.0,
)
```

### KafkaConfig

```python
KafkaConfig(
    bootstrap_servers="localhost:9092",
    topic_prefix="agent-runtime",
    default_topic="events",
    partition_by="job_id",
    acks="all",
    compression_type="gzip",
    batch_size=16384,
    linger_ms=10,
)
```

### RedisStreamsConfig

```python
RedisStreamsConfig(
    stream_prefix="agent:events",
    default_stream="all",
    max_len=100000,
    approximate_trim=True,
    consumer_group="default",
    consumer_name="worker-1",
    block_ms=5000,
)
```

---

## Integration with llm-client

### Context Propagation

```python
from llm_client import Agent, RequestContext
from agent_runtime import ExecutionContext

# Create execution context
exec_ctx = ExecutionContext(
    scope_id="tenant-123",
    principal_id="user-456",
    job_id="job-789",
)

# Convert to llm-client RequestContext
request_ctx = exec_ctx.to_request_context()

# Pass to Agent
result = await agent.run("Hello", context=request_ctx)
```

### Using Agent with Middleware

```python
from llm_client import Agent

# Enable middleware chain
agent = Agent(
    engine=engine,
    tools=tools,
    use_middleware=True,  # Uses production middleware defaults
)

# Middleware includes: logging, timeout, retry, budget, telemetry
```

### IdempotencyTracker in Engine

```python
from llm_client import ExecutionEngine, IdempotencyTracker

tracker = IdempotencyTracker(request_timeout=60.0)
engine = ExecutionEngine(
    provider=provider,
    idempotency_tracker=tracker,
)

# Idempotency works with agent-runtime job_id
result = await engine.complete(
    spec,
    context=exec_ctx.to_request_context(),
    idempotency_key=exec_ctx.job_id,
)
```

---

## Public API Index

### Context

- `ExecutionContext`, `BudgetSpec`, `PolicyRef`, `RunVersions`

### Jobs

- `JobStatus`, `JobRecord`, `JobManager`, `JobSpec`, `JobStore`, `InMemoryJobStore`, `JobFilter`

### Actions

- `ActionStatus`, `ActionType`, `ActionRecord`, `ActionManager`, `ActionSpec`, `ActionStore`, `InMemoryActionStore`, `ActionFilter`, `ActionRequiredError`

### Policy

- `PolicyEngine`, `PolicyDecision`, `PolicyDenied`, `PolicyRule`, `ToolPolicy`, `ModelPolicy`, `ConstraintPolicy`, `PolicyContext`, `PolicyResult`

### Ledger

- `Ledger`, `LedgerEvent`, `LedgerEventType`, `LedgerWriter`, `InMemoryLedgerWriter`, `Budget`, `BudgetDecision`, `UsageRecord`, `BudgetExceededError`

### Events

- `RuntimeEvent`, `RuntimeEventType`, `EventBus`, `InMemoryEventBus`, `EventSubscription`, `SSEEventAdapter`, `WebhookEventAdapter`

### Plugins

- `Plugin`, `PluginManifest`, `PluginRegistry`, `PluginType`, `PluginCapability`

### Runtime

- `RuntimeKernel`, `ExecutionRequest`, `ExecutionResult`, `ExecutionHandle`

### Orchestration

- `Operator`, `OperatorResult`, `OperatorContext`, `AgentRole`, `RoutingDecision`, `OrchestrationConfig`
- `Router`, `RoutingRule`, `DefaultRouter`
- `GraphExecutor`, `GraphNode`, `GraphEdge`, `ExecutionGraph`, `NodeResult`
- `ChainOperator`, `ParallelOperator`, `MapReduceOperator`, `PlannerExecutorOperator`, `DebateOperator`

### Replay

- `RunMetadata`, `EventFingerprint`, `ReplayValidationError`
- `ReplayRecorder`, `RecordedEvent`, `Recording`
- `ReplayPlayer`, `ReplayMode`, `ReplayResult`

### Storage (optional)

- `PostgresJobStore`, `PostgresActionStore`, `PostgresLedgerWriter` (requires `asyncpg`)
- `SignalMessage`, `RedisSignalChannel`, `RedisJobCache`, `RedisActionStore` (requires `redis`)

### Observability (optional)

- `OpenTelemetryAdapter`, `OTelConfig` (requires `opentelemetry-api`)

### Queue Adapters (optional)

- `KafkaEventAdapter`, `KafkaConfig` (requires `aiokafka`)
- `RedisStreamsAdapter`, `RedisStreamsConfig` (requires `redis`)
