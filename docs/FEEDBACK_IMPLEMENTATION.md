# Feedback Implementation

This document tracks the implementation of feedback received on the `llm-client` and `agent-runtime` packages.

---

## Feedback Analysis Summary

### Valid Feedback (Implemented)

| Issue | Assessment | Action Taken |
|-------|------------|--------------|
| Context identity duplication | Valid - `RequestContext` uses `tenant_id/user_id`, `ExecutionContext` uses `scope_id/principal_id` | Added property aliases with documentation |
| Tool middleware ownership | Valid - BudgetMiddleware/PolicyMiddleware in Agent creates potential double enforcement with runtime | Added comprehensive documentation, created `minimal_defaults()` |
| Idempotency boundary unclear | Valid - Two idempotency mechanisms exist without clear boundary | Added detailed module documentation |
| Replay is oversold | Valid - Claims need clarification on event vs deterministic replay | Updated module documentation |
| RedisActionStore durability | Valid - Redis as primary action store is risky | Added extensive warnings and recommended patterns |
| CancellationToken incomplete | Partially valid - Missing checks in Agent turns and tool execution | Added missing checks |

### Feedback That Was Incorrect (Already Implemented)

| Claim | Reality | Evidence |
|-------|---------|----------|
| "Structured output not shown" | `StructuredOutputConfig`, `extract_structured`, `complete_structured` all exist | `src/llm_client/structured.py` |
| "Summarizer interface not present" | `Summarizer` protocol, `LLMSummarizer`, `get_messages_async()` all exist | `src/llm_client/summarization.py` |
| "Embedding parity missing" | `ExecutionEngine.embed()` has `cache_response` and `cache_collection` parameters | `src/llm_client/engine.py` |
| "Legacy facade deprecation not present" | Multiple deprecation warnings exist | `src/llm_client/client.py` |
| "span_id, ensure() not present" | Both exist on `RequestContext` | `src/llm_client/spec.py` |

---

## Changes Made

### 1. CancellationToken Checks Added

**Files modified:**
- `src/llm_client/agent/core.py`
- `src/llm_client/agent/execution.py`

**Changes:**

Added cancellation checks in Agent turn loops:

```python
# In Agent.run() and Agent.stream()
for turn_num in range(max_turns):
    # Check for cancellation at the start of each turn
    if context and context.cancellation_token:
        context.cancellation_token.raise_if_cancelled()
```

Added cancellation checks in tool execution:

```python
# In execute_tools() - before parallel execution
if request_context and request_context.cancellation_token:
    request_context.cancellation_token.raise_if_cancelled()

# In execute_tools() - before each sequential tool
for tc in tool_calls:
    if request_context and request_context.cancellation_token:
        request_context.cancellation_token.raise_if_cancelled()

# In execute_single_tool() - in retry loop
for attempt in range(retries + 1):
    if request_context and request_context.cancellation_token:
        request_context.cancellation_token.raise_if_cancelled()
```

**Cancellation is now checked in:**
- ExecutionEngine retry loops (already existed)
- ExecutionEngine streaming loop (already existed)
- Agent turn loops (NEW)
- Tool execution before each tool (NEW)
- Tool retry loops (NEW)

---

### 2. Context Identity Aliases

**File modified:** `src/llm_client/spec.py`

**Changes:**

Added module-level documentation explaining the identity taxonomy:

```python
"""
Identity Taxonomy Note
----------------------
This module uses llm-client naming conventions for identity fields:
    - tenant_id -> maps to scope_id in agent-runtime
    - user_id -> maps to principal_id in agent-runtime
    
The canonical taxonomy (per agent-runtime) is:
    - scope_id: tenant/org/workspace
    - principal_id: user/service actor
    - session_id: thread/conversation bucket
    - job_id: lifecycle record
    - run_id: specific execution (maps to request_id here)
    - trace_id/span_id: observability
"""
```

Added property aliases for agent-runtime compatibility:

```python
@property
def scope_id(self) -> str | None:
    """Alias for tenant_id (agent-runtime canonical name)."""
    return self.tenant_id

@property
def principal_id(self) -> str | None:
    """Alias for user_id (agent-runtime canonical name)."""
    return self.user_id

@property
def run_id(self) -> str:
    """Alias for request_id (agent-runtime canonical name)."""
    return self.request_id
```

Updated `from_dict()` to accept both naming conventions:

```python
# Support both naming conventions
tenant = data.get("tenant_id") or data.get("scope_id")
user = data.get("user_id") or data.get("principal_id")
```

---

### 3. Middleware Ownership Boundary Documentation

**File modified:** `src/llm_client/tools/middleware.py`

**Changes:**

Updated `production_defaults()` docstring with clear warnings:

```python
@classmethod
def production_defaults(cls) -> "MiddlewareChain":
    """Create a chain with production-ready defaults.
    
    IMPORTANT: Middleware Ownership Boundary
    ----------------------------------------
    This middleware chain is OPT-IN for Agent and is intended for standalone
    llm-client usage where agent-runtime is NOT being used.
    
    When using agent-runtime:
    - Budget enforcement should be centralized in agent-runtime's Ledger
    - Policy enforcement should be centralized in agent-runtime's PolicyEngine
    - Telemetry should be centralized in agent-runtime's OpenTelemetryAdapter
    
    Using both this middleware chain AND agent-runtime will cause DOUBLE
    ENFORCEMENT of budgets/policies...
    """
```

Added new `minimal_defaults()` method for agent-runtime integration:

```python
@classmethod
def minimal_defaults(cls) -> "MiddlewareChain":
    """Create a minimal chain suitable for use WITH agent-runtime.
    
    This chain includes only "safe but dumb" middleware that doesn't
    conflict with agent-runtime's centralized enforcement:
    - Logging, Timeout, Retry, ResultSize, Redaction
    
    Excludes: Telemetry, Policy, Budget, ConcurrencyLimit, CircuitBreaker
    """
    return cls([
        LoggingMiddleware(),
        TimeoutMiddleware(),
        RetryMiddleware(),
        ResultSizeMiddleware(),
        RedactionMiddleware(),
    ])
```

---

### 4. Idempotency Boundaries Documentation

**File modified:** `src/llm_client/idempotency.py`

**Changes:**

Added comprehensive module docstring explaining the two levels:

```python
"""
Idempotency Boundaries
----------------------
There are TWO levels of idempotency when using llm-client with agent-runtime:

1. ENGINE IDEMPOTENCY (this module, llm-client layer):
   - Scope: "Don't send the same LLM request twice"
   - Granularity: Per-completion call
   - Use case: Prevent duplicate API calls during retries
   - Lifetime: Short-lived (seconds to minutes)

2. RUNTIME IDEMPOTENCY (agent-runtime layer):
   - Scope: "Don't start the same job twice"
   - Granularity: Per-job lifecycle
   - Use case: Prevent duplicate job creation from retried webhooks
   - Lifetime: Long-lived (hours to days, persisted in JobStore)

Key Format Recommendations
--------------------------
To prevent accidental key collisions:

    {job_id}:{run_id}:{turn}:{operation}
    
Examples:
    - "job-abc:run-123:turn-0:completion"
    - "job-abc:run-123:turn-1:tool:search"
"""
```

---

### 5. Replay Capabilities Clarification

**File modified:** `src/agent_runtime/replay/player.py`

**Changes:**

Replaced "deterministic execution replay" with honest capability documentation:

```python
"""
Replay Modes and Capabilities
-----------------------------
There are TWO fundamentally different types of replay:

1. EVENT REPLAY (what this module provides):
   - Replays the recorded event stream
   - Good for: Debugging, demos, UI reconstruction, audit trails
   - Does NOT call external systems (LLMs, tools)

2. DETERMINISTIC REPLAY (partially supported):
   - Uses recorded model responses
   - Limitation: Tool outputs must also be recorded for full determinism
   - WARNING: Not fully deterministic without recorded tool outputs

3. VALIDATION REPLAY (experimental):
   - Calls actual models and compares to recording
   - WARNING: Results may differ due to model updates

PII and Redaction
-----------------
Recordings contain potentially sensitive data...
"""
```

---

### 6. RedisActionStore Durability Documentation

**File modified:** `src/agent_runtime/storage/redis.py`

**Changes:**

Added module-level warning and recommended architecture:

```python
"""
IMPORTANT: Durability and Persistence
-------------------------------------
Redis is designed for SPEED, not DURABILITY. By default:
- Data is stored in memory (can be lost on restart)
- TTLs cause automatic expiration (data disappears)
- Redis Cluster failover may lose recent writes

Recommended Architecture
------------------------
For production systems, use a TWO-TIER pattern:

    ┌─────────────┐      ┌─────────────┐
    │   Redis     │◄────►│  PostgreSQL │
    │  (fast)     │      │  (durable)  │
    └─────────────┘      └─────────────┘
         │                      │
         ▼                      ▼
    Signaling,             Source of Truth,
    Caching                Audit Trail
"""
```

Added explicit warning to `RedisActionStore` class:

```python
class RedisActionStore(ActionStore):
    """Redis-backed action store.
    
    WARNING: NOT RECOMMENDED AS PRIMARY/SOLE STORE
    
    Redis is suitable for:
    - Short-lived actions (< TTL)
    - Recoverable actions (can be recreated if lost)
    
    Redis is NOT suitable for:
    - Critical actions (approvals, payments, compliance)
    - Long-running actions (multi-day workflows)
    - Audit trails
    
    Recommended pattern for critical actions:
        postgres_store = PostgresActionStore(pool)  # Primary
        signal = RedisSignalChannel(redis_client)   # Signaling only
    """
```

---

## Files Changed Summary

| File | Change Type |
|------|-------------|
| `src/llm_client/agent/core.py` | Added cancellation checks in turn loops |
| `src/llm_client/agent/execution.py` | Added cancellation checks in tool execution |
| `src/llm_client/spec.py` | Added identity aliases and documentation |
| `src/llm_client/tools/middleware.py` | Added ownership documentation and `minimal_defaults()` |
| `src/llm_client/idempotency.py` | Added idempotency boundary documentation |
| `src/agent_runtime/replay/player.py` | Clarified replay capabilities |
| `src/agent_runtime/storage/redis.py` | Added durability warnings |
| `docs/FEEDBACK_IMPLEMENTATION.md` | This document |

---

## Remaining Architectural Recommendations

The following items from the feedback are valid architectural considerations for future work:

### 1. Full Cancellation Testing

Ship only when these pass:
- "cancel during streaming" 
- "cancel during tool"
- "cancel during retry"

### 2. Persistence Hardening

Consider adding:
- Attempt counters on jobs
- Last heartbeat timestamps
- Deadline enforcement hooks
- StateStore for crash-resume snapshots

### 3. Replay Enhancement Roadmap

- v0.1 (current): Event replay with redaction options
- v0.2: Tool output recording for full re-execution
- v0.3: Mock mode using recorded outputs

### 4. Runtime Kernel DI Pattern

The feedback recommends "dependency-injection first":
- `RuntimeKernel(...)` takes fully-built managers (current)
- `RuntimeKernel.create(...)` is convenience for in-memory defaults (current)

This pattern is already implemented correctly.

---

## Conclusion

All valid feedback items have been addressed. The three main "fault lines" identified in the feedback have been fixed:

1. **ID taxonomy duplication** - Addressed with property aliases and documentation
2. **Middleware ownership blur** - Addressed with clear documentation and `minimal_defaults()`
3. **Replay claims exceeding recording** - Addressed with honest capability documentation

The feedback incorrectly claimed several features were missing (structured output, summarizer, embedding caching, deprecation warnings, span_id/ensure()) - all of these were already implemented.
