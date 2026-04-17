# Tool Creation Guide For `llm-client`

This guide is the focused package-level rulebook for creating tools with
`llm-client`. It covers the tool surfaces that actually exist in this repo and
the conventions the package already enforces.

## Start With The Right Tool Surface

There are two different categories of "tools" in this package:

### 1. Executable package tools

Use these when the package should execute your Python handler locally.

Primary types:

- `llm_client.tools.Tool`
- `@tool`
- `@sync_tool`

This is the default choice for application tools, agents, and adaptor-backed
capabilities.

### 2. Provider-native OpenAI Responses tools

Use these when OpenAI should receive a typed tool descriptor directly rather
than a locally executable Python handler.

Primary types:

- `ResponsesBuiltinTool`
- `ResponsesMCPTool`
- `ResponsesCustomTool`
- `ResponsesGrammar`

These are only for OpenAI Responses-native workflows. They are not substitutes
for local `Tool` handlers.

## Default Recommendation

For most new tools in this package:

1. Create a local executable `Tool`.
2. Keep the handler thin and domain-specific.
3. Return plain dicts or `ToolResult`.
4. Put operational behavior in `ToolExecutionMetadata` and middleware, not in
   ad hoc handler logic.
5. Only use OpenAI-native tool descriptors when you explicitly want hosted
   tools, MCP, connectors, or grammar-based custom tools in the Responses API.

## How Local Tools Should Be Created

### Preferred Path: `@tool` For Simple Async Tools

Use `@tool` when:

- the handler is already async
- the function signature can cleanly express the schema
- a lightweight docstring is enough to describe parameters

Example:

```python
from llm_client.tools import tool


@tool
async def get_incident_snapshot(service: str) -> dict[str, str]:
    """Get the current incident snapshot for a service.

    service: Canonical service name.
    """
    return {
        "service": service,
        "status": "active",
        "severity": "SEV-1",
    }
```

What the decorator gives you:

- tool name defaults to the function name
- description defaults to the docstring's first paragraph
- parameter schema is inferred from annotations
- `additionalProperties` is set to `False`
- required fields come from parameters without defaults

Use `@sync_tool` instead of `@tool` for synchronous functions that should run in
an executor.

### Explicit Path: `Tool(...)` For Stable Production Tools

Construct `Tool(...)` directly when:

- you need an exact JSON Schema
- you want explicit execution metadata
- you are building tools programmatically
- schema inference from annotations would be too loose

Example:

```python
from llm_client.tools import Tool, ToolExecutionMetadata


async def summarize_release_notes(release_notes: str) -> dict[str, object]:
    return {
        "summary": "Release is close but still blocked on validation.",
        "source_excerpt": release_notes[:180],
    }


release_notes_tool = Tool(
    name="summarize_release_notes",
    description="Summarize the release notes into the main readiness themes.",
    parameters={
        "type": "object",
        "properties": {
            "release_notes": {"type": "string"},
        },
        "required": ["release_notes"],
        "additionalProperties": False,
    },
    handler=summarize_release_notes,
    strict=True,
    execution=ToolExecutionMetadata(
        timeout_seconds=10.0,
        retry_attempts=0,
        concurrency_limit=2,
        safety_tags=("read-only",),
        trust_level="high",
    ),
)
```

This is the better choice for package-quality tools because the schema and
runtime behavior stay explicit.

## Design Rules For Local Tool Handlers

### Keep the handler small

A tool handler should usually:

- validate or normalize small input details only when the schema cannot express
  them
- call a service, adaptor, repository, or domain function
- return structured output

It should not become the place where unrelated policy, retry, logging, or
budget logic accumulates.

### Prefer async handlers

`Tool.handler` is async-first. If the underlying work is sync, use `@sync_tool`
or wrap sync code intentionally.

### Return simple, model-friendly output

Best return shapes:

- `dict[str, Any]`
- `str`
- `ToolResult`

If the handler returns:

- `dict`, the runtime wraps it as a success result
- `str`, the runtime wraps it as a success result
- `ToolResult`, the runtime preserves explicit success/error/metadata

Do not return arbitrary custom objects if a dict would do.

### Use `ToolResult` when partial or error semantics matter

Use `ToolResult` when you need:

- partial-success signaling
- structured metadata
- explicit tool-level error results without raising

Example:

```python
from llm_client.tools import ToolResult


async def draft_next_actions(owner: str) -> ToolResult:
    return ToolResult(
        content={"actions": [f"Ask {owner} to validate dashboards."]},
        success=True,
        metadata={
            "partial": True,
            "partial_reason": "Suggested actions still need operator review.",
        },
    )
```

## Schema Rules

The package expects tool schemas to be valid JSON Schema objects. In practice,
new tools should follow these rules:

### Use an object schema

Prefer:

```python
{
    "type": "object",
    "properties": {...},
    "required": [...],
    "additionalProperties": False,
}
```

### Close the schema unless you have a strong reason not to

`additionalProperties: False` is the package norm in examples and inferred
decorator tools. Keep the schema closed so model calls are easier to validate
and reason about.

### Use `strict=True` when schema enforcement matters

`Tool.execute()` only validates arguments against the schema when `strict=True`.
If you care about hard argument validation, set it explicitly.

### Keep names and descriptions model-usable

Tool names should be:

- stable
- specific
- action-oriented
- safe to expose to a model

Descriptions should tell the model when to call the tool and what it returns,
not repeat the name.

Good:

- `identify_operational_risks`
- `Compute a concrete list of release risks from blocker details.`

Avoid:

- vague names like `lookup`
- descriptions that describe implementation rather than capability

## Execution Metadata Rules

Attach runtime behavior with `ToolExecutionMetadata`, not ad hoc handler code.

Fields already supported by the package:

- `timeout_seconds`
- `retry_attempts`
- `concurrency_limit`
- `safety_tags`
- `trust_level`

Use them like this:

- `timeout_seconds` for expected maximum runtime
- `retry_attempts` for transient failures
- `concurrency_limit` when backend fan-out must be bounded
- `safety_tags` for policy and observability
- `trust_level` for downstream reasoning or audit context

This metadata is used by the tool execution engine and surfaced in execution
envelopes.

## Where Policy Should Live

Do not bury production controls inside each tool handler unless the control is
truly domain-specific.

Prefer middleware for cross-cutting concerns such as:

- logging
- timeout enforcement
- retries
- allow/deny policy
- budgets
- concurrency controls
- circuit breaking
- redaction
- telemetry

Relevant package surface:

- `llm_client.tools.MiddlewareChain`
- `LoggingMiddleware`
- `TimeoutMiddleware`
- `RetryMiddleware`
- `PolicyMiddleware`
- `BudgetMiddleware`
- `ConcurrencyLimitMiddleware`
- `CircuitBreakerMiddleware`
- `ResultSizeMiddleware`
- `RedactionMiddleware`
- `TelemetryMiddleware`
- `ToolOutputPolicyMiddleware`

## Registration And Execution

For local executable tools, the standard flow is:

1. create `Tool` objects
2. register them in `ToolRegistry`
3. execute them through `ToolExecutionEngine` or an `Agent`

Minimal pattern:

```python
from llm_client.agent import ToolExecutionMode
from llm_client.providers.types import ToolCall
from llm_client.tools import ToolExecutionEngine, ToolRegistry

registry = ToolRegistry([release_notes_tool])
engine = ToolExecutionEngine(registry)

batch = await engine.execute_calls(
    [ToolCall(id="call_1", name="summarize_release_notes", arguments='{"release_notes":"..."}')],
    mode=ToolExecutionMode.SEQUENTIAL,
)
```

Choose execution mode intentionally:

- `SINGLE` when only one call should run
- `SEQUENTIAL` when order matters
- `PARALLEL` when calls are independent
- `PLANNER` when higher-level planning decides branching

## Agent Integration Rule

If a tool is meant for an agent, create it the same way as any other local
`Tool`. The `Agent` composes with the tool runtime; it does not require a
different tool definition style.

Canonical pattern:

```python
agent = Agent(
    provider=provider,
    tools=[get_incident_snapshot, get_recent_alerts],
    definition=AgentDefinition(
        execution_policy=AgentExecutionPolicy(
            tool_execution_mode=ToolExecutionMode.PARALLEL,
            max_tool_calls_per_turn=8,
        ),
    ),
)
```

## Shared Infrastructure Tools Should Usually Be Builders

For reusable service-backed tools, follow the package's adaptor-builder pattern
instead of rewriting similar handlers repeatedly.

Existing examples:

- SQL: `build_sql_query_tool`, `build_sql_execute_tool`
- Redis: `build_redis_get_tool`, `build_redis_set_tool`, and related helpers
- Vector: `build_vector_search_tool`, `build_vector_upsert_tool`,
  `build_vector_delete_tool`

Use this pattern when:

- multiple apps need the same capability shape
- the tool is just a thin wrapper around an adaptor
- execution metadata should be injectable by the caller

## OpenAI Responses-Native Tools

Use provider-native descriptors only when you are targeting OpenAI Responses
API features that are not local Python handlers.

### Built-in hosted tools

Use `ResponsesBuiltinTool` for hosted tools such as:

- web search
- file search
- code interpreter
- computer use
- image generation

Example:

```python
from llm_client.tools import ResponsesBuiltinTool

tool = ResponsesBuiltinTool.file_search(
    vector_store_ids=["vs_123"],
    max_num_results=5,
)
```

### MCP and connectors

Use `ResponsesMCPTool` for remote MCP servers or connector-backed tools.

Example:

```python
from llm_client.tools import ResponsesConnectorId, ResponsesMCPTool

connector = ResponsesMCPTool.connector(
    ResponsesConnectorId.GMAIL,
    authorization="Bearer oauth-token",
)
```

### Grammar-backed custom tools

Use `ResponsesCustomTool` with `ResponsesGrammar` when the Responses API expects
a grammar-defined custom tool rather than a Python handler.

Example:

```python
from llm_client.tools import ResponsesCustomTool, ResponsesGrammar

planner = ResponsesCustomTool(
    name="planner",
    description="Emit a terse plan.",
    grammar=ResponsesGrammar(syntax="lark", definition='start: "done"'),
)
```

## Important OpenAI-Specific Constraints

When passing local function tools through the OpenAI provider:

- tool names are sanitized to valid OpenAI-compatible aliases when needed
- duplicate aliases are disambiguated
- unsupported schema shapes are sanitized

When using the Responses API specifically:

- function tools are flattened into Responses function-tool shape
- function tools default to `strict=True` in request translation
- native descriptors like `ResponsesBuiltinTool` and `ResponsesMCPTool` are
  supported directly

Implication:

- keep your original tool names stable for package code
- do not hand-author code that depends on OpenAI alias names

## Testing Expectations For New Tools

At minimum, new package-quality tools should usually have tests for:

1. successful execution
2. bad or malformed arguments when strict validation matters
3. error or partial-result behavior
4. execution metadata if retries, timeouts, or concurrency limits are part of
   the contract

If the tool is adaptor-backed, test the builder contract separately from the
backend implementation.

## Recommended Creation Checklist

Use this checklist before adding a new tool:

1. Decide whether this should be a local executable `Tool` or an OpenAI
   Responses-native descriptor.
2. Pick `@tool` for simple async handlers or `Tool(...)` for explicit package
   contracts.
3. Use a closed object schema and set `strict=True` when validation matters.
4. Keep the handler thin and return dicts or `ToolResult`.
5. Attach `ToolExecutionMetadata` instead of embedding operational policy in the
   handler.
6. Register via `ToolRegistry` and execute via `ToolExecutionEngine` or
   `Agent`.
7. Add tests for success, failure, and any runtime contract you are depending
   on.

## Best Package References

Read these files when implementing a new tool:

- `llm_client/tools/base.py`
- `llm_client/tools/decorators.py`
- `llm_client/tools/execution_engine.py`
- `llm_client/tools/runtime.py`
- `llm_client/adapters/tools.py`
- `docs/llm-client-tool-runtime-guide.md`
- `examples/08_tool_execution_modes.py`
- `examples/09_tool_calling_agent.py`
- `examples/37_sql_adaptor_tools.py`
- `examples/50_openai_mcp_and_connector_workflows.py`

## Short Version

If you want the package-standard answer:

- create local tools as `Tool` objects
- prefer `@tool` for simple async handlers
- prefer explicit `Tool(...)` for reusable or production-grade contracts
- keep schemas closed and handlers thin
- push policy into metadata and middleware
- use OpenAI-native tool descriptors only for Responses hosted tools, MCP, or
  connector workflows
