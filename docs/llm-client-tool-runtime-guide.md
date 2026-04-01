# llm-client Tool Runtime Guide

This guide covers the package-level tool system: tool definition, registry,
middleware, execution modes, and agent integration.

Runnable examples:

- [08_tool_execution_modes.py](/home/namiral/Projects/Packages/llm-client-v1/examples/08_tool_execution_modes.py)
- [09_tool_calling_agent.py](/home/namiral/Projects/Packages/llm-client-v1/examples/09_tool_calling_agent.py)

## Core concepts

The standalone tool system has four layers:

1. `Tool`
2. `ToolRegistry`
3. `ToolExecutionEngine`
4. middleware chain

That separation matters:

- a `Tool` is one callable capability
- the registry is lookup/validation
- the execution engine is batching and execution policy
- middleware handles cross-cutting controls

## Defining tools

You can define tools either:

- with the `@tool` decorator
- by constructing `Tool(...)` directly

Use the decorator for most cookbook and application code. Use direct `Tool(...)`
construction when you need explicit schemas or programmatic generation.

## Execution modes

The tool runtime supports:

- `single`
- `sequential`
- `parallel`
- `planner`

Use them intentionally:

- `single`: one tool only, ignore the rest
- `sequential`: preserve order and simplify dependency reasoning
- `parallel`: maximize throughput for independent calls
- `planner`: execute only the planner-selected call and leave branching logic
  above the runtime

The runnable reference is
[08_tool_execution_modes.py](/home/namiral/Projects/Packages/llm-client-v1/examples/08_tool_execution_modes.py).

## Middleware

The middleware stack is where tool execution becomes production-grade.

Available controls include:

- logging
- timeout
- retry
- allowlist/denylist policy
- budget enforcement
- concurrency limits
- circuit breaking
- result-size limits
- redaction
- telemetry

This means tool policy should usually live in middleware, not in the tool
handler itself.

## Agent integration

`Agent` composes directly with the tool runtime. The agent layer does not need
to reimplement tool execution mechanics; it selects tools, hands them to the
runtime, and interprets the resulting tool outputs inside the conversation
loop.

The runnable reference is
[09_tool_calling_agent.py](/home/namiral/Projects/Packages/llm-client-v1/examples/09_tool_calling_agent.py).

## Practical recommendation

For standalone projects:

1. start with decorator-defined tools
2. keep handlers pure and small
3. move operational controls into middleware
4. choose execution mode explicitly
5. only then layer an agent over the tool runtime
