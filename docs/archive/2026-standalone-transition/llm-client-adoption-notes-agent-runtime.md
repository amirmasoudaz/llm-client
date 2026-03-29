# llm-client Adoption Notes for `agent_runtime`

These notes describe how `agent_runtime` should consume `llm_client` after the
modernization work.

## Current 1.0 Status

As of `1.0.0`, `agent_runtime` is materially a consumer of `llm_client`, not a
shadow owner of the generic LLM substrate.

The generic pieces now owned by `llm_client` include:

- execution context and budget types
- runtime events
- replay primitives
- budget and ledger primitives
- engine-backed agent execution substrate

## Use `llm_client` As the Runtime Substrate

`agent_runtime` should treat `llm_client` as the canonical source for:

- `ExecutionEngine`
- `Agent`
- execution context and budget types
- runtime events
- replay primitives
- structured parsing helpers where needed

## Recommended Import Posture

Prefer:

- `llm_client.engine`
- `llm_client.agent`
- `llm_client.context`
- `llm_client.observability`
- `llm_client.types`

Avoid introducing new dependencies on:

- `llm_client.container`
- top-level compatibility aliases
- low-level helper modules when a stable namespace exists

## What Should Stay In `agent_runtime`

- job state management
- storage and persistence
- transports and workflow delivery
- deployment/runtime glue
- repo-specific orchestration and lifecycle semantics

## Migration Guidance

1. Prefer engine-backed agent creation over raw-provider wiring.
2. Consume package-level context/event/replay types directly.
3. Keep storage/transports outside `llm_client`.
4. Use stable module namespaces rather than top-level convenience imports.
