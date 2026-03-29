# llm-client Adoption Notes for `intelligence_layer`

These notes describe how `intelligence_layer` should consume `llm_client`
without pulling business/domain concerns down into the package.

## Current 1.0 Status

As of `1.0.0`, `intelligence_layer` is materially using `llm_client` as the
generic LLM and agentic kernel.

The generic pieces now consumed from `llm_client` include:

- engine-backed execution
- content envelopes and content projection
- structured output runtime helpers
- tool runtime helpers
- generic context planning and assembly primitives

## Use `llm_client` For Generic LLM Mechanics

`intelligence_layer` should rely on `llm_client` for:

- `ExecutionEngine`
- canonical request/result/content/context types
- structured output runtime helpers
- tool runtime helpers
- generic context-planning heuristics

## Keep Domain Logic Outside The Package

The following should remain in `intelligence_layer`:

- domain operators
- manifests and workflow contracts
- prompts and business policies
- switchboard/domain planning logic
- platform-specific tools and loaders

## Recommended Import Posture

Prefer:

- `llm_client.engine`
- `llm_client.content`
- `llm_client.types`
- `llm_client.structured`
- `llm_client.tools`
- `llm_client.context`

Avoid introducing new dependencies on:

- `llm_client.container`
- top-level compatibility aliases
- provider-specific modules unless the call site is intentionally low-level

## Migration Guidance

1. Prefer engine-backed execution paths over raw provider calls.
2. Use package-level structured/tool runtime primitives for generic LLM loops.
3. Keep domain result shaping and operator contracts above `llm_client`.
4. Continue moving generic heuristics into package-level context/observability
   layers when they stop being domain-specific.

## Post-Extraction Rule

Provider objects may still be accepted in constructors for bootstrap
convenience, but `intelligence_layer` should normalize them into an
`ExecutionEngine` immediately and treat the engine as the canonical runtime
dependency.

That means:

- higher-level structured execution should require an engine-backed path
- provider inspection for capability selection is acceptable when derived from
  the engine
- new provider-first execution branches should not be introduced unless the
  call site is intentionally low-level
