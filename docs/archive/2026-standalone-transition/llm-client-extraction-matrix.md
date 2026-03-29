# llm-client Extraction Matrix

This matrix captures what should move into `llm_client`, what should stay
outside it, and what should be redesigned before promotion.

## Classification Keys

- `Extract`: generic runtime substrate that belongs in `llm_client`
- `Redesign`: concept belongs in `llm_client`, but current shape is too
  repo-specific
- `External`: should stay outside the package
- `Deprecate`: legacy surface to phase out rather than promote

## `agent_runtime`

| Area | Classification | Notes |
| --- | --- | --- |
| execution context / budgets / policy refs | Extract | already migrated into package-level context types |
| runtime events and replay primitives | Extract | already promoted into package runtime/replay modules |
| cancellation bridging and engine-backed agent execution | Extract | already on canonical package path |
| storage and job persistence | External | deployment/runtime concern |
| transport adapters / workflow delivery | External | not part of core LLM runtime |
| plugin storage and deployment glue | External | package may reserve plugin namespace later, but not current storage/wiring |
| business/job lifecycle orchestration | External | repo/runtime specific |

## `intelligence_layer`

| Area | Classification | Notes |
| --- | --- | --- |
| structured tool/runtime loop primitives | Extract | largely promoted; remaining operator policy stays outside |
| context-planning heuristics | Extract | generic truncation/scoring logic belongs in package |
| request shaping and content-envelope helpers | Extract | promoted into package helpers |
| domain operators and manifests | External | business/domain layer |
| prompts, policies, and workflow contracts | External | not generic runtime substrate |
| switchboard intent and domain planning logic | External | product-specific behavior |
| platform tools and loaders | External | environment/project specific |
| operator result coercion / contract enforcement | Redesign | generic envelope pieces belong in package, domain result shaping stays outside |

## Current Direction

The current modernization work should keep following this pattern:

- extract generic LLM runtime substrate upward
- redesign ambiguous repo-specific abstractions before promotion
- keep business/domain logic outside the package
- deprecate legacy compatibility surfaces instead of quietly expanding them
- normalize consumer runtime execution onto `ExecutionEngine` once provider
  bootstrap is complete
