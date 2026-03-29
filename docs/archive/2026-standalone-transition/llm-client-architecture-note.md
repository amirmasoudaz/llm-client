# llm-client Architecture Note

This note defines the intended role of `llm_client` inside this repository and
the extraction boundary between it, `agent_runtime`, and `intelligence_layer`.

It is maintainer-facing and should be read alongside:

- [llm-client-public-api-v1.md](./llm-client-public-api-v1.md)
- [llm-client-modernization-roadmap-2026-03-09.md](./llm-client-modernization-roadmap-2026-03-09.md)

## Layer 0 Responsibilities

`llm_client` is the core framework layer for generic LLM and agentic runtime
capabilities. Its responsibility boundary is:

- provider abstraction and provider registry
- model metadata and model selection
- canonical request/result/content/context types
- execution engine behavior: retry, timeout, idempotency, failover, hooks
- agent loop and tool execution runtime
- structured outputs and schema-aware response handling
- canonical observability/runtime-events/replay primitives
- generic cache/routing/validation/error taxonomy

## Non-Goals

`llm_client` should not own:

- business operators or product workflows
- domain prompts, manifests, routing policy, or intent policy
- application storage, transports, deployment glue, or web handlers
- tenant/business authorization semantics
- SSE/UI projections or app-specific event formatting
- domain entity loaders and product-specific memory stores

## Module Classification

| Module / Namespace | Classification | Notes |
| --- | --- | --- |
| `llm_client.providers` | stable | canonical provider entry point |
| `llm_client.models` | stable | model profiles |
| `llm_client.types` | stable | canonical request/result/event types |
| `llm_client.content` | stable | canonical content model |
| `llm_client.context` | stable | execution context / budgets / policy refs |
| `llm_client.engine` | stable | orchestration runtime |
| `llm_client.agent` | stable | agent loop |
| `llm_client.tools` | stable | tool definitions and middleware |
| `llm_client.cache` | stable | cache abstractions and supported backends |
| `llm_client.observability` | stable | hooks, diagnostics, runtime events, replay |
| `llm_client.validation` | stable | validation entry point |
| `llm_client.errors` | stable | error taxonomy |
| `llm_client.config` | stable | settings and env loading |
| `llm_client.compat` | compatibility | legacy facade namespace |
| `llm_client.advanced` | advanced | lower-level helper/integration namespace |
| `llm_client.container` | legacy | compatibility/integration surface, not preferred |
| `llm_client.client` | legacy | retained for `OpenAIClient` compatibility |
| `llm_client.idempotency` | advanced | low-level helper, prefer engine APIs in most code |
| `llm_client.hashing` / `perf` / `serialization` / `streaming` | advanced | helper modules, use via `llm_client.advanced` when possible |
| provider-specific translator internals | private | not part of public contract |
| request-builder internals and extraction shims | provisional | may be promoted or folded later |

## Repo Import Inventory

Current high-level import pattern across repo consumers:

### `agent_runtime`

Primary `llm_client` usage:

- execution context types from `llm_client.context`
- runtime events from `llm_client.runtime_events`
- replay primitives from `llm_client.replay`
- agent/engine use through top-level `llm_client` imports in runtime boot paths
- structured validation in `agent_runtime.orchestration.router`

Assessment:

- most generic substrate has already been extracted successfully
- remaining top-level imports should continue moving toward canonical module
  namespaces over time

### `intelligence_layer`

Primary `llm_client` usage:

- `ExecutionEngine`
- `ContentResponseEnvelope` and content helpers
- request builders
- structured runtime helpers
- tool runtime helpers
- context-planning heuristics
- retry classification
- selected provider internals in bootstrap paths/tests

Assessment:

- `intelligence_layer` now depends on `llm_client` for the right generic core
- remaining direct provider/internal imports should stay narrow and explicit

## Post-Extraction Layering Rule

After the extraction program, the intended layering is:

- `llm_client`: canonical generic LLM/agent runtime framework
- `agent_runtime`: runtime host shell, storage, transport, and deployment glue
- `intelligence_layer`: domain/operator/policy layer

Practical rule:

- provider-only constructor inputs are acceptable as compatibility or bootstrap
  convenience
- once a higher-level runtime object exists, it should normalize to
  `ExecutionEngine` and execute through the engine
- direct-provider execution outside `llm_client` should be treated as an
  exception that must be justified, not as a normal integration style

## Direct-Provider Execution Paths

Current direct-provider paths that bypass the engine:

| Path | Classification | Notes |
| --- | --- | --- |
| `llm_client.structured` low-level provider branch | intentional | needed for low-level/provider-first structured use |
| `llm_client.tools.runtime` provider branch | intentional | low-level runtime helper fallback |
| `llm_client.streaming` adapter helpers over `provider.stream(...)` | intentional | utility layer, not engine replacement |
| `llm_client.client` image/pdf helper flows | legacy | compatibility-only; should not grow |

Policy:

- direct-provider execution is allowed only in explicit low-level/advanced or
  legacy compatibility surfaces
- higher-level application/runtime code should prefer `ExecutionEngine`
- repo consumers should not preserve parallel provider-first and engine-first
  paths for the same generic runtime behavior

## Semver and Deprecation Policy

Until `v1` is cut, the package should behave as if semver discipline already
applies to the stable namespaces.

Stable namespace policy:

- breaking changes require an explicit migration note
- removals should follow a deprecation cycle unless urgent for correctness
- top-level convenience exports should not expand casually

Compatibility policy:

- compatibility surfaces may warn
- new code should not be directed toward them
- removals should happen only after migration guidance exists

Advanced policy:

- advanced helpers are supported but lower-commitment than stable namespaces
- prefer additive change over silent breakage
- keep them out of top-level convenience imports

## Experimental Module Policy

An experimental/provisional module must:

- define `__all__`
- say its status in the module docstring or docs
- avoid accidental promotion through top-level imports
- ship with focused tests before being referenced as a recommended API

## Extraction Matrix

| Candidate | Source Area | Category | Notes |
| --- | --- | --- | --- |
| execution context / budgets / policy refs | `agent_runtime` | extract | now canonical in `llm_client.context` |
| runtime events / event bus | `agent_runtime` | extract | now canonical in `llm_client.runtime_events` / `observability` |
| replay primitives | `agent_runtime` | extract | now canonical in `llm_client.replay` / `observability` |
| structured tool runtime core | `intelligence_layer` | extract | now in `llm_client.tools.runtime` |
| context-planning heuristics | `intelligence_layer` | extract | now in `llm_client.context_planning` |
| operator-specific result shaping | `intelligence_layer` | external | stays outside package |
| manifests/prompts/policies/intent logic | `intelligence_layer` | external | business/domain layer |
| job/actions/storage/transports | `agent_runtime` | external | deployment/runtime layer |
| plugin/extension registry | `agent_runtime` | redesign | decide later under reserved plugin namespace |
| app event bridges / SSE projection | `intelligence_layer` | external | keep outside core package |
| compatibility client helper image/pdf flows | `llm_client.client` | deprecate | do not expand; eventually split or remove |

## Freeze Policy

Until API/core stabilization is complete, new `llm_client` work should be
limited to:

- modernization and extraction completion
- reliability/observability hardening
- public API discipline
- tests, docs, benchmarks, and examples

Avoid unrelated feature expansion unless it is required to unblock the
stabilization program.
