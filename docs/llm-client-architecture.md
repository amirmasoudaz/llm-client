# llm-client Architecture Note

This note defines the intended role of `llm_client` as a standalone package.

## Layer Position

`llm_client` is the reusable LLM systems layer for this codebase.

It should be the canonical home for:

- provider abstraction
- model metadata and routing
- request/result/content/context types
- execution engine behavior
- structured outputs
- tool definition and tool runtime
- agent loop runtime
- observability, replay, and runtime event primitives

It should not own:

- product/business policies
- domain operators or manifests
- repo-specific prompts and workflow semantics
- storage/transports/deployment glue
- UI/SSE projection code
- tenant/business ledgers and billing persistence

## Layer 0 Responsibilities

The package is acting as a Layer 0 or foundational runtime package. That means
its job is to provide the lowest reusable abstraction layer above vendor SDKs.

Current Layer 0 responsibilities:

- normalize provider behavior and capability differences
- provide engine-level retry, timeout, failover, idempotency, and hooks
- standardize message/content/result/event types
- provide generic budget, usage-ledger, and execution-governance primitives
- support agent and tool-call execution loops generically
- expose diagnostics and runtime/replay primitives
- provide pluggable cache policies plus storage-agnostic metadata/summary cache helpers

## Non-Goals

The following are explicitly out of scope for the core package:

- business-domain policies
- product-specific tool allowlists and trust decisions
- domain entity loaders
- repo-specific orchestration contracts
- API/web framework wiring
- persistence models for jobs, ledgers, or tenant state

## Module Classification

### Stable

- `llm_client.providers`
- `llm_client.models`
- `llm_client.types`
- `llm_client.content`
- `llm_client.context`
- `llm_client.budgets`
- `llm_client.adapters`
- `llm_client.context_assembly`
- `llm_client.engine`
- `llm_client.agent`
- `llm_client.benchmarks`
- `llm_client.tools`
- `llm_client.memory`
- `llm_client.cache`
- `llm_client.observability`
- `llm_client.validation`
- `llm_client.errors`
- `llm_client.config`

### Compatibility

- `llm_client.compat`

### Advanced

- `llm_client.advanced`

### Provisional

These modules are useful and intentionally exposed, but should still be treated
as subject to boundary refinement:

- `llm_client.model_catalog`
- `llm_client.provider_registry`
- `llm_client.routing`
- `llm_client.resilience`
- `llm_client.structured`
- `llm_client.summarization`
- `llm_client.replay`
- `llm_client.runtime_events`
- `llm_client.context_planning`

### Legacy

- `llm_client.client`
- `llm_client.container`

### Internal / Private

- provider-specific translation internals
- request-construction helpers
- internal concurrency helpers
- low-level retry/resilience plumbing that is not promoted into stable modules

## Integration Boundary

`llm_client` is the canonical home for generic LLM and agentic runtime
mechanics.

Rule of thumb:

- if the code is generic LLM execution, content handling, structured/tool loop
  runtime, retry/failover, context assembly, observability, or reusable
  adaptor/tooling substrate, it belongs in `llm_client`
- if the code is domain policy, application workflow behavior, UI delivery, or
  deployment-specific glue, it stays outside the package

Typical external consumers should treat `llm_client` as the reusable kernel and
compose their own application/runtime layers above it.

Provider-only construction may still exist as consumer convenience, but
higher-level execution paths should normalize to `ExecutionEngine` immediately
and run through the engine from that point onward.

### Tests

The test suite imports both stable and provisional modules directly. That is
acceptable for package-level regression coverage, but it should not be confused
with the recommended public API for external adopters.

## Direct-Provider Execution Paths

Documented remaining direct-provider paths that intentionally bypass the engine:

- `llm_client.structured`
  - direct provider path when no `ExecutionEngine` is supplied
- `llm_client.tools.runtime`
  - direct provider completion fallback when no engine is supplied
- `llm_client.streaming`
  - adapter utilities that wrap `provider.stream(...)` directly
- provider implementation modules
  - direct vendor SDK execution is expected here

Interpretation:

- these are low-level or fallback paths
- higher-level application/runtime flows should prefer `ExecutionEngine`
- higher-level consumers should not keep adding new provider-first execution
  paths outside explicitly low-level helpers

## Semver and Deprecation Policy

For this modernization program, the intended policy is:

- stable modules:
  - additive changes are acceptable in minor releases
  - breaking changes require a major version
  - benchmarking/report APIs live in `llm_client.benchmarks`
- compatibility modules:
  - warn before removal or major behavior changes
  - prefer migration notes over silent drift
- provisional modules:
  - allowed to evolve faster, but changes should still be documented clearly

## Experimental Module Policy

A module should only be presented as stable if it has:

- explicit `__all__`
- package-level documentation
- tests covering its public boundary
- a clear role relative to stable, advanced, or compatibility namespaces

Otherwise it should remain provisional, advanced, compatibility-only, or
internal.

## Stabilization Rule

Until the core modernization and package-boundary work is complete, unrelated
feature expansion should be deferred. Priority should stay on:

- API stabilization
- extraction/gap closure
- testing/benchmarking
- packaging and documentation hardening
