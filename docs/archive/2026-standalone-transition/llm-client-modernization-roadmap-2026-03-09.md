# LLM Client Modernization Roadmap

Date: 2026-03-09
Repo: `intelligence-layer-bif`
Primary package: `llm_client`
Related audit: [llm-client-capability-audit-2026-03-09.md](./llm-client-capability-audit-2026-03-09.md)

## Purpose

This document is the comprehensive execution plan for turning `llm_client` from a strong internal LLM utility package into a standalone, production-grade, reusable package with a stable API, strong provider abstractions, high operational reliability, strong test coverage, credible benchmarks, and clear examples for adoption across multiple projects.

This roadmap assumes:

- `llm_client` should remain a general-purpose infrastructure package
- product-specific business logic should stay outside the package
- the package should support multiple projects without inheriting Dana-specific workflow semantics
- stability, observability, and correctness matter more than feature count

## Linked Inputs

- Capability audit: [llm-client-capability-audit-2026-03-09.md](./llm-client-capability-audit-2026-03-09.md)
- Superseded short roadmap pointer: [llm-client-roadmap-2026-03-09.md](./llm-client-roadmap-2026-03-09.md)
- Current package docs: [`llm_client/README.md`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/README.md)
- Current package source: [`llm_client`](/home/namiral/Projects/Packages/llm-client-v1/llm_client)

## Current Implementation Frontier

As of the latest refactor slice, the modernization program has crossed the point where the package direction is no longer in question. The core provider, engine, content, structured-output, tool-runtime, replay/event, and agent execution layers now live primarily in `llm_client`, and the remaining work is mostly about closing consistency and operational gaps rather than inventing the substrate.

Current practical status:

- canonical content envelopes are now adopted across the major engine-backed helper, planner, legacy client, tool runtime, agent, and structured execution paths
- provider registry, model catalog, model-aware routing, and routing observability are in place
- shared runtime substrate previously duplicated in `agent_runtime` and `intelligence_layer` has been substantially pulled upward into `llm_client`
- the highest-value remaining code work is to finish the residual bypass audit and then move into unresolved framework gaps such as unsupported-content rules, stream/non-stream shape unification, metadata persistence, and the remaining structured-runtime duplication above the package

Current next-step lane:

1. classify the remaining raw provider and raw `RequestSpec` paths into intentional low-level APIs versus migration targets
2. finish the last worthwhile migrations where higher-level engine-backed code still bypasses canonical content envelopes
3. shift focus to unresolved Phase 3 / Phase 4 / Phase 8 gaps rather than continuing to chase low-value internal wrappers

## Desired End State

By the end of this roadmap, `llm_client` should provide:

- a stable, versioned, well-documented public API
- capability-aware multi-provider support
- canonical typed multimodal input/output models
- a single, canonical orchestration path for provider execution
- production-ready agent, tool, context, and optional memory abstractions
- strong caching, retries, routing, observability, and validation
- reproducible benchmark suites
- broad automated test coverage with contract tests across providers
- examples that demonstrate realistic usage across different application shapes
- packaging and release practices suitable for standalone use

## Strategic Principles

1. Keep the layer boundary clean.
   - `llm_client` owns LLM infrastructure primitives.
   - Domain workflows, business policy, and product contracts stay outside.

2. Make the public API smaller and stronger.
   - Prefer a smaller stable core over a broad but inconsistent export surface.

3. Standardize core abstractions before adding more features.
   - Provider registry, content model, engine path, memory interface, and error model come before feature expansion.

4. Treat OpenAI maturity as the baseline, not the finish line.
   - Cross-provider parity should be explicit and testable.

5. Build for operational clarity.
   - Observability, failure semantics, and debuggability must be first-class.

6. Make testing and benchmarks part of the product, not an afterthought.
   - A standalone package without strong contracts will drift quickly.

## Scope

### In Scope

- public API redesign and stabilization
- provider abstraction hardening
- provider registry and routing
- model metadata system
- canonical content and result types
- execution engine consolidation
- agent/tool/context/memory abstractions
- validation, caching, logging, telemetry, retries, and circuit handling
- package structure, docs, examples, tests, benchmarks, release process

### Out of Scope

- Dana-specific workflow/operator contracts
- product-specific prompts, intents, or policy semantics
- frontend or API product integration details outside package examples
- application-specific orchestration above the package boundary

## Program Structure

The work should run across ten coordinated workstreams:

1. Public API and package structure
2. Provider platform and registry
3. Typed content and unified I/O
4. Engine, execution, and reliability
5. Agent, tools, context, and memory
6. Observability, logging, and cost/usage
7. Validation, compatibility, and security
8. Tests, benchmarks, and examples
9. Packaging, release, and adoption
10. Runtime substrate extraction from higher layers

These workstreams should be delivered in phases, not all at once.

## Phase 0: Program Setup and Boundary Freeze

### Goal

Create the conditions to refactor safely without expanding the package surface accidentally.

### Outcomes

- package modernization charter
- public/private module classification
- adoption inventory
- compatibility policy
- release and deprecation policy draft
- extraction matrix for `agent_runtime` and `intelligence_layer`

### Actions

1. Create a package architecture note.
   - Define Layer 0 responsibilities and non-goals.
   - Record what must stay inside `llm_client` and what must move out or remain outside.

2. Inventory all current imports and classify them.
   - Categories:
     - public and should remain public
     - public but should be deprecated
     - internal and should become private
     - unused or legacy

3. Define a compatibility contract.
   - Establish semver policy.
   - Define how deprecations are announced and for how long they are supported.
   - Define which modules are stable vs experimental.

4. Freeze new feature additions outside this roadmap.
   - Limit work to modernization, gap closure, and API stabilization until Phase 2 is complete.

5. Create adoption notes for current consumers.
   - Document how `agent_runtime` and `intelligence_layer` currently use:
     - `OpenAIProvider`
     - `Agent`
     - `Tool`
     - `RequestContext`
     - `Usage`
     - `ExecutionEngine`

6. Create an extraction matrix for higher-layer generic runtime code.
   - classify candidate components as:
     - extract directly
     - redesign inside `llm_client`
     - keep external
     - deprecate after migration

### Exit Criteria

- clear package boundary agreed
- current public API mapped
- modernization tracked as a defined program
- extraction candidates are documented before large refactors start

## Phase 1: Public API Rationalization

### Goal

Turn the package into a predictable standalone library with a coherent top-level surface.

### Desired Public API

Target stable modules:

- `llm_client.providers`
- `llm_client.models`
- `llm_client.types`
- `llm_client.content`
- `llm_client.context`
- `llm_client.engine`
- `llm_client.agent`
- `llm_client.tools`
- `llm_client.memory`
- `llm_client.cache`
- `llm_client.observability`
- `llm_client.validation`
- `llm_client.errors`
- `llm_client.config`

Target internal modules:

- provider-specific translators
- low-level serializer helpers
- internal container/factory wiring unless deliberately promoted

### Actions

1. Introduce an explicit API map.
   - Add a maintainer-facing document listing:
     - stable modules
     - provisional modules
     - private modules

2. Reduce top-level `__init__.py` sprawl.
   - Keep only the stable, high-value exports.
   - Move secondary exports behind module namespaces.

3. Create a `types` module strategy.
   - Split general data types from provider implementation concerns.
   - Re-export canonical request/result/event/content types from one place.

4. Mark legacy APIs.
   - Decide whether `OpenAIClient` remains:
     - supported
     - compatibility-only
     - deprecated
   - If retained, define its limits clearly.

5. Make experimental modules explicit.
   - For example:
     - DI container
     - provider factories
     - some middleware stacks
   - Avoid presenting partially mature features as stable package guarantees.

### Deliverables

- public API matrix
- deprecation list
- refactored export structure
- module docstrings aligned to the new boundary

### Exit Criteria

- consumers can see the intended stable API surface clearly
- package layout reflects actual maturity

## Phase 2: Provider Platform and Registry

### Goal

Replace ad hoc provider creation and routing with a real provider platform.

### Target Design

Add a provider registry subsystem that supports:

- provider registration
- provider capability declaration
- provider discovery
- provider health state
- provider priority/cost/latency classes
- capability-aware selection
- failover and routing policy

### Provider Capability Model

Each provider/model combination should declare:

- provider name
- model key
- provider model name
- supports_text_generation
- supports_streaming
- supports_tools
- supports_embeddings
- supports_reasoning
- supports_images_input
- supports_audio_input
- supports_file_input
- supports_json_mode
- supports_json_schema
- supports_parallel_tool_calls
- context_window
- max_output_tokens
- tokenizer name
- pricing metadata
- rate limits
- maturity tier

### Actions

1. Add provider registry objects.
   - `ProviderRegistry`
   - `ProviderFactory`
   - `ProviderDescriptor`
   - `ModelCapabilityDescriptor`

2. Replace or absorb `create_provider()` and `StaticRouter`.
   - Preserve compatibility shims temporarily.
   - Route them through the registry.

3. Add a routing policy subsystem.
   - Static priority routing
   - capability-aware routing
   - failover routing
   - cost-aware routing
   - latency-aware routing
   - manual policy override hooks

4. Add provider health tracking.
   - rolling error rate
   - last successful call
   - circuit state
   - recent latency bands

5. Add provider selection traces.
   - Log why provider X/model Y was selected.
   - Log why failover occurred.

6. Add provider contract tests.
   - Shared test suite for all providers:
     - complete
     - stream
     - tool calls
     - embeddings if supported
     - usage handling
     - error mapping

7. Fix provider factory inconsistency.
   - Ensure Google is supported in the same creation path.
   - Remove defaults that do not exist in the model registry.

### Deliverables

- provider registry package
- routing and selection policies
- provider capability descriptors
- compatibility shim for old factory APIs

### Exit Criteria

- provider creation is registry-based
- routing is capability-aware
- cross-provider support is testable instead of assumed

## Phase 3: Canonical Content Model and Unified I/O

### Goal

Move from loose message dicts and string-heavy content handling to a typed, extensible multimodal content model.

### Target Design

Introduce canonical types such as:

- `ContentBlock`
- `TextBlock`
- `ImageBlock`
- `FileBlock`
- `AudioBlock`
- `ReasoningBlock`
- `ToolCallBlock`
- `ToolResultBlock`
- `AssistantOutput`
- `CompletionOutput`

Messages should become:

- role
- blocks
- optional metadata
- optional provenance

### Actions

1. Design the canonical content schema.
   - Keep it provider-neutral.
   - Model structured multimodal content explicitly.

2. Add translation layers per provider.
   - Translate canonical content to OpenAI, Google, Anthropic request formats.
   - Translate provider responses into canonical content blocks.

3. Refactor `Message` gradually.
   - Support backward-compatible simple text access.
   - Add content-block-based constructors and converters.

4. Redesign output/result normalization.
   - Replace string-only assumptions with:
     - text output
     - structured output
     - multimodal output
     - tool calls
     - reasoning traces

5. Add canonical serialization support.
   - Stable JSON representations for:
     - request specs
     - message content
     - responses
     - streaming deltas

6. Add migration shims.
   - Maintain `normalize_messages()` compatibility while encouraging block-based inputs.

### Deliverables

- `content` module
- canonical typed I/O model
- provider translators
- compatibility converters for old `Message` usage

### Exit Criteria

- all providers can accept and emit canonical content
- multimodal handling is part of the model, not ad hoc dicts

### Phase 3 Follow-Up: Native FileBlock Transport

This is a remaining content-model maturity gap.

`FileBlock` already exists in the canonical content model, but current provider
transport support is weaker than the rest of the multimodal stack. In practice,
the package can represent files canonically, but it does not yet provide a
fully mature, provider-native, end-to-end file transport path comparable to the
current text and image flows.

Current limitation:

- `FileBlock` is canonical at the model layer
- provider adapters do not yet implement a full native file lifecycle across
  OpenAI, Anthropic, and Google
- unsupported file inputs are currently degraded rather than transported
  natively in many cases
- example coverage should currently prefer extracted file text over implying
  native file transport parity

Implementation objective:

- make `FileBlock` a real transport feature, not just a canonical placeholder

Required design questions:

1. What canonical file forms must `FileBlock` support?
   - local path
   - raw bytes or base64 payload
   - uploaded provider file ID
   - remote URL
   - MIME type
   - optional extracted text sidecar

2. What is the canonical file lifecycle?
   - accept file input
   - validate file metadata
   - upload where provider-native upload is required
   - attach provider-native reference to the request
   - reuse or cache file handles where appropriate
   - expose cleanup or retention semantics

3. What is the fallback policy when native file transport is unavailable?
   - degrade to extracted text
   - degrade to metadata placeholder
   - fail in strict mode
   - surface explicit degradation diagnostics

4. What are the security constraints?
   - file size limits
   - MIME allowlist
   - path safety for local files
   - redaction and logging rules for file-derived content

Recommended implementation order:

1. Define canonical `FileBlock` semantics precisely.
   - Add explicit support decisions for:
     - `file_path`
     - `file_bytes` or encoded payload
     - `file_id`
     - `file_url`
     - `mime_type`
     - `name`
     - optional `extracted_text`

2. Add a file preparation layer.
   - Resolve local files safely
   - normalize MIME metadata
   - optionally extract or inject text sidecars
   - prepare upload payloads

3. Implement provider-native OpenAI file transport first.
   - Treat OpenAI as the first production path
   - support upload/reference semantics where the provider expects them
   - ensure request translation and response handling stay canonical

4. Add fallback extraction behavior for unsupported providers.
   - If Anthropic or Google cannot accept the file natively in the current
     transport path, convert to extracted text plus explicit degradation record
   - preserve strict-mode failure behavior

5. Add cache and lifecycle semantics.
   - reuse uploaded file handles where safe
   - include file transport shape in cache/versioning decisions
   - define invalidation rules when file contents change

6. Add tests and examples.
   - real local PDF example
   - native OpenAI file path
   - extracted-text fallback path
   - strict-mode failure path
   - provider translation and normalization tests

Implemented canonical `FileBlock` support in the current codebase:

- canonical forms:
  - `file_path`
  - `data` (inline/base64 or bytes normalized to base64)
  - `file_id`
  - `file_url`
  - `mime_type`
  - `name`
  - optional `extracted_text`
  - derived `sha256` and `size_bytes` for cache/version semantics
- native OpenAI support:
  - OpenAI Responses API transport now maps prepared `FileBlock` values to
    `input_file` parts using either `file_id` or inline `file_data`
- fallback policy:
  - non-native provider paths degrade to `extracted_text` when available
  - strict mode still fails explicitly when native transport is required but
    unavailable
- request-building/cache semantics:
  - engine/request-builder preparation now computes file transport metadata so
    local-file inputs carry stable version signals into request serialization
- coverage:
  - focused contract tests cover preparation, native OpenAI transport,
    extracted-text fallback, strict-mode failure, and request-builder versioning
  - cookbook example `35_file_block_transport.py` demonstrates the supported
    path and fallback behavior

Deliverables for this follow-up:

- upgraded `FileBlock` schema and semantics
- file preparation/transport helpers
- native OpenAI file transport path
- explicit fallback policy for non-native providers
- contract tests and cookbook examples using real local files

Exit criteria for this follow-up:

- `FileBlock` can be used as a real end-to-end feature on at least one
  production provider path
- unsupported providers degrade or fail explicitly according to policy
- examples no longer need to work around file transport by pre-extracting text
- file transport behavior is documented, tested, and observable

## Phase 4: Model Metadata System

### Goal

Replace static code-only model metadata with a managed metadata system.

### Target Design

Support:

- code-level fallback metadata
- external metadata manifests
- provider overrides
- versioning
- persistence
- validation

### Actions

1. Redesign `ModelProfile`.
   - Separate:
     - static class helpers
     - data records
     - registry/storage

2. Add a model catalog subsystem.
   - `ModelCatalog`
   - `ModelMetadataRecord`
   - `PricingRecord`
   - `CapabilityRecord`

3. Allow metadata to load from:
   - built-in package manifests
   - local override files
   - environment override hooks

4. Add persistence and versioning.
   - Track metadata revision and source.
   - Enable pinned metadata versions for reproducibility.

5. Add catalog validation.
   - Ensure defaults and provider factories only reference valid models.

6. Add model-selection debugging.
   - Surface why a given model was chosen.

### Deliverables

- model catalog subsystem
- externalized metadata manifests
- migration path from class-based profiles

### Exit Criteria

- model defaults are consistent
- metadata drift is manageable
- capabilities and pricing are not hardcoded in scattered classes

## Phase 5: Engine Consolidation and Reliability Core

### Goal

Make `ExecutionEngine` the canonical path for LLM execution and the single place for common reliability behavior.

### Actions

1. Define engine as the preferred execution path.
   - `Agent` should use engine-backed execution by default when provided.
   - public docs should present engine-first usage.

2. Normalize provider access patterns.
   - Decide which direct provider calls remain allowed.
   - Provide explicit “raw provider” and “engine-managed provider” modes.

3. Consolidate retry/backoff semantics.
   - Standardize retry classification across:
     - engine
     - providers
     - tool middleware

4. Harden idempotency.
   - Define guarantees for:
     - non-streaming requests
     - streaming requests
     - agent turns

5. Improve circuit and failover policies.
   - Add configurable failure windows and recovery rules.
   - Expose state for observability.

6. Add request policy hooks.
   - Request interception before dispatch
   - post-processing after completion
   - tracing and redaction hooks

7. Add better timeout semantics.
   - Separate:
     - request timeout
     - connect timeout
     - streaming inactivity timeout
     - overall execution deadline

8. Expose engine diagnostics.
   - cache hit/miss reasons
   - provider switch reasons
   - retry history
   - final normalized error classification

### Deliverables

- engine API revisions
- diagnostics and tracing structures
- canonical dispatch semantics

### Exit Criteria

- engine behavior is the main documented and tested path
- direct-provider use is an intentional low-level choice, not accidental drift

## Phase 6: Agent, Tooling, and Tool Call Engine

### Goal

Make agent and tool execution robust enough for standalone use across different projects.

### Agent Enhancements

- first-class agent definition object
- reusable prompt/template attachment
- explicit turn policy
- execution policy
- output policy
- optional memory policy

### Tool Engine Enhancements

- explicit tool execution planner
- single, sequential, and parallel modes
- per-tool constraints
- partial failure policy
- retry policy
- result normalization policy

### Actions

1. Introduce agent definition/config objects.
   - `AgentDefinition`
   - `PromptTemplateRef`
   - `AgentExecutionPolicy`

2. Separate agent definition from agent runtime state.
   - Definitions should be serializable and reusable across projects.
   - Runtime instances should hold mutable conversation and execution state.

3. Add a dedicated tool execution engine module.
   - Wrap current behavior from `agent/execution.py`.
   - Make execution mode explicit:
     - single
     - sequential
     - parallel
     - planner-managed

4. Add tool result semantics.
   - Standardize success/error/partial output envelopes.
   - Define truncation/redaction provenance.

5. Add per-tool policy metadata.
   - timeout
   - retry count
   - redaction policy
   - side-effect class
   - idempotency class

6. Add tool call transcript support.
   - Capture inputs, outputs, timings, failures, retries, and truncation.

7. Improve streaming around tool usage.
   - Standardize tool call and tool result events in both agent and engine paths.

### Deliverables

- agent definition model
- tool execution engine
- richer tool metadata and execution traces

### Exit Criteria

- agent and tool execution can be configured and reused cleanly across projects
- tool behavior is observable and policy-controllable

## Phase 7: Context Management and Memory

### Goal

Move from conversation-history management to full context assembly with optional memory support.

### Target Architecture

Separate:

- conversation history
- context assembly
- summarization
- memory retrieval
- memory persistence
- relevance/ranking policy

### Proposed Interfaces

- `ContextAssembler`
- `ContextBudget`
- `ContextSelectionPolicy`
- `Summarizer`
- `MemoryStore`
- `MemoryRetriever`
- `MemoryWriter`
- `MemoryPolicy`

### Memory Categories

- short-term turn memory
- episodic memory
- semantic memory
- tool-derived memory
- retrieved external memory

### Actions

1. Keep `Conversation` focused on history.
   - Avoid making it the memory abstraction.

2. Add a context assembler.
   - Inputs:
     - conversation history
     - system instructions
     - tool transcript
     - retrieved memory
     - external context
     - budget
   - Output:
     - assembled canonical message set
     - inclusion/exclusion trace

3. Add memory interfaces only first.
   - Do not hardwire one backend into core abstractions.

4. Provide reference memory implementations.
   - in-memory memory store
   - filesystem-backed memory store
   - vector-store adapter interfaces

5. Improve summarization.
   - add summary quality metadata
   - add optional summary types:
     - conversational summary
     - facts summary
     - decisions summary
     - task state summary

6. Add “smart” context strategies.
   - recency-only
   - recency + summary
   - recency + retrieval
   - budgeted relevance ranking

7. Add memory safety controls.
   - max retention
   - redaction and PII policy hooks
   - source provenance for retrieved memory

### Deliverables

- context assembly subsystem
- memory interfaces
- reference memory backends
- smarter context strategies

### Exit Criteria

- the package supports optional memory without forcing one product-specific pattern
- context assembly is inspectable and policy-driven

## Phase 8: Structured Outputs and Validation Hardening

### Goal

Make structured output a first-class, dependable feature across providers and agent flows.

### Actions

1. Unify structured output modes.
   - plain JSON mode
   - native JSON-schema mode where provider supports it
   - repair-loop mode when native strictness is unavailable

2. Add capability-aware structured execution.
   - Choose strictest viable structured mode per provider/model.

3. Improve repair diagnostics.
   - store repair attempts
   - store validation errors
   - expose repair trace for debugging

4. Add structured-stream handling contracts.
   - define how partial structured output should be emitted
   - define when repair can happen after stream completion

5. Add schema tool compatibility rules.
   - ensure tool schemas and output schemas are both normalized correctly for each provider.

6. Expand validation suite.
   - malformed JSON
   - schema mismatch
   - missing required fields
   - additional properties
   - tool-call mixed outputs

### Deliverables

- stronger structured output API
- provider-aware structured execution selection
- structured output debug traces

### Exit Criteria

- structured outputs are dependable and diagnosable across providers

## Phase 9: Observability, Logging, Usage, and Cost Intelligence

### Goal

Make package behavior transparent in production.

### Actions

1. Consolidate observability APIs.
   - unify hooks, telemetry, and logging around common request lifecycle events.

2. Define a standard event model.
   - request start
   - provider selected
   - cache hit/miss
   - retry
   - failover
   - stream event
   - tool call start/end
   - structured repair
   - request end

3. Add usage and cost reporting improvements.
   - separate:
     - prompt tokens
     - completion tokens
     - cached tokens
     - reasoning tokens where available
   - expose normalized cost reports

4. Add request and session reports.
   - request report object
   - session usage summary
   - provider distribution summary

5. Improve log safety.
   - central redaction controls
   - allow safe previews without leaking sensitive content

6. Add benchmark instrumentation hooks.
   - structured latency and throughput measurements

### Deliverables

- common lifecycle event taxonomy
- request/session reporting
- redaction-safe structured logging

### Exit Criteria

- operators can debug provider behavior, retries, costs, and streaming failures without source spelunking

## Phase 10: Error Model and Failure Semantics

### Goal

Make error handling coherent and programmatically useful.

### Actions

1. Decide and enforce one model.
   - internal code should raise typed errors consistently
   - public API may still normalize to result objects where appropriate

2. Map provider SDK errors into the unified taxonomy.
   - auth
   - quota
   - rate limit
   - timeout
   - content filter
   - invalid request
   - context overflow
   - provider unavailable

3. Add retryability rules centrally.
   - all retry policy decisions should use one classifier.

4. Add rich failure reports.
   - error code
   - retryability
   - provider
   - model
   - attempt count
   - fallback history
   - partial stream state if applicable

5. Add chaos-style failure injection tests.
   - malformed responses
   - timeout mid-stream
   - provider returns tool-call fragments incorrectly
   - circuit open

### Deliverables

- normalized error mapping layer
- shared retry classifier
- failure report objects

### Exit Criteria

- consumers can reliably handle failures without provider-specific branching

## Phase 11: Caching and Persistence Evolution

### Goal

Strengthen cache and persistence beyond simple response storage.

### Actions

1. Add cache versioning.
   - cache keys should include schema/version dimensions where appropriate.

2. Add cache policy objects.
   - response cache
   - embedding cache
   - negative-cache policy
   - TTL and invalidation strategy

3. Improve cache diagnostics.
   - why a read was skipped
   - why a write was suppressed
   - version mismatch visibility

4. Add persistent request/session artifacts where useful.
   - serialized request
   - normalized response
   - replay artifact for debugging

5. Add optional persistence modules for:
   - request logs
   - benchmark runs
   - model catalog snapshots

### Deliverables

- versioned cache system
- persistence extension points

### Exit Criteria

- cache behavior is predictable and debuggable across package versions

## Phase 12: Security, Privacy, and Hardening

### Goal

Make the package safe for production use in varied environments.

### Actions

1. Add explicit secret-handling policy.
   - never log API keys
   - avoid leaking raw content by default

2. Add redaction framework.
   - config-driven redaction for logs and tool outputs
   - provider response redaction hooks

3. Review unsafe serialization paths.
   - ensure persisted artifacts are safe and bounded.

4. Add input safety checks.
   - file size limits
   - malformed content block protection
   - oversized schema and tool definition protections

5. Add supply-chain and dependency review for provider extras.

### Deliverables

- security hardening checklist
- redaction and safety controls

### Exit Criteria

- package behavior is safe by default in logs, persistence, and examples

## Phase 13: Test Strategy

### Goal

Establish a test system that gives confidence in standalone package behavior.

### Test Pyramid

#### Unit Tests

- provider parameter translation
- content model conversion
- message normalization
- tool schema extraction
- validation helpers
- retry logic
- limiter logic
- cache orchestration
- error mapping

#### Contract Tests

Shared contract suite per provider:

- completion result contract
- streaming event contract
- usage contract
- tool-call contract
- embedding contract if supported
- structured output contract

#### Integration Tests

- engine with cache/hook/retry/failover
- agent with tool execution
- context assembler with summarization and memory retrieval
- observability hook wiring
- config loading and package initialization

#### Live Tests

Optional env-gated tests:

- OpenAI live smoke tests
- Google live smoke tests
- Anthropic live smoke tests

#### Property and Fuzz Tests

- schema validation robustness
- message/content serialization round-trips
- streaming delta assembly
- tool argument normalization

#### Regression Tests

- all bugs from issue history and benchmark regressions

### Coverage Goals

- high branch coverage in core modules
- mandatory contract coverage for all providers
- zero untested stable public APIs

### Deliverables

- test matrix by module and provider
- CI split for unit/contract/live benchmarks

## Phase 14: Benchmark Strategy

### Goal

Make performance and reliability measurable.

### Benchmark Categories

1. Provider latency
   - completion latency
   - first-token latency
   - full-stream latency

2. Engine overhead
   - provider direct vs engine-managed
   - cache hit vs miss
   - retry/failover cost

3. Tool loop performance
   - single tool
   - sequential tools
   - parallel tools

4. Structured output performance
   - valid on first pass
   - repair loop frequency
   - schema mode vs JSON mode

5. Context management cost
   - trim only
   - summarize
   - retrieve + assemble

6. Memory performance
   - write latency
   - retrieval latency
   - ranking latency

7. Concurrency and throughput
   - sustained request throughput
   - saturation behavior
   - rate-limiter fairness

8. Cost benchmarking
   - per-task token and cost profiles
   - provider/model cost tradeoffs

### Deliverables

- reproducible benchmark harness
- benchmark baselines committed to repo
- threshold alerts for regressions

## Current Capability Inventory

This section is the current, implementation-grounded capability map for
`llm_client`. It is the source of truth for the cookbook/examples program in
Phase 15. The list is intentionally broad: it covers stable standalone
surfaces first, then advanced and compatibility surfaces that still exist in
the package.

### 1. Provider Abstraction and Multi-Provider Access

`llm_client` can talk to multiple LLM providers through one provider contract.

- OpenAI completions, streaming, tool calling, structured outputs, and
  embeddings
- Anthropic completions, streaming, and content translation
- Google completions, streaming, content translation, and embeddings where the
  provider supports them
- provider-local token counting and usage parsing
- provider-local raw-response retention for diagnostics

### 2. Provider Registry, Catalog, and Routing

The package can discover and select providers/models instead of hard-wiring one
backend everywhere.

- provider registry with descriptors, aliases, and capability metadata
- model catalog with asset-backed metadata and catalog-driven defaults
- provider/model compatibility checks
- registry-backed routing by capability requirements
- explicit provider override ordering
- routing preferences by latency tier, cost tier, and compliance tags
- provider health tracking and degradation-aware routing

### 3. Unified Core Types

The package has normalized request/response/event types that can be shared
across providers and runtimes.

- `RequestSpec` and `RequestContext`
- `Message`, `ToolCall`, `ToolCallDelta`
- `CompletionResult`, `EmbeddingResult`, `Usage`
- `StreamEvent` and `StreamEventType`
- cancellation-aware request contexts
- richer execution context, budgets, policy refs, and run versions

### 4. Canonical Content Model

The package can represent multimodal and tool-oriented exchanges through one
content layer instead of raw provider payloads.

- text, image, audio, file, reasoning, tool-call, tool-result, and metadata
  blocks
- `ContentMessage`, `ContentRequestEnvelope`, and `ContentResponseEnvelope`
- adapters between classic `Message` objects and content envelopes
- lossy-vs-strict unsupported-content handling
- content degradation tracking for observability and testing
- stream/completion terminal normalization between content envelopes and
  `CompletionResult`

### 5. Execution Engine

The engine is now the main orchestration layer for request execution.

- non-streaming completion execution
- streaming execution
- content-envelope execution (`complete_content`, `stream_content`)
- embeddings execution
- batch completion execution
- request timeout handling
- retry/backoff handling
- failover across providers
- circuit breaker integration
- idempotency handling for request and stream paths
- cache integration and cache-policy support
- request/session hooks and diagnostics emission

### 6. Structured Output Runtime

The package can do structured extraction and validation as a first-class
runtime, not just ad hoc JSON parsing.

- schema-driven structured extraction
- provider-aware response format selection
- JSON mode vs strict schema mode selection
- schema normalization for provider transport
- repair loops with bounded retry attempts
- structured streaming
- parse-only / validate-only structured handling
- structured diagnostics and attempt traces
- normalized structured result envelopes
- shared structured execution loop reusable above the provider layer

### 7. Agent Framework

The package can define and execute agentic flows above the provider/runtime
core.

- `AgentDefinition`
- prompt/template attachment references
- execution, output, and memory policy objects
- agent runtime state separate from static definition
- multi-turn agent loop
- engine-backed agent execution
- tool-enabled agent execution
- explicit tool execution modes on agents
- quick agent construction helpers

### 8. Tool System

The package can define, validate, execute, and govern tools as first-class
runtime objects.

- `Tool`, `ToolRegistry`, and decorators
- tool schema inference from Python functions
- strict tool argument validation
- tool execution middleware chain
- dedicated tool execution engine
- `single`, `sequential`, `parallel`, and `planner` execution modes
- per-tool execution metadata: timeout, retries, concurrency, safety tags,
  trust level
- normalized tool execution envelopes and results
- runtime tool-loop support for LLM tool calling

### 9. Conversation Management

The package can maintain conversation state outside the raw request layer.

- `Conversation` and `ConversationConfig`
- message history management
- system-message management
- truncation strategies: sliding, drop-oldest, drop-middle, summarize
- async truncation with summarizer integration
- serialization and forking support
- sync wrappers for selected async conversation/summarization flows

### 10. Context Planning, Assembly, and Memory

The package can prepare context and memory inputs before LLM execution.

- context planner interfaces
- trimming strategy interfaces
- summarization strategy interfaces
- heuristic context planner
- memory retrieval and write interfaces
- short-term memory store
- persistent summary store
- retrieval-strategy hooks
- semantic relevance selection hooks
- multi-source context assembly contracts

### 11. Caching and Persistence

The package can cache and persist LLM-adjacent outputs in a structured way.

- versioned cache-key generation
- canonical cache policies and invalidation modes
- completion and embedding cache support
- metadata and summary cache stores
- cache diagnostics and cache stats
- filesystem, PostgreSQL+Redis, and Qdrant cache backends
- repository-style PostgreSQL persistence for advanced use

### 12. Observability, Replay, and Telemetry

The package can emit rich operational data without being tied to one specific
vendor.

- hook manager and hook protocol
- engine diagnostics recorder
- context planning recorder
- benchmark recorder
- runtime lifecycle taxonomy and request/session reports
- runtime event bus and normalized runtime events
- replay recorder/player and replay validation
- in-memory telemetry/metrics support
- logging and telemetry lifecycle adapters
- OpenTelemetry and Prometheus hook surfaces

### 13. Error Model, Failure Semantics, and Resilience

The package can normalize failures instead of leaking provider-specific errors
everywhere.

- normalized failure taxonomy
- retryable vs non-retryable classification
- provider failure normalization across OpenAI/Anthropic/Google paths
- normalized tool failure categories
- normalized structured-runtime failure categories
- remediation hints in failure payloads
- circuit breakers and fallback policy

### 14. Security, Privacy, and Safety Controls

The package can sanitize, preview, and redact payloads across runtime surfaces.

- central redaction policy
- payload preview modes
- provider payload capture modes
- tool output redaction and truncation policy
- lifecycle-safe logging
- observability-safe payload sanitization
- threat model and secure-default documentation

### 15. Benchmarking

The package can benchmark itself in deterministic and opt-in live modes.

- general benchmark harness
- deterministic local benchmark runs
- explicitly labeled live benchmark runs
- committed baseline report storage
- baseline comparison / trend comparison support
- structured output quality benchmarking
- benchmark hooks and instrumentation events

### 16. Batch and Throughput Utilities

The package can help with throughput-oriented execution patterns.

- `ExecutionEngine.batch_complete(...)`
- async batch manager with checkpointing/resume support
- legacy request-manager adapter
- benchmark support for throughput-oriented measurements

### 17. Rate Limiting and Throughput Governance

The package has lightweight built-in rate-limit primitives.

- token-bucket limiter
- request bucket limiter
- model-profile-aware default limit loading
- async context-manager interface for request/token accounting

### 18. Compatibility and Advanced Surfaces

The package still exposes lower-level or compatibility-oriented APIs for
projects that need them.

- compatibility `OpenAIClient` facade
- compatibility/integration container helpers
- advanced namespace for hashing, idempotency, serialization, and streaming
  adapters
- sync wrappers for non-async environments
- direct provider usage for low-level integrations

### 19. Configuration and Environment Loading

The package can be configured as a reusable library instead of only through
hard-coded values.

- explicit `.env` loading
- provider/cache/agent/logging settings models
- configuration schema helpers
- catalog-driven provider defaults

### 20. What Phase 15 Must Demonstrate

The cookbook should demonstrate:

- each core standalone capability in isolation
- the expected ergonomics of the stable module namespaces
- safe local/demo usage without real provider credentials
- opt-in live-provider usage where that matters
- combined flows that reflect real application design:
  engine + routing + cache + observability, agent + tools, structured +
  validation, context + memory + summarization, and benchmarking

## Phase 15: Examples and Developer Experience

### Goal

Make the package easy to adopt in other projects.

### Example Categories

1. Simple one-shot completion
2. Streaming CLI
3. Structured extraction
4. Tool-calling agent
5. Engine-managed failover
6. Cache-enabled usage
7. FastAPI SSE service
8. Memory-backed assistant
9. Batch processing
10. Observability-enabled service
11. Multimodal request example
12. Custom provider registration example

### Documentation Artifacts

- Quickstart
- Concepts guide
- Provider guide
- Agent guide
- Tool guide
- Memory guide
- Observability guide
- Migration guide
- Compatibility guide
- Cookbook

### Deliverables

- `examples/` refresh
- docs information architecture
- “recommended patterns” documentation

## Phase 16: Packaging, Release, and Standalone Readiness

### Goal

Make the package consumable outside this repo.

### Actions

1. Clean packaging metadata.
   - package extras for provider and infra dependencies
   - minimal install by default

2. Add standalone release workflow.
   - changelog discipline
   - versioning
   - wheel/sdist publishing

3. Add package-level CI.
   - unit
   - contract
   - lint/type
   - packaging validation
   - optional live suites

4. Add compatibility matrix.
   - Python versions
   - optional extras
   - provider SDK versions

5. Add release gates.
   - public API diff review
   - benchmark regression review
   - doc/example validation

### Exit Criteria

- package can be versioned and adopted independently from the host app

## Implementation Sequence

The work should be executed in this order:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 8
9. Phase 7
10. Phase 9
11. Phase 10
12. Phase 11
13. Phase 12
14. Phase 13
15. Phase 14
16. Phase 15
17. Phase 16

Rationale:

- API, provider, content, model, and engine foundations must come before memory and deep examples.
- Structured outputs should be hardened before advanced context/memory because they affect canonical output semantics.
- Tests and benchmarks should begin early, but full formalization should lock in after the main architecture stabilizes.

## Milestone Plan

### Milestone A: Foundation Stabilization

Includes:

- Phase 0
- Phase 1
- Phase 2

Success means:

- package boundary is clear
- public API is controlled
- provider creation/routing is no longer ad hoc

### Milestone B: Core Runtime Unification

Includes:

- Phase 3
- Phase 4
- Phase 5

Success means:

- canonical content model exists
- model metadata is trustworthy
- engine is the primary execution path

### Milestone C: Reusable Agent Platform

Includes:

- Phase 6
- Phase 8
- Phase 7

Success means:

- agent and tools are reusable and inspectable
- structured outputs are strong
- context and memory abstractions exist

### Milestone D: Operational Excellence

Includes:

- Phase 9
- Phase 10
- Phase 11
- Phase 12

Success means:

- failures are diagnosable
- caching is robust
- logs and telemetry are production-grade
- security posture is acceptable

### Milestone E: Standalone Package Launch

Includes:

- Phase 13
- Phase 14
- Phase 15
- Phase 16

Success means:

- package can be consumed and trusted by external projects

## Cross-Module Extraction Findings

This section captures what should be absorbed from `agent_runtime` and `intelligence_layer` if `llm_client` is to become the standalone core framework for LLM and agentic systems.

### What Should Move From `agent_runtime` Into `llm_client`

#### 1. Execution Context Should Evolve Beyond Today’s `RequestContext`

Current evidence:

- `agent_runtime/context.py`

`ExecutionContext`, `BudgetSpec`, `PolicyRef`, and `RunVersions` represent a richer runtime contract than the current `llm_client.RequestContext`. They carry budgets, replay/version metadata, job identity, and runtime correlation. That is generic agentic infrastructure, not application policy.

Roadmap action:

- extend `llm_client` with a formal execution context model
- keep `RequestContext` as the provider-facing transport subset if needed
- define conversions between `ExecutionContext`, provider request context, and event context

#### 2. Runtime Event Model And Replay Should Become First-Class Package Features

Current evidence:

- `agent_runtime/events/types.py`
- `agent_runtime/events/bus.py`
- `agent_runtime/replay/recorder.py`
- `agent_runtime/replay/player.py`

The runtime event schema, event bus abstraction, and replay recorder/player are generic. They are useful for streaming, auditability, debugging, demos, regression testing, and deterministic replay. These belong close to the agent runtime, not in a project-specific layer.

Roadmap action:

- add a package-level runtime event model in `llm_client`
- add event bus abstractions and in-memory implementation
- add replay capture and replay playback as optional subsystems
- keep database-backed persistence adapters outside the core package or behind extras

#### 3. Human-In-The-Loop Actions Should Be Part Of The Core Agentic Runtime

Current evidence:

- `agent_runtime/actions/types.py`
- `agent_runtime/actions/manager.py`

Pause/resume actions, resume tokens, action expiry, and waiting/resolution semantics are generic agent runtime concerns. They are not Dana-specific. If `llm_client` is meant to support real agentic workflows, this should be part of the framework.

Roadmap action:

- add an `actions` or `human` module in `llm_client`
- standardize action lifecycle, pause/resume protocol, and event emission
- keep UI-specific action payload design outside the package

#### 4. Plugin Discovery And Capability Registry Should Be Generalized

Current evidence:

- `agent_runtime/plugins/types.py`
- `agent_runtime/plugins/registry.py`

The plugin manifest and registry model is reusable, especially for tools, memory providers, policies, connectors, and extensions. `llm_client` already has tools, but not a broader extension model.

Roadmap action:

- add a generic plugin/extension registry to `llm_client`
- support tool, memory, provider, policy, and connector capabilities
- separate plugin registration from business workflows and operator manifests

#### 5. Budget And Usage Enforcement Should Be Lifted Closer To The Core Runtime

Current evidence:

- `agent_runtime/ledger/types.py`
- `agent_runtime/ledger/ledger.py`
- `agent_runtime/context.py`

`llm_client` already calculates usage, but `agent_runtime` adds budget enforcement, aggregate usage tracking, tool/connectors accounting, and audit events. That is generally useful for any production agent runtime.

Roadmap action:

- introduce a `cost` or `budgets` subsystem in `llm_client`
- support request, session, tenant, and principal scoped budgets
- keep billing system integration and finance policy outside the package

#### 6. DAG And Multi-Node Execution Is A Good Candidate For An Experimental Orchestration Layer

Current evidence:

- `agent_runtime/orchestration/executor.py`

The graph executor is generic enough to be useful for multi-agent and multi-stage flows. It should not be positioned as the main `llm_client` abstraction early, but it is a strong candidate for an experimental `workflow` or `orchestration` module after the core runtime is stable.

Roadmap action:

- keep this out of the initial `v1` stable surface
- plan an experimental orchestration layer once context, tools, and actions are stabilized

### What Should Stay Outside `llm_client` From `agent_runtime`

- job stores and persisted job lifecycle backends
- Postgres/Redis event and storage adapters
- repo-specific event projection and delivery wiring
- deployment-specific runtime kernel assembly
- any policy enforcement tied to repo auth or tenant semantics

`JobRecord` and `JobManager` are borderline. Their semantics are generic, but they imply a full execution service. The right compromise is:

- standardize job lifecycle interfaces in `llm_client`
- keep persistence and deployment orchestration outside the core package or in optional extras

### What Should Move From `intelligence_layer` Into `llm_client`

#### 1. Structured LLM Execution With Tool Loops Should Be Consolidated Into `llm_client`

Current evidence:

- `intelligence_layer/kernel/operators/execution/llm_structured.py`
- `intelligence_layer/kernel/operators/execution/tools_runtime.py`
- `intelligence_layer/kernel/operators/execution/resilience.py`

This code contains real generic infrastructure:

- schema-driven prompting and repair loops
- provider turn completion with streamed token extraction
- tool-call normalization
- tool argument parsing and validation
- sequential and parallel tool execution
- tool-call depth and count limits
- fallback and retry handling around structured generation

This logic overlaps with what a mature `llm_client` should own. Keeping it in `intelligence_layer` means the best structured-output runtime is living above the package that should be the LLM foundation.

Roadmap action:

- move structured tool-loop execution into `llm_client`
- unify it with existing `Agent` and `structured` modules
- expose one supported structured runtime instead of parallel implementations

#### 2. Context Assembly, Truncation, And Relevance Scoring Should Inform The Context/Memory Subsystem

Current evidence:

- `intelligence_layer/kernel/conversation/context_assembler.py`
- `intelligence_layer/kernel/conversation/history_truncation.py`
- `intelligence_layer/kernel/conversation/history_scoring.py`

These modules are exactly the kind of context-management capabilities identified as missing in the audit:

- multi-source context assembly
- constraints-aware history trimming
- tiered truncation
- semantic and recency scoring
- memory layer composition

The current implementation is wired to project loaders, but the abstractions are reusable.

Roadmap action:

- add pluggable context assemblers and planners in `llm_client`
- adopt tiered truncation and semantic relevance selection
- make loaders and retrieval backends interfaces instead of project services

#### 3. Conversation Orchestrator Protocols Should Be Distilled Into Generic Runtime Interfaces

Current evidence:

- `intelligence_layer/kernel/conversation/orchestrator.py`

The current orchestrator contains both generic and domain-specific pieces. The generic parts are:

- planner protocol
- tool runtime protocol
- conversation loop guards
- action-required versus tool-call versus final-result control flow

The domain-specific directive helpers for onboarding and outreach do not belong in `llm_client`.

Roadmap action:

- extract only the generic conversation loop protocol and guard concepts
- keep workflow-specific directives and action payload semantics outside the package

#### 4. A Generic Asset And Manifest Story Should Be Designed, But Not By Lifting The Current Kernel Contracts Whole

Current evidence:

- `intelligence_layer/kernel/contracts/registry.py`

There is useful architecture here around versioned prompts, schemas, manifests, and validation. But the current registry is tightly coupled to intents, operators, capability manifests, and project contracts.

Roadmap action:

- do not move `ContractRegistry` directly into `llm_client`
- design a slimmer asset subsystem for prompt templates, schemas, and tool manifests only if needed
- keep business intents, operator manifests, and plan templates outside the package

#### 5. Prompt Loading And Planner Patterns Are Useful As Examples, Not Core Primitives

Current evidence:

- `intelligence_layer/kernel/runtime/switchboard_planner.py`
- `intelligence_layer/kernel/runtime/context_assembly.py`

The package should support building planners and routers, but the current switchboard runtime is application-intent specific. Its value for `llm_client` is as a cookbook pattern, not as a core module.

Roadmap action:

- capture this as an example and design reference
- avoid baking intent routing semantics into `llm_client`

### What Should Stay Outside `llm_client` From `intelligence_layer`

- operator registry and operator manifests
- intent registry, plan templates, and business contracts
- auth scope enforcement and principal-type access policy
- platform tools, database loaders, and platform plugin implementations
- product-specific conversation prompts and tool mappings
- switchboard intent taxonomy and workflow planning
- SSE/UI projection logic and app event writing

### Required Architecture Actions For The Final Plan

To reflect the findings above, the modernization program must explicitly include these actions:

1. Add a new sub-workstream for runtime substrate extraction.
   - execution context
   - event model
   - replay
   - actions
   - plugin registry
   - budgets and usage enforcement

2. Expand the public API target to account for runtime concerns.
   - `llm_client.actions`
   - `llm_client.events` or `llm_client.runtime.events`
   - `llm_client.replay`
   - `llm_client.plugins`
   - `llm_client.budgets` or `llm_client.cost`
   - experimental `llm_client.workflow`

3. Make structured execution consolidation an explicit deliverable.
   - fold `intelligence_layer` structured tool-loop runtime into `llm_client`
   - retire duplicate structured execution paths over time

4. Make context subsystem extraction explicit.
   - tiered truncation
   - semantic history scoring
   - multi-source context assembly
   - memory/retrieval interfaces

5. Define the post-extraction package layering.
   - `llm_client` becomes the reusable LLM and agent runtime framework
   - `agent_runtime` becomes either a thin adapter package or is partially merged
   - `intelligence_layer` remains a business/domain orchestration layer using `llm_client`

6. Treat storage and transport backends as adapters, not core logic.
   - Postgres/Redis/Kafka/event projection remain extras or downstream packages

## Suggested Repository Changes

### Package Layout

Potential target structure:

```text
llm_client/
  actions/
  agent/
  cache/
  config/
  content/
  context/
  cost/
  engine/
  errors/
  memory/
  observability/
  plugins/
  providers/
    registry/
    routing/
    openai/
    google/
    anthropic/
  replay/
  runtime/
  tools/
  types/
  validation/
  workflow/
```

This does not need to happen in one move. It should follow API stabilization and compatibility shims.

### Repository Boundary After Extraction

After the extraction work, the repo should trend toward this split:

- `llm_client`
  - generic providers, engine, agent runtime, tools, actions, replay, plugins, context, memory, budgets, workflow primitives
- `agent_runtime`
  - optional adapters and deployment glue for jobs, actions, event persistence, transport, and operational backends
- `intelligence_layer`
  - business operators, contracts, policy, prompts, platform connectors, workflow and product semantics

This is important because the current repo still has valuable generic runtime code in higher layers. That should be reduced before calling `llm_client` the canonical framework.

### Docs Layout

Potential target docs structure:

```text
docs/
  llm-client/
    architecture/
    api/
    guides/
    cookbook/
    benchmarks/
    migration/
```

## Acceptance Criteria for “Production-Ready”

`llm_client` should not be considered production-ready until all of the following are true:

- public API is explicitly documented and versioned
- all stable provider paths pass shared contract tests
- direct-provider vs engine-managed behavior differences are explicit and intentional
- structured output behavior is validated and benchmarked
- observability works in engine-managed and agent-managed paths
- cache semantics are versioned and diagnosable
- examples run and are validated in CI
- benchmark baselines exist and regressions are detectable
- security/logging defaults are safe
- standalone installation and release workflow is working

## Risks and Mitigations

### Risk: API churn without adoption discipline

Mitigation:

- freeze stable modules early
- deprecate before removing
- add API compatibility tests

### Risk: Building too much before consolidating fundamentals

Mitigation:

- do not add memory and advanced multimodal work before provider/content/engine foundations are fixed

### Risk: Cross-provider parity remains nominal only

Mitigation:

- require contract tests and live smoke coverage for every supported provider tier

### Risk: Engine remains bypassed

Mitigation:

- refactor package docs and examples to engine-first patterns
- add warnings for feature paths that bypass engine-managed reliability features

### Risk: Memory becomes product-specific

Mitigation:

- keep memory interfaces generic
- ship reference implementations only
- avoid embedding domain assumptions in retrieval or write policies

## Recommended Immediate Next Steps

These are the first concrete execution tasks to begin now:

1. Create the package boundary and API classification document.
2. Add a modernization tracker issue set aligned to the phases in this roadmap.
3. Fix provider factory/default inconsistencies before any larger redesign.
4. Design the provider registry and canonical content model before implementing them.
5. Decide the future of `OpenAIClient`, the DI container, and other compatibility surfaces.
6. Add a contract test harness skeleton for providers.
7. Add a benchmark harness skeleton even before full benchmarks are implemented.
8. Create an extraction inventory for `agent_runtime` and `intelligence_layer` and assign each candidate to:
   - move into `llm_client`
   - remain downstream
   - become optional adapter/extra
9. Prioritize three immediate extraction targets:
   - structured tool-loop runtime
   - context assembly and history scoring primitives
   - runtime event and replay substrate

## Recommended Ownership Model

Assign clear ownership by workstream:

- API and package structure
- provider platform
- content/types
- engine/reliability
- agent/tools
- context/memory
- observability/security
- docs/examples
- tests/benchmarks/release
- extraction and migration coordination

This work is large enough that a single unstructured branch will slow down and regress. It should be run like a program with milestones, explicit ownership, and compatibility gates.

## Final Recommendation

The package is worth serious investment.

The correct strategy is not to bolt on more features around the edges. It is to:

- tighten the public API
- unify provider and content abstractions
- make the engine canonical
- add memory and context assembly as real subsystems
- formalize testing, benchmarks, docs, and release discipline

If that sequence is followed, `llm_client` can become a credible standalone LLM systems package. If not, it will remain a capable but uneven internal dependency.

## Master Task Checklist

Use this section as the execution tracker for the modernization program. Mark items as done in place as work lands.

### Already Completed

- [x] Create the `llm_client` capability audit
- [x] Create the canonical modernization roadmap
- [x] Supersede the shorter duplicate roadmap with a pointer document
- [x] Link the capability audit to the canonical modernization roadmap
- [x] Add cross-module extraction findings for `agent_runtime` and `intelligence_layer`
- [x] Define the target package boundary at a high level
- [x] Align provider factory/config defaults with supported model keys and add Google factory support
- [x] Switch initial runtime `Agent` construction paths to pass the engine through
- [x] Fix `CompletionResult` integration in conversation title generation
- [x] Add the first shared provider contract test harness skeleton
- [x] Route initial higher-level operator and planner LLM paths through `ExecutionEngine` when available
- [x] Extract the generic structured tool-loop runtime into `llm_client.tools.runtime`
- [x] Move canonical execution-context primitives into `llm_client.context` and keep `agent_runtime.context` as a compatibility shim
- [x] Make `extract_structured` and `LLMSummarizer` engine-capable so reusable package helpers prefer the canonical execution path
- [x] Extract generic history truncation and scoring heuristics into `llm_client.context_planning`
- [x] Promote canonical runtime events, in-memory event bus, and replay primitives into `llm_client`
- [x] Add deterministic scripted-provider test utilities and expand engine reliability coverage
- [x] Make `Agent` engine-first by default and remove its internal raw-provider execution branch

### Phase 0: Program Setup and Boundary Freeze

- [x] Create the package architecture note
- [x] Define Layer 0 responsibilities and non-goals
- [x] Classify all current `llm_client` modules as stable, provisional, private, or legacy
- [x] Inventory all repo imports of `llm_client`
- [x] Document all direct-provider execution paths that bypass the engine
- [x] Define semver and deprecation policy
- [x] Define experimental module policy
- [x] Create the extraction matrix for `agent_runtime` and `intelligence_layer`
- [x] Classify extraction candidates as extract, redesign, external, or deprecate
- [x] Write adoption notes for `agent_runtime` consumers
- [x] Write adoption notes for `intelligence_layer` consumers
- [x] Freeze unrelated feature expansion until API/core stabilization is complete

### Phase 1: Public API Rationalization

- [x] Define the `v1` public API map
- [x] Define stable namespaces for `providers`, `models`, `types`, `content`, `context`, `engine`, `agent`, `tools`, `memory`, `cache`, `observability`, `validation`, `errors`, and `config`
- [x] Decide whether `OpenAIClient` is supported, compatibility-only, or deprecated
- [x] Decide whether the DI container remains public, becomes internal, or is removed
- [x] Reduce `llm_client.__init__` to stable exports only
- [x] Remove accidental hidden exports from the top-level package namespace
- [x] Align package docstrings and reference examples with stable module namespaces
- [x] Add explicit `__all__` discipline to public modules
- [x] Add regression tests for top-level hidden exports and public module export lists
- [x] Move weak or compatibility APIs under a clearer compatibility/internal namespace
- [x] Standardize request, result, usage, and event types under canonical modules
- [x] Reserve canonical modules for actions, events/runtime events, replay, plugins, and budgets/cost
- [x] Add public API documentation for every stable module
- [x] Add compatibility tests for the public API surface

### Phase 2: Provider Platform and Registry

- [x] Implement a real `ProviderRegistry`
- [x] Define a `ProviderCapabilities` model
- [x] Register OpenAI through the common registry path
- [x] Register Google through the common registry path
- [x] Register Anthropic through the common registry path
- [x] Replace ad hoc provider factory logic with registry-backed resolution
- [x] Replace or evolve `StaticRouter` into policy-driven routing
- [x] Add capability-aware routing policies
- [x] Add provider health tracking
- [x] Add provider priority, latency, cost, and compliance metadata
- [x] Add explicit provider override support
- [x] Add provider routing observability
- [x] Add provider contract tests for overlapping capabilities

### Phase 3: Canonical Content Model and Unified I/O

- [x] Create the canonical `llm_client.content` module
- [x] Define content block types for text, image, audio, file, reasoning, tool call, tool result, and metadata
- [x] Define canonical request and response envelopes around those blocks
- [x] Add backward-compatible adapters from current message representations
- [x] Update OpenAI provider to translate through the content model
- [x] Extend canonical content translation to Anthropic and Google adapters
- [x] Adopt canonical content envelopes in engine-backed helper paths
- [x] Adopt canonical content envelopes in engine-backed planner paths
- [x] Adopt canonical content envelopes in legacy client facade paths
- [x] Adopt canonical content envelopes in tool-runtime engine paths
- [x] Adopt canonical content envelopes in agent engine paths
- [x] Adopt canonical content envelopes in structured engine paths
- [x] Update Google provider to translate through the content model
- [x] Update Anthropic provider to translate through the content model
- [x] Define unknown/unsupported content handling rules
- [x] Materialize `MetadataBlock` into prompt text by default, with explicit opt-out for skip behavior
- [x] Unify streaming and non-streaming output shapes
- [x] Add normalization round-trip tests
- [x] Define precise canonical `FileBlock` semantics
- [x] Add a file preparation and normalization layer
- [x] Implement native OpenAI `FileBlock` transport
- [x] Define and implement non-native provider fallback behavior for `FileBlock`
- [x] Add file transport cache/versioning semantics
- [x] Add `FileBlock` contract tests and real-file cookbook examples

### Phase 4: Model Metadata System

- [x] Design the model catalog format
- [x] Externalize static model profiles into metadata assets
- [x] Add schema validation for model metadata assets
- [x] Add metadata loader and resolution APIs
- [x] Add local override support for deployments
- [x] Add context window, output limit, tokenizer, and pricing fields
- [x] Add modality/tool/JSON/reasoning capability flags
- [x] Add deprecation and replacement metadata
- [x] Make provider defaults resolve through the metadata catalog
- [x] Fix stale/invalid default model values
- [x] Fix provider factory/model support mismatches
- [x] Add metadata drift tests

### Phase 5: Engine Consolidation and Reliability Core

- [x] Make engine-first execution the documented default
- [x] Update `Agent` to prefer engine-backed execution when available
- [x] Identify and refactor all repo-local engine bypasses
- [x] Standardize retry and backoff classification
- [x] Standardize timeout semantics
- [x] Harden idempotency semantics for non-streaming requests
- [x] Harden idempotency semantics for streaming requests
- [x] Harden idempotency semantics for agent turns
- [x] Improve circuit breaker policies
- [x] Improve provider failover policies
- [x] Add request lifecycle hooks for pre-dispatch, post-response, and error handling
- [x] Add engine diagnostics for retries, cache, routing, and final error status
- [x] Add a stable engine diagnostics recorder hook for observability consumers
- [x] Unify runtime context handling so repo-local context shims are no longer needed
- [x] Add engine behavior tests for retries, fallback, stream failures, and cancellation
- [x] Route initial operator executor, switchboard planner, and title generation LLM calls through `ExecutionEngine` when present
- [x] Centralize `RequestSpec` construction helpers for engine-backed paths

### Phase 6: Agent, Tooling, and Tool Call Engine

- [x] Define `AgentDefinition`
- [x] Define prompt/template attachment references
- [x] Define execution, output, and memory policy objects for agents
- [x] Separate agent definition from mutable runtime state
- [x] Create a dedicated tool execution engine module
- [x] Support explicit single-tool execution mode
- [x] Support explicit sequential tool execution mode
- [x] Support explicit parallel tool execution mode
- [x] Support planner-managed tool execution mode
- [x] Standardize tool result envelopes for success, partial, and error cases
- [x] Add per-tool timeout metadata
- [x] Add per-tool retry metadata
- [x] Add per-tool concurrency metadata
- [x] Add per-tool safety/trust metadata
- [x] Extract generic structured tool-loop runtime from `intelligence_layer`
- [x] Extract generic tool-call normalization into `llm_client`
- [x] Add tool runtime regression tests for malformed calls, depth limits, and allowlist violations

### Phase 7: Context Management and Memory

- [x] Define a context planner interface
- [x] Define summarization strategy interfaces
- [x] Define trimming strategy interfaces
- [x] Define memory retrieval and write interfaces
- [x] Add a minimal short-term memory implementation
- [x] Add optional persistent summary support
- [x] Add retrieval-backed memory interface hooks
- [x] Extract generic history truncation strategies from `intelligence_layer`
- [x] Extract generic history scoring heuristics from `intelligence_layer`
- [x] Add tiered truncation support
- [x] Add semantic relevance selection support
- [x] Add multi-source context assembly interfaces
- [x] Keep domain entity loaders outside the package
- [x] Add observability for context planner decisions
- [x] Add context assembly and memory tests

### Phase 8: Structured Outputs and Validation Hardening

- [x] Unify JSON mode, strict schema mode, and repair-loop mode under one API
- [x] Add capability-aware structured execution mode selection
- [x] Consolidate parallel structured execution implementations
- [x] Extract generic structured repair-loop core into `llm_client`
- [x] Extract generic structured result-envelope normalization into `llm_client`
- [x] Add structured repair diagnostics
- [x] Add structured output trace/debug objects
- [x] Add structured streaming contracts
- [x] Add provider-aware schema normalization
- [x] Add validation coverage for malformed JSON
- [x] Add validation coverage for schema mismatch
- [x] Add validation coverage for missing required fields
- [x] Add validation coverage for tool-call mixed outputs
- [x] Benchmark structured output success and repair rates

### Phase 9: Observability, Logging, Usage, and Cost Intelligence

- [x] Consolidate hooks, telemetry, and logging around one lifecycle event model
- [x] Define standard runtime event taxonomy
- [x] Define request and session report objects
- [x] Improve normalized usage breakdowns
- [x] Improve normalized cost reporting
- [x] Add central redaction controls
- [x] Add safe payload preview modes
- [x] Add benchmark instrumentation hooks
- [x] Promote generic runtime events and replay primitives from `agent_runtime`
- [x] Add package-level replay recording
- [x] Add package-level replay playback
- [x] Add initial stable namespace modules for `types`, `content`, `observability`, and compatibility imports
- [x] Keep storage-backed event persistence outside core package APIs

### Phase 10: Error Model and Failure Semantics

- [x] Decide and enforce one internal error-handling model
- [x] Map provider SDK failures into the unified taxonomy
- [x] Normalize retryable vs non-retryable classification
- [x] Normalize timeout/auth/quota/rate-limit/content-filter/request errors
- [x] Normalize tool runtime failure categories
- [x] Normalize structured output failure categories
- [x] Define remediation hints where useful
- [x] Update providers to use the chosen error model consistently
- [x] Add failure injection tests across providers and tool runtime

### Phase 11: Caching and Persistence Evolution

- [x] Define canonical cache key strategy
- [x] Version cache keys by prompt/content/model/schema shape
- [x] Define cache invalidation policy
- [x] Add cache diagnostics
- [x] Add metadata cache support
- [x] Add summary/context cache support where safe
- [x] Keep persistence pluggable
- [x] Keep ledger storage and billing persistence outside the package
- [x] Add cache correctness tests

### Phase 12: Security, Privacy, and Hardening

- [x] Define prompt/tool/output redaction policy
- [x] Classify log fields as safe, sensitive, or forbidden
- [x] Add safe defaults for provider payload capture
- [x] Add tool output redaction/truncation options
- [x] Document prompt-injection and tool-misuse threat model
- [x] Document secure production deployment defaults
- [x] Add security-focused regression tests where practical

### Phase 13: Test Strategy

- [x] Build provider contract test harness
- [x] Build deterministic fake-provider test utilities
- [x] Build stream transcript fixtures
- [x] Add unit tests for provider request translation
- [x] Add unit tests for provider response parsing
- [x] Add unit tests for content normalization
- [x] Add unit tests for model metadata resolution
- [x] Add unit tests for engine reliability behavior
- [x] Add unit tests for context planner and memory behavior
- [x] Add regression tests for GPT-5-specific OpenAI handling
- [x] Add regression tests for tool name sanitization and schema sanitization
- [x] Add property/fuzz tests for normalization and schema transforms
- [x] Add opt-in live integration tests per provider
- [x] Add failure-injection coverage for 429, 5xx, malformed payloads, and stream interruption

### Phase 14: Benchmark Strategy

- [x] Build benchmark harness structure
- [x] Add deterministic local performance baselines
- [x] Add live benchmark mode with explicit labeling
- [x] Benchmark completion latency
- [x] Benchmark first-token stream latency
- [x] Benchmark full stream latency
- [x] Benchmark embeddings latency and throughput
- [x] Benchmark tool-loop overhead
- [x] Benchmark cache hit/miss performance
- [x] Benchmark failover overhead
- [x] Benchmark context-planning overhead
- [x] Benchmark structured output success and repair rates
- [x] Store benchmark results for trend comparison

### Phase 15: Examples and Developer Experience

Current cookbook entry points:

- `examples/01_one_shot_completion.py`
- `examples/02_streaming.py`
- `examples/03_embeddings.py`
- `examples/04_content_blocks.py`
- `examples/05_structured_extraction.py`
- `examples/06_provider_registry_and_routing.py`
- `examples/07_engine_cache_retry_idempotency.py`
- `examples/08_tool_execution_modes.py`
- `examples/09_tool_calling_agent.py`
- `examples/10_context_memory_planning.py`
- `examples/11_observability_and_redaction.py`
- `examples/12_benchmarks.py`
- `examples/13_batch_processing.py`
- `examples/14_sync_wrappers.py`
- `examples/15_rate_limiting.py`
- `examples/16_fastapi_sse.py`
- `examples/17_persistence_repository.py`
- `examples/18_memory_backed_assistant.py`

- [x] Write architecture overview docs
- [x] Write public API reference docs
- [x] Write provider setup guides
- [x] Write routing and failover guide
- [x] Write tool runtime guide
- [x] Write structured outputs guide
- [x] Write context and memory guide
- [x] Write observability and redaction guide
- [x] Write migration guide from direct SDK usage
- [x] Add one-shot completion example
- [x] Add structured extraction example
- [x] Add embeddings example
- [x] Add streaming example
- [x] Add tool-calling agent example
- [x] Add provider failover example
- [x] Add engine-with-cache-and-retry example
- [x] Add memory-backed assistant example
- [x] Add FastAPI SSE example
- [x] Add batch processing example
- [x] Validate examples in CI

### Phase 16: Packaging, Release, and Standalone Readiness

- [x] Review dependency footprint and import side effects
- [x] Add optional dependency extras for providers and integrations
- [x] Add package installation matrix docs
- [x] Add changelog process
- [x] Add semantic versioning policy docs
- [x] Add support policy docs
- [x] Add release automation
- [x] Add artifact build verification in CI
- [x] Add publish workflow
- [x] Test standalone installation across representative environments

### Extraction and Migration Program

- [x] Define the post-extraction package layering explicitly
- [x] Migrate `agent_runtime.context` toward the canonical `llm_client` execution context
- [x] Migrate generic `agent_runtime` replay primitives into `llm_client.replay`
- [x] Migrate `agent_runtime` router structured parsing onto `llm_client.structured`
- [x] Migrate generic `agent_runtime` budget/usage concepts into package-level types
- [x] Decide whether plugin/extension registry lands in `llm_client.plugins` now or later
- [x] Keep `agent_runtime` storage, transports, and deployment glue outside the core package
- [x] Migrate generic `intelligence_layer` structured tool-loop runtime into `llm_client`
- [x] Migrate generic `intelligence_layer` context-planning heuristics into `llm_client.context`
- [x] Keep `intelligence_layer` operators, manifests, prompts, and policies outside the package
- [x] Refactor repo consumers to use canonical `llm_client` APIs after extraction
- [x] Remove superseded repo-local parallel implementations once replacements are proven

Implemented in the final hardening phase:

- canonical ledger, budget, and usage primitives now live in `llm_client.budgets`
- `agent_runtime.ledger` now acts as a compatibility export layer over the
  package-owned runtime substrate
- the plugin/extension registry was explicitly deferred from `llm_client 1.0`
  because the current registry remains host-runtime shaped
- the `intelligence_layer` conversation assembler now composes
  `llm_client.context_assembly` and `llm_client.context_planning` rather than
  owning a parallel generic assembly path

Expanded 1.0 scope:

- service-driver/adaptor program for optional generic integrations such as
  PostgreSQL, MySQL, Redis, and Qdrant is now in scope for the `1.0.0`
  program
- generic tool-wrapping helpers over those drivers remain in scope, while
  business queries and domain workflows must still stay out of the core
  package

### Immediate Next Implementation Targets

- [x] Fix provider factory/default inconsistencies
- [x] Build the provider contract test harness skeleton
- [x] Start engine-first consolidation
- [x] Extract structured tool-loop runtime
- [x] Unify execution context between `agent_runtime` and `llm_client`
- [x] Continue refactoring remaining direct-provider execution paths to prefer `ExecutionEngine`
- [x] Extract generic history truncation and scoring heuristics into `llm_client`
- [x] Promote generic runtime event and replay primitives from `agent_runtime`
- [x] Build deterministic fake-provider utilities and broader engine reliability tests

### Current Frontier

The package has already crossed the architectural threshold where `llm_client`
is clearly the reusable runtime core. The current frontier is no longer proving
the design. It is finishing consistency work and closing the remaining gaps that
still keep the package from being a clean standalone framework.

Current implementation focus:

- Extend operational observability now that retry/timeout, idempotency, and
  failover behavior are more consistent across engine and provider layers.
- Continue tightening the remaining low-level/public API boundaries so direct
  provider calls stay intentional and clearly separated from engine-first
  helper/runtime layers.
- Prepare the next standalone-package maturity pass around operational
  diagnostics and production policy controls.

Additional release-documentation deliverables:

- write a comprehensive package API guide covering canonical input/output
  shapes, the stable module map, and the recommended entry points for common
  use cases
- write a package usage and capabilities guide that explains how to use each
  major capability, what to avoid, and where provider-specific differences
  matter

This means the next work is primarily consolidation, API discipline, and
operational hardening rather than foundational architecture creation.
