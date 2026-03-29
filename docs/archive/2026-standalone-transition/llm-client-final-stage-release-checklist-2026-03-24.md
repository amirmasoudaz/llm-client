# llm-client Final-Stage Release Checklist

Date: 2026-03-24
Repo: `intelligence-layer-bif`
Primary package: `llm_client`
Primary execution tracker: [llm-client-modernization-roadmap-2026-03-09.md](./llm-client-modernization-roadmap-2026-03-09.md)
Related audit: [llm-client-capability-audit-2026-03-09.md](./llm-client-capability-audit-2026-03-09.md)
Public API map: [llm-client-public-api-v1.md](./llm-client-public-api-v1.md)

## Purpose

This document is the final-stage release checklist for turning `llm_client`
into a true standalone package release candidate and then a `1.0.0` package.

It exists because the modernization roadmap is broad and historical. At this
stage, we need a smaller execution document that answers:

- what still blocks a credible `1.0.0`
- what decisions have already been made
- what must be finished before a release candidate
- what can wait until after `1.0.0`

This document should be treated as the last-mile release gate, not as a
replacement for the full modernization roadmap.

## Current Position

`llm_client` has already crossed the architectural threshold where it is
clearly the reusable LLM runtime core for this repository.

That is now visible in the codebase:

- provider abstractions, engine, structured execution, tools, content
  envelopes, routing, model catalog, observability, replay, and agent runtime
  substrate are centered in `llm_client`
- `agent_runtime` is increasingly a host/runtime shell
- `intelligence_layer` is increasingly a domain/product layer

The release question is no longer whether the package is real. The release
question is whether the remaining inconsistencies, packaging gaps, and
boundary-cleanup tasks are small enough that we can defend `1.0.0`.

Current answer:

- `v1.0.0` should happen
- `v1.0.0` should not happen until the checklist below is complete
- the right next milestone is `1.0.0-rc1`, not immediate `1.0.0`

## Release Decision Record

### Decision 1: Keep `llm_client` as the package name for now

Status:
- provisional keep

Reasoning:

- the import path already exists and is widely used inside the repo
- renaming now would create extra migration churn during the final hardening
  phase
- the name is not ideal because the package is broader than a thin "client",
  but that problem can be mitigated with better documentation and positioning

Implication:

- keep the import path as `llm_client` for `1.0.0`
- document clearly that the package is a runtime framework, not just an SDK
  wrapper
- revisit naming only if there is a future repo split or standalone branding
  pass

### Decision 2: `llm_client` remains library-first

Status:
- accepted

Reasoning:

- the package itself is the durable reusable asset
- a centralized API service may be valuable later, but that is a separate
  deployment product, not the core package contract
- forcing all consumers through a service now would add latency, tenancy, and
  operational complexity at exactly the moment the package boundary is being
  finalized

Implication:

- finish `llm_client` as a strong standalone library first
- if desired later, build a separate API server on top of it
- do not let "future central service" distort the core package API

### Decision 3: Examples remain examples unless the abstraction is generic

Status:
- accepted

Reasoning:

- many newer cookbook examples are intentionally application-shaped
- they are useful as proofs and adoption guides, but that does not mean every
  technique shown there belongs in the stable package core
- features like RAG orchestration, incident control planes, or SQL assistants
  should only move into the package if they can be expressed as generic
  primitives

Implication:

- keep application workflows out of the stable package surface
- only promote reusable abstractions such as content blocks, routing,
  retrieval-friendly interfaces, or generic tool/runtime helpers

### Decision 4: `agent_runtime` and `intelligence_layer` stay, but thinner

Status:
- accepted

Reasoning:

- `agent_runtime` still owns runtime shell concerns:
  - storage
  - transports
  - deployment glue
  - job/runtime hosting behavior
- `intelligence_layer` still owns domain/product concerns:
  - operators
  - manifests/contracts
  - prompts
  - domain policy
  - app/API behavior
- the goal was never to erase those layers entirely
- the goal was to stop them from owning reusable generic LLM substrate

Implication:

- continue moving generic mechanisms downward into `llm_client`
- continue keeping business logic upward in the higher layers
- document the post-extraction layering explicitly before `1.0.0`

## Release Gates

`1.0.0-rc1` should not be cut until all `RC1 Gate` items are complete.

`1.0.0` should not be cut until all `GA Gate` items are complete.

## RC1 Gate

### A. Roadmap Completion and Drift Cleanup

- [x] Complete the remaining unchecked items in
  [llm-client-modernization-roadmap-2026-03-09.md](./llm-client-modernization-roadmap-2026-03-09.md)
- [x] Resolve current cookbook drift between:
  - [examples/README.md](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/README.md)
  - [scripts/ci/run_llm_client_examples.py](/home/namiral/Projects/Packages/intelligence-layer-bif/scripts/ci/run_llm_client_examples.py)
  - the actual files under [examples/llm_client](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/llm_client)
- [x] Decide whether [34_end_to_end_mission_control.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/34_end_to_end_mission_control.py)
  is:
  - part of the canonical cookbook
  - intentionally excluded
  - experimental and separately documented
- [x] Update cookbook inventory tests to reflect the final decision about the
  example set
- [x] Update CI example runner scope to match the documented cookbook contract

Reasoning:

- a package cannot credibly claim a cookbook contract while the README, tests,
  and runner disagree
- this is exactly the kind of release drift that causes future maintenance
  confusion

### B. Remaining Core Code Gaps

- [x] Define precise canonical `FileBlock` semantics
- [x] Add a file preparation and normalization layer
- [x] Implement native OpenAI `FileBlock` transport
- [x] Define non-native provider fallback behavior for `FileBlock`
- [x] Add file transport cache/versioning semantics
- [x] Add `FileBlock` contract tests
- [x] Add at least one real-file cookbook example that exercises the supported 
  path and the fallback behavior clearly

Reasoning:

- the content model is now broad enough that `FileBlock` cannot remain vague
- file handling is a major source of provider divergence and user confusion
- if the package claims multimodal support, file semantics must be explicit

### C. Final Extraction and Layering Cleanup

- [x] Define the post-extraction package layering explicitly in docs
- [x] Migrate generic package-level budget and usage concepts fully into
  `llm_client`
- [x] Decide whether a plugin or extension registry lands in
  `llm_client.plugins` now or is deferred explicitly until after `1.0.0`
- [x] Keep `agent_runtime` storage, transports, and deployment glue explicitly
  outside `llm_client`
- [x] Migrate the remaining generic `intelligence_layer` context-planning
  heuristics into `llm_client`
- [x] Keep `intelligence_layer` operators, manifests, prompts, and policies
  explicitly outside `llm_client`
- [x] Refactor repo consumers to use canonical `llm_client` APIs after
  extraction
- [x] Remove superseded repo-local parallel implementations once the
  `llm_client` replacements are proven
- [x] Continue refactoring remaining direct-provider execution paths to prefer
  `ExecutionEngine`

Reasoning:

- if `1.0.0` ships before the layering is explicit, future developers will
  reintroduce duplication and bleed responsibilities back upward
- package stability is not only about code; it is also about ownership and
  boundaries
- `llm_client.plugins` is explicitly deferred because the current plugin layer
  is still host-runtime shaped and would weaken the package boundary if forced
  into `1.0.0`

### D. Open-Source and Packaging Hygiene

- [x] Add package metadata fields to [pyproject.toml](/home/namiral/Projects/Packages/intelligence-layer-bif/pyproject.toml):
  - `license`
  - `authors`
  - `maintainers`
  - `classifiers`
  - `keywords`
- [x] Add `py.typed` and package it correctly
- [x] Add a top-level `LICENSE`
- [x] Add `CONTRIBUTING.md`
- [x] Add `SECURITY.md`
- [x] Add `CODE_OF_CONDUCT.md`
- [x] Review whether the repo root README should become more package-centric or
  whether the package should later move to its own repo
- [x] Ensure built artifacts include the package assets and typing marker
- [x] Ensure package install smoke tests cover the supported extras matrix

Reasoning:

- the package is installable today, but that is not the same thing as being
  open-source-release-ready
- external adopters expect standard metadata and governance files
- `py.typed` is part of a serious Python package story if the public API is
  intended to be stable

### E. Documentation Freeze for the Stable Surface

- [x] Perform one final public API review against
  [llm-client-public-api-v1.md](./llm-client-public-api-v1.md)
- [x] Confirm which modules are stable, compatibility, advanced, provisional,
  and internal
- [x] Update [llm_client/README.md](/home/namiral/Projects/Packages/intelligence-layer-bif/llm_client/README.md)
  to match the final stable package story exactly
- [x] Add a "what this package is not" section to the package docs
- [x] Add explicit "do use / do not use" guidance for:
  - stable namespaces
  - compatibility APIs
  - advanced/integration helpers
- [x] Confirm semver and support policy docs reflect the actual intended
  package surface

Reasoning:

- `1.0.0` only means something if the stable surface is explicit
- documentation drift after the API freeze would undermine the release

### E1. Package API and Capability Guides

- [x] Write a comprehensive package API guide covering:
  - canonical input/output shapes
  - request/response/content models
  - engine/provider/agent/tool relationships
  - the stable module map
  - common use cases and the recommended entry points for each
- [x] Write a package usage and capabilities guide covering:
  - how to install and configure the package
  - how to use each major capability
  - what to use versus what to avoid
  - capability-specific caveats, limits, and provider differences
  - how to choose the right layer for a task

Reasoning:

- the package README and cookbook are necessary but not sufficient for `1.0.0`
- external adopters should not have to reverse-engineer the codebase to
  understand the package contract or learn how to use the package correctly
- these guides are part of making `llm_client` a real reusable framework rather
  than just a well-refactored internal module

### F. Example Documentation Expansion

- [x] Expand the cookbook docs beyond the current index-style README
- [x] Add an examples guide that explains:
  - prerequisites
  - expected env vars
  - which examples make real provider calls
  - which examples require optional infra
  - how to interpret outputs
  - common failure/skip modes
- [x] Add one short purpose-and-scope note for each cookbook example
- [x] Clearly label which examples are:
  - core capability demos
  - application-shaped reference examples
  - experimental or infra-heavy examples
- [x] Document how the examples map back to stable package APIs
- [x] Confirm every example reflects current package behavior and naming

Reasoning:

- the examples are now substantial enough that discoverability and explanation
  matter as much as code correctness
- cookbook examples are one of the main adoption surfaces for a package like
  this

### G. Release Candidate Validation

- [x] Run the focused package test suites required for release
- [x] Run packaging/artifact verification
- [x] Run the cookbook CI path end-to-end
- [x] Run live provider integration tests in the supported configuration
- [x] Run the benchmark/report path and save the release-candidate artifacts
- [x] Verify no unresolved package-facing regressions remain in the issue list
- [x] Cut `1.0.0-rc1`

Reasoning:

- the release candidate exists to validate the package contract under realistic
  usage, not just local confidence
- local RC status:
  - focused package suites passed
  - artifact verification passed against rebuilt wheel and sdist
  - full cookbook runner completed under credential-cleared skip mode
  - deterministic RC benchmark artifacts were generated at
    `artifacts/benchmarks/llm_client_rc_deterministic.json` and
    `artifacts/benchmarks/llm_client_rc_deterministic_comparison.json`
  - benchmark strategy and structured benchmarking test suites passed locally
  - live provider smoke tests passed for OpenAI and Anthropic using the
    repository `.env` credentials
  - no separate package-facing regression issue was found beyond the main
    umbrella modernization issue (`CAN-12`)
  - version metadata and release notes were updated for `1.0.0rc1`
  - `1.0.0rc1` wheel and sdist were rebuilt successfully and passed
    `twine check`
  - final `1.0.0` wheel and sdist were rebuilt successfully and passed
    `twine check`
  - final `1.0.0` artifact verification passed against the GA distributions

## GA Gate

### H. Post-RC Feedback and Final Corrections

- [x] Resolve any RC1 issues discovered during package install, examples, or
  live provider validation
- [x] Resolve any RC1 documentation ambiguities
- [x] Resolve any RC1 public API naming or packaging surprises
- [x] Confirm no additional breaking API changes are needed

Reasoning:

- if meaningful API or packaging churn is still required after RC1, the package
  was not actually ready for `1.0.0`

### I. Version Freeze

- [x] Freeze the stable API for `1.x`
- [x] Document any compatibility-only surfaces that remain transitional
- [x] Ensure release notes state the final stable module contract clearly
- [ ] Tag and publish `1.0.0`

Reasoning:

- after `1.0.0`, changes to stable namespaces must be backward-compatible
- the package should not continue behaving like a moving internal utility after
  this point
- the durable `1.0.0` git tag should be created on the final standalone
  package repository history after extraction, not on a monorepo history that
  is about to be split

## Explicit Non-Blockers for `1.0.0`

These items are valuable, but should not block the `1.0.0` package release
unless they expose a concrete design flaw in the stable core.

- a centralized multi-tenant API server built on top of `llm_client`
- broad built-in RAG product features
- moving the package into a separate repository
- adding large new application workflows inspired by cookbook examples
- aggressive base-dependency minimization beyond what is already documented

Reasoning:

- these are follow-on strategic expansions, not release blockers for the core
  standalone package contract

## Expanded 1.0 Scope

This section was originally tracked as post-`1.0`. It is now explicitly in
scope for the `1.0.0` program.

### J. Service Drivers and Adaptors

- [x] Define the canonical extension boundary for service access:
  - low-level drivers/adaptors in `llm_client`
  - agent-facing tool wrappers built on top of them
  - project-specific queries, mutations, and business semantics outside the
    package
- [x] Decide final namespace naming:
  - `llm_client.adapters` is the public stable namespace target
  - lower-level drivers remain internal or advanced implementation detail
- [x] Start with optional extras and thin generic integrations for:
  - [x] PostgreSQL
  - [x] MySQL
  - [x] Redis
  - [x] Qdrant
- [x] Keep these integrations optional and installable by extras rather than
  core dependencies
- [x] Define typed request/result contracts for generic operations such as:
  - SQL read/query execution
  - SQL write/mutation execution
  - key-value and cache operations
  - vector upsert/search/delete
- [x] Add explicit safety and capability boundaries:
  - [x] read-only defaults where appropriate
  - [x] opt-in write support
  - [x] timeout support
  - [x] retry and observability hooks
  - [x] budget and usage accounting integration
- [x] Add generic tool-construction helpers so projects can wrap drivers as
  agent tools without duplicating transport logic
- [x] Keep business/domain queries out of the package
- [x] Add cookbook examples showing:
  - [x] direct driver usage
  - [x] tool wrapping over a driver
  - [x] safe read-only access patterns
  - [x] optional write workflows with explicit gating

Reasoning:

- there is real value in centralizing reusable service-access substrate next to
  the agent/tool runtime
- the package should own generic connectivity and execution mechanics, not
  product-specific queries or business workflows
- this work is important, but it is an expansion program, not a release
  blocker for `1.0.0`

## Questions We Must Answer Before `1.0.0`

- [x] Are we satisfied keeping the import path `llm_client` for `1.x`?
  - Answer: `Yes`.
  - Reasoning: the package is broader than a thin client now, but renaming at
    the `1.0.0` boundary would create migration churn across the repo and
    weaken the stabilization effort. Keep the import path for `1.x`; revisit
    naming only in a future major version or repo split.
- [x] Is the stable API map truly small enough and strong enough to defend?
  - Answer: `Yes, with the current frozen stable namespaces`.
  - Reasoning: the stable contract is now concentrated around providers,
    models, types, content, context, budgets, engine, agent, tools, memory,
    cache, observability, validation, config, and adapters. Compatibility and
    advanced surfaces are explicitly separated and documented.
- [x] Are there any remaining direct-provider paths that are accidental rather
  than intentional low-level APIs?
  - Answer: `No known accidental higher-layer paths remain`.
  - Reasoning: higher-layer consumer code was normalized onto engine-backed
    execution earlier in the program. Remaining direct-provider paths live
    inside `llm_client` as intentional low-level or compatibility surfaces.
- [x] Is the package README sufficient for someone outside this repo to adopt
  the package without reverse-engineering internals?
  - Answer: `Yes, for v1 adoption`.
  - Reasoning: the package README, API guide, usage guide, installation
    matrix, examples guide, and guide index now form a coherent external
    adoption surface. Future polish is still possible, but reverse-engineering
    internals is no longer required for the core package path.
- [x] Are example expectations and setup requirements explicit enough?
  - Answer: `Yes, with the current cookbook contract`.
  - Reasoning: the cookbook README, examples guide, runner subsets, inventory
    tests, and skip/fail-fast behavior now align. The RC smoke also validated
    the credential-cleared skip contract across the full cookbook.
- [x] Are `agent_runtime` and `intelligence_layer` now clean consumers rather
  than shadow owners of generic runtime mechanisms?
  - Answer: `Yes, materially`.
  - Reasoning: the generic runtime substrate, context planning, budgets,
    replay/events, and structured/tool-loop substrate have been moved down
    into `llm_client`. The higher layers now read as host/application
    consumers rather than shadow owners of generic mechanisms.
- [x] Are we comfortable publishing the current repository structure, or should
  we position this as a monorepo-hosted package explicitly?
  - Answer: `Publish as a monorepo-hosted package explicitly`.
  - Reasoning: the package itself is publishable, but the repository is still
    broader than a package-only OSS repo. The current docs and root README
    should continue to present this honestly as a monorepo-hosted standalone
    package.

## Execution Order

Recommended order for the final stage:

1. finish roadmap leftovers and cookbook drift cleanup
2. finish extraction/layering cleanup and residual migration work that is still genuinely open
3. freeze docs and stable API wording
4. write the package API guide and full capabilities/how-to guide
5. expand cookbook documentation
6. run RC validation and cut `1.0.0-rc1`
7. resolve RC findings
8. cut `1.0.0`

## Definition of Done

`llm_client` is ready for `1.0.0` when all of the following are true:

- the roadmap has no remaining release-critical unchecked items
- the cookbook contract is internally consistent across code, docs, tests, and
  CI
- the stable API is frozen and documented clearly
- generic runtime substrate has been extracted cleanly from higher layers
- package metadata and OSS hygiene are complete
- release-candidate validation passes
- no unresolved breaking issues remain

At that point, `llm_client` can be treated as:

- a standalone reusable package
- the canonical LLM and agentic runtime substrate for this suite
- a library whose future stable changes must respect semantic versioning
