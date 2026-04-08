# Implementation Plan

## Title

OpenAI Docs-Parity and Package Robustness Plan for `llm-client` `1.2`

## Metadata
- Date: 2026-04-08
- Author: Codex
- Status: Approved
- Related issue / task: Comprehensive OpenAI docs-parity audit and `1.2` implementation planning
- Related analysis report:
  - [docs/openai-docs-ledger-completeness-audit-2026-03-31.md](/home/namiral/Projects/Packages/llm-client-v1/docs/openai-docs-ledger-completeness-audit-2026-03-31.md)
  - [docs/openai-provider-capability-audit.md](/home/namiral/Projects/Packages/llm-client-v1/docs/openai-provider-capability-audit.md)
  - [docs/llm-client-evaluation-report-2026-04-01.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-evaluation-report-2026-04-01.md)
- Related PRs:
  - merged: `release/1.1.1`
  - open at planning time: `docs/tool-creation-guide`
- Related docs:
  - [llm_client/README.md](/home/namiral/Projects/Packages/llm-client-v1/llm_client/README.md)
  - [docs/llm-client-package-api-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-package-api-guide.md)
  - [docs/llm-client-public-api-v1.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-public-api-v1.md)
  - [docs/llm-client-usage-and-capabilities-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-usage-and-capabilities-guide.md)

## Executive Summary

`llm-client` is already operational and release-usable for its current OpenAI-focused contract, but it does not yet have full parity with the latest OpenAI docs surface. The `1.2` opportunity is not basic provider viability; it is breadth expansion, correctness hardening against the latest official docs, and clearer package boundaries around what is fully supported, partially supported, or intentionally out of scope.

This plan proposes a docs-driven `1.2` program that uses two complementary sources of truth:

- the local docs-ledger API as the exhaustive inventory and audit corpus
- the official OpenAI docs MCP as the final validator for volatile or high-risk surfaces

The main remaining implementation areas are:

- broader hosted retrieval / file-search resource management
- broader Realtime product coverage
- broader connectors / MCP / skills product coverage
- `tool_search` and namespaced tools from the current function-calling docs
- fuller model-registry coverage
- one more end-to-end docs and example re-audit after the implementation tranche lands

Locked planning decisions for `1.2`:

- `tool_search` will be implemented as an advanced, OpenAI-specific surface rather than a stable generic tool abstraction.
- tool namespaces will be implemented as an OpenAI-specific capability rather than a provider-agnostic package abstraction.
- “skills” will be explicitly deferred as a first-class package abstraction.
- model-registry expansion will target supported product-family models, not full docs-corpus coverage.
- video / Sora and other high-churn preview-like surfaces are excluded from `1.2`.

## Problem / Opportunity Statement

The package’s OpenAI support is strong but uneven. The core Responses, conversation, moderation, files, vector stores, speech, image, webhook, fine-tuning, and staged deep-research surfaces are already present. The remaining problem is that the package still lags the current OpenAI docs in a few high-value product families and lacks a final `1.2`-grade capability statement grounded in both the local docs-ledger corpus and official OpenAI docs.

The opportunity is to move from “strong implementation with known partials” to “robust, intentionally complete `1.2` contract,” where the remaining exclusions are explicit product decisions rather than audit leftovers.

## Goals
- [ ] Produce a fresh, comprehensive docs-driven gap inventory for the `1.2` scope using both the local docs-ledger API and official OpenAI docs MCP validation.
- [ ] Close the highest-value remaining OpenAI gaps that fit the package’s architecture and semver story.
- [ ] Add or widen tests, examples, docs, and model metadata so the implemented surface is durable and accurately documented.
- [ ] Ship `1.2` with a defensible capability statement and a minimal intentionally deferred backlog.

## Non-Goals
- [ ] Full one-to-one coverage of every current and future OpenAI product, preview surface, or experimental SDK helper.
- [ ] Re-architecting `llm_client` into the OpenAI Agents SDK or making OpenAI-specific abstractions dominate the package’s provider-agnostic design.
- [ ] Broad repo cleanup unrelated to OpenAI docs parity.
- [ ] Silent expansion of the stable public API without corresponding docs, tests, and semver analysis.
- [ ] First-class package “skills” abstractions in `1.2`.
- [ ] Sora / video-generation support in `1.2`.
- [ ] Full docs-corpus model registry coverage in `1.2`.

## Context and Background

The repo already completed a substantial OpenAI provider expansion during the `1.1.0` line and a packaging/metadata hardening patch in `1.1.1`. The current audits show the package is strong in core OpenAI workflows and validated by the full cookbook run for the current contract. The remaining work is not emergency repair. It is a strategic extension of the package toward a more complete and current `1.2` release.

Recent cross-validation against the official OpenAI docs MCP confirmed most earlier conclusions from the local docs-ledger audit, but it also surfaced at least one concrete new gap category not explicitly emphasized in the local audit:

- `tool_search`
- namespaced tools

The official docs also reinforced that some areas remain intentionally partial, particularly Realtime breadth, deep-research lifecycle breadth, and broader hosted file-search/resource management.

## Codebase / System Analysis Summary
- Areas inspected:
  - `llm_client/providers/openai.py`
  - `llm_client/engine.py`
  - `llm_client/tools/base.py`
  - `llm_client/models.py`
  - `llm_client/assets/model_catalog.json`
  - docs-ledger audit docs and release/evaluation reports
- Relevant modules / services:
  - provider surface in [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py)
  - engine wrappers in [`llm_client/engine.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/engine.py)
  - OpenAI tool abstractions in [`llm_client/tools/base.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/tools/base.py)
  - model registry in [`llm_client/models.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/models.py) and [`llm_client/assets/model_catalog.json`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/assets/model_catalog.json)
  - cookbook runner and examples under `examples/` and `scripts/ci/`
- Current behavior:
  - strong OpenAI coverage for Responses, conversations, background mode, moderation, files, vector stores, speech APIs, image APIs, fine-tuning jobs, webhooks, realtime helpers, MCP/connectors descriptors, and staged deep-research helpers
  - partial coverage for broader Realtime, retrieval/file-search management, deep-research lifecycle, connectors/skills product surface, and total model-catalog breadth
  - no first-class `tool_search` or tool namespaces support
- Known pain points:
  - local docs-ledger API reliability is imperfect (`/docs/catalog` and occasional endpoint instability)
  - official docs MCP search is authoritative but noisier for broad inventory-style queries
  - several OpenAI surfaces evolve quickly, so a “done once” audit will go stale
- Architecture notes:
  - the package is not a thin OpenAI SDK wrapper; it has a normalized runtime/engine/provider contract
  - widening package surface must respect stable vs advanced vs provider-specific boundaries
  - first-class support should be reserved for surfaces that can be documented, tested, and maintained coherently

## Dependency and Interface Impact
- Internal dependencies:
  - `llm_client.providers.base`
  - `llm_client.providers.types`
  - `llm_client.spec`
  - `llm_client.tools.base`
  - `llm_client.models` / `model_catalog`
  - docs/examples/tests
- External dependencies:
  - OpenAI Python SDK surface
  - OpenAI docs MCP tooling for cross-validation
  - local docs-ledger API for exhaustive audit inventory
- API / schema / contract impact:
  - likely additions to provider and engine methods
  - likely additions to tool abstractions for `tool_search` and namespaced tools
  - likely additions to typed result objects for richer Realtime / retrieval workflows
  - stable public API docs must be updated with any new `1.2` surface
- Build / deployment impact:
  - mostly Python package and docs changes
  - examples and tests may require new environment-gated setup paths
- Backward compatibility considerations:
  - `1.2` should remain source-compatible for existing `1.1.x` users where possible
  - new APIs should be additive
  - behavior changes must be documented explicitly if stricter validation or tool behavior changes are introduced

## Constraints and Assumptions
### Constraints
- The package must preserve its provider-agnostic architecture and semver discipline.
- Public docs, examples, and tests must land with code, not later.
- The plan must treat the local docs-ledger inventory and official OpenAI docs as complementary sources, not interchangeable ones.
- Some OpenAI product surfaces may not fit the package cleanly enough to justify first-class support in `1.2`.

### Assumptions
- `1.2` is the correct release line for the next major OpenAI breadth expansion.
- The user wants a comprehensive audit and planning artifact first, with implementation following in disciplined branches/PRs.
- The local docs-ledger API remains available enough for inventory work, even if some endpoints are intermittently unreliable.
- Each meaningful implementation change will land in a dedicated scoped branch and commit sequence, with PR/MR-based merges only.

## Facts, Assumptions, and Unknowns

### Facts
- Core OpenAI coverage is already broad and validated for the current contract.
- `tool_search` and tool namespaces do not currently appear in the repo as first-class features.
- The official docs MCP confirms broader Realtime, deep-research, and file-search surfaces than the package currently wraps.
- The package’s cookbook and evaluation state are positive for the `1.1.x` contract.

### Assumptions
- Not every OpenAI preview or niche product family should become a first-class package abstraction in `1.2`.
- The most valuable `1.2` work is additive expansion on top of the current stable package, not another provider rewrite.
- The package should prefer robust support for supported model/product families over exhaustive parity for every documented model or surface.

### Unknowns
- None currently blocking planning approval.

## Risks and Failure Modes
### Risk 1
- Impact: Expanding too much OpenAI-specific product surface could erode the package’s provider-agnostic coherence and make `1.x` maintenance harder.
- Detection: APIs or docs start depending on OpenAI-only concepts that do not fit the rest of the runtime.
- Mitigation: keep a strict first-class vs passthrough vs intentionally unsupported matrix; prefer additive, clearly namespaced abstractions.

### Risk 2
- Impact: The docs-ledger corpus and official OpenAI docs drift, producing stale or contradictory audit conclusions.
- Detection: the two sources disagree materially during re-audit, or official MCP fetches show feature changes not reflected locally.
- Mitigation: use the local ledger as inventory, but require official MCP validation before marking high-risk surfaces “implemented.”

### Risk 3
- Impact: Realtime and hosted-tool breadth work can balloon into a product-management layer too large for `1.2`.
- Detection: helper additions start multiplying without a clear package contract or test boundary.
- Mitigation: split Realtime and hosted-tool work into bounded phases with explicit no-go points.

### Risk 4
- Impact: Model-registry expansion becomes endless churn without improving package usability.
- Detection: large registry diffs produce little runtime value and constant doc churn.
- Mitigation: prioritize models tied to actually supported feature families and document the policy for inclusion.

### Risk 5
- Impact: Large feature branches or mixed commits make `1.2` harder to review, debug, and release safely.
- Detection: branches contain unrelated changes, or commit history mixes docs, runtime, and examples without clear boundaries.
- Mitigation: require scoped branches and one meaningful change per commit wherever practical, with explicit PR/MR review boundaries.

## Proposed Approach

Use a two-source audit model and a bounded implementation program:

1. Re-audit the repo against the local docs-ledger corpus to refresh exhaustive inventory coverage.
2. Cross-validate the high-value and volatile gaps against the official OpenAI docs MCP.
3. Convert the validated gaps into a `1.2` implementation matrix with three statuses:
   - first-class in `1.2`
   - passthrough/advanced only
   - explicitly deferred
4. Implement only the first-class `1.2` set, pairing each addition with docs, tests, and examples.
5. Finish with a final dual-source re-audit, release validation, and a `1.2` capability statement.

This approach fits the repo because it respects the package’s existing strengths:

- audit-driven expansion
- durable docs artifacts
- cookbook-backed validation
- scoped release branches and semver discipline

Execution discipline for `1.2`:

- every meaningful change gets a dedicated scoped branch
- every meaningful change is committed separately with a clear commit message
- docs, tests, and examples move with the relevant code change, not afterward
- merges to `main` happen only through PR/MR flow

## Alternatives Considered
### Alternative A
- Description: Continue implementing gaps ad hoc without a refreshed comprehensive plan, using only the existing March audits.
- Why not chosen: the package has moved since those audits, and the official docs cross-check already surfaced at least one new gap (`tool_search` / namespaces). This would risk implementing against stale assumptions.

### Alternative B
- Description: Aim for total OpenAI-doc parity in one release.
- Why not chosen: too broad for one bounded `1.2` cycle, and it would blur the line between a normalized runtime package and a full OpenAI product SDK clone.

### Alternative C
- Description: Rely only on the official OpenAI docs MCP and drop the local docs-ledger API from the workflow.
- Why not chosen: official fetches are better for final truth, but the local ledger is superior for exhaustive corpus traversal and inventory-style auditing.

## Phase Plan

### Phase 1: Refresh the dual-source audit baseline
- Objective: Produce an up-to-date gap matrix for `1.2` using both the local docs-ledger inventory and official OpenAI docs validation.
- Why now: planning on stale audit data would be irresponsible and would likely miss new doc-surface gaps.
- Type: Sequential
- Depends on:
  - current repo state on `main`
  - local docs-ledger availability
  - official OpenAI docs MCP availability
- Deliverables:
  - updated audit report under `docs/`
  - explicit `1.2` scope matrix
  - source-of-truth notes documenting where the two doc sources differ
- Checklist:
  - [ ] Re-run the docs-ledger inventory pass over the current OpenAI docs corpus.
  - [ ] Re-run official-docs MCP validation for high-value and volatile surfaces.
  - [ ] Add newly confirmed gaps, including `tool_search` and namespaced tools, to the audit matrix.
  - [ ] Classify each remaining gap as `first-class`, `advanced/passthrough`, or `deferred`.
- Validation:
  - audit matrix cites both local-ledger and official-doc sources where needed
  - gap set is traceable to inspected code paths
- Exit criteria:
  - no major gap category remains unclassified
  - source-of-truth ambiguity is documented instead of implied away
- Green-light cue:
  - the `1.2` scope is small enough to execute intentionally
- Rollback / containment:
  - if source disagreement is too large, hold implementation and publish a `Needs Clarification` audit delta

### Phase 2: Close hosted retrieval and file-search breadth gaps
- Objective: Expand hosted retrieval beyond current generic Files API/vector-store basics to the stable resource workflows justified by the docs.
- Why now: retrieval/file search is one of the clearest remaining product gaps and a recurring building block for higher-level workflows.
- Type: Sequential
- Depends on:
  - Phase 1 classification
- Deliverables:
  - widened retrieval/file-search resource helpers
  - tests and examples for those flows
  - updated capability docs
- Checklist:
  - [ ] Identify stable missing retrieval resources and workflows from the current docs.
  - [ ] Add provider and engine support for the selected hosted retrieval workflows.
  - [ ] Extend typed result surfaces where needed.
  - [ ] Add cookbook examples and focused tests.
- Validation:
  - provider/engine regression tests
  - cookbook examples for hosted retrieval and file-search workflows
- Exit criteria:
  - hosted retrieval/file-search support is no longer limited to the current vector-store/file-batch slice
- Green-light cue:
  - retrieval workflows can be expressed without raw SDK drop-down for supported cases
- Rollback / containment:
  - keep unsupported resources explicitly documented as deferred rather than half-exposed

### Phase 3: Expand Realtime to the next stable boundary
- Objective: Move Realtime from helper-grade support toward a clearer `1.2` first-class contract.
- Why now: official docs show much broader Realtime lifecycle and control coverage than the package currently wraps.
- Type: Sequential
- Depends on:
  - Phase 1 classification
- Deliverables:
  - widened Realtime session/control/event support
  - clarified boundary for what remains outside `1.2`
  - tests and examples covering the new supported slice
- Checklist:
  - [ ] Audit current Realtime docs for the next stable server-side and session-lifecycle surfaces.
  - [ ] Add the bounded set of provider/engine wrappers that fit the package contract.
  - [ ] Add output/event normalization only where the semantics are stable.
  - [ ] Add examples and tests for the widened Realtime lifecycle.
- Validation:
  - focused unit tests
  - example coverage for new Realtime flows
  - final docs parity check for the widened scope
- Exit criteria:
  - Realtime support has a clearly documented supported boundary instead of a vague “helpers only” story
- Green-light cue:
  - major Realtime use cases in package docs no longer require raw SDK access
- Rollback / containment:
  - if Realtime breadth starts to require too much low-level event plumbing, stop at the documented stable boundary

### Phase 4: Add `tool_search`, tool namespaces, and broaden hosted-tool orchestration
- Objective: Close the latest official function-calling gaps and improve hosted tool ergonomics.
- Why now: the official docs cross-check surfaced this as a real missing area, and it fits naturally with the existing tool abstractions.
- Type: Sequential
- Depends on:
  - Phase 1 classification
  - existing `ResponsesBuiltinTool` / `ResponsesCustomTool` surfaces
- Deliverables:
  - advanced OpenAI-specific `tool_search` abstractions
  - OpenAI-specific namespaced tool abstractions
  - broader hosted-tool workflow helpers and docs
- Checklist:
  - [ ] Add advanced/provider-specific `tool_search` support for OpenAI.
  - [ ] Add OpenAI-specific namespaced tool support.
  - [ ] Add translation and validation logic in the provider/tool layer.
  - [ ] Add docs, examples, and tests covering these new tool patterns.
- Validation:
  - request translation tests
  - docs/examples that show model/tool behavior coherently
- Exit criteria:
  - function-calling docs no longer have obvious first-class gaps for `tool_search` and namespaces
- Green-light cue:
  - hosted-tool orchestration reads as intentional package design, not raw dict passthrough
- Rollback / containment:
  - keep these abstractions explicitly OpenAI-specific and out of the stable generic tool surface

### Phase 5: Deepen connectors / MCP / skills and deep-research lifecycle coverage
- Objective: Decide and implement the `1.2` boundary for connectors, MCP, skills, and deep research.
- Why now: these surfaces are connected and should be planned as a product-family group rather than one-off helper additions.
- Type: Sequential
- Depends on:
  - Phase 1 classification
  - Phases 2 to 4 where relevant
- Deliverables:
  - deeper connector/MCP position for the package
  - widened deep-research lifecycle support where justified
  - security and risk notes for these agentic surfaces
- Checklist:
  - [ ] Expand connector/MCP workflows only to the supported stable boundary.
  - [ ] Broaden deep-research lifecycle support where docs and package fit justify it.
  - [ ] Document “skills” as explicitly out of scope for `1.2`.
  - [ ] Add examples, tests, and security notes for the supported agentic flows.
- Validation:
  - targeted tests
  - examples
  - docs review against prompt-injection and exfiltration guidance
- Exit criteria:
  - connectors/MCP/deep-research support is deliberate, documented, and bounded
- Green-light cue:
  - users can see exactly what the package supports without guessing from provider internals
- Rollback / containment:
  - preserve “skills” as a documented `1.2` defer rather than forcing a weak abstraction

### Phase 6: Finish model-registry expansion and metadata alignment
- Objective: Bring model metadata in line with the `1.2` supported feature surface.
- Why now: model coverage should follow supported product surfaces, not run ahead of them.
- Type: Parallelizable
- Depends on:
  - Phases 2 to 5 for final supported-family decisions
- Deliverables:
  - expanded model catalog for supported OpenAI families only
  - updated capability flags
  - docs on model support policy
- Checklist:
  - [ ] Expand the registry for model families actually supported by `1.2`.
  - [ ] Remove or mark clearly any misleading capability metadata.
  - [ ] Add or update model-catalog tests.
  - [ ] Document the inclusion policy for future model additions.
- Validation:
  - model-catalog tests
  - docs consistency review
- Exit criteria:
  - supported model families no longer materially lag the implementation surface
- Green-light cue:
  - model metadata is no longer a notable audit weakness
- Rollback / containment:
  - avoid registry churn for intentionally unsupported or excluded product families such as video/Sora

### Phase 7: Final re-audit, cookbook hardening, and `1.2` release prep
- Objective: Convert the implementation tranche into a release-ready `1.2` state.
- Why now: the expanded surface needs a fresh docs parity statement and end-to-end validation before release.
- Type: Sequential
- Depends on:
  - Phases 2 to 6
- Deliverables:
  - updated audits
  - final cookbook validation
  - `1.2` release notes
  - clear deferred backlog after release
- Checklist:
  - [ ] Re-run the local-ledger audit against the post-implementation repo.
  - [ ] Re-run official-docs MCP cross-validation for the widened surfaces.
  - [ ] Update docs, examples, and public API references.
  - [ ] Run the full cookbook and targeted provider/runtime tests.
  - [ ] Prepare release notes and final capability statement for `1.2`.
- Validation:
  - targeted tests
  - full cookbook run
  - final audit diff review
- Exit criteria:
  - `1.2` docs, tests, examples, and audits agree on the package state
- Green-light cue:
  - remaining items are explicit product decisions, not audit leftovers
- Rollback / containment:
  - if cookbook or docs parity remains unstable, defer release and publish a narrowed scope adjustment

## Milestones

### Milestone 1: `1.2` audit baseline approved
- Target outcome:
  - refreshed and credible `1.2` gap inventory
- Includes phases:
  - Phase 1
- Completion criteria:
  - all known gap families classified
  - official-docs cross-check added to the baseline
- Evidence / demo expected:
  - updated audit report
  - explicit `1.2` scope matrix
- Go / no-go note:
  - no implementation should begin until this milestone is approved

### Milestone 2: product-surface breadth gaps closed
- Target outcome:
  - hosted retrieval, Realtime, and tool-system gaps are materially reduced
- Includes phases:
  - Phases 2, 3, 4
- Completion criteria:
  - supported hosted retrieval, Realtime, and tool docs no longer obviously outpace the package
- Evidence / demo expected:
  - provider/engine APIs
  - examples
  - tests
- Go / no-go note:
  - if Realtime or hosted-tool scope balloons, stop at a documented stable boundary

### Milestone 3: agentic and metadata surfaces stabilized
- Target outcome:
  - connectors/MCP/deep-research boundary is explicit, “skills” are explicitly deferred, and model metadata reflects the supported contract
- Includes phases:
  - Phases 5 and 6
- Completion criteria:
  - docs, model catalog, and public APIs tell the same story
- Evidence / demo expected:
  - updated docs
  - tests
  - model-catalog diff
- Go / no-go note:
  - if “skills” remains ill-defined, defer it explicitly rather than forcing an abstraction

### Milestone 4: `1.2` release candidate
- Target outcome:
  - release-ready `1.2` branch with aligned docs, cookbook, and audits
- Includes phases:
  - Phase 7
- Completion criteria:
  - full validation and final capability statement complete
- Evidence / demo expected:
  - updated release notes
  - passing validation
  - final audit artifacts
- Go / no-go note:
  - release only if docs, tests, and cookbook all align

## Validation Strategy
### Automated validation
- Unit tests:
  - request translation
  - response parsing
  - model catalog
  - public namespace exports
  - new hosted-tool / Realtime / retrieval helpers
- Integration tests:
  - provider/engine workflow tests
  - example-focused smoke flows
- End-to-end tests:
  - full cookbook runner with documented expected skips only
- Static analysis / lint / type checks:
  - `py_compile`
  - targeted `pytest`
  - any repo-standard type/lint checks already in use

### Manual validation
- Manual test flows:
  - representative Realtime flow
  - representative file-search/retrieval flow
  - representative `tool_search` / namespace flow if implemented
  - representative deep-research flow
- Inspection points:
  - docs vs code parity
  - provider surface vs engine surface parity
  - examples vs public docs parity

### Acceptance thresholds
- Functional:
  - no known first-class feature left undocumented or untested
- Performance:
  - no major regression in cookbook runtime caused by new orchestration surfaces
- Reliability:
  - new surfaces degrade cleanly when optional setup is absent
- Security:
  - docs and examples do not encourage unsafe MCP / deep-research patterns
- Usability / DX:
  - new OpenAI abstractions are meaningfully better than raw dict passthrough

### Change management discipline
- Every meaningful implementation change should land as its own scoped commit.
- Branches should stay single-purpose and PR-ready.
- Docs/examples/tests should be committed with the relevant runtime change, not bundled later into unrelated cleanups.

## Rollout / Migration Strategy
- Rollout pattern:
  - audit refresh first, then bounded feature branches, then `release/1.2.0`
- Feature flags:
  - not required by default, unless a specific unstable surface needs experimental containment
- Backward compatibility:
  - additive APIs where possible
  - stricter validation only when it prevents clearly unsupported combinations
- Data migration:
  - none expected
- Rollback path:
  - revert the specific feature branch or defer that feature from the release branch
- Safe deployment notes:
  - release tags should continue to be cut only from merged `main`

## Observability / Debugging Plan
- Logs:
  - provider request/response translation traces where already supported
- Metrics:
  - not adding new metrics by default; rely on existing observability surfaces
- Traces:
  - preserve and document provider-specific traces for complex tool/research flows where useful
- Alerts:
  - not relevant at package level
- Debug hooks:
  - retain low-level provider items where useful for exact replay/debugging
- Failure signals:
  - cookbook regressions
  - docs/code drift
  - provider/runtime test failures

## Open Questions
- [ ] None currently blocking `1.2` execution. Re-open only if official docs or package direction change materially.

## Decision Log
### Decision 1
- Decision:
  - Use the local docs-ledger API as the inventory source and the official OpenAI docs MCP as the final validator.
- Why:
  - this combines exhaustive traversal with authoritative current guidance
- Alternatives rejected:
  - ledger only
  - official MCP only
- Implications:
  - the audit workflow is slightly heavier, but much more trustworthy

### Decision 2
- Decision:
  - Treat `1.2` as a bounded breadth-and-hardening release, not a “clone the full OpenAI platform” release.
- Why:
  - it keeps the package maintainable and aligned with its architecture
- Alternatives rejected:
  - full product-surface parity in one cycle
- Implications:
  - some official docs surfaces will still be explicitly deferred after `1.2`

### Decision 3
- Decision:
  - Implement `tool_search` as an advanced, OpenAI-specific surface in `1.2`.
- Why:
  - it is a concrete official-docs gap, but it does not yet justify widening the stable generic tool model.
- Alternatives rejected:
  - stable generic tool-system support for `tool_search`
- Implications:
  - package docs must clearly mark it as provider-specific.

### Decision 4
- Decision:
  - Implement tool namespaces as an OpenAI-specific capability in `1.2`.
- Why:
  - the official function-calling docs justify support, but there is no evidence yet that a provider-agnostic namespace abstraction is warranted.
- Alternatives rejected:
  - generalizing namespaces across the package tool model immediately
- Implications:
  - validation and docs must keep the scope explicitly OpenAI-specific.

### Decision 5
- Decision:
  - Defer “skills” as a first-class package abstraction from `1.2`.
- Why:
  - connectors and MCP add concrete runtime value; “skills” remains too product-specific and underspecified for a clean package abstraction.
- Alternatives rejected:
  - introducing a weak or speculative skills layer in `1.2`
- Implications:
  - the defer should be explicit in audits and release notes.

### Decision 6
- Decision:
  - Expand the model catalog only for supported product-family models in `1.2`.
- Why:
  - this keeps the registry truthful, useful, and maintainable.
- Alternatives rejected:
  - full docs-corpus model coverage
  - purely “most current practical” coverage without a support-policy anchor
- Implications:
  - some docs-listed models will remain intentionally absent until their product families are supported.

### Decision 7
- Decision:
  - Exclude Sora/video and other high-churn low-priority surfaces from `1.2`.
- Why:
  - they add substantial scope and churn without matching the highest-value remaining package gaps.
- Alternatives rejected:
  - folding video/Sora into the already broad `1.2` release
- Implications:
  - `1.2` stays focused on retrieval, Realtime, tools, connectors/MCP, and metadata parity.

## Implementation Log
| Date | Phase / Step | What was done | Files / Modules touched | Validation run | Outcome | Notes / blockers |
|---|---|---|---|---|---|---|
| 2026-04-08 | Planning | Drafted the `1.2` comprehensive implementation plan from current repo audits plus official-docs cross-validation. | `plans/2026-04-08-openai-v1-2-implementation-plan.md` | Inspected audits, provider surface, tool surface, official docs MCP fetches | Complete | Official-docs cross-check surfaced additional likely gap categories: `tool_search` and namespaced tools. |
| 2026-04-08 | Planning | Locked `1.2` strategic scope decisions with the user, including OpenAI-specific `tool_search` and namespaces, skills defer, supported-family model coverage, and Sora/video exclusion. | `plans/2026-04-08-openai-v1-2-implementation-plan.md` | Plan review against repo audits and user decisions | Complete | Plan is now implementation-ready. |

## Final Approval Gate
- [x] Scope is clear
- [x] Critical unknowns are resolved
- [x] Plan is evidence-based
- [x] Phases are actionable
- [x] Validation is defined
- [x] Rollout / rollback is defined where relevant
- [x] Parallelizable work is clearly marked
- [x] Ready for implementation
