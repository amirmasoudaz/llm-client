# llm-client 1.0.0 Release Notes

Release date: 2026-03-26

`llm_client` `1.0.0` is the first stable release of the package as a
standalone, typed, reusable LLM and agentic runtime framework.

This release freezes the `1.x` public package contract defined in:

- [llm-client-public-api-v1.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-public-api-v1.md)

## What 1.0.0 Means

`1.0.0` means:

- the stable namespace map is no longer a draft
- new integrations should be built against the stable module namespaces
- compatibility layers remain available for migration, but they are not the
  preferred package surface
- future `1.x` releases should preserve backward compatibility for the stable
  package contract

## Stable Namespace Contract

The stable `1.x` surface is:

- `llm_client.providers`
- `llm_client.models`
- `llm_client.types`
- `llm_client.content`
- `llm_client.context`
- `llm_client.budgets`
- `llm_client.context_assembly`
- `llm_client.engine`
- `llm_client.agent`
- `llm_client.benchmarks`
- `llm_client.tools`
- `llm_client.adapters`
- `llm_client.cache`
- `llm_client.memory`
- `llm_client.observability`
- `llm_client.validation`
- `llm_client.errors`
- `llm_client.config`

Compatibility-only or advanced surfaces remain available, but they are outside
the primary `1.x` promise.

## What Shipped In The 1.0 Program

- provider, engine, agent, tool, cache, context, observability, and structured
  output layers were tightened into a standalone package boundary
- engine-first execution became the canonical higher-level runtime path
- generic runtime substrate extracted from higher layers now lives in the
  package:
  - context
  - budget and ledger primitives
  - runtime events
  - replay
  - structured runtime helpers
  - generic context-planning and assembly primitives
- `FileBlock` became a real transport feature
- stable service adaptors were added under `llm_client.adapters`:
  - SQL
  - Redis
  - vector/Qdrant
- standalone package guides, examples guide, and cookbook alignment work were
  completed
- OSS/package hygiene and packaging verification were completed

## Validation Completed For 1.0.0

- focused package suites passed
- guide, packaging, and public API inventory suites passed
- cookbook contract validation passed
- deterministic benchmark artifacts were generated and compared
- live provider smoke tests passed for OpenAI and Anthropic
- wheel and sdist verification passed
- `twine check` passed for the final `1.0.0` distributions

## Final Adjustments Between RC1 And GA

- Anthropic defaults were aligned to current Claude 4 naming while preserving
  legacy `claude-4-5-*` compatibility keys
- the GPT-5 Mini live smoke path was corrected to avoid false failures caused
  by a too-small completion budget
- a practical build guide was added:
  [llm-client-build-and-recipes-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-build-and-recipes-guide.md)
- the public API map was promoted from draft to frozen `1.x` contract
- support and semver docs were updated to reflect the `1.x` freeze explicitly

## Documentation Starting Points

If you are adopting the package in another project, start with:

1. [llm-client-package-api-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-package-api-guide.md)
2. [llm-client-build-and-recipes-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-build-and-recipes-guide.md)
3. [llm-client-usage-and-capabilities-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-usage-and-capabilities-guide.md)
4. [llm-client-examples-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-examples-guide.md)
5. [examples/README.md](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/README.md)

## Compatibility Note

The package still contains compatibility layers for migration, including
`llm_client.compat` and some top-level convenience aliases. Those remain valid
for migration, but new code should target the stable module namespaces above.
