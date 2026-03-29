# llm-client 1.0.0-rc1 Release Notes

Release date: 2026-03-26

`1.0.0-rc1` is the first release candidate for `llm_client` as a standalone,
typed, monorepo-hosted package. This release freezes the intended `1.x`
package boundary and validates the package through focused tests, cookbook
execution, packaging verification, deterministic benchmarks, and live provider
smoke tests.

## What This Release Candidate Represents

- the `llm_client` package is now the canonical reusable kernel for:
  - provider integrations
  - content and request/response normalization
  - engine-based execution
  - agent and tool execution
  - context assembly and budget tracking
  - observability, validation, caching, and replay substrate
- `agent_runtime` and `intelligence_layer` now consume that substrate instead of
  shadow-owning most generic mechanisms
- the package is documented and installable as a standalone distribution from
  this monorepo

## Major Changes Included In The 1.0 Program

- engine-first execution became the canonical path across higher-level
  consumers
- structured tool-loop runtime and generic context-management substrate were
  extracted into `llm_client`
- budget and usage/ledger substrate moved into `llm_client`
- content envelopes and `FileBlock` transport semantics were finalized
- the stable API surface, compatibility layers, and advanced surfaces were
  documented explicitly
- standalone package guides were added:
  - package API guide
  - usage and capabilities guide
  - examples guide
- package metadata and OSS hygiene were completed:
  - `py.typed`
  - `LICENSE`
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`
- service adaptors were added under the stable `llm_client.adapters` namespace:
  - SQL
  - Redis
  - vector/Qdrant
- cookbook examples were expanded and aligned with the runner and inventory
  tests

## RC Validation Performed

- focused package release suites passed
- packaging and artifact verification passed against rebuilt wheel and sdist
- full cookbook runner completed locally in credential-cleared skip mode
- deterministic benchmark artifacts were generated:
  - `artifacts/benchmarks/llm_client_rc_deterministic.json`
  - `artifacts/benchmarks/llm_client_rc_deterministic_comparison.json`
- benchmark strategy and structured benchmarking test suites passed locally
- live provider smoke tests passed for:
  - OpenAI
  - Anthropic

Notes:

- the live smoke environment used the repository `.env` credentials
- Google live smoke was not part of the validated RC set in this environment
  because a corresponding live key was not enabled

## Late RC Fixes Captured During Validation

- Anthropic defaults and live smoke assumptions were updated to current Claude 4
  naming by making `claude-sonnet-4` the canonical default while preserving the
  legacy `claude-4-5-*` keys as compatibility aliases
- the OpenAI GPT-5 Mini live smoke was updated to avoid a false failure caused
  by a too-small completion budget on a reasoning-capable model
- SQL adaptor cookbook examples now skip cleanly when local database services
  are unavailable instead of hard-failing the cookbook run
- a reusable service stack compose file was added at
  `docker-compose.llm-client-services.yml`

## Known Non-GA Items

`1.0.0-rc1` is not `1.0.0`.

The remaining GA work is intentionally smaller and narrower:

- resolve any RC feedback
- freeze `1.x` policy and compatibility-only surfaces for GA
- finalize `1.0.0` release notes
- tag and publish `1.0.0`
