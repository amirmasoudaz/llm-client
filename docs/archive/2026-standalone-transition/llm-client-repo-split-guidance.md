# llm-client Repository Split Guidance

This document describes what should stay in the repository when the codebase is
turned into a standalone `llm_client` package repository, and what should move
out with the application/runtime layers.

The goal is simple:

- keep the package and everything needed to install, test, document, and
  release it
- move out product/runtime/application code that only consumes the package

## Keep In The `llm_client` Repository

These are package-owned or package-adjacent and should remain:

- `llm_client/`
- `examples/`
- `tests/llm_client/`
- `docs/llm-client-*.md`
- `docs/CHANGELOG.md` only if it is replaced or repurposed for package release
  history; otherwise create a package-specific changelog and move the current
  API changelog out
- `pyproject.toml`
- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `CODE_OF_CONDUCT.md`
- `.gitignore`
- `.dockerignore`
- `.github/workflows/llm-client-package-ci.yml`
- `.github/workflows/llm-client-publish.yml`
- `scripts/ci/run_llm_client_examples.py`
- `scripts/ci/run_llm_client_rc_benchmarks.py`
- `scripts/ci/verify_llm_client_artifacts.py`
- `docker-compose.llm-client-services.yml`
- `contracts/benchmarks/llm_client_deterministic_baseline.v1.json`

Keep only package-relevant top-level metadata and automation.

## Move Out Of The `llm_client` Repository

These belong to the higher-level product/runtime layers and should move to
their own repositories or to a separate monorepo:

- `agent_runtime/`
- `intelligence_layer/`
- `scenarios/`
- `tests/api/`
- `tests/kernel/`
- `tests/scenarios/`
- `tests/bench/`
- `tests/benchmarks/` unless a benchmark module is strictly package-owned
- `scripts/bench/`
- `scripts/deploy/`
- `scripts/refactor/`
- `.github/workflows/deploy.yml`
- `.github/workflows/docker-publish.yml`
- `docker-compose.yml`
- `docker-compose-staging.yml`
- `docker-compose-prod.yml`
- `Dockerfile` if it is for the application stack rather than package-specific
  docs/test automation
- `flow_script.py`
- `docs/ARCHITECTURE-MULTI-TURN.md`
- `docs/FRONTEND-INTEGRATION.md`
- `docs/TEST-CURLS.md`
- `docs/TEST-CURLS.postman_collection.json`
- `docs/TEST-CURLS.postman_environment.json`
- `docs/CODEBASE-REFACTOR-V-0.4.md`
- `docs/system-inventory-v2026-02-24/`

These files describe or support the application layers, not the reusable
package.

## Review Before Deciding

These paths need a judgment call during the split:

- `requirements.txt`
  Keep only if it is repurposed for development convenience in the standalone
  package repo. `pyproject.toml` should remain the canonical dependency source.
- `contracts/`
  Keep only package-owned benchmark contracts or other package-owned schemas.
  Move application/operator/workflow contracts out.
- `docs/CHANGELOG.md`
  The current file appears to describe API/application changes rather than the
  standalone package. Either move it out or replace it with a package
  changelog.
- `tests/__init__.py`
  Keep if the standalone test layout still needs it.

## Generated Or Local-Only Paths

Do not move these into the standalone package repository as committed release
content:

- `dist/`
- `build/`
- `artifacts/`
- `.coverage`
- `.pytest_cache/`
- `.venv/`
- `tmp/`
- `llm_client.egg-info/`
- editor folders such as `.idea/`
- local env files such as `.env`, `.env.staging`

These should remain ignored or local-only.

## Package-Owned Test Scope After The Split

The standalone package repo should keep:

- `tests/llm_client/`

It may also keep a very small number of integration-smoke tests that exercise
installed-package behavior, but only if they do not depend on
`agent_runtime` or `intelligence_layer`.

## Package-Owned Documentation Scope After The Split

The standalone package repo should keep:

- package API/reference docs
- build/how-to guides
- provider/tool/context/cache/observability/adaptor guides
- examples guide and cookbook docs
- packaging, release, semver, support, and threat-model docs
- release notes

It should not keep:

- application API docs
- frontend integration docs
- product workflow/system inventory docs

## Recommended Split Order

1. Copy package-owned paths into the new standalone repository.
2. Move package CI and publish workflows with them.
3. Re-run package tests, cookbook validation, benchmark validation, and
   artifact verification in the new repo.
4. Only after the standalone repo is validated, create the real `1.0.0` git
   tag there.

That last point matters: the durable release tag should live on the standalone
package repository history, not on the old monorepo history if the package is
about to be extracted.
