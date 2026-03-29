# Contributing to llm-client

Thanks for contributing.

This repository currently hosts the standalone `llm_client` package alongside
other application layers that consume it. Contributions that touch
`llm_client` should preserve the package boundary and avoid leaking
application-specific logic into the package.

## Ground Rules

1. Keep `llm_client` generic.
   Business policies, product workflows, domain prompts, and app-specific
   orchestration belong outside the package.

2. Prefer stable namespaces for public package changes.
   If a new API is intended for long-term users, place it in a documented
   stable namespace. If it is lower-level or specialized, prefer
   `llm_client.advanced` or keep it internal until it is ready.

3. Do not expand compatibility surfaces casually.
   Compatibility modules exist to support migration. New features should not be
   added there by default.

4. Keep docs, tests, and examples in sync with code.
   For package-facing changes, update the relevant docs and add or adjust
   focused coverage.

## Development Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,anthropic,google,telemetry,performance,server]"
```

Optional infrastructure-heavy examples may also need PostgreSQL, Redis, or
Qdrant running locally.

## Recommended Workflow

1. Make a focused change.
2. Update package docs if the public contract changes.
3. Update examples if the recommended usage changes.
4. Run the most relevant test slice for the change.
5. If packaging behavior changes, run artifact verification too.

## Contribution Checklist

- public API changes are reflected in `docs/llm-client-public-api-v1.md`
- stable vs compatibility vs advanced status is still clear
- example and README usage stays accurate
- tests cover the changed behavior
- packaging metadata or artifact expectations stay valid

## What to Avoid

- repo-specific business logic in `llm_client`
- adding public exports without documenting their intended status
- widening legacy compatibility APIs instead of improving stable ones
- changing stable semantics silently

## Reporting Problems

For security-sensitive issues, use the process in [SECURITY.md](./SECURITY.md)
instead of opening a public issue with exploit details.

