# Security Policy

## Supported Release Line

Security hardening and fixes are expected to focus on the active `1.x` line
once `1.0.0` is released. Pre-`1.0` versions may still receive fixes, but that
should not be treated as a long-term guarantee.

## Reporting a Vulnerability

Please do not publish detailed exploit information in a public issue first.

Instead:

1. Prepare a concise report describing the issue, impact, and affected areas.
2. Include reproduction details only as needed for maintainers to verify it.
3. Share the report privately with the package maintainers through the
   repository’s preferred private disclosure channel.

If no dedicated security contact is configured yet, treat the issue as
maintainer-private until a formal disclosure channel is published.

## Scope

Security-sensitive areas for this package include:

- provider credential handling
- logging and redaction behavior
- cache isolation and cache key correctness
- multi-tenant safety assumptions in higher-level integrations
- tool execution and tool output handling
- replay/event persistence containing potentially sensitive data

## Expectations for Contributors

- avoid introducing secret-bearing fixtures into committed test data
- prefer sanitized payload examples in docs and examples
- document any change that affects redaction, replay, persistence, or caching
- raise concerns early if a new feature weakens tenant or data isolation

