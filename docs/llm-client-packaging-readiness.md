# llm-client Packaging Readiness Review

This document records the current packaging and import-side-effect posture of
the standalone `llm-client` package.

## Dependency Footprint Summary

### Base Dependencies

The base install currently includes:

- `openai`
- `numpy`
- `httpx`
- `aiohttp`
- `blake3`
- `jsonschema`
- `pydantic`
- `docstring_parser`
- `python-dotenv`
- `PyYAML`
- `tiktoken`
- `typing_extensions`

This is broader than a minimal provider-only client because the current stable
package boundary intentionally includes:

- provider execution
- cache namespace
- content and structured output runtime
- observability and benchmarking support

### Optional Extras

Optional extras currently cover:

- Anthropic provider support
- Google provider support
- PostgreSQL-backed cache/persistence support
- Redis-backed hot-cache support
- telemetry integrations
- performance helpers
- server examples
- development tooling

## Import Side Effects Review

### Accepted Base Import Behavior

The package currently treats these as accepted for a standalone install:

- importing `llm_client` exposes the stable public runtime surface
- importing `llm_client.cache` exposes the stable cache namespace
- importing `llm_client.providers` exposes the provider registry and provider
  implementations

### Notable Consequences

- `llm_client.cache` remains a stable namespace, but the `pg_redis` backend is
  now isolated behind lazy imports so PostgreSQL and Redis dependencies are not
  required for a base install
- OpenAI support remains part of the base install because it is the baseline
  provider path and the current compatibility layer depends on it
- Anthropic and Google remain optional because their adapters are present but
  their SDKs are not required for the base runtime

## Packaging Position

For `1.0.x`, the current packaging posture is considered acceptable because it
matches the declared public API surface. Future optimization work may reduce the
base dependency footprint by further isolating optional backends behind
lazy-loading boundaries, but that is not required for this release-readiness
phase.

## Release Readiness Conclusion

The package is considered packaging-ready when:

- the `pyproject.toml` metadata is valid
- wheel and sdist artifacts are built in CI
- live cookbook entrypoints are validated in CI with credential-less skip allowance
- standalone installation is smoke-tested across supported Python versions
- publish automation exists for tagged and manual releases
