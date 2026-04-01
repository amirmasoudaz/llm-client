# llm-client Semantic Versioning Policy

`llm-client` uses semantic versioning for the standalone package contract.

As of `1.0.0`, the stable namespace map is frozen for the `1.x` line.

## Version Meaning

- `MAJOR`: incompatible changes to the stable public API or stable behavior
- `MINOR`: backward-compatible feature additions and meaningful capability
  expansions
- `PATCH`: backward-compatible fixes, reliability improvements, documentation
  updates, and benchmark/test improvements

## Stable API Scope

The stable API scope is defined in:

- [llm-client-public-api-v1.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-public-api-v1.md)

Breaking changes are evaluated against that stable API map, not against
internal helper modules or compatibility surfaces.

For the `1.x` line, new stable surface should be added deliberately and
sparingly. Compatibility layers may remain, but they should not expand in a
way that obscures the canonical module map.

## Deprecation Policy

- Deprecations must be documented in release notes.
- Compatibility surfaces may emit `DeprecationWarning`.
- Removal of a deprecated stable API requires a major version bump.

## Experimental and Advanced Surfaces

Modules explicitly labeled `advanced`, `compat`, or internal are not held to
the same stability guarantees as the stable namespace set, but significant
behavioral changes should still be documented in release notes.
