# llm-client Support Policy

This document defines the support expectations for the standalone `llm-client`
package.

## Supported Python Versions

- Python `3.10`
- Python `3.11`
- Python `3.12`

## Supported Operating Environments

The package is designed primarily for Linux-based server and CI environments.
Other environments may work, but CI validation focuses on Linux runners.

## Support Tiers

### Stable Namespace Support

The stable namespaces defined in
[llm-client-public-api-v1.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-public-api-v1.md)
are the primary support commitment and the frozen `1.x` package contract.

### Compatibility Namespace Support

Compatibility surfaces are supported for migration, but may emit warnings and
should not be the basis for new integrations.

Current transitional compatibility surfaces include:

- `llm_client.compat`
- top-level convenience aliases retained for migration
- `llm_client.container` helpers when used as migration/integration shims

### Advanced Namespace Support

Advanced surfaces are available for expert use, but are not guaranteed to have
the same long-term stability as the stable namespace set.

## Provider Support Posture

- OpenAI path: baseline production path
- Anthropic path: supported optional provider
- Google path: supported optional provider

Provider parity is tested where the abstractions overlap, but vendor-specific
behavior still differs by capability and API shape.
