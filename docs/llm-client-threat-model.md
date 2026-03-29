# llm-client Threat Model

This note captures the security model for `llm_client` as a reusable LLM and
agent runtime package.

## Main Threats

- prompt injection through user input, retrieved context, tool results, or
  provider outputs
- tool misuse through unsafe arguments, undeclared tools, or over-trusting tool
  output
- sensitive data disclosure through logs, diagnostics, cached payloads, raw
  provider responses, or tool output reflection
- replay of unsafe or stale cached responses across tenants or incompatible
  request shapes
- provider payload leakage through debug or error capture surfaces

## Security Stance

`llm_client` should provide:

- safe-by-default observability redaction
- explicit field classification: safe, sensitive, forbidden
- explicit tool-output hardening hooks
- versioned cache keys with tenant and shape isolation
- storage-agnostic primitives rather than product/business persistence

`llm_client` should not claim to solve:

- business authorization policy
- domain-specific prompt injection defenses
- product-specific approval workflows
- secure secret storage or deployment infrastructure

## Prompt Injection Model

The package assumes that any of the following can contain adversarial
instructions:

- user messages
- retrieved memory/context
- tool outputs
- documents/files passed into prompts
- model outputs that are later fed back into subsequent turns

Implications:

- system/developer instructions must remain structurally separate from
  untrusted content
- tool selection and execution must not rely on assistant text alone
- tool outputs should be treated as untrusted input unless explicitly verified
- observability and replay should not store raw sensitive prompt material by
  default

## Tool Misuse Model

Tool calls are high-risk because they bridge the model to side effects and
trusted data.

Controls expected at or above `llm_client`:

- allowlists/denylists and tool policy checks
- argument validation
- explicit execution modes
- execution budgets and timeouts
- output hardening before logging or reinjection into model context

`llm_client` provides generic mechanisms for these controls, but product teams
must still define domain-specific policies.

## Sensitive Data Model

Data classes relevant to this package:

- safe:
  - provider name
  - model name
  - status
  - latency
  - token counts
- sensitive:
  - secrets, tokens, credentials
  - end-user prompt content
  - tool output that may contain PII or secrets
- forbidden by default:
  - raw provider request payloads
  - raw provider response payloads
  - SDK-native request/response bodies

The default package stance is:

- sanitize sensitive fields
- omit forbidden fields unless explicitly captured under a safer policy
- prefer metadata-only capture for provider payload debugging

## Cache Threats

Main cache risks:

- cross-tenant leakage
- stale responses reused after request shape changes
- unsafe caching of error or moderation/content-filter payloads
- accidental persistence of sensitive raw payloads

Package response:

- request-shape versioned cache keys
- tenant-aware key material
- explicit invalidation policy
- cache-backed metadata/summary stores separated from ledgers/billing state

## Boundary Rules

Remain outside core package responsibility:

- tenant ledgers
- billing persistence
- secret managers
- key rotation
- deployment-time network perimeter controls
- product approval workflows
