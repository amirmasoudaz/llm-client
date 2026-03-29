# llm-client Service Adaptors Design

This document defines the package-level design for service adaptors that are
now in scope for the `1.0.0` program.

## Goal

Provide reusable, generic integrations for external services that agents and
tools commonly need, without pushing business-domain queries or product
workflow logic into the package core.

## Boundary

`llm_client` should own:

- normalized service adaptor interfaces
- backend-specific adaptor implementations behind optional extras
- generic execution helpers
- tool-construction helpers for wrapping those adaptors as agent tools
- safety defaults, observability, retry, timeout, and budget hooks

`llm_client` should not own:

- domain-specific SQL queries
- product-owned cache key semantics
- tenant/business authorization logic
- business workflows composed on top of those services

## Namespace Decision

Public package namespace:

- `llm_client.adapters`

Reasoning:

- `adapters` describes the stable package contract better than `drivers`
- the public concern is normalized integration, not raw backend clients
- lower-level backend drivers can remain internal or advanced implementation
  detail

Planned submodules:

- `llm_client.adapters.sql`
- `llm_client.adapters.redis`
- `llm_client.adapters.vector`
- `llm_client.adapters.tools`

## Initial Backend Scope

The `1.0.0` adaptor program starts with optional integrations for:

- PostgreSQL
- MySQL
- Redis
- Qdrant

Current implementation status:

- PostgreSQL: first concrete adaptor shipped
- MySQL: first concrete adaptor shipped
- Redis: first concrete adaptor shipped
- Qdrant: first concrete adaptor shipped

Each backend should be installed by an extra, not as a mandatory core
dependency.

## Capability Model

### SQL adaptors

Generic operations:

- read/query execution
- parameterized statement execution
- explicit write execution when enabled
- schema inspection helpers where safe

Safety defaults:

- read-only by default
- writes require explicit opt-in
- parameter binding required
- statement timeout support required

### Redis adaptors

Generic operations:

- get/set/delete
- hash operations
- key expiration
- optional pub/sub helpers if added later

Safety defaults:

- bounded payload sizes
- explicit TTL behavior
- key-prefix controls

### Vector adaptors

Generic operations:

- collection/index ensure
- point/document upsert
- search/query
- delete

Safety defaults:

- explicit collection configuration
- bounded result counts
- opt-in destructive operations

## Tool-Wrapping Helpers

The package should provide generic helpers for turning adaptors into tools
without making the adaptor modules themselves business-specific.

Examples:

- read-only SQL query tool
- write-enabled SQL tool with explicit approval gate
- Redis lookup tool
- Qdrant search tool

These helpers should live under:

- `llm_client.adapters.tools`

## Runtime Integration Requirements

Adaptor operations should integrate with package runtime concerns:

- `ExecutionContext`
- `llm_client.budgets`
- tool middleware
- observability hooks and runtime events
- retry/timeout behavior where appropriate

Current status:

- shared adaptor runtime now carries execution context, event-bus publishing,
  ledger integration, timeout normalization, and retry behavior
- SQL, Redis, and Qdrant implementations use the shared runtime path

## Packaging Model

Planned extras:

- `postgres`
- `mysql`
- `redis`
- `qdrant`

Possible combined extra:

- `adapters`

## 1.0 Delivery Rule

The adaptor program can ship in `1.0.0` only if:

- the namespace is explicit
- installs remain optional by extras
- safety defaults are documented
- the public contracts are generic rather than product-specific
- cookbook examples show the intended usage without embedding business logic
