# llm-client Usage and Capabilities Guide

This guide explains how to use the package by capability rather than by module
inventory.

See also:

- [llm-client-package-api-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-package-api-guide.md)
- [llm-client-build-and-recipes-guide.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-build-and-recipes-guide.md)
- [llm_client/README.md](/home/namiral/Projects/Packages/intelligence-layer-bif/llm_client/README.md)
- [llm-client-guides-index.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-guides-index.md)

## Installation and Configuration

Minimum:

```bash
pip install -e .
```

Optional providers and integrations are installed by extras. The exact matrix
is documented in
[llm-client-installation-matrix.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-installation-matrix.md).

Environment loading is explicit:

```python
from llm_client.config import load_env

load_env()
```

## Capability Guide

### Direct provider completions

Use:

- `llm_client.providers`

Best for:

- low-level control
- provider-specific workflows
- simple direct execution

Avoid when:

- you want retry, cache, failover, or hooks across providers

### Engine-managed execution

Use:

- `llm_client.engine`

Best for:

- production execution paths
- retry/failover/cache/idempotency
- higher-level application runtimes

Preferred default:

- normalize to `ExecutionEngine` as early as possible in application code

### Structured outputs

Use:

- `llm_client.structured`

Best for:

- extraction
- schema-constrained generation
- repair loops and diagnostics

Provider note:

- provider capabilities still differ; the structured runtime normalizes the
  surface but cannot erase vendor differences completely

### Tools and tool execution

Use:

- `llm_client.tools`

Best for:

- exposing callable runtime capabilities to agents
- middleware such as timeout, logging, or output policy
- single, sequential, and parallel tool execution

Avoid:

- putting business-specific platform policy in the generic tool runtime

### Agents

Use:

- `llm_client.agent`

Best for:

- multi-turn conversation
- tool-calling assistants
- application-shaped LLM runtimes that still need generic internals

### Context, memory, and summaries

Use:

- `llm_client.context`
- `llm_client.context_assembly`
- `llm_client.memory`

Best for:

- generic execution envelope and request governance
- assembling neutral context from multiple sources
- memory retrieval and persistent summary abstractions

Avoid:

- domain entity loading inside the package core

### Budgeting and usage tracking

Use:

- `llm_client.budgets`

Best for:

- generic usage ledgering
- execution budget checks
- tool/provider usage accounting around a generic runtime

Avoid:

- treating it as a billing or tenancy policy system

### Observability and replay

Use:

- `llm_client.observability`

Best for:

- runtime events
- diagnostics
- hooks
- replay
- telemetry

### Caching

Use:

- `llm_client.cache`

Best for:

- completion caching
- embedding caching
- storage-agnostic cache policy handling

### Benchmarks

Use:

- `llm_client.benchmarks`

Best for:

- deterministic regression baselines
- benchmark comparisons
- release candidate validation

### Service adaptors

Use:

- `llm_client.adapters`
- `llm_client.adapters.tools`

Best for:

- normalized SQL, Redis, or vector service access
- exposing controlled service operations as tools
- keeping generic connectivity and safety rules inside the package

Concrete backends now available:

- `PostgresSQLAdaptor`
- `MySQLSQLAdaptor`
- `RedisKVAdaptor`
- `QdrantVectorAdaptor`

Avoid:

- putting business-specific queries or authorization semantics into the
  package-owned adaptor layer

## Recommended Package Paths By Use Case

### Small service that just needs completions

Start with:

- `llm_client.providers`
- `llm_client.types`

### Production backend with retries, cache, and observability

Start with:

- `llm_client.engine`
- `llm_client.providers`
- `llm_client.observability`

### Tool-calling assistant

Start with:

- `llm_client.agent`
- `llm_client.engine`
- `llm_client.tools`

### Memory-backed copilot

Start with:

- `llm_client.agent`
- `llm_client.context_assembly`
- `llm_client.memory`

### LLM runtime with cost governance

Start with:

- `llm_client.context`
- `llm_client.budgets`
- `llm_client.engine`

### Agent with controlled database access

Start with:

- `llm_client.adapters`
- `llm_client.adapters.tools`
- `llm_client.agent`

## Use / Avoid Summary

Use:

- stable namespaces for package integrations
- `ExecutionEngine` for higher-level runtime paths
- `compat` only for migration
- package guides plus cookbook examples together

Avoid:

- importing internals as if they were public
- anchoring new integrations on migration-era compatibility stories or
  package-external wrapper layers
- assuming provider parity where vendor capabilities differ
- treating examples as proof that every application pattern belongs in the
  core package
