# llm-client Package API Guide

This guide explains the standalone package contract for `llm_client` as a
package consumer would see it.

Use this document when you need to answer:

- which module to import from
- what the canonical input and output shapes are
- how providers, engine, agent, tools, context, and observability fit together
- which entry point to choose for a specific integration

See also:

- [llm-client-public-api-v1.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-public-api-v1.md)
- [llm_client/README.md](/home/namiral/Projects/Packages/llm-client-v1/llm_client/README.md)
- [llm-client-build-and-recipes-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-build-and-recipes-guide.md)
- [llm-client-usage-and-capabilities-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-usage-and-capabilities-guide.md)

## Stable Module Map

The stable package surface is:

- `llm_client.providers`
- `llm_client.models`
- `llm_client.types`
- `llm_client.content`
- `llm_client.context`
- `llm_client.budgets`
- `llm_client.context_assembly`
- `llm_client.engine`
- `llm_client.agent`
- `llm_client.benchmarks`
- `llm_client.tools`
- `llm_client.adapters`
- `llm_client.cache`
- `llm_client.memory`
- `llm_client.observability`
- `llm_client.validation`
- `llm_client.errors`
- `llm_client.config`

Treat `llm_client.compat` as compatibility-only and `llm_client.advanced` as
specialized lower-level surface, not as the default entry point for new work.

## Canonical Input and Output Shapes

### Provider-Level Inputs

Use `llm_client.types` and `llm_client.content` for canonical request data:

- `Message`
- `RequestContext`
- `RequestSpec`
- typed content blocks such as text, image, tool-call, tool-result, and file
  blocks

Use provider modules when you need to execute a direct provider call:

- `OpenAIProvider`
- `AnthropicProvider`
- `GoogleProvider`

Provider-level outputs are:

- `CompletionResult`
- `NormalizedOutputItem`
- `BackgroundResponseResult`
- `ConversationResource`
- `CompactionResult`
- `DeletionResult`
- `ConversationItemResource`
- `ConversationItemsPage`
- `EmbeddingResult`
- `ModerationResult`
- `ImageGenerationResult`
- `GeneratedImage`
- `AudioTranscriptionResult`
- `AudioSpeechResult`
- `FileResource`
- `FilesPage`
- `FileContentResult`
- `VectorStoreResource`
- `VectorStoresPage`
- `VectorStoreSearchResult`
- `VectorStoreFileResource`
- `VectorStoreFilesPage`
- `VectorStoreFileContentResult`
- `FineTuningJobResult`
- `FineTuningJobsPage`
- `FineTuningJobEventsPage`
- `RealtimeClientSecretResult`
- `RealtimeCallResult`
- `RealtimeConnection`
- `WebhookEventResult`
- `StreamEvent`
- `Usage`

### Engine-Level Inputs

Use `llm_client.engine` when you want:

- retries
- failover/routing
- cache handling
- idempotency-aware execution
- lifecycle hooks and diagnostics
- provider-native workflow orchestration around background/conversation/state APIs

Canonical execution context above request correlation lives in
`llm_client.context`:

- `ExecutionContext`
- `BudgetSpec`
- `PolicyRef`
- `RunVersions`

### Budget and Usage Inputs

Use `llm_client.budgets` when you need:

- `Ledger`
- `Budget`
- `UsageRecord`
- `LedgerWriter`
- budget enforcement or usage aggregation around a generic execution runtime

This layer is generic. Product billing rules and tenant-specific quota
policies should remain outside the package.

## Relationship Between The Main Layers

### Providers

Providers are the vendor-facing execution layer.

Use them when:

- you want direct control over one provider
- you do not need engine-level retry/cache/failover behavior
- you are building a lower-level integration
- you need provider-native workflows such as OpenAI Responses background retrieval or resumed streaming
- you need OpenAI-first product surfaces such as moderation, image generation/editing, speech APIs, generic file uploads/content retrieval, hosted vector stores, vector-store files and file batches, fine-tuning jobs, realtime connection/call/transcription helpers, hosted Responses tool workflows, webhook verification, or staged deep-research orchestration

Use `llm_client.engine` instead of direct provider calls when you want those
provider-native workflows but still need a shared engine surface for hook
correlation, timeout handling, or provider selection.

### Engine

The engine is the preferred execution layer for higher-level application code.

Use it when:

- you want stable request execution behavior
- you want caching, retry, failover, hooks, or idempotency
- you want higher-level flows to be provider-agnostic
- you want one orchestration surface for provider-native workflows beyond `complete(...)`, including moderation, media APIs, generic file APIs, vector stores, vector-store files and file batches, fine-tuning, realtime connection/call/transcription helpers, hosted Responses tool workflows, webhook verification, and staged deep-research orchestration

### Agent

The agent layer builds on the engine/provider layer and the tool system.

Use it when:

- you need a multi-turn loop
- you need tool calling
- you need a structured conversation runtime rather than one-shot execution

### Tools

The tool layer defines callable runtime capabilities:

- `Tool`
- `ToolRegistry`
- `ResponsesBuiltinTool`
- `ResponsesAttributeFilter`
- `ResponsesFileSearchHybridWeights`
- `ResponsesFileSearchRankingOptions`
- `ResponsesToolSearch`
- `ResponsesFunctionTool`
- `ResponsesToolNamespace`
- `ResponsesConnectorId`
- `ResponsesMCPTool`
- `ResponsesMCPApprovalPolicy`
- `ResponsesMCPToolFilter`
- `ResponsesCustomTool`
- `ResponsesGrammar`
- `tool`
- `ToolExecutionEngine`

The package owns the generic tool runtime. Domain-specific tools still belong
outside the package.

For OpenAI Responses-native hosted tools, prefer the typed descriptors instead
of raw dicts. That includes convenience aliases such as
`ResponsesBuiltinTool.web_search_preview(...)`,
`ResponsesBuiltinTool.remote_mcp(...)`, and
`ResponsesBuiltinTool.connector(...)`, plus the richer MCP/connector-specific
descriptor `ResponsesMCPTool`. When you want docs-aligned connector ids without
hand-typed strings, use `ResponsesConnectorId`.

For advanced OpenAI-specific deferred-tool workflows, use:

- `ResponsesToolSearch` for hosted or client-executed `tool_search`
- `ResponsesFunctionTool` when a function needs provider metadata such as
  `defer_loading=True`
- `ResponsesToolNamespace` to group deferred tools under a namespace like
  `crm` or `billing`
- `ResponsesAttributeFilter`, `ResponsesFileSearchRankingOptions`, and
  `ResponsesFileSearchHybridWeights` when hosted retrieval or file-search needs
  typed filtering and ranking controls instead of raw provider dicts

Provider-native Responses tools are request-side descriptors, not executable
runtime tools. Keep using `ToolRegistry` for locally executed function tools.

On the provider/engine workflow side, the OpenAI retrieval helpers now expose
first-class `attribute_filter`, `ranking_options`, `max_num_results`, and
`rewrite_query` controls on `search_vector_store(...)`, plus
`include_search_results=True` on `respond_with_file_search(...)` for requesting
`file_search_call.results` in the Responses payload.

### Service Adaptors

Use `llm_client.adapters` when you need normalized access to supporting
services that agents or tools depend on:

- `PostgresSQLAdaptor`
- `MySQLSQLAdaptor`
- `RedisKVAdaptor`
- `QdrantVectorAdaptor`
- typed SQL/Redis/vector request and result contracts
- generic tool builders such as `build_sql_query_tool`

This layer is for generic connectivity and execution mechanics. Business
queries, authorization decisions, and workflow-specific service semantics stay
outside the package.

### Context and Memory

Use:

- `llm_client.context` for execution envelope and budget/policy metadata
- `llm_client.context_assembly` for generic multi-source assembly contracts
- `llm_client.memory` for generic memory and summary abstractions

### Observability

Use `llm_client.observability` for:

- hooks
- runtime events
- replay
- telemetry and diagnostics

## Which Entry Point Should You Use?

### One-shot completion

Use:

- `llm_client.providers`
- or `llm_client.engine` if you want retry/cache/failover

### Structured extraction

Use:

- `llm_client.structured`
- optionally through `llm_client.engine`

### Tool execution without a full agent loop

Use:

- `llm_client.tools`
- optionally `ToolExecutionEngine`

### Multi-turn assistant or copilot

Use:

- `llm_client.agent`
- `llm_client.engine`
- `llm_client.tools`

### Context-aware assistant with summaries or memory

Use:

- `llm_client.context`
- `llm_client.context_assembly`
- `llm_client.memory`
- `llm_client.agent` or `llm_client.engine`

### Usage governance or budget enforcement

Use:

- `llm_client.context`
- `llm_client.budgets`

### Controlled database or vector access through tools

Use:

- `llm_client.adapters`
- `llm_client.adapters.tools`
- `llm_client.agent` if the adaptor will be exposed to a model through a tool

### Benchmarks or release validation

Use:

- `llm_client.benchmarks`

## Top-Level Import Policy

Top-level `llm_client` imports are allowed as convenience imports, but they are
not the canonical contract for long-term integrations.

Preferred:

```python
from llm_client.engine import ExecutionEngine
from llm_client.providers import OpenAIProvider
from llm_client.tools import tool
```

Avoid for long-term integrations:

```python
from llm_client import ExecutionEngine, OpenAIProvider, tool
```

Compatibility-only imports should remain under `llm_client.compat`.

## What Is Outside The Package Contract

`llm_client` does not define the contract for:

- domain workflows
- business prompts
- operator manifests
- product policy engines
- application servers
- tenant billing persistence
- host-runtime storage/transports

Those layers may consume `llm_client`, but they are not the package API.
