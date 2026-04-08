# llm-client Usage and Capabilities Guide

This guide explains how to use the package by capability rather than by module
inventory.

See also:

- [llm-client-package-api-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-package-api-guide.md)
- [llm-client-build-and-recipes-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-build-and-recipes-guide.md)
- [llm_client/README.md](/home/namiral/Projects/Packages/llm-client-v1/llm_client/README.md)
- [llm-client-guides-index.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-guides-index.md)

## Installation and Configuration

Minimum:

```bash
pip install -e .
```

Optional providers and integrations are installed by extras. The exact matrix
is documented in
[llm-client-installation-matrix.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-installation-matrix.md).

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

Special note:

- OpenAI Responses background workflows now live at the provider layer via `retrieve_background_response`, `cancel_background_response`, `wait_background_response`, and `stream_background_response`.
- OpenAI Responses conversation-state workflows now also live at the provider layer via `create_conversation`, `retrieve_conversation`, `update_conversation`, `delete_conversation`, `create_conversation_items`, `list_conversation_items`, `retrieve_conversation_item`, `delete_conversation_item`, and `compact_response_context`.
- MCP approval workflows can now continue through `submit_mcp_approval_response(...)` without raw provider-shaped request payloads.
- OpenAI request controls `include`, `prompt_cache_key`, and `prompt_cache_retention` are first-class parameters on the OpenAI provider.
- Stored OpenAI Responses can now be deleted through `delete_response(...)` without dropping to the raw SDK.
- OpenAI moderation, direct image generation/editing, speech-to-text, text-to-speech, generic file upload/retrieve/content helpers, hosted vector-store CRUD/search, and fine-tuning job workflows are now available through first-class provider and engine methods instead of raw SDK escape hatches.
- OpenAI realtime connection plus client-secret/call-control/transcription helpers, typed `RealtimeEventResult` receive-side wrappers, conversation-item retrieve/delete/truncate helpers, `response.cancel`, webhook verification/unwrapping, vector-store file CRUD/content/polling and file-batch helpers, hosted Responses tool workflows, and staged deep-research orchestration are also available through first-class provider and engine methods.

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

### Background Responses and long-running jobs

Use:

- `llm_client.providers`
- specifically `OpenAIProvider(..., use_responses_api=True)`

Best for:

- long-running GPT-5.x / reasoning tasks that should continue after the initial request returns
- polling for terminal completion state
- canceling background work
- resuming a background stream with provider-emitted sequence cursors

Current package boundary:

- available both at the provider layer and through `ExecutionEngine`
- the remaining gap is broader platform breadth, not Responses lifecycle/state handling itself

### OpenAI Responses-native tools

Use:

- `llm_client.tools`
- specifically `ResponsesBuiltinTool`, `ResponsesToolSearch`,
  `ResponsesFunctionTool`, `ResponsesToolNamespace`, `ResponsesCustomTool`,
  `ResponsesGrammar`, `ResponsesAttributeFilter`,
  `ResponsesChunkingStrategy`, `ResponsesExpirationPolicy`,
  `ResponsesFileSearchRankingOptions`,
  `ResponsesFileSearchHybridWeights`, and `ResponsesVectorStoreFileSpec`

Best for:

- OpenAI Responses built-in hosted tools without raw provider dict payloads
- OpenAI-specific deferred-tool workflows using `tool_search`
- namespaced OpenAI function tools with deferred loading
- typed MCP and connector descriptors with `allowed_tools` and approval-policy shaping
- typed connector-specific allowlists via enums such as `ResponsesGmailTool`
- MCP and connector deferred loading for `tool_search` workflows
- typed hosted retrieval controls for file-search filters and ranking
- typed hosted vector-store resource controls for expiration, chunking, and
  per-file batch metadata
- grammar-backed custom tools on `OpenAIProvider(..., use_responses_api=True)`
- request-time tool descriptors that should stay in the stable package surface

Current package boundary:

- provider-level and engine request-envelope compatible in `1.1.0`
- richer MCP/connector descriptors are available through `ResponsesMCPTool`
- `ResponsesConnectorId` provides docs-aligned connector ids for typed connector requests
- connector-specific enums provide docs-aligned `allowed_tools` values without
  hand-typed strings
- `OpenAIProvider.submit_tool_search_output(...)` continues client-executed
  `tool_search` loops by returning the loaded tool set from your application
- `ToolRegistry` and the agent tool runtime remain function-tool execution layers
- the package now exposes a normalized subset of rich Responses output items and retains raw `provider_items` for exact replay
- OpenAI retrieval helpers now expose typed `attribute_filter`,
  `ranking_options`, `max_num_results`, and `rewrite_query` controls, plus
  `include_search_results=True` on `respond_with_file_search(...)`
- OpenAI vector-store helpers now expose typed `expiration_policy`,
  `chunking_strategy`, and `files=[ResponsesVectorStoreFileSpec(...)]`
  controls on the vector-store creation and file-batch workflows
- Store-level ingestion can now be awaited with `poll_vector_store(...)` and
  `create_vector_store_and_poll(...)` when hosted vector-store creation starts
  with initial `file_ids`

### Rich Responses output items

Use:

- `CompletionResult.output_items`
- `CompletionResult.refusal`

Best for:

- normalized refusal handling without parsing provider-native payloads
- inspecting hosted-tool outputs like web/file search, code interpreter, and image generation
- preserving a stable subset of rich Responses output while still keeping raw `provider_items` for exact replay

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
