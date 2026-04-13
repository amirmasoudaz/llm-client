# llm-client Public API Map (v1)

This document defines the intended public package boundary for `llm_client`.
It is the contract to use when modernizing imports, writing examples, and
deciding whether a symbol should be considered stable, compatibility-only,
advanced, reserved, or internal.

This is the frozen `1.x` public package map. It defines the semver-protected
surface for `llm_client` starting with `1.0.0`.

## Stability Levels

- `Stable`: Recommended for new projects. Backward-compatibility should be
  preserved deliberately.
- `Compatibility`: Retained for existing callers. New projects should avoid it.
- `Advanced`: Supported lower-level helpers for specialized use, but not part
  of the preferred standalone-package surface.
- `Reserved`: Namespace is intentionally reserved for future promotion, but is
  not yet implemented as a stable module.
- `Internal`: Not part of the public contract. Behavior and exports may change
  without notice.

## Canonical Import Rule

For new integrations:

- Prefer module namespaces over top-level `llm_client` imports.
- Treat `llm_client.__init__` as a convenience layer, not the canonical source
  of truth for long-term integrations.
- Use `llm_client.compat` for legacy API access.
- Use `llm_client.advanced` for lower-level helper and integration surfaces.

## Stable Namespaces

### `llm_client.providers`

Purpose:
- Provider protocol and concrete provider entry points.

Use for:
- `Provider`, `BaseProvider`
- `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`
- provider-level message/result/event types when working directly at provider
  level

Notes:
- Provider-specific translator internals remain internal.

### `llm_client.models`

Purpose:
- Stable model profile definitions.

Use for:
- `ModelProfile`
- named model profiles such as `GPT5`, `GPT5Mini`, `Gemini20Flash`

Notes:
- The metadata catalog is exposed separately through `llm_client.model_catalog`
  and the top-level stable surface.

### `llm_client.types`

Purpose:
- Canonical request/result/event/cancellation data types.

Use for:
- `RequestContext`, `RequestSpec`
- `Message`, `Role`
- `ToolCall`, `ToolCallDelta`
- `Usage`
- `CompletionResult`, `NormalizedOutputItem`, `BackgroundResponseResult`, `DeepResearchRunResult`, `ConversationResource`, `CompactionResult`, `DeletionResult`, `ConversationItemResource`, `ConversationItemsPage`, `EmbeddingResult`
- `ModerationResult`, `ImageGenerationResult`, `GeneratedImage`, `AudioTranscriptionResult`, `AudioSpeechResult`
- `FileResource`, `FilesPage`, `FileContentResult`, `UploadResource`, `UploadPartResource`
- `VectorStoreResource`, `VectorStoresPage`, `VectorStoreSearchResult`, `VectorStoreFileResource`, `VectorStoreFilesPage`, `VectorStoreFileContentResult`, `VectorStoreFileBatchResource`
- `FineTuningJobResult`, `FineTuningJobsPage`, `FineTuningJobEventsPage`
- `RealtimeClientSecretResult`, `RealtimeTranscriptionSessionResult`, `RealtimeCallResult`, `RealtimeEventResult`, `RealtimeResponseOutput`, `RealtimeConnection`, `WebhookEventResult`
- `StreamEvent`, `StreamEventType`
- `CancellationToken`, `CancelledError`

Notes:
- New code should prefer this namespace over importing the same core types from
  provider-specific modules.

### `llm_client.content`

Purpose:
- Canonical structured content model and content-envelope boundary.

Use for:
- typed content blocks
- `ContentMessage`
- `ContentRequestEnvelope`
- `ContentResponseEnvelope`
- content projection and response/envelope normalization helpers

Notes:
- This is the canonical multimodal/content boundary for engine and provider
  integrations.

### `llm_client.context`

Purpose:
- Canonical execution-context layer above request correlation.

Use for:
- `BudgetSpec`
- `PolicyRef`
- `RunVersions`
- `ExecutionContext`

Notes:
- `RequestContext` stays canonical in `llm_client.types` / `llm_client.spec`.

### `llm_client.budgets`

Purpose:
- Canonical budget enforcement and usage-ledger primitives.

Use for:
- `Ledger`, `LedgerWriter`, `InMemoryLedgerWriter`
- `LedgerEvent`, `LedgerEventType`
- `UsageRecord`
- `Budget`, `BudgetDecision`, `BudgetExceededError`

Notes:
- This namespace owns generic execution-usage and budget concepts.
- Product billing policy, tenant ledgers, and application-owned quota
  persistence stay outside `llm_client`.

### `llm_client.context_assembly`

Purpose:
- Generic multi-source context assembly contracts above the planner and memory
  subsystems.

Use for:
- `ContextSourceLoader`
- `ContextSourceRequest`
- `ContextSourcePayload`
- `ContextAssemblyRequest`
- `ContextAssemblyResult`
- `MultiSourceContextAssembler`

Notes:
- This namespace is intentionally generic.
- Domain-specific entity loaders and product context loaders must stay outside
  `llm_client`.

### `llm_client.engine`

Purpose:
- Canonical engine runtime for retries, failover, hooks, cache, and
  idempotency-aware orchestration.

Use for:
- `ExecutionEngine`
- `RetryConfig`
- `FailoverPolicy`

### `llm_client.agent`

Purpose:
- Canonical multi-turn agent layer.

Use for:
- `Agent`
- `AgentConfig`
- `AgentResult`
- `TurnResult`
- `quick_agent`

### `llm_client.benchmarks`

Purpose:
- Canonical benchmark harness, report format, and trend-comparison helpers.

Use for:
- `BenchmarkCase`, `BenchmarkReport`, `BenchmarkRunMode`
- deterministic local benchmark runs
- explicitly labeled live benchmark runs
- committed baseline report storage and comparison

Notes:
- benchmark cases should default to deterministic providers and local
  primitives
- live benchmark runs should stay opt-in and explicitly labeled

### `llm_client.tools`

Purpose:
- Canonical tool definition and middleware surface.

Use for:
- `Tool`, `ToolRegistry`, `ToolResult`
- `ResponsesBuiltinTool`, `ResponsesAttributeFilter`, `ResponsesChunkingStrategy`, `ResponsesExpirationPolicy`, `ResponsesFileSearchHybridWeights`, `ResponsesFileSearchRankingOptions`, `ResponsesToolSearch`, `ResponsesFunctionTool`, `ResponsesToolNamespace`, `ResponsesVectorStoreFileSpec`, `ResponsesConnectorId`, `ResponsesDropboxTool`, `ResponsesGmailTool`, `ResponsesGoogleCalendarTool`, `ResponsesGoogleDriveTool`, `ResponsesMicrosoftTeamsTool`, `ResponsesOutlookCalendarTool`, `ResponsesOutlookEmailTool`, `ResponsesSharePointTool`, `ResponsesMCPTool`, `ResponsesMCPApprovalPolicy`, `ResponsesMCPToolFilter`, `ResponsesCustomTool`, `ResponsesGrammar`, `ResponsesShellCallChunk`, `ResponsesShellCallOutcome`, `ResponsesShellCallOutput`, `ResponsesApplyPatchCallOutput`
- `tool`, `sync_tool`, `tool_from_function`
- tool middleware stack for advanced use

Notes:
- `ToolRegistry` remains the execution/runtime surface for local function tools.
- `ResponsesBuiltinTool`, `ResponsesAttributeFilter`,
  `ResponsesChunkingStrategy`, `ResponsesExpirationPolicy`,
  `ResponsesFileSearchHybridWeights`, `ResponsesFileSearchRankingOptions`,
  `ResponsesToolSearch`, `ResponsesFunctionTool`, `ResponsesToolNamespace`,
  `ResponsesVectorStoreFileSpec`, `ResponsesMCPTool`, the connector-tool enums,
  `ResponsesShellCallChunk`, `ResponsesShellCallOutcome`,
  `ResponsesShellCallOutput`, `ResponsesApplyPatchCallOutput`, and
  `ResponsesCustomTool` are provider-native request/continuation descriptors
  for OpenAI Responses workflows, not locally executable tools.

### `llm_client.cache`

Purpose:
- Cache abstractions and supported backends.

Use for:
- `CacheCore`
- `CachePolicy`, `CacheInvalidationMode`
- `MetadataCacheStore`, `SummaryCacheStore`
- cache backend types/settings
- supported cache backend implementations

Notes:
- Persistence-specific SQL helpers are not part of the preferred stable
  surface.
- Cache-backed metadata and summary stores remain storage-agnostic and sit
  above `CacheCore`, not above repo-specific ledgers or billing models.

### `llm_client.memory`

Purpose:
- Generic memory and persistent-summary abstractions.

Use for:
- `MemoryRecord`, `MemoryQuery`, `MemoryWrite`
- `MemoryReader`, `MemoryWriter`, `MemoryStore`
- `SummaryRecord`, `SummaryStore`
- `ShortTermMemoryStore`, `InMemorySummaryStore`

Notes:
- This namespace holds generic in-process abstractions only.
- Domain-backed entity loaders, tenant-specific retrieval services, and
  product-owned memory policies stay outside `llm_client`.

### `llm_client.observability`

Purpose:
- Canonical hooks, diagnostics, runtime eventing, replay, and telemetry entry
  point.

Use for:
- hooks and metrics hooks
- `EngineDiagnosticsRecorder`
- runtime events/event bus
- replay primitives
- telemetry registry/usage tracking

Notes:
- `RuntimeEvent` / replay types are canonical here, even though their concrete
  implementations live in dedicated modules.

### `llm_client.validation`

Purpose:
- Canonical validation entry point for requests, tools, and schemas.

Use for:
- `ValidationError`
- request/tool/schema validation functions

### `llm_client.errors`

Purpose:
- Canonical error taxonomy.

Use for:
- `LLMClientError`
- provider/tool/cache/agent/config error types
- retryability-aware error mapping helpers

### `llm_client.config`

Purpose:
- Supported configuration and `.env` loading surface.

Use for:
- settings models
- `get_settings`
- `configure`
- `load_env`

## Compatibility Namespace

### `llm_client.compat`

Status:
- `Compatibility`

Use for:
- `OpenAIClient`
- `ResponseTimeoutError`

Policy:
- Retained for existing users.
- New code should prefer `llm_client.providers`, `llm_client.engine`, and
  `llm_client.agent`.
- Top-level `llm_client.OpenAIClient` should be treated as compatibility-only.

## Advanced Namespace

### `llm_client.advanced`

Status:
- `Advanced`

Purpose:
- Explicit home for lower-level helpers and integration surfaces that are still
  useful but are not the preferred standalone-package API.

Contains:
- container/factory helpers
- idempotency helpers
- hashing/performance/serialization utilities
- streaming adapters/utilities

Policy:
- Supported for specialized use.
- Not the primary package entry point for typical integrations.

## Stable And Reserved Namespaces

### `llm_client.adapters`

Status:
- `Stable`

Purpose:
- Canonical namespace for generic service adaptors such as SQL, Redis, and
  vector backends.

Use for:
- normalized adaptor contracts
- typed request/result shapes
- concrete backend adaptors behind optional extras
- generic tool-construction helpers over adaptors

Current concrete backend surface:
- `PostgresSQLAdaptor`
- `MySQLSQLAdaptor`
- `RedisKVAdaptor`
- `QdrantVectorAdaptor`

Policy:
- The public concern is normalized adaptors, not raw backend drivers.
- Lower-level drivers may exist as internal or advanced implementation detail.
- Business queries and domain workflows stay outside the package.

### `llm_client.plugins`

Status:
- `Reserved`
- `Deferred For 1.0`

Intent:
- Future canonical namespace for pluggable runtime/plugin interfaces if that
  layer is promoted from repo-specific runtime integrations.

Policy:
- `1.0.0` does not introduce a stable plugin registry in `llm_client`.
- Plugin lifecycle and host-runtime extension concerns remain external to the
  core package for now.
- Revisit only after the standalone package API is frozen and real extension
  requirements are proven across projects.

## Internal / Non-Contract Modules

These modules may exist and be useful internally, but they are not part of the intended standalone-package contract:

- provider-specific translator internals
- request-builder internals
- resilience implementation details
- retry-policy implementation details
- low-level runtime extraction helpers not promoted into stable namespaces
- repo-specific orchestration glue in other packages

## Canonical Type Placement

The intended canonical placement is:

- request/result/usage/event/cancellation core types: `llm_client.types`
- structured content and envelopes: `llm_client.content`
- execution context / budgets / policy refs: `llm_client.context`
- budget enforcement and usage-ledger primitives: `llm_client.budgets`
- runtime events and replay access: `llm_client.observability`
- error taxonomy: `llm_client.errors`

This means new API work should avoid introducing duplicate type entry points
unless there is a strong compatibility reason.

## Top-Level `llm_client` Policy

Top-level `llm_client` exports should stay intentionally small:

- stable high-value convenience exports may remain
- compatibility aliases may remain with warnings
- hidden accidental exports should not leak into the module namespace

Long-term callers should still import from the canonical module namespaces
above rather than relying on top-level convenience imports.
