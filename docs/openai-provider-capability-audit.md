# OpenAI Provider Capability Audit

Last updated: 2026-04-08

This document tracks how `llm_client`'s OpenAI provider aligns with the local scraped OpenAI docs index exposed by the docs API container.

## 2026-04-08 refresh

This document was refreshed against both the local docs-ledger inventory and the official OpenAI docs MCP before `1.2` implementation work starts.

Refresh findings:

- the local docs-ledger is still the better inventory source, but localhost reachability remains inconsistent from the default sandbox namespace
- the official docs MCP confirmed the `tool_search` and namespace work completed earlier on this branch, and the next missing hosted-tool gap was output-side continuation helpers for `shell` and `apply_patch`
- the official Realtime conversations guidance now explicitly enumerates `conversation.item.added` and `conversation.item.done`, confirming the current Realtime base is useful but still narrower than the latest documented event surface
- the official retrieval/file-search guidance now explicitly calls out vector-store-file `attributes` and `attribute_filter`-based filtering, which reframes the remaining retrieval gap as ergonomics and first-class workflow support rather than missing vector-store CRUD

## Canonical sources

- Index inventory: `GET http://127.0.0.1:8000/docs/index`
- Catalog: `GET http://127.0.0.1:8000/docs/catalog`
- Exported docs used in this audit:
  - `guides/migrate-to-responses.md`
  - `guides/conversation-state.md`
  - `guides/function-calling.md`
  - `guides/structured-outputs.md`
  - `guides/pdf-files.md`
  - `guides/background.md`
  - `guides/predicted-outputs.md`
  - `guides/tools*.md`

## Scope

In scope for this audit pass:

- `llm_client/providers/openai.py`
- `llm_client/content.py`
- `llm_client/models.py`
- OpenAI-specific provider/content tests

Out of scope for this pass:

- Full provider-agnostic normalization of every OpenAI Responses output-item subtype

## Audit method

This pass used the docs index as the source-of-truth inventory and then ran a bidirectional check:

1. Docs to repo: identify missing or partial OpenAI capabilities in the provider.
2. Repo to docs: verify implemented behavior against the current docs and SDK surface.
3. Implement only must-have and high-impact fixes first.
4. Lock the fixes with focused regression tests.

## Current status summary

### Fixed in this pass

- Responses API request forwarding now preserves the provider surface instead of collapsing to `model + input + reasoning`.
- Responses API now translates custom function tools and forced `tool_choice` into the correct request shape.
- Responses API now preserves advanced `tool_choice` objects, including `allowed_tools`, through the public provider surface.
- Responses API now supports structured outputs through `text.format` and `responses.parse(...)`.
- Responses API now converts assistant/tool history into proper input items, including `function_call` and `function_call_output`.
- Responses API now preserves raw output items on completion results and replays reasoning items through manual tool loops.
- Responses API non-streaming parsing now returns text, tool calls, reasoning summaries, finish reason, and usage.
- Responses API streaming is now implemented instead of incorrectly routing through chat-completions streaming.
- Native `file_url` transport is now supported for Responses file inputs.
- Shared usage parsing now handles Responses token fields (`input_tokens`, `output_tokens`, `input_tokens_details`).
- Shared usage parsing now surfaces reasoning-token counts from `output_tokens_details.reasoning_tokens`.
- Responses finish-reason parsing now normalizes `max_output_tokens` incompletes to a stable length-style finish reason and avoids fabricating terminal reasons for queued background responses.
- OpenAI background Responses lifecycle is now surfaced through provider APIs for retrieve, cancel, polling, and resumed streaming.
- Stream events now preserve provider sequence cursors for resumed background Responses streams.
- OpenAI stored Responses can now be deleted through a first-class provider helper instead of raw SDK access.
- OpenAI Conversations item endpoints are now surfaced through first-class create/list/retrieve/delete provider APIs.
- `llm_client.tools` now exposes first-class Responses built-in tool descriptors instead of requiring raw dict payloads.
- `llm_client.tools` now exposes grammar-backed Responses custom tools instead of requiring raw dict payloads.
- Responses function tools now default to `strict=True` unless the caller explicitly overrides that field.
- OpenAI Responses tool validation now fails early when built-in/custom Responses descriptors are used on the chat-completions path or with non-OpenAI providers.
- The model catalog and provider registry now expose explicit Responses-first capability flags for OpenAI completions models and the OpenAI provider surface.
- OpenAI provider and engine workflows now expose direct moderation, image generation/editing, speech-to-text, translation, text-to-speech, generic file upload/retrieve/content helpers, Uploads lifecycle helpers, hosted vector stores, vector-store files, and fine-tuning job APIs.
- OpenAI provider and engine workflows now expose realtime websocket connection helpers, client-secret/call-control helpers, webhook verification/unwrapping, and vector-store file polling/batch helpers.
- OpenAI provider and engine workflows now expose realtime transcription-session helpers plus hosted Responses workflow helpers for web search, file search, code interpreter, shell, apply-patch, computer-use, image generation, remote MCP, and connectors.
- OpenAI provider and engine workflows now expose first-class `submit_shell_call_output(...)` and `submit_apply_patch_call_output(...)` helpers, plus typed continuation payload helpers in `llm_client.tools`, so hosted shell/apply-patch loops no longer require raw provider dicts.
- OpenAI retrieval/file-search workflows now expose first-class typed tuning controls for `attribute_filter`, `ranking_options`, `max_num_results`, `rewrite_query`, and `include_search_results`.
- Deep-research workflows now include first-class clarify, rewrite, kickoff, and staged runner helpers aligned with the docs-ledger guidance.

### Still partial or deferred

- `CompletionResult.provider_items` should remain available as a low-level replay/debug escape hatch, but not as a documented stable provider-agnostic contract. The stable surface is `output_items` plus the typed result objects.
- The package now exposes a normalized subset of rich Responses outputs via `CompletionResult.output_items` and `CompletionResult.refusal`, while retaining `provider_items` for exact replay of provider-specific details that are not yet part of the stable normalized shape.
- Realtime coverage is now materially broader, but it is still not the full OpenAI Realtime product surface.
- Hosted retrieval now covers the generic Files API, the Uploads API lifecycle, vector stores, vector-store files, polling, file batches, typed filters, ranking options, query rewriting, and hosted file-search result inclusion, but broader file-search product/resource management is still incomplete.
- MCP/connectors now have typed descriptors, typed connector allowlists, deferred loading for tool-search workflows, and helper workflows, but they still do not cover the full skills/connectors product surface from the docs.
- Deep research now covers clarify/rewrite/kickoff/staged orchestration, but it is still not the full lifecycle/product surface from the docs.
- The local docs API remains intermittently unavailable on `127.0.0.1:8000`; live re-audit should be treated as best-effort until `/health` and `/docs/index` stabilize consistently.

## Capability matrix

| Capability | Docs source | Repo status | Priority | Notes |
| --- | --- | --- | --- | --- |
| Chat Completions text generation | `guides/text-generation.md`, `api-reference.md` | Implemented | Must-have | Existing path remained valid in this pass. |
| Chat Completions streaming | `api-reference.md` | Implemented | Must-have | Existing transcript/contract coverage remains green. |
| Chat Completions embeddings | `guides/embeddings.md` | Implemented | Must-have | Existing embedding path unchanged. |
| Chat Completions predicted outputs | `guides/predicted-outputs.md` | Implemented via passthrough kwargs | Important | No provider-specific shim needed; request surface already accepts `prediction`. |
| Chat prompt caching fields | `guides/prompt-caching.md` | Implemented via passthrough kwargs | Important | Supported by SDK signature; no extra translation required. |
| Responses API basic text generation | `guides/migrate-to-responses.md` | Implemented | Must-have | Fixed request forwarding and response parsing. |
| Responses API function tools | `guides/function-calling.md` | Implemented | Must-have | Fixed flattened tool shape and forced function `tool_choice`. |
| Responses API tool-call history replay | `guides/function-calling.md`, `guides/conversation-state.md` | Implemented | Must-have | Provider now emits `function_call` and `function_call_output` input items. |
| Responses API structured outputs | `guides/structured-outputs.md` | Implemented | Must-have | Fixed `text.format` mapping and `responses.parse(...)` path. |
| Responses API streaming | `guides/migrate-to-responses.md`, SDK surface | Implemented | Must-have | Added native responses streaming with text, reasoning, tool-call, usage, and done events. |
| Responses API conversation chaining | `guides/conversation-state.md` | Implemented via passthrough kwargs | Must-have | `previous_response_id` and `conversation` now survive the Responses path. |
| Conversations API lifecycle | `guides/conversation-state.md` | Implemented | Important | Added first-class create/retrieve/update/delete conversation APIs. |
| Conversations API item CRUD and pagination | `guides/conversation-state.md` | Implemented | Important | Added first-class create/list/retrieve/delete conversation item APIs plus stable page/item result types. |
| Responses API compaction | `guides/conversation-state.md` | Implemented | Important | Added first-class `compact_response_context(...)` support and normalized compaction results. |
| Responses API background creation | `guides/background.md` | Implemented | Important | `background=True` survives request translation and now has lifecycle support in the provider. |
| Responses API file ID/file data inputs | `guides/pdf-files.md` | Implemented | Must-have | Existing native support retained. |
| Responses API file URL inputs | `guides/pdf-files.md` | Implemented | Must-have | Fixed native `file_url` transport; previously degraded to text. |
| Responses API usage accounting | SDK `ResponseUsage`, `guides/prompt-caching.md`, `guides/reasoning.md` | Implemented | Must-have | Shared parser now understands Responses usage fields, cached input tokens, and reasoning-token counts. |
| Responses API finish-reason normalization for incomplete/background states | `guides/structured-outputs.md`, `guides/background.md` | Implemented | Important | `max_output_tokens` incompletes normalize to `length`; queued/in-progress background responses do not get a fake terminal finish reason. |
| Responses API reasoning item persistence across manual tool loop | `guides/function-calling.md`, `guides/conversation-state.md` | Implemented | Important | Completion results now preserve raw Responses output items and replay reasoning items with tool outputs. |
| Responses API advanced `tool_choice` including `allowed_tools` | `guides/function-calling.md` | Implemented | Important | Public surface now accepts dict tool-choice payloads and aliases nested function names correctly. |
| Responses built-in tools as first-class package abstractions | `guides/tools*.md` | Implemented | Important | Added stable `ResponsesBuiltinTool` descriptors in `llm_client.tools` and OpenAI translation support. |
| Responses custom tools with CFG grammar | `guides/function-calling.md` | Implemented | Important | Added stable `ResponsesCustomTool` and `ResponsesGrammar` descriptors in `llm_client.tools`. |
| Responses MCP approval continuation | `guides/tools-remote-mcp.md`, `guides/tools-connectors-mcp.md` | Implemented | Important | Added `submit_mcp_approval_response(...)` helper for approval loops, including MCP/connector convenience kwargs so continuation calls can resend auth-bearing tool definitions. |
| Encrypted reasoning continuity controls | `guides/reasoning.md`, `guides/migrate-to-responses.md` | Implemented | Important | Added first-class `include=["reasoning.encrypted_content"]` request control support. |
| Prompt caching request controls | `guides/prompt-caching.md` | Implemented | Important | Added first-class `prompt_cache_key` and `prompt_cache_retention` controls. |
| Background response retrieve/cancel/resume APIs | `guides/background.md` | Implemented | Important | Provider now exposes retrieval, cancellation, polling, and resumed background streaming. |
| Stored Responses deletion | `guides/background.md`, SDK surface | Implemented | Important | Added first-class `delete_response(...)` helper instead of requiring raw SDK access. |
| Responses rich output-item normalization | `guides/migrate-to-responses.md`, SDK `ResponseOutputItem` union | Implemented | Important | Added normalized `output_items` plus `refusal`, while preserving raw `provider_items` for exact replay. |
| Responses function-tool strict defaults | `guides/function-calling.md` | Implemented | Important | Responses function tools now default `strict=True` unless the caller sets it explicitly. |
| OpenAI `tool_search` | Official docs MCP `guides/function-calling#tool-search` | Implemented | Important | Added first-class advanced `ResponsesToolSearch` plus `respond_with_tool_search(...)` and `submit_tool_search_output(...)` helpers for hosted and client-executed workflows. |
| OpenAI-specific tool namespaces | Official docs MCP `guides/function-calling#tool-search` best-practices section | Implemented | Important | Added `ResponsesToolNamespace` and `ResponsesFunctionTool`, plus recursive alias sanitization and output normalization so namespace intent is preserved on the OpenAI path. |
| Hosted shell/apply-patch continuation helpers | Official docs MCP `guides/tools-shell.md`, `guides/tools-apply-patch.md` | Implemented | Important | Added typed `ResponsesShellCallChunk`, `ResponsesShellCallOutput`, and `ResponsesApplyPatchCallOutput`, plus `submit_shell_call_output(...)` and `submit_apply_patch_call_output(...)` on the provider and engine. |
| Realtime conversation item lifecycle events | Official docs MCP `guides/realtime-conversations#text-inputs-and-outputs`, `guides/realtime-conversations#interruption-and-truncation`, `guides/realtime-conversations#push-to-talk`, `guides/realtime-mcp#realtime-mcp-flow` | Implemented | Important | `RealtimeConnection` now exposes `create_text_message(...)`, `append_input_audio_chunks(...)`, `commit_audio_and_create_response(...)`, `disable_vad(...)`, `send_audio_turn(...)`, `update_session_tools(...)`, `create_response_with_tools(...)`, realtime `mcp_approval_response` creation, `conversation.item.retrieve`, `conversation.item.delete`, `conversation.item.truncate`, `response.cancel`, and typed `RealtimeEventResult` / `RealtimeMCPToolListingResult` / `RealtimeResponseOutput` helpers via `recv_event()` / `recv_until_type(...)` / `wait_for_mcp_tool_listing(...)` / `collect_response_output(...)`. |
| Retrieval attributes, chunking, and ingestion ergonomics | Official docs MCP `guides/tools-file-search#metadata-filtering`, `guides/retrieval#attributes`, `guides/retrieval#attribute-filtering`, `guides/retrieval#batch-operations`, `assistants/tools/file-search#creating-vector-stores-and-adding-files` | Implemented | Important | Added typed `ResponsesAttributeFilter`, `ResponsesFileSearchRankingOptions`, `ResponsesFileSearchHybridWeights`, `ResponsesExpirationPolicy`, `ResponsesChunkingStrategy`, and `ResponsesVectorStoreFileSpec`, plus direct `search_vector_store(...)`, `create_vector_store(...)`, `poll_vector_store(...)`, `create_vector_store_and_poll(...)`, vector-store file, and vector-store batch controls for typed retrieval tuning, expiration, chunking, per-file ingestion metadata, store provisioning from typed file specs, and store-level ingestion waits. |
| Uploads API lifecycle | Official docs MCP OpenAPI `/uploads`, SDK `uploads.upload_file_chunked(...)` | Implemented | Important | Added first-class `create_upload(...)`, `add_upload_part(...)`, `complete_upload(...)`, `cancel_upload(...)`, and `upload_file_chunked(...)` provider/engine helpers with typed `UploadResource` and `UploadPartResource` results so larger hosted ingestion flows no longer require raw SDK access. |

## Implementation roadmap

### 1.2 next pass

The next bounded `1.2` implementation slice should be:

1. deeper connectors/MCP and hosted file-search product management
2. then the next broader Realtime product-management follow-up

Why this order:

- `tool_search`, namespaces, hosted shell/apply-patch continuation, retrieval tuning/ingestion ergonomics, store-level vector-store polling, and the first Realtime lifecycle/event wrapper slice are now closed as the early advanced OpenAI-specific `1.2` slices
- deeper connectors/MCP and hosted file-search product management now represent the clearest remaining high-value gaps, although MCP approval continuation no longer requires raw tool reconstruction
- broader Realtime product-management work still remains, but the connection-level lifecycle contract is materially stronger than before, including higher-level text/audio-turn helpers and collected response output helpers on `RealtimeConnection`

This roadmap covers the remaining work needed to extend `llm_client` toward broader official-docs parity. It is ordered by package impact: fix compliance gaps first, then add missing capabilities that need package-contract expansion, then harden metadata, tests, and user-facing docs.

### Phase 1: Close remaining core Responses parity gaps

Milestone: `llm_client` can run docs-compliant Responses workflows for text, tools, state, and structured outputs without silent behavior loss.

Deliverables:

- Reasoning-item persistence across manual tool loops.
- Explicit support and tests for advanced Responses request controls already representable by the package.
- Clear separation between fully supported and passthrough-only Responses features.

Exit criteria:

- Reasoning models can complete a tool loop without losing required reasoning items.
- Responses request translation has focused coverage for all currently supported core controls.
- The capability matrix can mark core Responses parity as implemented instead of partial.

[x] Add a provider-safe representation for Responses reasoning items in [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py) and shared result/request models as needed.
[x] Preserve reasoning items when replaying assistant output into subsequent Responses tool-loop calls.
[x] Add explicit regression coverage for `previous_response_id`, `conversation`, `store`, `background`, `parallel_tool_calls`, and advanced `tool_choice` passthrough on the Responses path.
[x] Verify finish-reason and usage behavior against additional docs/API examples and tighten parsing where the current implementation is permissive.

### Phase 2: Add background response lifecycle support

Milestone: background Responses move from request-only support to a complete provider workflow.

Deliverables:

- Provider methods for retrieve, cancel, and stream-resume behavior for background responses.
- Result normalization for polling background responses to completion.
- Tests covering the non-blocking lifecycle.

Exit criteria:

- Package users can create, retrieve, cancel, and continue processing background responses through provider APIs.
- Audit matrix can mark background lifecycle support as implemented.

[x] Extend the provider contract with background lifecycle methods in the shared provider interface and [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py).
[x] Map OpenAI SDK background retrieve/cancel APIs into package result types, preserving text, tool calls, reasoning summaries, and usage.
[x] Add stream-resume or equivalent polling helpers where the docs expose resumable background workflows.
[x] Add contract and smoke tests covering create, retrieve, cancel, and completed-background parsing.

### Phase 3: Introduce first-class Responses tool abstractions

Milestone: built-in and advanced Responses tools are representable through package-native APIs instead of opaque dict passthrough.

Deliverables:

- New package abstractions for built-in Responses tools.
- Support for custom-tool/grammar-style tool definitions where the docs require richer schemas than function tools.
- Compatibility rules documenting which tool classes work on Chat Completions versus Responses.

Exit criteria:

- Users can express built-in Responses tools without raw provider-specific dicts.
- Tool normalization and validation occur before provider dispatch.
- Capability matrix records built-in/custom tool support as implemented.

[x] Design and add shared tool model extensions for built-in Responses tools and grammar-based custom tools in the package tool layer.
[x] Update [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py) to translate the new tool abstractions into official Responses request shapes.
[x] Add validation rules for incompatible tool/model/API combinations so unsupported mixes fail early.
[x] Add translation and round-trip tests for built-in tool descriptors, custom grammar tools, and function-tool strictness behavior.

### Phase 4: Normalize richer Responses output items

Milestone: Responses-only output structures are preserved in package results instead of being partially flattened.

Deliverables:

- Shared result types or extension fields for richer Responses output items.
- Parsing support for refusals, built-in tool outputs, file/image outputs, and other surfaced item classes that matter to package consumers.
- Compatibility notes documenting which item classes remain provider-specific.

Exit criteria:

- Provider output no longer silently drops materially useful Responses item data.
- Package tests lock in the normalized subset and document intentional exclusions.

[x] Audit the current docs/API examples for output item classes that are still discarded or collapsed by the provider.
[x] Extend shared result/content models only where stable cross-provider semantics exist; otherwise add clearly namespaced OpenAI extensions.
[x] Update non-streaming and streaming parsing paths in [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py) to emit the richer item data.
[x] Add fixture-backed tests for refusal outputs, built-in tool outputs, and file/image-related output items that the package chooses to preserve.

### Phase 5: Tighten model metadata and capability exposure

Milestone: package capability metadata accurately reflects the official docs and the actual implementation state.

Deliverables:

- Updated model/capability metadata for Responses, structured outputs, tools, files, streaming, and background support.
- Clear public documentation on supported versus passthrough-only features.
- Reduced ambiguity in feature-detection behavior used by higher-level package code.

Exit criteria:

- Model metadata matches the implemented provider surface.
- Public docs/examples do not claim unsupported behavior.

[x] Audit model profiles and capability flags in [`llm_client/models.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/models.py) and any provider capability helpers against the expanded implementation.
[x] Mark features as first-class, passthrough-only, or unsupported so calling code can make correct decisions.
[x] Update user-facing docs/examples to steer users toward Responses-first features where the official docs recommend them.
[x] Reconcile any Chat Completions legacy behavior that now conflicts with the Responses-preferred package story.

### Phase 6: Re-audit, harden, and release

Milestone: the expanded provider surface is regression-safe and the audit becomes the durable maintenance baseline.

Deliverables:

- Updated capability matrix with final statuses.
- Broad regression coverage for OpenAI provider behavior.
- Release notes or changelog entry summarizing new OpenAI support.

Exit criteria:

- The audit no longer contains stale statuses for implemented work.
- Focused and broad provider suites pass after each phase.
- The repo contains a clear next-pass backlog only for intentionally deferred work.

[x] Re-run the docs-to-repo and repo-to-docs audit after each implementation phase and update this document in place.
[x] Expand targeted tests in [`tests/llm_client`](/home/namiral/Projects/Packages/llm-client-v1/tests/llm_client) for new request translation, parsing, streaming, and lifecycle behavior.
[x] Run focused OpenAI/provider suites plus broader package contract coverage before merging each phase.
[x] Add a concise release summary to package docs or changelog once the next major capability tranche lands.

## Milestones checklist

[x] M1: Core Responses parity is fully compliant for supported text/tool/state workflows.
[x] M2: Background response lifecycle support is available through provider APIs.
[x] M3: First-class Responses tool abstractions are available to package consumers.
[x] M4: Rich Responses output items are normalized or explicitly documented as provider-specific.
[x] M5: Model metadata, docs, and tests accurately reflect the expanded implementation.
[x] M6: OpenAI media/moderation/fine-tuning/vector-store/realtime/webhook/deep-research workflow surfaces are first-class package APIs.

## Code changes made in this pass

- [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py)
  - Added correct Responses input-item translation.
  - Added Responses tool flattening and tool-alias rewriting.
  - Added Responses structured-output mapping and parse path.
  - Added native Responses stream handling and output extraction.
  - Added first-class conversation lifecycle, context compaction, MCP approval continuation, and explicit prompt-cache/encrypted-reasoning controls.
- [`llm_client/providers/types.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/types.py)
  - Added completion-result preservation of provider-specific Responses output items for replay through `to_message()`.
  - Added stable `NormalizedOutputItem` plus `CompletionResult.refusal` and `CompletionResult.output_items`.
  - Added `BackgroundResponseResult` and resumable stream `sequence_number` support.
  - Added stable `ConversationResource` and `CompactionResult`.
- [`llm_client/content.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/content.py)
  - Added native `file_url` transport for Responses file inputs.
  - Added envelope round-tripping for preserved OpenAI Responses output items and normalized output-item metadata.
  - Added first-class request-envelope fields for `include`, `prompt_cache_key`, and `prompt_cache_retention`.
- [`llm_client/request_builders.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/request_builders.py)
  - Added first-class request-spec handling for `include`, `prompt_cache_key`, and `prompt_cache_retention`.
- [`llm_client/cache/serializers.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/cache/serializers.py)
  - Added cache persistence for preserved and normalized Responses output items.
- [`llm_client/models.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/models.py)
  - Added Responses usage-field support, including reasoning-token counts.
- [`llm_client/model_catalog.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/model_catalog.py)
  - Added explicit model metadata flags for `responses_api`, `background_responses`, `responses_native_tools`, and `normalized_output_items`.
- [`llm_client/provider_registry.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/provider_registry.py)
  - Added matching provider capability flags and capability filtering for the expanded OpenAI Responses surface.
- [`llm_client/assets/model_catalog.json`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/assets/model_catalog.json)
  - Updated the asset-backed model catalog with explicit Responses capability flags.
- [`llm_client/assets/model_catalog.schema.json`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/assets/model_catalog.schema.json)
  - Extended schema validation for the new model capability flags.
- [`llm_client/spec.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/spec.py)
  - Widened `tool_choice` typing to permit official Responses tool-choice objects.
- [`llm_client/providers/base.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/base.py)
  - Added shared background lifecycle methods and polling helper on the provider contract.
- [`llm_client/engine.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/engine.py)
  - Added engine-level orchestration for background response lifecycle, conversation lifecycle, conversation item CRUD/listing, response compaction, MCP approval continuation, and stored-response deletion.
- [`llm_client/providers/__init__.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/__init__.py)
  - Exported `BackgroundResponseResult` and `NormalizedOutputItem` from the stable provider namespace.
- [`llm_client/types.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/types.py)
  - Exported `BackgroundResponseResult` and `NormalizedOutputItem` from the stable shared types namespace.
- [`llm_client/tools/base.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/tools/base.py)
  - Added `ResponsesBuiltinTool`, `ResponsesCustomTool`, and `ResponsesGrammar`.
  - Added early validation helpers for providers that only support executable function tools.
- [`llm_client/tools/__init__.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/tools/__init__.py)
  - Exported the new Responses tool descriptors from the stable tools namespace.
- [`examples/38_openai_background_responses.py`](/home/namiral/Projects/Packages/llm-client-v1/examples/38_openai_background_responses.py)
  - Added a live example for background Responses lifecycle through the package surface.
- [`examples/39_openai_conversation_state_workflow.py`](/home/namiral/Projects/Packages/llm-client-v1/examples/39_openai_conversation_state_workflow.py)
  - Added a live example for Conversations lifecycle, item management, and compaction.
- [`examples/40_openai_normalized_output_items.py`](/home/namiral/Projects/Packages/llm-client-v1/examples/40_openai_normalized_output_items.py)
  - Added a live example showing the stable `output_items` surface versus low-level `provider_items`.
- [`examples/17_persistence_repository.py`](/home/namiral/Projects/Packages/llm-client-v1/examples/17_persistence_repository.py)
  - Hardened the PostgreSQL example so unreachable or misconfigured database infrastructure cleanly skips under cookbook validation instead of hard-failing.
- [`scripts/ci/run_llm_client_examples.py`](/home/namiral/Projects/Packages/llm-client-v1/scripts/ci/run_llm_client_examples.py)
  - Fixed example-root resolution, added resume support via `--from-example`, and registered the new OpenAI workflow examples in the canonical cookbook run.

## Validation completed

Focused and broader OpenAI/provider regression suites passed after the implementation:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_provider_request_translation.py \
  tests/llm_client/test_provider_response_parsing.py \
  tests/llm_client/test_content_model.py \
  tests/llm_client/test_file_block_transport.py \
  tests/llm_client/test_openai_provider_contract_smoke.py \
  tests/llm_client/test_provider_overlap_contracts.py \
  tests/llm_client/test_openai_content_translation.py \
  tests/llm_client/test_public_api_namespaces.py \
  tests/llm_client/test_guides_inventory.py \
  tests/llm_client/test_request_builders.py \
  tests/llm_client/test_model_catalog.py \
  tests/llm_client/test_provider_registry.py
```

Result: `89 passed`

A broader package validation slice also passed after the engine/orchestration and
normalization expansion:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_engine_reliability.py \
  tests/llm_client/test_openai_provider_contract_smoke.py \
  tests/llm_client/test_provider_overlap_contracts.py \
  tests/llm_client/test_openai_content_translation.py \
  tests/llm_client/test_provider_response_parsing.py \
  tests/llm_client/test_provider_request_translation.py \
  tests/llm_client/test_file_block_transport.py \
  tests/llm_client/test_content_model.py \
  tests/llm_client/test_public_api_namespaces.py \
  tests/llm_client/test_guides_inventory.py \
  tests/llm_client/test_request_builders.py \
  tests/llm_client/test_model_catalog.py \
  tests/llm_client/test_provider_registry.py
```

Result: `109 passed`

Live cookbook validation was also completed after:

- hardening `17_persistence_repository.py` so unavailable PostgreSQL
  infrastructure becomes a clean skip under `LLM_CLIENT_EXAMPLE_ALLOW_SKIP=1`
- resuming the ring from `17_persistence_repository.py`
- rerunning the final heavy examples from `34_end_to_end_mission_control.py`
  with a higher per-example timeout (`300s`) because that showcase is
  intentionally broader than the generic `120s` budget

The completed live run successfully exercised the new OpenAI workflow examples
(`38`-`40`), the broader application examples, and the end-to-end mission
control flow. The final PostgreSQL adaptor examples (`36`, `37`) degraded to
clean infrastructure skips under invalid local DB credentials rather than
failing the ring.

## Follow-up backlog

### Next must-have candidates

- Continue Phase 6 by re-auditing the remaining lower-priority official-docs surface as the docs snapshot evolves.
- Tighten or recalibrate the cookbook eval baseline in
  [`examples/30_eval_and_regression_gate.py`](/home/namiral/Projects/Packages/llm-client-v1/examples/30_eval_and_regression_gate.py)
  if the current first-token thresholds remain stricter than live-provider
  behavior.

### Deferred until package-contract expansion

- Promote additional Responses-native output item families into shared provider-agnostic result types only where stable cross-provider semantics emerge.

## Re-audit workflow

Use this when rerunning the audit against a newer docs snapshot:

```bash
curl -sS http://127.0.0.1:8000/docs/index
curl -sS http://127.0.0.1:8000/docs/catalog
curl -sS http://127.0.0.1:8000/docs/export/file/guides/migrate-to-responses.md
curl -sS http://127.0.0.1:8000/docs/export/file/guides/function-calling.md
curl -sS http://127.0.0.1:8000/docs/export/file/guides/structured-outputs.md
curl -sS http://127.0.0.1:8000/docs/export/file/guides/conversation-state.md
curl -sS http://127.0.0.1:8000/docs/export/file/guides/pdf-files.md
curl -sS http://127.0.0.1:8000/docs/export/file/guides/background.md
```

Recheck the resulting matrix against:

- [`llm_client/providers/openai.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/providers/openai.py)
- [`llm_client/content.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/content.py)
- [`llm_client/models.py`](/home/namiral/Projects/Packages/llm-client-v1/llm_client/models.py)
- OpenAI-focused tests under [`tests/llm_client`](/home/namiral/Projects/Packages/llm-client-v1/tests/llm_client)
