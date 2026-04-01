# llm-client 1.1.0 Release Notes

Last updated: 2026-04-01

`llm_client` `1.1.0` extends the stable `1.x` package line with a broader
OpenAI Responses API implementation, including background response lifecycle
support, resumed background streaming, first-class Responses tool descriptors,
conversation-state lifecycle helpers, response compaction, MCP approval
continuations, conversation item lifecycle helpers, stored-response deletion,
explicit prompt-caching controls, tighter Responses result normalization, and
first-class OpenAI media, moderation, fine-tuning, realtime, webhook, and
hosted retrieval workflow surfaces, plus staged deep-research orchestration,
typed MCP/connector policy descriptors, documented connector ids, generic
Files API helpers, and realtime transcription sessions.

## Added

- OpenAI provider background lifecycle APIs:
  - `retrieve_background_response(...)`
  - `cancel_background_response(...)`
  - `wait_background_response(...)`
  - `stream_background_response(...)`
- OpenAI provider conversation-state APIs:
  - `create_conversation(...)`
  - `retrieve_conversation(...)`
  - `update_conversation(...)`
  - `delete_conversation(...)`
  - `create_conversation_items(...)`
  - `list_conversation_items(...)`
  - `retrieve_conversation_item(...)`
  - `delete_conversation_item(...)`
  - `compact_response_context(...)`
  - `submit_mcp_approval_response(...)`
  - `delete_response(...)`
- OpenAI provider product/workflow APIs:
  - `moderate(...)`
  - `generate_image(...)`
  - `edit_image(...)`
  - `transcribe_audio(...)`
  - `translate_audio(...)`
  - `synthesize_speech(...)`
  - `create_file(...)`
  - `retrieve_file(...)`
  - `list_files(...)`
  - `delete_file(...)`
  - `get_file_content(...)`
  - `create_vector_store(...)`
  - `retrieve_vector_store(...)`
  - `update_vector_store(...)`
  - `delete_vector_store(...)`
  - `list_vector_stores(...)`
  - `search_vector_store(...)`
  - `create_vector_store_file(...)`
  - `upload_vector_store_file(...)`
  - `list_vector_store_files(...)`
  - `retrieve_vector_store_file(...)`
  - `update_vector_store_file(...)`
  - `delete_vector_store_file(...)`
  - `get_vector_store_file_content(...)`
  - `poll_vector_store_file(...)`
  - `create_vector_store_file_and_poll(...)`
  - `upload_vector_store_file_and_poll(...)`
  - `create_vector_store_file_batch(...)`
  - `retrieve_vector_store_file_batch(...)`
  - `cancel_vector_store_file_batch(...)`
  - `poll_vector_store_file_batch(...)`
  - `list_vector_store_file_batch_files(...)`
  - `create_vector_store_file_batch_and_poll(...)`
  - `upload_vector_store_file_batch_and_poll(...)`
  - `create_fine_tuning_job(...)`
  - `retrieve_fine_tuning_job(...)`
  - `cancel_fine_tuning_job(...)`
  - `list_fine_tuning_jobs(...)`
  - `list_fine_tuning_events(...)`
  - `create_realtime_client_secret(...)`
  - `create_realtime_transcription_session(...)`
  - `connect_realtime(...)`
  - `connect_realtime_transcription(...)`
  - `create_realtime_call(...)`
  - `accept_realtime_call(...)`
  - `reject_realtime_call(...)`
  - `hangup_realtime_call(...)`
  - `refer_realtime_call(...)`
  - `unwrap_webhook(...)`
  - `verify_webhook_signature(...)`
  - `clarify_deep_research_task(...)`
  - `rewrite_deep_research_prompt(...)`
  - `respond_with_web_search(...)`
  - `respond_with_file_search(...)`
  - `respond_with_code_interpreter(...)`
  - `respond_with_remote_mcp(...)`
  - `respond_with_connector(...)`
  - `start_deep_research(...)`
  - `run_deep_research(...)`
- Stable OpenAI Responses tool descriptors in `llm_client.tools`:
  - `ResponsesBuiltinTool`
  - `ResponsesConnectorId`
  - `ResponsesMCPTool`
  - `ResponsesMCPApprovalPolicy`
  - `ResponsesMCPToolFilter`
  - `ResponsesCustomTool`
  - `ResponsesGrammar`
  - convenience aliases for OpenAI docs terminology such as
    `web_search_preview(...)`, `remote_mcp(...)`, and `connector(...)`
- `BackgroundResponseResult` in the stable shared types surface.
- `ConversationResource` in the stable shared types surface.
- `CompactionResult` in the stable shared types surface.
- `DeletionResult` in the stable shared types surface.
- `ConversationItemResource` in the stable shared types surface.
- `ConversationItemsPage` in the stable shared types surface.
- `NormalizedOutputItem` in the stable shared types surface.
- `RealtimeClientSecretResult` in the stable shared types surface.
- `RealtimeTranscriptionSessionResult` in the stable shared types surface.
- `RealtimeCallResult` in the stable shared types surface.
- `RealtimeConnection` in the stable shared types surface.
- `WebhookEventResult` in the stable shared types surface.
- `FileResource` in the stable shared types surface.
- `FilesPage` in the stable shared types surface.
- `FileContentResult` in the stable shared types surface.
- `VectorStoreFileResource` in the stable shared types surface.
- `VectorStoreFilesPage` in the stable shared types surface.
- `VectorStoreFileContentResult` in the stable shared types surface.
- `VectorStoreFileBatchResource` in the stable shared types surface.
- Stream-event `sequence_number` support for providers that expose resumable
  cursors, such as OpenAI background Responses streaming.

## Changed

- `OpenAIProvider(..., use_responses_api=True)` now supports the full Phase 1
  Responses parity tranche plus Phase 2 background lifecycle workflows and
  Phase 3 request-side tool descriptors, conversation-state helpers, and
  response compaction.
- `tool_choice` in the shared provider contract now accepts official Responses
  object forms in addition to simple string modes.
- OpenAI request/engine surfaces now accept first-class Responses built-in and
  grammar-backed custom tool descriptors instead of requiring raw dict payloads.
- OpenAI request surfaces now expose first-class `include`,
  `prompt_cache_key`, and `prompt_cache_retention` controls.
- `ExecutionEngine` now orchestrates provider-native OpenAI workflow methods for
  background responses, conversations, conversation items, compaction, MCP
  approval continuation, stored-response deletion, moderation, media APIs,
  generic Files API operations, vector stores, vector-store files, fine-tuning jobs, realtime
  connection/call/transcription helpers, hosted Responses tool workflows,
  webhook verification, and staged deep-research orchestration.
- Responses function tools now default to strict JSON-schema mode unless the
  caller explicitly overrides `strict`.
- `CompletionResult` now exposes `refusal` and normalized rich `output_items`
  for OpenAI Responses outputs while retaining raw `provider_items` for exact
  replay. `provider_items` remains a low-level escape hatch rather than a
  stable provider-agnostic contract.
- The model catalog and provider registry now expose explicit Responses-first
  capability flags for OpenAI completions models and the OpenAI provider.
- The OpenAI model registry now covers a much broader docs-aligned set of
  current and compatibility model families, including GPT-4.1, GPT-5
  chat/codex variants, o-series reasoning families, image/audio/realtime
  variants, and deprecated replacement metadata where appropriate.
- `Usage` now exposes reasoning-token counts when the provider reports
  `output_tokens_details.reasoning_tokens`.
- Responses finish reasons now normalize `max_output_tokens` truncation to a
  stable length-style finish reason.

## Fixed

- Responses manual tool-loop replay now preserves required reasoning items for
  reasoning-capable models.
- Responses usage parsing now preserves cached-input and reasoning-token
  accounting.
- Responses background retrieval no longer forces callers to inspect raw SDK
  objects to determine lifecycle status or final completion output.
- MCP approval continuations no longer require raw `mcp_approval_response`
  payload construction at call sites.
- Stored Responses deletion and conversation item lifecycle operations no longer
  require raw OpenAI SDK access at call sites.
- OpenAI moderation, image, speech, fine-tuning, realtime
  connection/call/transcription, generic file workflows, hosted Responses tool workflows, webhook,
  vector-store-file polling/batches, and staged deep-research flows no longer
  require raw OpenAI SDK access at call sites.
- The PostgreSQL cookbook persistence example now degrades to a clean
  infrastructure skip when the configured database is unavailable or
  misconfigured, instead of failing the whole example ring.

## Documentation

- Updated the package reference and package/API guides for:
  - background response lifecycle methods
  - engine-level workflow orchestration methods
  - conversation lifecycle and response compaction methods
  - conversation item lifecycle and stored-response deletion methods
  - MCP approval continuation helper
  - first-class Responses tool descriptors
  - normalized rich Responses output items
  - Responses-first model/provider capability flags
  - the new generic Files API methods and file shared result types
  - the new `BackgroundResponseResult`, `ConversationResource`, `CompactionResult`, `DeletionResult`, `ConversationItemResource`, and `ConversationItemsPage` types
  - the new realtime/webhook/vector-store-file shared result types
  - resumed stream cursor handling
  - first-class OpenAI media, moderation, fine-tuning, realtime, webhook, hosted retrieval, hosted Responses tool workflow methods, and staged deep-research workflow methods
- Updated the OpenAI capability audit through the Phase 6 live re-audit and hardening pass.
- Expanded the cookbook with examples for realtime connection wrapping,
  vector-store file batches, deep-research clarify/rewrite kickoff flows,
  realtime transcription sessions, MCP/connector workflows, and staged
  deep-research orchestration, plus generic OpenAI Files API workflows.

## Validation

The OpenAI/provider-focused regression slice passed after the `1.1.0` changes:

```bash
./.venv/bin/pytest -q \
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

Result: `89 passed`

The broader package regression slice also passed:

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

After the realtime/webhook/deep-research/vector-store-file tranche, the focused
provider/runtime validation slice also passed:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_openai_extended_api_surfaces.py \
  tests/llm_client/test_engine_reliability.py \
  tests/llm_client/test_public_api_namespaces.py \
  tests/llm_client/test_model_catalog.py
```

Result: `49 passed`

The broader provider/runtime slice remained green:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_provider_response_parsing.py \
  tests/llm_client/test_provider_overlap_contracts.py \
  tests/llm_client/test_request_builders.py \
  tests/llm_client/test_public_api_namespaces.py \
  tests/llm_client/test_model_catalog.py \
  tests/llm_client/test_openai_extended_api_surfaces.py \
  tests/llm_client/test_engine_reliability.py
```

Result: `79 passed`

The registry/examples/tool-alias tranche also passed:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_model_catalog.py \
  tests/llm_client/test_request_builders.py \
  tests/llm_client/test_public_api_namespaces.py \
  tests/llm_client/test_openai_extended_api_surfaces.py \
  tests/llm_client/test_engine_reliability.py
```

Result: `57 passed`

The realtime connection / hosted retrieval / deeper-research tranche also
passed its focused runtime validation:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_openai_extended_api_surfaces.py \
  tests/llm_client/test_engine_reliability.py \
  tests/llm_client/test_public_api_namespaces.py
```

Result: `42 passed`

The typed MCP/connector, realtime-transcription, and staged deep-research
tranche also passed the broader provider/runtime slice:

```bash
./.venv/bin/pytest -q \
  tests/llm_client/test_provider_overlap_contracts.py \
  tests/llm_client/test_provider_response_parsing.py \
  tests/llm_client/test_openai_extended_api_surfaces.py \
  tests/llm_client/test_engine_reliability.py \
  tests/llm_client/test_public_api_namespaces.py \
  tests/llm_client/test_request_builders.py \
  tests/llm_client/test_provider_request_translation.py
```

Result: `87 passed`

The live cookbook ring was also completed with the new OpenAI workflow
examples. The only package-level robustness fix required was hardening the
PostgreSQL persistence example so unavailable DB infrastructure becomes a clean
skip under `LLM_CLIENT_EXAMPLE_ALLOW_SKIP=1`. The heavier
`34_end_to_end_mission_control.py` showcase completed successfully once rerun
with a higher per-example timeout, and the final PostgreSQL adaptor examples
(`36`, `37`) also degraded cleanly to infrastructure skips under invalid local
DB credentials.

After the final 2026-04-01 rerun with corrected provider credentials and the
last example fixes in place, the full cookbook sweep completed successfully:

```bash
set -a && source .env && set +a && \
LLM_CLIENT_EXAMPLE_ALLOW_SKIP=1 \
./.venv/bin/python scripts/ci/run_llm_client_examples.py --subset all --timeout-seconds 120
```

Result: completed `52` example scripts for `subset=all` with exit code `0`.

The last cookbook-specific fixes for that successful rerun were:

- extending
  [39_openai_conversation_state_workflow.py](/home/namiral/Projects/Packages/llm-client-v1/examples/39_openai_conversation_state_workflow.py)
  to a `240s` runner timeout
- updating
  [52_openai_files_api.py](/home/namiral/Projects/Packages/llm-client-v1/examples/52_openai_files_api.py)
  so `assistants`-purpose files are not incorrectly treated as downloadable
  content

## Follow-up

- Lower-priority surface now mainly means additional provider-agnostic
  normalization beyond the currently documented stable subset and continued
  live docs re-audits as the local docs API snapshot evolves.
