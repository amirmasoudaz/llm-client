# llm_client Cookbook

These examples are the runnable cookbook for the standalone `llm_client`
package.

Design goals:

- real provider calls for all LLM-facing examples
- fail fast when required credentials or services are missing
- one script per major capability
- a few combined flows that look like real application designs
- optional dependency or infrastructure examples fail with clear setup guidance

Run examples from the repository root:

```bash
python examples/01_one_shot_completion.py
```

Or run the full live cookbook ring:

```bash
python scripts/ci/run_llm_client_examples.py --subset all
```

Run only the core capability ring:

```bash
python scripts/ci/run_llm_client_examples.py --subset core
```

Run only the application-shaped examples:

```bash
python scripts/ci/run_llm_client_examples.py --subset application
```

The default runner tracks the currently validated cookbook ring. Newly added
advanced provider-specific examples may appear in this index before they are
promoted into that default ring, and can be run directly by filename.

## Core Capability Examples

- `01_one_shot_completion.py`: direct provider completion
- `02_streaming.py`: token streaming and final stream result handling
- `03_embeddings.py`: embedding generation through the engine
- `04_content_blocks.py`: content blocks, envelopes, and content projection
- `05_structured_extraction.py`: schema validation, repair loop, diagnostics
- `06_provider_registry_and_routing.py`: provider registry, capability lookup,
  routing, and failover
- `07_engine_cache_retry_idempotency.py`: retries, cache hits, idempotency, and
  engine diagnostics
- `08_tool_execution_modes.py`: single, sequential, and parallel tool execution
- `09_tool_calling_agent.py`: multi-turn agent with tool calling
- `10_context_memory_planning.py`: memory retrieval, summaries, and context
  planning
- `11_observability_and_redaction.py`: diagnostics, lifecycle reports, metrics,
  and redaction
- `12_benchmarks.py`: deterministic benchmark harness and saved report
- `13_batch_processing.py`: engine batch execution and batch manager
- `14_sync_wrappers.py`: sync access to conversation and summarization helpers
- `15_rate_limiting.py`: token/request limiter usage
- `35_file_block_transport.py`: canonical file preparation, native OpenAI
  responses transport, and explicit fallback behavior for non-native providers
- `38_openai_background_responses.py`: engine-managed OpenAI background
  response lifecycle with polling and deletion
- `39_openai_conversation_state_workflow.py`: engine-managed OpenAI
  conversation creation, item listing, and context compaction
- `40_openai_normalized_output_items.py`: normalized `output_items` versus
  low-level provider replay items across a Responses tool loop
- `41_openai_background_resume_stream.py`: background stream attach/reconnect
  with `sequence_number` resume support
- `42_openai_prompt_cache_and_encrypted_reasoning.py`: first-class prompt
  caching controls and encrypted reasoning continuity inspection
- `43_openai_long_running_compaction.py`: longer conversation threads,
  compaction, and item retrieval against stored OpenAI state
- `46_openai_realtime_connection_wrapper.py`: realtime client-secret creation
  and websocket connection wrapper usage
- `47_openai_vector_store_file_batches.py`: vector-store file batches, polling,
  and batch file listing
- `48_openai_deep_research_clarify_rewrite.py`: deep-research clarification,
  prompt rewrite, and kickoff flow
- `49_openai_realtime_transcription_session.py`: realtime transcription
  session creation and transcription websocket connection wrapper usage
- `50_openai_mcp_and_connector_workflows.py`: hosted web-search plus typed
  remote MCP / connector workflow helpers
- `51_openai_run_deep_research_staged.py`: staged deep-research orchestration
  with clarification, rewrite, kickoff, and optional wait-for-completion
- `52_openai_files_api.py`: generic OpenAI Files API upload, retrieval,
  listing, content fetch, and optional cleanup
- `53_openai_realtime_conversation_lifecycle.py`: realtime text-turn
  lifecycle with `create_text_message(...)`, `create_response(...)`, and typed
  event waiting
- `54_openai_tool_search_and_namespaces.py`: advanced OpenAI `tool_search`
  plus namespaced deferred tools and optional `submit_tool_search_output(...)`
  continuation
- `55_openai_uploads_api.py`: OpenAI Uploads API lifecycle with create, part
  upload, completion, cancellation, and chunked-upload helper coverage
- `56_openai_realtime_output_collection.py`: realtime text-turn output
  collection via `collect_response_output(...)`
- `57_openai_realtime_push_to_talk.py`: optional realtime push-to-talk helper
  flow with `disable_vad(...)` and `send_audio_turn(...)`

## Combined / Application-Shaped Examples

- `16_fastapi_sse.py`: FastAPI streaming endpoint built on `llm_client`
- `17_persistence_repository.py`: persistence repository dry run and safety
  checks
- `18_memory_backed_assistant.py`: context planning + memory + engine response
- `19_multi_provider_failover_gateway.py`: injected failure + router fallback
  + gateway diagnostics
- `20_rag_with_citations.py`: Qdrant-backed retrieval + citations + grounded
  answer generation
- `21_document_review_diff.py`: draft diffing + structured review + approval
  framing
- `22_human_in_the_loop_approvals.py`: approval checkpoint + memory-backed
  revision loop
- `23_async_job_queue_sse.py`: FastAPI job queue with polling + SSE progress
- `24_customer_support_copilot.py`: Qdrant-backed support copilot with
  retrieval + structured routing
- `25_incident_war_room_assistant.py`: agentic war-room workflow with tools and
  live synthesis
- `26_research_briefing_agent.py`: Qdrant-backed research briefing workflow
- `27_sql_analytics_assistant.py`: NL-to-SQL drafting with safety checks
- `28_release_readiness_control_plane.py`: structured go/no-go control-plane
  decision
- `29_multimodal_intake_pipeline.py`: multimodal content projection +
  intake-brief generation
- `30_eval_and_regression_gate.py`: live evaluation suite with ship/hold gate
- `31_tool_calling_with_partial_failures.py`: graceful degradation around
  partial and failed tool calls
- `32_cache_strategy_showdown.py`: no-cache vs FS vs Qdrant vs idempotency
  comparison
- `33_compliance_redaction_pipeline.py`: safe payload handling + audit artifact
  generation
- `34_end_to_end_mission_control.py`: full-stack incident/release mission
  control showcase across routing, context, tools, redaction, replay, cache,
  and evaluation-minded decisioning
- `36_sql_adaptor_direct.py`: direct PostgreSQL adaptor usage with read-only
  querying, safety enforcement, and explicit write enablement on a temporary
  table
- `37_sql_adaptor_tools.py`: live tool-calling agent using a read-only SQL
  adaptor tool against temporary incident data in PostgreSQL
- `44_engine_orchestrated_openai_workflow.py`: engine-level orchestration
  across conversation state, background responses, follow-up turns, and
  compaction
- `45_openai_mcp_approval_continuation.py`: continue a stored MCP approval
  loop with a first-class approval-response helper

## Notes

- The examples now use real provider calls through
  [cookbook_support.py](/home/namiral/Projects/Packages/llm-client-v1/examples/cookbook_support.py).
- By default, the cookbook expects `OPENAI_API_KEY` and uses OpenAI models.
- The OpenAI Responses lifecycle/state examples (`38`-`45`) expect
  `LLM_CLIENT_EXAMPLE_PROVIDER=openai`.
- Additional OpenAI capability examples use:
  - `LLM_CLIENT_EXAMPLE_REALTIME_MODEL` for example `46`
  - `LLM_CLIENT_EXAMPLE_REALTIME_TRANSCRIPTION_MODEL` for example `49`
  - `LLM_CLIENT_EXAMPLE_VECTOR_STORE_ID`,
    `LLM_CLIENT_EXAMPLE_VECTOR_STORE_FILE_IDS`, and/or
    `LLM_CLIENT_EXAMPLE_VECTOR_STORE_UPLOAD_PATHS` for example `47`
  - `LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_MODEL` and
    `LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_PROMPT` for examples `48` and `51`
  - `LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_CLARIFICATIONS` and optionally
    `LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_WAIT=0|1` for example `51`
  - `LLM_CLIENT_EXAMPLE_OPENAI_TOOLS_MODEL` for example `50`
  - `LLM_CLIENT_EXAMPLE_MCP_SERVER_URL`,
    `LLM_CLIENT_EXAMPLE_MCP_SERVER_LABEL`,
    `LLM_CLIENT_EXAMPLE_MCP_AUTHORIZATION`,
    `LLM_CLIENT_EXAMPLE_MCP_REQUIRE_APPROVAL`,
    `LLM_CLIENT_EXAMPLE_CONNECTOR_ID`,
    `LLM_CLIENT_EXAMPLE_CONNECTOR_LABEL`,
    `LLM_CLIENT_EXAMPLE_CONNECTOR_AUTHORIZATION`, and
    `LLM_CLIENT_EXAMPLE_CONNECTOR_REQUIRE_APPROVAL` for example `50`
  - `LLM_CLIENT_EXAMPLE_UPLOAD_FILE_PATH` for example `52`
  - optionally `LLM_CLIENT_EXAMPLE_FILE_PURPOSE` and
    `LLM_CLIENT_EXAMPLE_KEEP_UPLOADED_FILE=0|1` for example `52`
- The MCP approval continuation example also expects:
  - `LLM_CLIENT_EXAMPLE_MCP_PREVIOUS_RESPONSE_ID`
  - `LLM_CLIENT_EXAMPLE_MCP_APPROVAL_REQUEST_ID`
  - optionally `LLM_CLIENT_EXAMPLE_MCP_APPROVE=0|1`
  - and optionally the same connector / remote-MCP env vars used by example
    `50` when approval continuation needs to resend an auth-bearing tool
    definition
- Example `53` reuses `LLM_CLIENT_EXAMPLE_REALTIME_MODEL`.
- Example `54` reuses `LLM_CLIENT_EXAMPLE_OPENAI_TOOLS_MODEL`.
- Example `55` reuses `LLM_CLIENT_EXAMPLE_UPLOAD_FILE_PATH`.
- Example `56` reuses `LLM_CLIENT_EXAMPLE_REALTIME_MODEL`.
- Example `57` reuses `LLM_CLIENT_EXAMPLE_REALTIME_MODEL` and expects
  `LLM_CLIENT_EXAMPLE_REALTIME_AUDIO_PATH`.
- You can switch providers with:
  - `LLM_CLIENT_EXAMPLE_PROVIDER=openai|anthropic|google`
  - `LLM_CLIENT_EXAMPLE_MODEL=...`
  - `LLM_CLIENT_EXAMPLE_SECONDARY_PROVIDER=...`
  - `LLM_CLIENT_EXAMPLE_SECONDARY_MODEL=...`
  - `LLM_CLIENT_EXAMPLE_EMBEDDINGS_PROVIDER=...`
  - `LLM_CLIENT_EXAMPLE_EMBEDDINGS_MODEL=...`
- The persistence example also requires `LLM_CLIENT_EXAMPLE_PG_DSN`.
- The SQL adaptor examples also require `LLM_CLIENT_EXAMPLE_PG_DSN`.
- SQL adaptor examples require the optional PostgreSQL extra:
  - `pip install llm-client[postgres]`
- The retrieval/cache examples that use Qdrant require:
  - `QDRANT_URL=http://127.0.0.1:6333`
  - optionally `QDRANT_API_KEY=...`
- The FastAPI app examples also require the optional FastAPI/uvicorn
  dependencies to be installed.
