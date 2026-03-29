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

## Notes

- The examples now use real provider calls through
  [cookbook_support.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/cookbook_support.py).
- By default, the cookbook expects `OPENAI_API_KEY` and uses OpenAI models.
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
