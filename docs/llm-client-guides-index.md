# llm-client Guides and Cookbook Index

This index ties the standalone package documentation to the runnable cookbook
examples under [`examples/`](../examples/README.md).

## Core Reference

- Architecture overview:
  [llm-client-architecture.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-architecture.md)
- Public API map:
  [llm-client-public-api-v1.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-public-api-v1.md)
- Package reference:
  [llm_client/README.md](/home/namiral/Projects/Packages/llm-client-v1/llm_client/README.md)
- Package API guide:
  [llm-client-package-api-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-package-api-guide.md)
- Build and recipes guide:
  [llm-client-build-and-recipes-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-build-and-recipes-guide.md)
- Usage and capabilities guide:
  [llm-client-usage-and-capabilities-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-usage-and-capabilities-guide.md)

## Guides

- Service adaptors:
  [llm-client-service-adaptors-design.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-service-adaptors-design.md)
- Provider setup:
  [llm-client-provider-setup-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-provider-setup-guide.md)
- Routing and failover:
  [llm-client-routing-and-failover-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-routing-and-failover-guide.md)
- Tool runtime:
  [llm-client-tool-runtime-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-tool-runtime-guide.md)
- Structured outputs:
  [llm-client-structured-outputs-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-structured-outputs-guide.md)
- Context and memory:
  [llm-client-context-and-memory-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-context-and-memory-guide.md)
- Observability and redaction:
  [llm-client-observability-and-redaction-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-observability-and-redaction-guide.md)
- Migration from direct SDK usage:
  [llm-client-migration-from-direct-sdk-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-migration-from-direct-sdk-guide.md)

## Packaging and Release

- Installation matrix:
  [llm-client-installation-matrix.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-installation-matrix.md)
- Packaging readiness review:
  [llm-client-packaging-readiness.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-packaging-readiness.md)
- Changelog process:
  [llm-client-changelog-process.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-changelog-process.md)
- Semantic versioning policy:
  [llm-client-semver-policy.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-semver-policy.md)
- Support policy:
  [llm-client-support-policy.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-support-policy.md)
- Release automation:
  [llm-client-release-automation.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-release-automation.md)
- 1.1.0 release notes:
  [llm-client-release-notes-1.1.0.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-release-notes-1.1.0.md)
- 2026-04-01 evaluation report:
  [llm-client-evaluation-report-2026-04-01.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-evaluation-report-2026-04-01.md)
- 1.0.0 release notes:
  [llm-client-release-notes-1.0.0.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-release-notes-1.0.0.md)

## Cookbook Scripts

- Cookbook entry point:
  [examples/README.md](/home/namiral/Projects/Packages/llm-client-v1/examples/README.md)
- Examples guide:
  [llm-client-examples-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-examples-guide.md)
- One-shot completion:
  [01_one_shot_completion.py](/home/namiral/Projects/Packages/llm-client-v1/examples/01_one_shot_completion.py)
- Streaming:
  [02_streaming.py](/home/namiral/Projects/Packages/llm-client-v1/examples/02_streaming.py)
- Embeddings:
  [03_embeddings.py](/home/namiral/Projects/Packages/llm-client-v1/examples/03_embeddings.py)
- Content blocks and envelopes:
  [04_content_blocks.py](/home/namiral/Projects/Packages/llm-client-v1/examples/04_content_blocks.py)
- Structured extraction:
  [05_structured_extraction.py](/home/namiral/Projects/Packages/llm-client-v1/examples/05_structured_extraction.py)
- Routing and failover:
  [06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/llm-client-v1/examples/06_provider_registry_and_routing.py)
- Engine cache, retry, and idempotency:
  [07_engine_cache_retry_idempotency.py](/home/namiral/Projects/Packages/llm-client-v1/examples/07_engine_cache_retry_idempotency.py)
- Tool execution modes:
  [08_tool_execution_modes.py](/home/namiral/Projects/Packages/llm-client-v1/examples/08_tool_execution_modes.py)
- Tool-calling agent:
  [09_tool_calling_agent.py](/home/namiral/Projects/Packages/llm-client-v1/examples/09_tool_calling_agent.py)
- Context and memory planning:
  [10_context_memory_planning.py](/home/namiral/Projects/Packages/llm-client-v1/examples/10_context_memory_planning.py)
- Observability and redaction:
  [11_observability_and_redaction.py](/home/namiral/Projects/Packages/llm-client-v1/examples/11_observability_and_redaction.py)
- Benchmarks:
  [12_benchmarks.py](/home/namiral/Projects/Packages/llm-client-v1/examples/12_benchmarks.py)
- Batch processing:
  [13_batch_processing.py](/home/namiral/Projects/Packages/llm-client-v1/examples/13_batch_processing.py)
- Sync wrappers:
  [14_sync_wrappers.py](/home/namiral/Projects/Packages/llm-client-v1/examples/14_sync_wrappers.py)
- Rate limiting:
  [15_rate_limiting.py](/home/namiral/Projects/Packages/llm-client-v1/examples/15_rate_limiting.py)
- OpenAI background Responses lifecycle:
  [38_openai_background_responses.py](/home/namiral/Projects/Packages/llm-client-v1/examples/38_openai_background_responses.py)
- OpenAI conversation state workflow:
  [39_openai_conversation_state_workflow.py](/home/namiral/Projects/Packages/llm-client-v1/examples/39_openai_conversation_state_workflow.py)
- OpenAI normalized output items:
  [40_openai_normalized_output_items.py](/home/namiral/Projects/Packages/llm-client-v1/examples/40_openai_normalized_output_items.py)
- FastAPI SSE:
  [16_fastapi_sse.py](/home/namiral/Projects/Packages/llm-client-v1/examples/16_fastapi_sse.py)
- Persistence repository:
  [17_persistence_repository.py](/home/namiral/Projects/Packages/llm-client-v1/examples/17_persistence_repository.py)
- Memory-backed assistant:
  [18_memory_backed_assistant.py](/home/namiral/Projects/Packages/llm-client-v1/examples/18_memory_backed_assistant.py)
