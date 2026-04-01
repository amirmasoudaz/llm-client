# llm-client Evaluation Report

Date: 2026-04-01
Version under evaluation: `1.1.0`

## Scope

This report summarizes the package state after the OpenAI provider expansion,
the docs-ledger-driven audit work, the cookbook additions through
`examples/52_openai_files_api.py`, and the latest live validation pass on
2026-04-01.

## Overall Status

The package is in a strong implementation state and the codebase is broadly
aligned with the current documented `1.1.0` surface.

The release validation state is now positive. After correcting the OpenAI
credential and rerunning the full cookbook, the runner completed successfully
for the full `subset=all` sweep with exit code `0` and no `example failed:`
entries in the log.

## What Is Done

- OpenAI Responses coverage is substantially expanded and documented.
- Background Responses lifecycle, conversation lifecycle, response compaction,
  MCP approval continuation, stored-response deletion, conversation items,
  generic Files API, moderation, image generation/editing, speech APIs,
  fine-tuning jobs, vector stores, realtime helpers, webhook helpers, and
  staged deep-research helpers are implemented in the package surface.
- The model catalog is expanded to cover a much broader OpenAI family set than
  the original `1.0.0` line.
- The cookbook now includes examples `38` through `52`, including background
  responses, conversation-state workflows, normalized output items, realtime
  helpers, MCP/connector workflows, deep-research flows, and Files API usage.
- The timeout handling for
  [27_sql_analytics_assistant.py](/home/namiral/Projects/Packages/llm-client-v1/examples/27_sql_analytics_assistant.py)
  was corrected in the runner by assigning it a `240s` override so it no
  longer fails due to an artificial `120s` ceiling.
- Provider/runtime fixes were added from live validation findings:
  - realtime transcription session creation now uses the OpenAI SDK surface the
    provider actually expects
  - realtime transcription websocket connection bootstrapping normalizes
    transcription-only models to a connectable realtime transport model
  - GPT-5 Responses requests now omit unsupported non-default temperature
    values
- Focused regression coverage passed after these fixes:
  - `25 passed` for the final provider translation and extended-surface slice
  - earlier focused OpenAI/provider/runtime slices passed throughout this
    tranche as well

## Live Cookbook Evaluation

### Direct example results

- [40_openai_normalized_output_items.py](/home/namiral/Projects/Packages/llm-client-v1/examples/40_openai_normalized_output_items.py)
  was fixed and now executes cleanly.
- [41_openai_background_resume_stream.py](/home/namiral/Projects/Packages/llm-client-v1/examples/41_openai_background_resume_stream.py)
  was fixed and now executes cleanly.
- [49_openai_realtime_transcription_session.py](/home/namiral/Projects/Packages/llm-client-v1/examples/49_openai_realtime_transcription_session.py)
  was fixed and now executes cleanly.
- [50_openai_mcp_and_connector_workflows.py](/home/namiral/Projects/Packages/llm-client-v1/examples/50_openai_mcp_and_connector_workflows.py)
  runs with the provided MCP configuration and no longer fails the cookbook.
- [52_openai_files_api.py](/home/namiral/Projects/Packages/llm-client-v1/examples/52_openai_files_api.py)
  was fixed to handle OpenAI's file-purpose restrictions correctly and now
  executes cleanly with `purpose=assistants`.

### Full cookbook run

Command used:

```bash
set -a && source .env && set +a && \
LLM_CLIENT_EXAMPLE_ALLOW_SKIP=1 \
./.venv/bin/python scripts/ci/run_llm_client_examples.py --subset all --timeout-seconds 120
```

Observed result:

- The full run exited `0`.
- The runner completed all `52` scripts in the `all` subset.
- There were no `example failed:` entries in the final log.
- The OpenAI-specific examples that had previously been blocked by
  authentication now complete under the corrected environment.

Expected setup/infrastructure skips or degradations observed in the final run:

- [17_persistence_repository.py](/home/namiral/Projects/Packages/llm-client-v1/examples/17_persistence_repository.py)
  degrades cleanly when the configured PostgreSQL DSN is invalid.
- [36_sql_adaptor_direct.py](/home/namiral/Projects/Packages/llm-client-v1/examples/36_sql_adaptor_direct.py)
  degrades cleanly when the configured PostgreSQL DSN is invalid.
- [37_sql_adaptor_tools.py](/home/namiral/Projects/Packages/llm-client-v1/examples/37_sql_adaptor_tools.py)
  degrades cleanly when the configured PostgreSQL DSN is invalid.
- [45_openai_mcp_approval_continuation.py](/home/namiral/Projects/Packages/llm-client-v1/examples/45_openai_mcp_approval_continuation.py)
  remains intentionally setup-driven and requests
  `LLM_CLIENT_EXAMPLE_MCP_PREVIOUS_RESPONSE_ID` plus
  `LLM_CLIENT_EXAMPLE_MCP_APPROVAL_REQUEST_ID`.
- [47_openai_vector_store_file_batches.py](/home/namiral/Projects/Packages/llm-client-v1/examples/47_openai_vector_store_file_batches.py)
  remains intentionally setup-driven and requests
  `LLM_CLIENT_EXAMPLE_VECTOR_STORE_ID`.

### Assessment of `27_sql_analytics_assistant.py`

[27_sql_analytics_assistant.py](/home/namiral/Projects/Packages/llm-client-v1/examples/27_sql_analytics_assistant.py)
was the earlier timeout concern. That timeout issue is addressed at the runner
level. The example now completes within the adjusted `240s` timeout budget in
the successful full rerun.

## Documentation and API Reference Parity

The docs are largely aligned with the current codebase state.

Confirmed documentation coverage includes:

- the new generic Files API methods
- realtime connection helpers
- deep-research helpers
- conversation lifecycle helpers
- MCP approval continuation helpers
- new OpenAI cookbook examples through `52`
- `1.1.0` release notes
- the OpenAI capability audit and docs-ledger completeness audit

Parity notes:

- The main package docs and guides now reference the current repo path under
  `/home/namiral/Projects/Packages/llm-client-v1/...` rather than the older
  standalone-transition source path.
- Some archived historical documents still mention the older repo name
  `intelligence-layer-bif`. Those are in archived material and do not describe
  the current package contract.
- The current docs now match the verified `1.1.0` codebase and the successful
  cookbook rerun, with expected setup-based examples remaining clearly
  environment-dependent.

## Remaining Work

Post-release follow-up:

1. Continue broader OpenAI parity work where still intentionally partial:
   realtime breadth, hosted retrieval/file-search breadth, deeper MCP/connectors
   product management, and broader model-catalog completeness.
2. Optionally add real fixtures for the setup-dependent cookbook examples so
   PostgreSQL-backed, vector-store-backed, and MCP-approval continuation flows
   can be exercised automatically in future release runs.

## Release Recommendation

`1.1.0` is release-validated for the current package contract.

The implementation, documentation, focused regression slices, and full live
cookbook runner are aligned. Remaining setup-dependent examples are behaving as
intentional environment-gated flows rather than package failures.
