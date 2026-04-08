# OpenAI Docs-Ledger Completeness Audit

Last updated: 2026-04-08

This report audits `llm_client` against the local OpenAI docs ledger API running on `http://127.0.0.1:8000`, using only the `docs/` and `search/` endpoints as requested.

## Audit method

Sources used:

- `GET /docs/index`
- `GET /docs/export/file/{file_path}`
- `POST /search/query`

Important API notes from this audit:

- The docs-ledger API was reachable from the host namespace but not from the default sandbox namespace. To complete this audit, localhost reads had to be executed outside the sandbox and cached to `/tmp/openai_docs_audit_cache`.
- `GET /docs/index` worked once accessed from the host namespace.
- `POST /search/query` worked and returned reliable `md_relpath` hits.
- `GET /docs/catalog` returned `500 Internal Server Error` during this audit and should be treated as unreliable until fixed.
- Because the docs were cached locally after retrieval, the repo/code cross-check below is based on the actual exported docs corpus, not prior package notes.

Important API notes from the 2026-04-08 refresh:

- `GET /docs/index` and selected `GET /docs/export/file/...` reads are currently reachable from the host namespace, but localhost access is still inconsistent from the default sandbox namespace.
- `GET /docs/export/file/guides/function-calling.md` was readable from the sandbox while `GET /docs/export/file/guides/realtime.md` was not, so the ledger remains useful but operationally inconsistent.
- Official OpenAI docs MCP was used to cross-validate the highest-risk gaps from this refresh:
  - `https://developers.openai.com/api/docs/guides/function-calling#tool-search`
  - `https://developers.openai.com/api/docs/guides/realtime-conversations#text-inputs-and-outputs`
  - `https://developers.openai.com/api/docs/guides/tools-file-search#metadata-filtering`
  - `https://developers.openai.com/api/docs/guides/retrieval#attributes`
  - `https://developers.openai.com/api/docs/guides/retrieval#attribute-filtering`

Relevant docs corpus cached from the ledger included:

- `api-reference.md`
- `models.md`
- `changelog.md`
- `guides/text-generation.md`
- `guides/code-generation.md`
- `guides/migrate-to-responses.md`
- `guides/structured-outputs.md`
- `guides/function-calling.md`
- `guides/conversation-state.md`
- `guides/background.md`
- `guides/prompt-caching.md`
- `guides/reasoning.md`
- `guides/images-vision.md`
- `guides/image-generation.md`
- `guides/audio.md`
- `guides/speech-to-text.md`
- `guides/text-to-speech.md`
- `guides/moderation.md`
- `guides/embeddings.md`
- `guides/fine-tuning.md`
- `guides/distillation.md`
- `guides/supervised-fine-tuning.md`
- `guides/reinforcement-fine-tuning.md`
- `guides/deep-research.md`
- `guides/realtime.md`
- `guides/realtime-server-controls.md`
- `guides/retrieval.md`
- `guides/pdf-files.md`
- `guides/tools.md`
- `guides/tools-file-search.md`
- `guides/tools-web-search.md`
- `guides/tools-code-interpreter.md`
- `guides/tools-image-generation.md`
- `guides/tools-computer-use.md`
- `guides/tools-remote-mcp.md`
- `guides/tools-connectors-mcp.md`
- `guides/agents.md`
- `guides/agents-sdk.md`

## Executive summary

`llm_client` has strong OpenAI support for:

- text generation
- chat completions
- embeddings
- structured outputs
- function and tool calling
- Responses API core flows
- Responses streaming
- background Responses lifecycle
- Conversations API and conversation-state flows
- file/image/audio input transport on the text-generation path
- reasoning controls and reasoning-item replay
- moderation
- direct image generation and editing
- speech-to-text, translation, and text-to-speech
- generic Files API upload, retrieval, listing, content fetch, and deletion
- hosted vector-store CRUD, search, and vector-store file CRUD/content access
- fine-tuning jobs
- webhook verification and event unwrapping
- realtime websocket connection wrappers plus realtime and realtime-transcription session helpers and call-control helpers
- vector-store file polling and vector-store file batches
- hosted Responses tool workflow helpers for web search, file search, code interpreter, remote MCP, and connectors
- typed MCP/connector descriptors with structured approval-policy support
- deep-research clarify/rewrite helpers plus kickoff orchestration and a staged runner over Responses background mode
- expanded OpenAI model registry coverage across newer GPT-4.1, GPT-5 chat/codex, o-series, image, audio, realtime, and deprecated compatibility families
- runnable cookbook coverage for realtime connection, realtime transcription, vector-store file batches, MCP/connector workflows, and deep-research kickoff/staged workflows

`llm_client` does not implement the full OpenAI docs surface.

The biggest missing or incomplete areas are:

- first-class OpenAI `tool_search` support
- first-class OpenAI-specific tool namespaces
- broader realtime product coverage beyond the current websocket/session wrapper, transcription session helper, and call-control helpers
- broader hosted retrieval and file-search resource management beyond vector stores, vector-store files, and file batches
- broader connectors / MCP / skills product coverage beyond typed descriptors and helper workflows
- full OpenAI model registry coverage across the entire docs corpus

## Feature matrix

### Fully implemented or substantially implemented

| Feature area | Docs evidence | Repo evidence | Status | Notes |
| --- | --- | --- | --- | --- |
| Text generation | `guides/text-generation.md`, `api-reference.md` | `llm_client/providers/openai.py` uses `chat.completions.create` and `responses.create` | Implemented | Core completion path exists and is tested. |
| Code generation | `guides/code-generation.md` | Same text-generation path as above | Implemented through generic text generation | No dedicated Codex-specific wrapper, but code output is supported as normal text generation. |
| Structured outputs | `guides/structured-outputs.md`, `guides/function-calling.md` | `llm_client/providers/openai.py`, `llm_client/structured.py` | Implemented | Responses parse path and schema-driven formatting are present. |
| Function calling | `guides/function-calling.md` | `llm_client/providers/openai.py`, `llm_client/tools/base.py` | Implemented | Includes strict-mode handling, tool-choice handling, and tool-loop replay. |
| Responses API | `guides/migrate-to-responses.md`, `api-reference.md` | `llm_client/providers/openai.py` | Implemented | Core request/response translation, parsing, and usage handling are in place. |
| Streaming | `guides/function-calling.md`, `api-reference.md` | `llm_client/providers/openai.py`, `llm_client/providers/types.py` | Implemented | Includes text deltas, reasoning deltas, tool-call deltas, usage, and done events. |
| Background Responses lifecycle | `guides/background.md` | `retrieve_background_response`, `cancel_background_response`, `wait_background_response`, `stream_background_response` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | Includes resume cursor support through `sequence_number`. |
| Conversations and conversation state | `guides/conversation-state.md` | `create/retrieve/update/delete_conversation*` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | Includes conversation items and `/responses/compact`. |
| File inputs | `guides/pdf-files.md` | `llm_client/content.py` | Implemented | Supports `file_id`, `file_url`, file data, and file-path preparation for supported request shapes. |
| Embeddings | `guides/embeddings.md` | `llm_client/providers/openai.py` uses `embeddings.create` | Implemented | Provider embedding path exists. |
| Reasoning controls | `guides/reasoning.md`, `guides/prompt-caching.md`, `guides/function-calling.md` | `llm_client/providers/openai.py`, `llm_client/spec.py`, `llm_client/models.py` | Implemented | Includes reasoning effort, encrypted reasoning include controls, reasoning-token usage, and reasoning-item replay. |
| Moderation | `guides/moderation.md` | `moderate(...)` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | First-class provider and engine moderation workflow is present. |
| Direct image generation/editing | `guides/image-generation.md`, `guides/tools-image-generation.md` | `generate_image(...)` and `edit_image(...)` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | Direct images API wrappers are in place in addition to Responses image-generation tools. |
| Speech and transcription APIs | `guides/audio.md`, `guides/speech-to-text.md`, `guides/text-to-speech.md` | `transcribe_audio(...)`, `translate_audio(...)`, and `synthesize_speech(...)` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | First-class STT, translation, and TTS provider/engine methods are present. |
| Fine-tuning jobs | `guides/fine-tuning.md`, `guides/distillation.md`, `guides/supervised-fine-tuning.md`, `guides/reinforcement-fine-tuning.md` | Fine-tuning job CRUD/list/event methods in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | Job creation, retrieval, cancel, listing, and event listing are covered. |
| Fine-tuned model usage | Fine-tuning guides, `models.md` | `llm_client/providers/base.py`, `llm_client/models.py`, `llm_client/model_catalog.py` | Implemented | Fine-tuned ids are accepted through relaxed OpenAI model inference instead of a fixed profile-only path. |
| Webhooks | `guides/realtime-server-controls.md`, `guides/deep-research.md` | `unwrap_webhook(...)` and `verify_webhook_signature(...)` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Implemented | Verification and parsed event unwrapping are exposed as first-class package APIs. |

### Partially implemented

| Feature area | Docs evidence | Repo evidence | Status | Gap summary |
| --- | --- | --- | --- | --- |
| Images and vision | `guides/images-vision.md`, `guides/images.md` | `llm_client/content.py` supports `ImageBlock` and image inputs; `llm_client/providers/openai.py` normalizes some image-generation outputs | Partial | Image understanding on the text-generation path is supported, but the broader image product surface is not. |
| Audio understanding | `guides/audio.md` | `llm_client/content.py` supports `AudioBlock` / `input_audio`; speech APIs plus realtime transcription session helpers live in `llm_client/providers/openai.py` | Partial | Input-audio transport, speech endpoints, and realtime transcription session/bootstrap helpers are implemented, but the broader realtime-audio product surface is still incomplete. |
| Built-in tools | `guides/tools.md`, `guides/tools-*.md` | `ResponsesBuiltinTool` / `ResponsesMCPTool` in `llm_client/tools/base.py`; workflow helpers in `llm_client/providers/openai.py` and `llm_client/engine.py` | Partial | Typed request descriptors and helper workflows now exist for web search, file search, code interpreter, remote MCP, and connectors, but there are still no full management APIs around every hosted tool family. |
| MCP and connectors | `guides/tools-remote-mcp.md`, `guides/tools-connectors-mcp.md`, `guides/developer-mode.md` | `ResponsesMCPTool`, `ResponsesConnectorId`, `submit_mcp_approval_response(...)`, `respond_with_remote_mcp(...)`, `respond_with_connector(...)` | Partial | Typed remote-MCP and connector request surfaces, documented connector ids, authorization shaping, approval continuations, and helper workflows exist, but broader connector/skills product management remains outside the package. |
| Retrieval / file search | `guides/retrieval.md`, `guides/tools-file-search.md`, `guides/deep-research.md` | `create_file(...)`, `retrieve_file(...)`, `list_files(...)`, `delete_file(...)`, `get_file_content(...)`, `ResponsesBuiltinTool.file_search(...)`, `ResponsesAttributeFilter`, `ResponsesFileSearchRankingOptions`; vector-store CRUD/search, vector-store-file CRUD/content/polling, and vector-store file-batch helpers in `llm_client/providers/openai.py` | Partial | Generic Files API plus hosted vector stores, vector-store files, file batches, typed filters/ranking, and hosted file-search result inclusion are implemented, but broader file-search product/resource management is still incomplete. |
| Agents | `guides/agents.md`, `guides/agents-sdk.md` | `llm_client.agent` package, generic tool runtime, engine | Partial | The package has its own agent layer, but it is not a full implementation of the OpenAI Agents SDK / AgentKit product surface. |
| Realtime API | `guides/realtime.md`, `guides/realtime-server-controls.md`, `guides/realtime-transcription.md`, `api-reference.md` | `connect_realtime(...)`, `connect_realtime_transcription(...)`, realtime and realtime-transcription session helpers, and call helpers in `llm_client/providers/openai.py` and `llm_client/engine.py` | Partial | Stable websocket/session bootstrap is wrapped for both standard realtime and transcription flows, but the full Realtime product surface still extends beyond the current helper set. |
| Deep Research | `guides/deep-research.md` | `clarify_deep_research_task(...)`, `rewrite_deep_research_prompt(...)`, `start_deep_research(...)`, and `run_deep_research(...)` in `llm_client/providers/openai.py` and `llm_client/engine.py` | Partial | Clarify, rewrite, kickoff, optional background wait, and typed MCP/connectors are implemented, but the broader deep-research lifecycle/product surface is still incomplete. |
| Model registry | `models.md`, `index.md` models section | `llm_client/assets/model_catalog.json`, `llm_client/models.py`, `llm_client/model_catalog.py` | Partial | The registry now includes a much broader OpenAI set, including GPT-4.1, GPT-5 chat/codex variants, o-series reasoning families, image/audio/realtime families, and deprecated compatibility entries, but it still does not cover the full docs-ledger model corpus. |

### Missing

| Feature area | Docs evidence | Repo evidence | Status | Gap summary |
| --- | --- | --- | --- | --- |
| Skills as an OpenAI product surface | `guides/tools-connectors-mcp.md`, `guides/developer-mode.md`, broader docs index | No dedicated “skills” product abstractions in `llm_client` | Missing | The package has its own tool abstractions, but not OpenAI “skills/connectors” product coverage. |
| Full model coverage | `models.md`, docs index models section | Registry still omits several documented families and legacy/current variants | Missing | Many documented OpenAI models are still absent, especially the broader legacy/current catalog outside the newly added moderation, audio, image, realtime, and deep-research families. |

### Newly implemented in 1.2 work

| Feature area | Docs evidence | Repo evidence | Status | Notes |
| --- | --- | --- | --- | --- |
| OpenAI `tool_search` | Official docs MCP `guides/function-calling#tool-search` | `ResponsesToolSearch` in `llm_client/tools/base.py`; `respond_with_tool_search(...)` and `submit_tool_search_output(...)` in `llm_client/providers/openai.py` | Implemented | The package now supports hosted and client-executed `tool_search` as an OpenAI-specific advanced surface. |
| OpenAI-specific tool namespaces | Official docs MCP `guides/function-calling#defining-namespaces` | `ResponsesToolNamespace` and `ResponsesFunctionTool` in `llm_client/tools/base.py`; recursive tool aliasing/normalization in `llm_client/providers/openai.py` | Implemented | Namespace tool definitions now preserve deferred-tool intent instead of flattening everything to raw dict passthrough. |
| Retrieval tuning ergonomics | Official docs MCP `guides/tools-file-search#metadata-filtering`, `guides/retrieval#attribute-filtering`, `guides/retrieval#relevance-tuning` | `ResponsesAttributeFilter`, `ResponsesFileSearchRankingOptions`, and `ResponsesFileSearchHybridWeights` in `llm_client/tools/base.py`; `search_vector_store(...)` and `respond_with_file_search(...)` in `llm_client/providers/openai.py` | Implemented | The package now exposes typed retrieval filters/ranking, explicit `max_num_results`/`rewrite_query` controls, and `include_search_results=True` for hosted file-search responses. |

## Correctness notes for implemented areas

The implemented core areas are not just stubs. There is meaningful correctness coverage for the OpenAI Responses surface:

- request translation tests in `tests/llm_client/test_provider_request_translation.py`
- response parsing tests in `tests/llm_client/test_provider_response_parsing.py`
- engine workflow tests in `tests/llm_client/test_engine_reliability.py`
- request-spec serialization tests in `tests/llm_client/test_request_builders.py`
- public export overlap tests in `tests/llm_client/test_public_api_namespaces.py`

This means the package’s current OpenAI strengths are real:

- Responses request translation
- tool-call history replay
- conversation chaining and conversation resources
- compaction
- resumed background streams
- strict/default tool behavior
- normalized rich output items for the supported subset

## Key evidence and reasoning

### Evidence that the package is narrower than the docs corpus

- The docs-ledger `index.md` includes guides and models for image generation, speech-to-text, text-to-speech, moderation, fine-tuning, realtime, deep research, and many more model families.
- The OpenAI model registry in `llm_client/assets/model_catalog.json` now covers a much broader slice of the docs-ledger model corpus, but still not all documented families and variants.
- `llm_client/providers/openai.py` still does not expose the full product family set from the docs, especially broader hosted file-search resources beyond the generic Files API/vector-store layers, fuller connector/skills product management, and several newer platform families.

### Evidence that some product families are only partial

- `llm_client/tools/base.py` exposes `ResponsesBuiltinTool` descriptors for hosted tool families, but those are still request-shape abstractions, not complete product-management wrappers.
- `llm_client/providers/openai.py` normalizes `file_search_call`, `web_search_call`, `code_interpreter_call`, `image_generation_call`, and MCP-related output items, which proves hosted-tool awareness.
- That same file now exposes direct image APIs, speech APIs, moderation, fine-tuning jobs, vector stores, vector-store files, vector-store file batches, webhook helpers, realtime connection/call/transcription helpers, hosted Responses tool workflow helpers, and staged deep-research helpers, but it does not yet expose the full remaining hosted-resource or broader product-management surface.

### Evidence for the remaining 1.2 gaps

- The official realtime conversations docs explicitly enumerate `conversation.item.added` and `conversation.item.done` lifecycle events; the package currently parses Responses streaming events well, but it does not yet model that broader Realtime event surface as a first-class package contract.
- The official file-search and retrieval docs still extend beyond the current package surface into deeper hosted resource management and broader product lifecycle flows.

### Evidence that fine-tuned models are not a supported surface

- The provider contract was widened so OpenAI fine-tuned model ids can now resolve without predeclaring every `ft:` model key in `llm_client/models.py`.
- The remaining registry problem is breadth, not whether fine-tuned ids are accepted at all.

## Gap priority

### Priority 0: major remaining platform families

- broader realtime product coverage beyond the current websocket/session wrapper
- broader hosted retrieval / file-search resource management

### Priority 1: breadth and product-surface expansion

- broader built-in OpenAI tool product coverage beyond request descriptors and helper workflows
- deeper connectors / MCP / skills coverage

### Priority 2: completeness and ergonomics

- model registry expansion to match the docs corpus
- additional examples and tests for the new product families once implemented

## Recommended implementation plan

### Phase 1: Add the newly confirmed tool-surface gaps

Completed in this branch:

- OpenAI-specific advanced support for `tool_search`.
- OpenAI-specific tool namespace support without promoting namespaces into the stable generic tool contract.
- Typed retrieval/file-search filters, ranking controls, and hosted search-result inclusion helpers.

The next active phase is broader Realtime and product-surface follow-up work.

### Phase 2: Close the remaining research and retrieval gaps

Deliverables:

- Expand hosted retrieval/file-search management beyond vector stores, files, file batches, and the newly added tuning ergonomics.
- Add examples covering realtime transcription, connector/MCP helpers, and the staged deep-research workflow.

Exit criteria:

- The package can represent the docs-ledger deep-research and hosted retrieval workflows further end to end, not just the setup calls.

### Phase 3: Expand broader realtime and hosted-tool coverage

Deliverables:

- Expand the Realtime surface beyond the current websocket/server wrapper where the SDK and docs expose stable server-side controls.
- Add higher-level helpers/examples around hosted OpenAI tools where the docs expose stable workflows.

Exit criteria:

- Realtime and hosted-tool workflows from the docs can be expressed without dropping to raw SDK access.

### Phase 4: Continue model-registry and docs parity expansion

Deliverables:

- Expand `llm_client/assets/model_catalog.json` and `llm_client/models.py` to cover more of the docs-ledger model corpus.
- Re-run the docs-ledger audit and close the remaining “partial” entries where justified.

Exit criteria:

- The registry and feature matrix no longer materially lag the docs-ledger on supported model families.

## Status update after implementation

This audit triggered a follow-up implementation tranche in the package.

Completed from this report:

- first-class OpenAI moderation via `provider.moderate(...)` and `engine.moderate(...)`
- first-class direct image generation/editing via `provider.generate_image(...)`, `provider.edit_image(...)`, and engine mirrors
- first-class speech-to-text and translation via `provider.transcribe_audio(...)`, `provider.translate_audio(...)`, and engine mirrors
- first-class text-to-speech via `provider.synthesize_speech(...)` and the engine mirror
- first-class hosted vector-store CRUD/search via provider and engine methods
- first-class fine-tuning job create/retrieve/cancel/list/events via provider and engine methods
- realtime transcription session creation plus realtime-transcription connection wrappers
- typed MCP/connector descriptors with approval-policy shaping plus hosted workflow helpers for web search, file search, code interpreter, remote MCP, and connectors
- staged deep-research orchestration via `run_deep_research(...)`, including optional clarification, rewrite, kickoff, and background wait
- model-registry expansion for GPT-4.1, GPT-5 chat/codex variants, o-series reasoning families, image, audio, moderation, realtime, deep-research, and deprecated compatibility model entries
- fine-tuned model-id acceptance through dynamic `ft:` / `ft-` model profile synthesis
- cookbook examples for realtime connection wrapping, realtime transcription, vector-store file batches, MCP/connector workflows, and deep-research kickoff/staged flows

Validation status after the implementation tranche:

- broader provider/runtime slice: `87 passed`

## Remaining gaps

Still missing or only partial after the implementation tranche:

- broader Realtime API product coverage beyond the current websocket/session bootstrap, transcription-session helper, and call-control helpers
- hosted file-search resource management beyond vector-store primitives
- broader OpenAI-provided tool product coverage beyond the existing Responses-native descriptors and helper workflows
- broader model-registry expansion for the rest of the docs corpus, including newer and legacy families still omitted from the ledger-backed registry
- first-class wrappers for some newer product families in the docs corpus such as videos / Sora and broader skills/connectors management surfaces

## Immediate next tasks

1. Expand hosted retrieval/file-search resource management beyond the current vector-store surface.
2. Deepen connector / MCP / skills coverage beyond typed descriptors and helper workflows.
3. Broaden the Realtime product surface beyond the current wrapper/session helpers.
4. Re-run the docs-ledger audit after each tranche and keep the registry aligned with the documented model set.

## Bottom line

The package is in good shape for OpenAI text-generation and Responses-native agentic workflows, including typed hosted-tool helpers, staged deep-research flows, and realtime transcription bootstrap.

It is not complete against the OpenAI docs corpus exposed by the docs ledger API.

The current implementation should be described as:

- strong on text/Responses/tooling/conversation flows
- partial on multimodal and hosted-tool product families
- missing several major OpenAI platform product surfaces entirely
