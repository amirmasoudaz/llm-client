# llm-client Provider Setup Guide

This guide covers the provider side of `llm_client`: direct provider usage,
provider registry setup, and the practical environment setup for the live
cookbook examples.

Runnable examples:

- [01_one_shot_completion.py](/home/namiral/Projects/Packages/llm-client-v1/examples/01_one_shot_completion.py)
- [02_streaming.py](/home/namiral/Projects/Packages/llm-client-v1/examples/02_streaming.py)
- [03_embeddings.py](/home/namiral/Projects/Packages/llm-client-v1/examples/03_embeddings.py)
- [06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/llm-client-v1/examples/06_provider_registry_and_routing.py)

## Stable imports

For new projects, prefer:

```python
from llm_client.providers import OpenAIProvider, AnthropicProvider, GoogleProvider
from llm_client.providers.types import BackgroundResponseResult, Message, StreamEventType
from llm_client.engine import ExecutionEngine
```

## Direct provider usage

Use a provider directly when you want the thinnest path:

- one-shot completion
- raw streaming events
- embeddings
- provider-specific debugging
- provider-native lifecycle workflows such as OpenAI background Responses retrieval or cancellation

Minimal shape:

```python
provider = OpenAIProvider(model="gpt-5-mini")
result = await provider.complete([Message.user("Hello")])
```

## Engine-backed usage

Use `ExecutionEngine` when you want:

- retries
- failover
- cache
- hooks/diagnostics
- idempotency
- normalized error handling

Minimal shape:

```python
engine = ExecutionEngine(provider=OpenAIProvider(model="gpt-5-mini"))
result = await engine.complete(
    RequestSpec(provider="openai", model="gpt-5-mini", messages=[Message.user("Hello")])
)
```

## OpenAI / Anthropic / Google

The standalone package supports all three provider families through the same
provider contract.

Typical construction:

```python
OpenAIProvider(model="gpt-5-mini", api_key="...")
AnthropicProvider(model="claude-sonnet-4", api_key="...")
GoogleProvider(model="gemini-2.0-flash", api_key="...")
```

When loading credentials from environment, call
[`load_env()`](../README.md) explicitly first.

## OpenAI Responses lifecycle and state

When you construct `OpenAIProvider(..., use_responses_api=True)`, the provider now exposes the OpenAI background response lifecycle directly:

- `retrieve_background_response(response_id, **kwargs)`
- `cancel_background_response(response_id, **kwargs)`
- `stream_background_response(response_id, starting_after=None, **kwargs)`
- `wait_background_response(response_id, poll_interval=2.0, timeout=None, **kwargs)`
- `create_conversation(items=None, metadata=None, **kwargs)`
- `retrieve_conversation(conversation_id, **kwargs)`
- `update_conversation(conversation_id, metadata=None, **kwargs)`
- `delete_conversation(conversation_id, **kwargs)`
- `create_conversation_items(conversation_id, items, include=None, **kwargs)`
- `list_conversation_items(conversation_id, after=None, include=None, limit=None, order=None, **kwargs)`
- `retrieve_conversation_item(conversation_id, item_id, include=None, **kwargs)`
- `delete_conversation_item(conversation_id, item_id, **kwargs)`
- `compact_response_context(messages=None, model=None, instructions=None, previous_response_id=None, **kwargs)`
- `submit_mcp_approval_response(previous_response_id, approval_request_id, approve, tools=None, **kwargs)`
- `submit_shell_call_output(previous_response_id, call_id=None, output=..., tools=None, **kwargs)`
- `submit_apply_patch_call_output(previous_response_id, call_id=None, status=None, output=None, tools=None, **kwargs)`
- `delete_response(response_id, **kwargs)`

For OpenAI MCP and connector approval loops, `submit_mcp_approval_response(...)`
accepts the same convenience kwargs as `respond_with_remote_mcp(...)` and
`respond_with_connector(...)`, including `server_url`, `connector_id`,
`authorization`, `allowed_tools`, `require_approval`, and `defer_loading`.
This matters because OpenAI does not persist MCP/connector authorization on
stored Responses objects, so continuation requests must resend the tool
definition when approval is granted or denied.

Minimal polling example:

```python
provider = OpenAIProvider(model="gpt-5.2", use_responses_api=True)
queued = await provider.complete("Write a long report.", background=True, store=True)
response_id = queued.raw_response.id

state = await provider.wait_background_response(response_id, poll_interval=0.5, timeout=30.0)
if state.completion:
    print(state.completion.content)
```

Minimal resumed-stream example:

```python
cursor = None
async for event in provider.stream_background_response("resp_123", starting_after=cursor):
    cursor = event.sequence_number or cursor
```

Minimal conversation-state example:

```python
conversation = await provider.create_conversation(
    items=[{"role": "user", "content": "Summarize the launch plan."}],
    metadata={"team": "mission-control"},
)

updated = await provider.update_conversation(
    conversation.conversation_id,
    metadata={"team": "mission-control", "phase": "review"},
)

compacted = await provider.compact_response_context(
    messages=[{"role": "user", "content": "Summarize the launch plan."}],
    previous_response_id="resp_123",
)

items_page = await provider.list_conversation_items(
    conversation.conversation_id,
    limit=20,
    order="asc",
)

await provider.delete_response("resp_123")
```

## OpenAI Responses tool descriptors

`OpenAIProvider(..., use_responses_api=True)` now accepts first-class
Responses tool descriptors from `llm_client.tools`:

- `ResponsesBuiltinTool` for built-in hosted tools such as `web_search`,
  `file_search`, `computer_use`, `code_interpreter`, `image_generation`, `mcp`,
  `shell`, and `apply_patch`
- `ResponsesToolSearch` for OpenAI-specific advanced deferred-tool workflows
- `ResponsesFunctionTool` when a function tool needs provider metadata such as
  `defer_loading=True`
- `ResponsesToolNamespace` to group related OpenAI function tools under a
  namespace
- `ResponsesCustomTool` plus `ResponsesGrammar` for grammar-backed custom tools

Example:

```python
from llm_client.providers import OpenAIProvider
from llm_client.tools import (
    ResponsesAttributeFilter,
    ResponsesBuiltinTool,
    ResponsesConnectorId,
    ResponsesCustomTool,
    ResponsesFileSearchRankingOptions,
    ResponsesFunctionTool,
    ResponsesGmailTool,
    ResponsesGrammar,
    ResponsesMCPTool,
    ResponsesToolNamespace,
    ResponsesToolSearch,
)

provider = OpenAIProvider(model="gpt-5.2", use_responses_api=True)

result = await provider.complete(
    "Search the web and CRM tools, then return a compact plan.",
    tools=[
        ResponsesToolSearch.hosted(),
        ResponsesToolNamespace(
            name="crm",
            description="CRM tools for customer lookup and order management.",
            tools=(
                ResponsesFunctionTool(
                    name="lookup_profile",
                    description="Fetch a customer profile by customer id.",
                    parameters={
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"],
                        "additionalProperties": False,
                    },
                    defer_loading=True,
                ),
            ),
        ),
        ResponsesBuiltinTool.web_search(search_context_size="medium"),
        ResponsesMCPTool.connector(
            ResponsesConnectorId.GMAIL,
            server_label="Gmail",
            allowed_tools=(ResponsesGmailTool.SEARCH_EMAILS,),
            require_approval="never",
            defer_loading=True,
        ),
        ResponsesCustomTool(
            name="planner",
            description="Emit a compact plan.",
            grammar=ResponsesGrammar(syntax="lark", definition='start: "done"'),
        ),
    ],
)
```

These descriptors are provider-native request objects. They are not registered
or executed through `ToolRegistry`, which remains the runtime for local
function tools only.

If you use client-executed `tool_search`, return the loaded tool set with
`OpenAIProvider.submit_tool_search_output(...)` after the model emits a
`tool_search_call`.

If you use hosted `shell` or `apply_patch`, return your host-side execution
results with `OpenAIProvider.submit_shell_call_output(...)` or
`OpenAIProvider.submit_apply_patch_call_output(...)`. The typed helpers
`ResponsesShellCallChunk`, `ResponsesShellCallOutput`, and
`ResponsesApplyPatchCallOutput` are available from `llm_client.tools` so you do
not need to build raw provider dicts for those continuation items.

For typed MCP/connectors, `ResponsesMCPTool` now also supports
`defer_loading=True` for tool-search workflows, and `allowed_tools` can be
supplied with connector-specific enums such as `ResponsesGmailTool`.

For hosted retrieval/file-search tuning, the OpenAI provider now also accepts
typed first-class controls:

- `ResponsesAttributeFilter`
- `ResponsesFileSearchRankingOptions`

Use them either inside `ResponsesBuiltinTool.file_search(...)` or directly on
`search_vector_store(...)` / `respond_with_file_search(...)`. The file-search
helper also supports `include_search_results=True`, which requests
`file_search_call.results` in the Responses output.

## OpenAI Responses request controls

`OpenAIProvider(..., use_responses_api=True)` now exposes the following
OpenAI-specific controls as first-class keyword parameters on `complete(...)`
and `stream(...)`:

- `include`
- `prompt_cache_key`
- `prompt_cache_retention`

Typical uses:

- `include=["reasoning.encrypted_content"]` for stateless reasoning continuity
- `prompt_cache_key="tenant-a"` to improve prompt-cache routing for repeated prefixes
- `prompt_cache_retention="24h"` on supported models

## Live cookbook environment

The cookbook examples now use real provider calls. They do not fall back to
scripted or mock providers.

Default environment:

```bash
export OPENAI_API_KEY=...
```

Optional cookbook overrides:

```bash
export LLM_CLIENT_EXAMPLE_PROVIDER=openai
export LLM_CLIENT_EXAMPLE_MODEL=gpt-5-mini
export LLM_CLIENT_EXAMPLE_SECONDARY_PROVIDER=openai
export LLM_CLIENT_EXAMPLE_SECONDARY_MODEL=gpt-5-nano
export LLM_CLIENT_EXAMPLE_EMBEDDINGS_PROVIDER=openai
export LLM_CLIENT_EXAMPLE_EMBEDDINGS_MODEL=text-embedding-3-small
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY=...
export LLM_CLIENT_EXAMPLE_PROVIDER=anthropic
export LLM_CLIENT_EXAMPLE_MODEL=claude-sonnet-4
```

For Google:

```bash
export GEMINI_API_KEY=...
export LLM_CLIENT_EXAMPLE_PROVIDER=google
export LLM_CLIENT_EXAMPLE_MODEL=gemini-2.0-flash
```

## Provider registry

If you need runtime selection rather than a single hard-coded provider, use the
provider registry and router:

- provider descriptors
- capability flags
- model compatibility
- priority
- latency/cost/compliance hints

The default registry now exposes explicit OpenAI Responses capability flags in
addition to the broader completion/tool/streaming booleans:

- `responses_api`
- `background_responses`
- `responses_native_tools`
- `normalized_output_items`

The runnable reference is
[06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/llm-client-v1/examples/06_provider_registry_and_routing.py).

## Practical recommendation

Default application pattern:

1. construct providers through the registry or directly
2. run requests through `ExecutionEngine`
3. keep direct-provider usage for low-level tests and thin scripts

That keeps the package value concentrated in one path instead of bypassing the
engine and reimplementing reliability features at the call site.
