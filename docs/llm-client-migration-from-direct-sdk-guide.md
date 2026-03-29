# Migrate from Direct SDK Usage to llm-client

This guide is for teams currently calling vendor SDKs directly and wanting to
move onto `llm_client` without losing control of low-level behavior.

Relevant cookbook scripts:

- [01_one_shot_completion.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/01_one_shot_completion.py)
- [02_streaming.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/02_streaming.py)
- [05_structured_extraction.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/05_structured_extraction.py)
- [06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/06_provider_registry_and_routing.py)
- [07_engine_cache_retry_idempotency.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/07_engine_cache_retry_idempotency.py)

## Why migrate

Direct SDK usage is fine for quick experiments, but it usually leads to
repeated ad hoc work around:

- request/response normalization
- streaming event normalization
- retries and failover
- structured output validation
- hooks and diagnostics
- redaction
- tool execution
- cache/idempotency

`llm_client` centralizes those concerns.

## Minimal migration path

### Before

Typical direct-SDK shape:

```python
# pseudo-code
response = sdk.responses.create(...)
text = response.output_text
```

### After

Start with provider-level migration:

```python
provider = OpenAIProvider(model="gpt-5-mini")
result = await provider.complete(messages)
text = result.content
```

That already gives you normalized result types without forcing the full engine.

## Second step: move to the engine

Once provider-level parity is stable, move calls through `ExecutionEngine`.

That unlocks:

- retries
- fallback
- cache
- idempotency
- hooks
- normalized failure payloads

## Third step: move special logic up into package primitives

Typical migrations:

- raw JSON parsing -> `extract_structured(...)`
- custom tool loops -> `ToolRegistry` + `ToolExecutionEngine` + `Agent`
- custom logging/tracing wrappers -> hooks and lifecycle reports
- custom multimodal request shaping -> content blocks/envelopes

## When not to migrate a path

Keep direct provider usage if:

- you are doing a very thin benchmark/probe
- you need provider-specific debugging
- you explicitly do not want retries/cache/hooks on that path

Everything else should usually move to the engine.

## Practical migration sequence

1. replace raw SDK response parsing with `CompletionResult`/`StreamEvent`
2. route core request paths through `ExecutionEngine`
3. migrate structured output flows onto `llm_client.structured`
4. migrate tool flows onto `llm_client.tools` / `llm_client.agent`
5. attach lifecycle/diagnostic hooks
6. remove app-local retry/failover/cache logic once parity is proven
