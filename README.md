# llm-client Package Reference (v0.2.0)

This document is the canonical, user-facing reference for the `llm-client` Python package (import path: `llm_client`).
It describes capabilities, API surface, configuration, expected inputs/outputs, and known limitations based on the
current code in this repository.

> Scope: This is primarily an API/usage reference (how to use the library correctly). It is not a tutorial on LLMs.

---

## Table of Contents

- [What this package is](#what-this-package-is)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Configuration](#configuration)
- [Model Profiles (`llm_client.models`)](#model-profiles-llm_clientmodels)
- [Provider Types (`llm_client.providers.types`)](#provider-types-llm_clientproviderstypes)
- [Providers](#providers)
- [Execution Engine (`llm_client.engine`)](#execution-engine-llm_clientengine)
- [Legacy OpenAI Facade (`llm_client.client.OpenAIClient`)](#legacy-openai-facade-llm_clientclientopenaiclient)
- [Agent Framework (`llm_client.agent`)](#agent-framework-llm_clientagent)
- [Tool System (`llm_client.tools`)](#tool-system-llm_clienttools)
- [Conversation Management (`llm_client.conversation`)](#conversation-management-llm_clientconversation)
- [Streaming (`llm_client.streaming`)](#streaming-llm_clientstreaming)
- [Caching (`llm_client.cache`)](#caching-llm_clientcache)
- [Batch Processing (`llm_client.batch_req`)](#batch-processing-llm_clientbatch_req)
- [Dependency Injection Container (`llm_client.container`)](#dependency-injection-container-llm_clientcontainer)
- [Idempotency (`llm_client.idempotency`)](#idempotency-llm_clientidempotency)
- [Validation (`llm_client.validation`)](#validation-llm_clientvalidation)
- [Persistence (`llm_client.persistence`)](#persistence-llm_clientpersistence)
- [Telemetry, Hooks, and Logging](#telemetry-hooks-and-logging)
- [Errors and Exceptions](#errors-and-exceptions)
- [Cancellation Support](#cancellation-support)
- [Structured Output](#structured-output)
- [Tool Middleware](#tool-middleware)
- [Summarization Interface](#summarization-interface)
- [Limitations and Behavioral Notes](#limitations-and-behavioral-notes)
- [Public API Index (`llm_client.__init__` exports)](#public-api-index-llm_client__init__-exports)

---

## What this package is

`llm-client` is an async-first Python library that provides:

- A **provider abstraction** over multiple LLM vendors (OpenAI, Anthropic, Google Gemini).
- A unified set of **types** for messages, tool calls, streaming events, usage, and results.
- An **execution engine** that can add caching, retries, circuit breaking, and hooks to provider calls.
- An **agent framework** with a multi-turn loop, tool calling, tool execution, and optional streaming.
- A **tool system** (`Tool`, `ToolRegistry`, decorators) with JSON-schema inference and optional strict validation.
- **Tool middleware chain** for composable cross-cutting concerns (logging, timeout, retry, policy, budget, etc.).
- **Conversation management** with token-aware truncation and async summarization.
- **Cancellation support** via cooperative `CancellationToken` throughout the async stack.
- **Structured output** with JSON schema validation and automatic repair loop.
- Optional **cache backends** (filesystem, PostgreSQL+Redis hybrid, Qdrant) for both completions and embeddings.
- **Batch processing** helpers.
- **Observability** via hooks, plus internal telemetry utilities.

This package intentionally exposes both:

- A modern API (providers + `ExecutionEngine` + `Agent`)
- A backward-compatible facade (`OpenAIClient`)

---

## Installation

### Requirements

- Python `>= 3.10`
- Async runtime (you will typically use `asyncio`)

### Install (editable for development)

```bash
pip install -e .
```

### Optional extras

```bash
# Anthropic provider support
pip install -e ".[anthropic]"

# Google Gemini provider support
pip install -e ".[google]"

# Faster JSON (orjson)
pip install -e ".[performance]"

# Prometheus/OpenTelemetry hooks
pip install -e ".[telemetry]"

# All extras
pip install -e ".[all]"
```

---

## Quick Start

### 1) Load `.env` (explicit)

The package does **not** implicitly read `.env`. Call `llm_client.load_env()` yourself:

```python
from llm_client import load_env

load_env()  # looks for .env from current working directory
```

### 2) Simple completion (OpenAI)

```python
import asyncio
from llm_client import OpenAIProvider

async def main():
    provider = OpenAIProvider(model="gpt-5-nano")
    result = await provider.complete("Hello!")
    print(result.content)
    await provider.close()

asyncio.run(main())
```

### 3) Streaming tokens

```python
import asyncio
from llm_client import OpenAIProvider, StreamEventType

async def main():
    provider = OpenAIProvider(model="gpt-5-nano")
    async for event in provider.stream("Write a haiku about coding."):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.DONE:
            print("\n", event.data.usage)
    await provider.close()

asyncio.run(main())
```

### 4) Agent with tools

```python
import asyncio
from llm_client import Agent, OpenAIProvider, tool

@tool
async def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"

async def main():
    agent = Agent(provider=OpenAIProvider("gpt-5-nano"), tools=[get_weather])
    result = await agent.run("What's the weather in Tokyo?")
    print(result.content)
    await agent.provider.close()

asyncio.run(main())
```

---

## Core Concepts

### Provider

A **provider** is an object implementing the `llm_client.providers.base.Provider` protocol:

- `complete(...) -> CompletionResult` for non-streaming completions
- `stream(...) -> AsyncIterator[StreamEvent]` for streaming
- `embed(...) -> EmbeddingResult` for embeddings (if supported)
- `count_tokens(...)` and `parse_usage(...)` backed by a model profile

### Model profile

A `ModelProfile` (in `llm_client.models`) describes:

- a stable model key (`"gpt-5"`, `"text-embedding-3-large"`, etc.)
- the actual provider model name string
- tokenization encoding for counting
- rate limit defaults
- usage cost calculation helpers
- capability flags (reasoning, tool calling, streaming)

### RequestSpec + ExecutionEngine

`RequestSpec` is a deterministic request description (messages, model, tools, etc.). The `ExecutionEngine`:

- validates the spec
- optionally reads/writes cache
- applies retries (exponential backoff)
- uses a circuit breaker per provider+model
- emits hook events for observability
- can fallback to other providers if you supply a router

### Agent

An `Agent` wraps a provider (optionally an engine) plus a `Conversation` and a `ToolRegistry`, then runs multi-turn
loops:

1) call the model
2) if tool calls exist, execute tools
3) append tool results to conversation
4) repeat until a final answer or max turns

### Tool

A `Tool` is a callable function exposed to the model. Tools use JSON Schema parameter definitions. Tools can be created
manually or inferred from a Python function’s signature using decorators / `tool_from_function`.

---

## Configuration

The configuration system lives in `llm_client.config` and is optional for basic usage (you can pass keys directly to
provider constructors).

### `load_env(path: str | None = None, override: bool = False) -> bool`

Located in `llm_client.config.settings` and re-exported as `llm_client.load_env`.

- Loads environment variables from a `.env` file.
- If `path` is `None`, it uses `dotenv.find_dotenv(usecwd=True)` to locate a `.env`.
- Returns `True` if a file was found and loaded, else `False`.

### `Settings`

`Settings` is a dataclass that aggregates sub-configs:

- `openai: OpenAIConfig`
- `anthropic: AnthropicConfig`
- `google: GoogleConfig`
- `cache: CacheConfig` (or a subclass for backend specifics)
- `agent: AgentConfig`
- `logging: LoggingConfig`
- `metrics: MetricsConfig`
- `rate_limit: RateLimitConfig`

Key constructors:

- `Settings.from_env(prefix: str = "LLM_") -> Settings`
- `Settings.from_file(path: str | Path) -> Settings` (YAML or TOML)
- `Settings.default() -> Settings`
- `Settings.to_dict() -> dict[str, Any]`

### Environment variables (via `Settings.from_env`)

Supported (non-exhaustive; see `llm_client.config.settings.Settings.from_env` for exact behavior):

- Provider:
  - `LLM_OPENAI_API_KEY`
  - `LLM_OPENAI_BASE_URL`
  - `LLM_OPENAI_MODEL`
  - `LLM_ANTHROPIC_API_KEY`
  - `LLM_ANTHROPIC_MODEL`
  - `LLM_GOOGLE_API_KEY`
  - `LLM_GOOGLE_MODEL`
- Cache:
  - `LLM_CACHE_BACKEND` (one of `none|fs|pg_redis|qdrant`)
  - `LLM_CACHE_DIR` (enables FS cache config)
- Agent:
  - `LLM_AGENT_MAX_TURNS`
  - `LLM_AGENT_TOOL_TIMEOUT`
  - `LLM_AGENT_BATCH_CONCURRENCY`
- Logging:
  - `LLM_LOG_LEVEL`
  - `LLM_LOG_FORMAT`
- Metrics:
  - `LLM_METRICS_ENABLED` (`true|false`)
  - `LLM_METRICS_PROVIDER` (`none|prometheus|otel`)
  - `LLM_METRICS_PROMETHEUS_PORT`
  - `LLM_METRICS_OTEL_ENDPOINT`

Provider-specific defaults (also read directly by provider configs):

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY` (Google provider also checks `GEMINI_API_KEY`)

### Configuration files (YAML/TOML)

`Settings.from_file(...)` supports:

- YAML: requires `pyyaml`
- TOML: uses stdlib `tomllib` on modern Python, otherwise `tomli`

The input dict is validated against `CONFIG_SCHEMA` (in `llm_client.config_schema`) using `jsonschema`. If `jsonschema`
is missing, schema validation is skipped.

---

## Model Profiles (`llm_client.models`)

### `ModelProfile` (base class)

`ModelProfile` is a registry-based class. Subclasses register themselves by unique `key`.

Important class attributes (defined on subclasses):

- `key: str` (stable key used by `ModelProfile.get`)
- `model_name: str` (provider model identifier)
- `category: str` (`"completions"` or `"embeddings"`)
- `context_window: int`
- `max_output: int | None` (completions)
- `output_dimensions: int | None` (embeddings)
- `rate_limits: dict` (e.g. `{"tkn_per_min": ..., "req_per_min": ...}`)
- `usage_costs: dict` (token costs)
- Capability flags:
  - `reasoning_model: bool`
  - `reasoning_efforts: list[str]`
  - `default_reasoning_effort: str | None`
  - `function_calling_support: bool`
  - `token_streaming_support: bool`
- `encoding: str` (tiktoken encoding name)

Key methods:

- `ModelProfile.get(key: str) -> type[ModelProfile]`
- `ModelProfile.count_tokens(context: Any) -> int`
- `ModelProfile.parse_usage(usage: dict[str, Any]) -> dict[str, int | Decimal]`
- `ModelProfile.input_cost(n_tokens: int) -> Decimal`
- `ModelProfile.output_cost(n_tokens: int) -> Decimal`
- `ModelProfile.cached_input_cost(n_tokens: int) -> Decimal`

Token counting notes:

- Token counting uses `tiktoken` encoding and caches per-string tokenization (`lru_cache`).
- For lists of dicts (message-like objects), token counting is approximate: it iterates values and stringifies them.

### Included model keys

The package defines a catalog of model keys in `llm_client.models` (examples):

- Completions:
  - `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.1`, `gpt-5.2`
  - several Claude and Gemini entries are also defined as `ModelProfile` subclasses (keys vary)
- Embeddings:
  - `text-embedding-3-large`, `text-embedding-3-small`

Use either:

- model key string: `OpenAIProvider(model="gpt-5")`
- subclass: `OpenAIProvider(model=GPT5)`

---

## Provider Types (`llm_client.providers.types`)

These types define the stable “contract” between providers, the engine, and the agent.

### `Role` (enum)

- `Role.SYSTEM`, `Role.USER`, `Role.ASSISTANT`, `Role.TOOL`

### `Message` (dataclass)

Fields:

- `role: Role`
- `content: str | None`
- `name: str | None`
- `tool_calls: list[ToolCall] | None`
- `tool_call_id: str | None` (used for tool response messages)

Key methods:

- `Message.to_dict() -> dict[str, Any]` (OpenAI-style message dict)
- `Message.from_dict(data: dict[str, Any]) -> Message`
- Convenience constructors:
  - `Message.user(content: str)`
  - `Message.assistant(content: str | None = None, tool_calls: list[ToolCall] | None = None)`
  - `Message.system(content: str)`
  - `Message.tool_result(tool_call_id: str, content: str, name: str | None = None)`

### `ToolCall` and `ToolCallDelta`

- `ToolCall(id: str, name: str, arguments: str)` where `arguments` is a JSON string.
- `ToolCall.parse_arguments() -> dict[str, Any]` parses `arguments`.

`ToolCallDelta` is used during streaming tool-call assembly:

- `id: str`, `index: int`, `name: str | None`, `arguments_delta: str`

### `Usage` (dataclass)

Fields:

- `input_tokens`, `output_tokens`, `total_tokens`, `input_tokens_cached`
- `input_cost`, `output_cost`, `total_cost`

Methods:

- `Usage.to_dict() -> dict[str, Any]`
- `Usage.from_dict(data: dict[str, Any]) -> Usage`

### `CompletionResult` (dataclass)

Fields:

- `content: str | None`
- `tool_calls: list[ToolCall] | None`
- `usage: Usage | None`
- `reasoning: str | None`
- `model: str | None`
- `finish_reason: str | None`
- `status: int = 200`
- `error: str | None`
- `raw_response: Any | None` (debugging; not always set)

Properties:

- `ok: bool` (true if `status == 200` and `error is None`)
- `has_tool_calls: bool`

Methods:

- `to_message() -> Message` (assistant message with tool_calls)
- `to_dict() -> dict[str, Any]`

### `EmbeddingResult` (dataclass)

Fields:

- `embeddings: list[list[float]]`
- `usage: Usage | None`
- `model: str | None`
- `status: int = 200`
- `error: str | None`

Properties:

- `ok: bool`
- `embedding: list[float] | None` (first embedding, for single-input calls)

### `StreamEventType` and `StreamEvent`

`StreamEventType` includes:

- `TOKEN`, `REASONING`
- `TOOL_CALL_START`, `TOOL_CALL_DELTA`, `TOOL_CALL_END`
- `META`, `USAGE`
- `DONE`, `ERROR`

`StreamEvent` fields:

- `type: StreamEventType`
- `data: Any` (see below)
- `timestamp: float`

Expected `StreamEvent.data` shapes:

- `TOKEN`: `str`
- `REASONING`: `str`
- `TOOL_CALL_START`: `ToolCallDelta`
- `TOOL_CALL_DELTA`: `ToolCallDelta`
- `TOOL_CALL_END`: `ToolCall`
- `META`: `dict[str, Any]`
- `USAGE`: `Usage`
- `DONE`: `CompletionResult`
- `ERROR`: typically `dict[str, Any]` like `{"status": int, "error": str}`

### `normalize_messages(messages: MessageInput) -> list[Message]`

Accepted `MessageInput`:

- `str` -> single user message
- `dict` -> `Message.from_dict(...)`
- `Message` -> used as-is
- `Sequence[...]` of the above

---

## Providers

Providers live in `llm_client.providers.*` and are re-exported from `llm_client`.

### Provider protocol (`llm_client.providers.base.Provider`)

A provider implements:

- `complete(messages, tools=None, tool_choice=None, temperature=None, max_tokens=None, response_format=None, **kwargs)`
- `stream(messages, tools=None, tool_choice=None, temperature=None, max_tokens=None, **kwargs)`
- `embed(inputs, **kwargs)`
- `count_tokens(content)`
- `parse_usage(raw_usage)`
- `close()`
- async context manager (`__aenter__`, `__aexit__`)

### BaseProvider (`llm_client.providers.base.BaseProvider`)

`BaseProvider(model)` accepts either:

- a model key string (resolved via `ModelProfile.get`)
- a `ModelProfile` subclass

It provides:

- `model` (the `ModelProfile` subclass)
- `model_name`
- default `count_tokens` via the model profile
- default `parse_usage` via the model profile (wraps into `Usage`)

### OpenAIProvider (`llm_client.providers.openai.OpenAIProvider`)

Capabilities:

- Completions (non-streaming and streaming)
- Tool calling (OpenAI tool format)
- Reasoning-effort parameters for reasoning-capable models
- Embeddings
- Optional caching backends
- Rate limiting via `Limiter`

Constructor:

```python
OpenAIProvider(
    model: type[ModelProfile] | str,
    *,
    cache_dir: str | Path | None = None,
    cache_backend: Literal["qdrant", "pg_redis", "fs"] | None = None,
    cache_collection: str | None = None,
    pg_dsn: str | None = None,
    redis_url: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    redis_ttl_seconds: int = 86400,
    compress_pg: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    use_responses_api: bool = False,
)
```

Key methods:

- `await provider.complete(messages, tools=None, tool_choice=None, temperature=None, max_tokens=None, response_format=None, **kwargs) -> CompletionResult`
- `provider.stream(...) -> AsyncIterator[StreamEvent]`
- `await provider.embed(inputs: str | list[str], **kwargs) -> EmbeddingResult`
- `await provider.warm_cache() -> None`
- `await provider.close() -> None`

Notes:

- The provider uses `openai.AsyncOpenAI`.
- If `use_responses_api=True`, the provider will use OpenAI’s responses-style API behavior when implemented in this module.
- Additional `**kwargs` are provider-specific and may be forwarded to the underlying OpenAI SDK calls.

### AnthropicProvider (`llm_client.providers.anthropic.AnthropicProvider`)

Availability:

- `llm_client.ANTHROPIC_AVAILABLE` indicates whether `anthropic` SDK import succeeded.

Capabilities:

- Chat completions (non-streaming and streaming)
- Tool calling
- “Extended thinking” for supported models (subject to Anthropic API capabilities)
- Rate limiting via `Limiter`
- Optional cache backends

Limitations:

- **Embeddings are not supported** (Anthropic does not natively provide embeddings). Calls to `embed` are expected to be
  unsupported/raise or return an error result depending on implementation.

Constructor (high-level; see module for full set of cache args):

- Requires `anthropic` extra (`pip install llm-client[anthropic]`) and `ANTHROPIC_API_KEY` (or `api_key=`).

### GoogleProvider (`llm_client.providers.google.GoogleProvider`)

Availability:

- `llm_client.GOOGLE_AVAILABLE` indicates whether `google-genai` SDK import succeeded.

Capabilities:

- Chat completions (non-streaming and streaming, depending on SDK support path)
- Tool calling (mapped into Gemini API as supported by SDK)
- Embeddings if supported by the SDK/model
- Rate limiting via `Limiter`
- Optional cache backends

API key behavior:

- `api_key` parameter, else `GEMINI_API_KEY`, else `GOOGLE_API_KEY`.
- If none found, constructor raises `ValueError`.

---

## Execution Engine (`llm_client.engine`)

The engine orchestrates provider calls with validation, hooks, retry/backoff, circuit breaker, caching, and (optionally)
routing/fallback.

### `RetryConfig` (dataclass)

Fields:

- `attempts: int = 3`
- `backoff: float = 1.0`
- `max_backoff: float = 20.0`
- `retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)`

### `ExecutionEngine`

Constructor:

```python
ExecutionEngine(
    provider: Provider | None = None,
    *,
    router: ProviderRouter | None = None,
    cache: CacheCore | None = None,
    hooks: HookManager | None = None,
    retry: RetryConfig | None = None,
    breaker_config: CircuitBreakerConfig | None = None,
    fallback_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    max_concurrency: int = 20,
)
```

Rules:

- You must pass either `provider` or `router`.
- If `cache` is not provided and the provider has `.cache`, the engine will use `provider.cache`.
- A circuit breaker is maintained per `provider.__class__.__name__ + ":" + provider.model_name`.

#### `await engine.complete(spec: RequestSpec, ...) -> CompletionResult`

Arguments:

- `spec: RequestSpec` (validated by `llm_client.validation.validate_spec`)
- `context: RequestContext | None = None` (used for hook correlation)
- Cache control:
  - `cache_response: bool = False`
  - `cache_collection: str | None = None`
  - `rewrite_cache: bool = False`
  - `regen_cache: bool = False`
  - `cache_key: str | None = None` (if not given, engine derives from spec+provider)
- `retry: RetryConfig | None = None` (overrides engine default)

Behavior:

1) Validates `spec`.
2) Selects providers: `[provider]` or `router.select(spec)`.
3) Circuit breaker gate; if open, yields a synthetic failure result and may try fallbacks.
4) Optional cache read (only when `cache_response=True` and a cache exists).
5) Performs up to `attempts` provider calls with exponential backoff and jitter for retryable statuses.
6) On success, optionally writes to cache and emits hook events.
7) On failure with a status in `fallback_statuses`, will try next provider if a router is configured.

Return:

- A `CompletionResult` with `.ok` set accordingly.

#### `engine.stream(spec: RequestSpec, ...) -> AsyncIterator[StreamEvent]`

Behavior:

- Validates `spec`, then streams from the first provider that succeeds.
- If an early `ERROR` occurs before any tokens are seen and the status is fallback-eligible, the engine may switch to the
  next provider (router required).
- Emits hook events such as `stream.start`, `stream.event`, `stream.end`, etc.

#### `await engine.embed(inputs: str | Iterable[str], ...) -> Any`

Arguments:

- `inputs`: one string or iterable of strings (validated by `validate_embedding_inputs`)
- `context`, `timeout`, `**kwargs` forwarded to provider

Important limitations:

- **Embedding routing is not implemented**: if you pass a router without a default provider, embeddings will not be
  selected via the router.
- **Embedding caching is not implemented** in the engine (commented as future work).

#### `await engine.batch_complete(specs: Iterable[RequestSpec], ...) -> list[CompletionResult]`

Runs `complete(...)` concurrently under internal semaphore:

- engine has default semaphore from `max_concurrency`
- `max_concurrency=` here can reduce concurrency for this batch

---

## Legacy OpenAI Facade (`llm_client.client.OpenAIClient`)

`OpenAIClient` exists for backward compatibility with a previous API style. New code should generally prefer:

- provider classes directly (`OpenAIProvider`, `AnthropicProvider`, etc.), or
- `ExecutionEngine` for caching/retries/hooks, or
- `Agent` for tool-calling loops.

### `OpenAIClient.__init__(...)`

```python
OpenAIClient(
    model: type[ModelProfile] | str | None = None,
    *,
    cache_dir: str | Path | None = None,
    responses_api_toggle: bool = False,
    use_engine: bool = True,  # deprecated and ignored (always uses engine)
    engine: ExecutionEngine | None = None,
    cache_backend: Literal["qdrant", "pg_redis", "fs"] | None = None,
    cache_collection: str | None = None,
    pg_dsn: str | None = None,
    redis_url: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    redis_ttl_seconds: int = 86400,
    compress_pg: bool = True,
)
```

Behavior:

- Resolves `model`:
  - `ModelProfile` subclass -> used as-is
  - string -> `ModelProfile.get(...)`
  - `None` -> uses `llm_client.config.get_settings().openai.default_model`
- Builds a cache core from the provided cache parameters.
- Creates (or accepts) an `ExecutionEngine`. If it constructs its own engine, it uses an internal `OpenAIProvider`.
- If metrics are enabled in global settings, it may attach `PrometheusHook` or `OpenTelemetryHook` to the engine.

### `await client.get_response(...) -> dict[str, Any] | AsyncIterator[str]`

This is the main legacy entrypoint. It supports both completions and embeddings.

Signature (simplified; see `src/llm_client/client.py` for the full list):

- `identifier`, `attempts`, `backoff`
- cache flags: `cache_response`, `rewrite_cache`, `regen_cache`, `cache_collection`, `hash_as_identifier`
- request and provider kwargs are accepted via `**kwargs`
- streaming:
  - `stream: bool`
  - `stream_mode: "pusher" | "sse"`
  - `channel: str | None` (for pusher mode)

Behavior:

- Determines whether the request is embeddings based on `self.model.category == "embeddings"`.
- Normalizes `messages` from `kwargs["messages"]` or `kwargs["input"]` (fallback to `kwargs["prompt"]` for legacy).
- Builds a `RequestSpec` and runs it through `ExecutionEngine.complete(...)` or `ExecutionEngine.embed(...)`.
- Returns a dict with keys like:
  - `"params"`: `RequestSpec.to_dict()` payload
  - `"output"`: completion text (or embedding vectors)
  - `"usage"`: token usage dict (if available)
  - `"status"`: status code
  - `"error"`: `"OK"` or error message
  - `"identifier"`: cache key / request identifier
  - `"body"`: additional metadata wrapper
  - optionally `"response"` when `return_response=True`

Streaming:

- `stream_mode="sse"` returns an async iterator of SSE-formatted strings.
- `stream_mode="pusher"` returns a dict describing the pusher channel output.

### Other notable methods

- `await warm_cache()`: calls cache `warm()` if configured.
- `await batch(specs: Iterable[RequestSpec], **kwargs)`: forwards to `engine.batch_complete(...)`.
- `await close()`: closes cache (provider is not always closed here; if you manage the provider, close it explicitly).
- `encode_file(path) -> dict[str, Any]`: helper for image/pdf message blocks (OpenAI-style).
- `await transcribe_pdf(path) -> dict[str, Any]`:
  - uploads the PDF using `openai.AsyncOpenAI().files.create(...)` with purpose `"user_data"`
  - then asks the model to extract text via responses-style calls through `get_response(...)`
- `await transcribe_image(path) -> dict[str, Any]`: base64-embeds an image and requests transcription.

Important notes:

- `transcribe_pdf` uses direct file upload via the OpenAI SDK; it does not go through `ExecutionEngine` for file
  lifecycle. This is explicitly called out in code comments as a “leaky abstraction”.

---

## Agent Framework (`llm_client.agent`)

### `AgentConfig` (`llm_client.config.agent.AgentConfig`)

Fields:

- Turn limits:
  - `max_turns: int = 10`
  - `max_tool_calls_per_turn: int = 10`
- Tool execution:
  - `parallel_tool_execution: bool = True`
  - `tool_timeout: float = 30.0`
  - `max_tool_output_chars: int | None = None`
  - `tool_retry_attempts: int = 0`
- Context management:
  - `max_tokens: int | None = None`
  - `reserve_tokens: int = 2000`
- Behavior:
  - `stop_on_tool_error: bool = False`
  - `include_tool_errors_in_context: bool = True`
  - `stream_tool_calls: bool = True`
  - `batch_concurrency: int = 20`

### `Agent` (`llm_client.agent.core.Agent`)

Constructor:

```python
Agent(
    provider: Provider,
    *,
    tools: list[Tool] | ToolRegistry | None = None,
    system_message: str | None = None,
    conversation: Conversation | None = None,
    config: AgentConfig | None = None,
    engine: ExecutionEngine | None = None,
    use_engine: bool = False,
    max_turns: int = 10,
    max_tokens: int | None = None,
)
```

Notes:

- If `use_engine=True` (or `engine` is provided), the agent uses the engine for completions; otherwise it calls the
  provider directly.
- `conversation` is created automatically if not supplied; it uses `max_tokens` and `reserve_tokens` from config.

#### `await agent.run(prompt: str, *, max_turns: int | None = None, **kwargs) -> AgentResult`

Behavior:

1) Adds user message to conversation.
2) Repeats for each turn:
   - calls model (`provider.complete` or engine)
   - accumulates usage
   - if tool calls: executes tools (parallel or sequential) with timeout/retry
   - optionally truncates tool output (`max_tool_output_chars`)
   - appends tool results to conversation
3) Stops when model returns no tool calls (final answer), or max turns, or error.

`**kwargs` are forwarded to the provider call (temperature, max_tokens, response_format, reasoning params, etc.).

#### `agent.stream(prompt: str, ...) -> AsyncIterator[StreamEvent]`

The agent exposes streaming (token-by-token) and may optionally emit tool-call events depending on provider output and
`AgentConfig.stream_tool_calls`.

#### Session persistence

Utilities in `llm_client.agent.session`:

- `save_agent_session(agent: Agent, path: str | Path) -> None`
- `load_agent_session(path: str | Path) -> Agent`
- `quick_agent(...) -> Agent` (also re-exported from `llm_client`)

These store/restore conversation history and tool definitions sufficient to resume an agent run.

---

## Tool System (`llm_client.tools`)

### `ToolResult`

Fields:

- `content: str | dict[str, Any] | None`
- `success: bool = True`
- `error: str | None`
- `metadata: dict[str, Any]`

Methods:

- `to_string() -> str` (dicts become JSON; errors become `"Error: ..."` strings)
- `ToolResult.success_result(content) -> ToolResult`
- `ToolResult.error_result(error: str) -> ToolResult`

### `Tool`

Fields:

- `name: str`
- `description: str`
- `parameters: dict[str, Any]` (JSON schema)
- `handler: Callable[..., Awaitable[Any]]`
- `strict: bool = False` (enforces schema validation on inputs)

Methods:

- `to_openai_format() -> dict[str, Any]` (OpenAI “tools” schema)
- `execute(**kwargs) -> ToolResult` (validates if strict; normalizes return values)
- `execute_json(arguments_json: str) -> ToolResult`

### `ToolRegistry`

Responsibilities:

- register tools, lookup by name
- execute tool calls by name + JSON arguments
- convert to OpenAI format list

Key methods (names may vary; see `llm_client.tools.base.ToolRegistry`):

- `register(tool: Tool) -> None`
- `get(name: str) -> Tool | None`
- `to_openai_format() -> list[dict[str, Any]]`
- `execute(name: str, arguments_json: str) -> ToolResult`
- `execute_many(calls: list[dict]) -> list[ToolResult]` (parallel)

### Creating tools: `tool_from_function`, `@tool`, `sync_tool`

#### `tool_from_function(func, name=None, description=None, strict=False) -> Tool`

- Converts an async or sync function into a `Tool`.
- Sync functions are wrapped via `llm_client.concurrency.run_sync`.
- Parameters are inferred from Python annotations with a limited mapping to JSON Schema.

Type inference support (current implementation):

- `str`, `int`, `float`, `bool`, `None`, `Any`
- `Optional[T]` and `Union[...]` (as `anyOf`)
- `list[T]` / `List[T]` (as array)
- `dict[...]` / `Dict[...]` (as object, without property schema)
- Unknown types default to `{"type": "string"}` (stringified)

Docstring parameter descriptions:

- The schema generator does a simple scan for lines like `param_name: ...` in the docstring.

#### Decorators (`llm_client.tools.decorators`)

- `@tool` wraps a function and returns a `Tool`.
- `sync_tool` is used for sync functions but still returns a tool with async handler.

---

## Conversation Management (`llm_client.conversation`)

### `ConversationConfig`

Fields:

- `max_tokens: int | None` (enables truncation when a model is provided to `get_messages`)
- `truncation_strategy: Literal["sliding", "drop_oldest", "drop_middle", "summarize"]`
- `reserve_tokens: int = 1000` (reserved for model output)
- `system_message: str | None`
- `preserve_system: bool = True`
- `session_id: str | None` (auto-generated if not provided)

### `Conversation`

Constructors:

```python
Conversation(
    messages: list[Message] | None = None,
    *,
    system_message: str | None = None,
    max_tokens: int | None = None,
    truncation_strategy: TruncationStrategy = "sliding",
    reserve_tokens: int = 1000,
    session_id: str | None = None,
)
```

Key operations:

- Add messages:
  - `add_message(message: Message)`
  - `add_user(content: str)`
  - `add_assistant(content: str)`
  - `add_assistant_with_tools(content: str | None, tool_calls: list[ToolCall])`
  - `add_tool_result(tool_call_id: str, content: str, name: str | None = None)`
  - `add_system(content: str)` (replaces current system message)
- Retrieve messages:
  - `get_messages(model: type[ModelProfile] | None = None, include_system: bool = True) -> list[Message]`
  - `get_messages_dict(...) -> list[dict[str, Any]]`
- Token counting:
  - `count_tokens(model: type[ModelProfile]) -> int`
- Truncation strategies:
  - `sliding`: keeps the most recent messages that fit
  - `drop_oldest`: removes oldest until within limit
  - `drop_middle`: keeps first message and as many recent messages as fit
- Persistence:
  - `to_dict()`, `from_dict(...)`
  - `to_json()`, `from_json(...)`
  - `save(path)`, `load(path)`
- Forking:
  - `fork()` (copy all messages, new session id)
  - `branch(from_index=0)` (copy messages up to index, new session id)

Important limitation:

- `truncation_strategy="summarize"` is accepted by the type, but it is currently treated as a fallback to `"sliding"`
  (no summarization is implemented in the current code).

---

## Streaming (`llm_client.streaming`)

This module provides utilities for consuming `AsyncIterator[StreamEvent]`.

### `format_sse_event(event: str, data: str) -> str`

Returns an SSE-formatted string:

```text
event: {event}
data: {data}

```

### `SSEAdapter`

- `transform(stream: AsyncIterator[StreamEvent]) -> AsyncIterator[str]`: yields `event.to_sse()` for each event.
- `emit(event: StreamEvent) -> str`: converts one event to SSE (async method).
- `close()`: marks adapter closed.

### `CallbackAdapter`

Constructor callbacks:

- `on_token(str)`
- `on_reasoning(str)`
- `on_tool_call_start(ToolCallDelta)`
- `on_tool_call_delta(ToolCallDelta)`
- `on_tool_call_end(ToolCall)`
- `on_usage(Usage)`
- `on_done(CompletionResult)`
- `on_error(dict[str, Any])`
- `on_meta(dict[str, Any])`

Methods:

- `emit(event: StreamEvent) -> None`
- `consume(stream) -> CompletionResult | None` (returns the final DONE result if observed)

### `BufferingAdapter`

Buffers and accumulates:

- `content`, `reasoning`, `tool_calls`, `usage`, `result`, `error`

Methods:

- `wrap(stream) -> AsyncIterator[StreamEvent]` (passes through while buffering)
- `get_result() -> CompletionResult` (returns DONE result if present, else reconstructs best-effort)
- `close()` clears buffers

### `PusherStreamer`

Pushes stream events to a Pusher channel over HTTP.

Required environment variables:

- `PUSHER_AUTH_KEY`
- `PUSHER_AUTH_SECRET`
- `PUSHER_AUTH_VERSION`
- `PUSHER_APP_ID`
- `PUSHER_APP_CLUSTER`

Methods:

- async context manager (`async with PusherStreamer(channel)`)
- `push_event(name: str, data: str) -> dict | str`
- `emit(event: StreamEvent) -> None`
- `consume(stream) -> CompletionResult | None`
- `close()`

### Utilities

- `collect_stream(stream) -> CompletionResult` (buffers and returns final result)
- `stream_to_string(stream) -> str` (concatenates only token events)

---

## Caching (`llm_client.cache`)

Caching is implemented as:

- A `CacheCore` orchestration layer (stats, key resolution, safe read/write)
- A `CacheBackend` interface (per-backend read/write/resolve)
- Concrete backend implementations:
  - filesystem
  - Redis + Postgres hybrid
  - Qdrant

Caching is used by:

- Providers (each provider can own its own `.cache`)
- The `ExecutionEngine` (can use an external cache or provider cache)

### Cache core (`llm_client.cache.core.CacheCore`)

Key methods:

- `ensure_ready()`, `warm()`, `close()`
- `get_cached(identifier, rewrite_cache, regen_cache, only_ok=True, collection=None) -> (dict|None, effective_key)`
- `put_cached(identifier, rewrite_cache, regen_cache, response, model_name, log_errors, collection=None) -> effective_key`
- `get_stats() -> CacheStats`, `reset_stats() -> CacheStats`

### CacheStats (`llm_client.cache.core.CacheStats`)

Tracks:

- `hits`, `misses`, `writes`, `errors`
- `total_read_ms`, `total_write_ms`
- computed: `hit_rate`, `avg_read_ms`, `avg_write_ms`

### Backend selection (`llm_client.cache.factory.CacheSettings`)

Providers create cache cores via:

```python
from llm_client.cache import CacheSettings, build_cache_core
```

`CacheSettings` fields include:

- `backend: "none"|"fs"|"pg_redis"|"qdrant"`
- `client_type: str` (e.g. `"completions"` or `"embeddings"`)
- `default_collection: str | None`
- backend-specific connection settings:
  - `cache_dir` (fs)
  - `pg_dsn`, `redis_url`, `redis_ttl_seconds`, `compress` (pg_redis)
  - `qdrant_url`, `qdrant_api_key` (qdrant)

### Filesystem backend (`llm_client.cache.fs.FSCache`)

Stores cache records as JSON files under a directory. Supports collections/namespaces by directory structure.

### PostgreSQL + Redis backend (`llm_client.cache.postgres_redis.HybridRedisPostgreSQLCache`)

Hybrid behavior:

- Redis provides fast reads / TTL behavior.
- PostgreSQL provides durable storage.
- Optional compression via zlib.

External dependencies:

- `asyncpg`
- `redis` (async)
- a running Postgres and Redis service

### Qdrant backend (`llm_client.cache.qdrant.QdrantCache`)

Stores cache items in Qdrant (vector DB) using HTTP API calls (`aiohttp`). This backend is suitable when you want to
persist or index items externally.

External dependencies:

- `aiohttp`
- a running Qdrant service

### Cache serialization (`llm_client.cache.serializers`)

The engine and providers serialize `CompletionResult` to cache-friendly dicts via:

- `result_to_cache_dict(result, spec_dict) -> dict`
- `cache_dict_to_result(cache_dict) -> CompletionResult`

---

## Batch Processing (`llm_client.batch_req`)

This module provides a general-purpose async batch manager that can:

- process items with a bounded worker pool
- checkpoint results for resume
- optionally show a progress bar (if `tqdm` is installed)

### `BatchManager`

Constructor:

```python
BatchManager(
    max_workers: int = 50,
    checkpoint_file: str | Path | None = None,
    save_interval: int = 10,  # currently not used as a periodic write; checkpoint is appended per result
)
```

Key method:

- `await process_batch(items: Iterable[Any], processor: Callable[[Any], Coroutine], desc: str = "Processing") -> list[dict[str, Any]]`

Behavior:

- Each processed result is recorded as a dict and receives `"_batch_index": int`.
- Exceptions become a record like `{"error": "...", "status": 500, "_batch_index": i}`.
- If a checkpoint file exists, it is read line-by-line as JSONL and skipped indices are not reprocessed.

### `RequestManager` (legacy alias)

`RequestManager` is a backward-compatible alias with a different constructor:

- `RequestManager(max_semaphore: int = 1000)` maps to `BatchManager(max_workers=max_semaphore)`

Legacy method:

- `await run_batch(coros: list[Coroutine] | None = None) -> list[Any]`

Returns:

- For success: the raw `result`
- For error: the error dict

---

## Dependency Injection Container (`llm_client.container`)

This module provides a lightweight DI/service-locator system and convenience factories.

### `ServiceRegistry`

Responsibilities:

- Register singleton instances and factories.
- Resolve services by Python type.

Key methods:

- `register_singleton(service_type, instance)`
- `register_factory(service_type, factory, singleton: bool = True)`
- `resolve(service_type, default=...)`
- `try_resolve(service_type) -> instance | None`
- `has(service_type) -> bool`
- `clear()`

### Provider factories

- `create_openai_provider(api_key=None, model="gpt-4o", **kwargs)`
- `create_anthropic_provider(api_key=None, model="claude-sonnet-4-20250514", **kwargs)`
- `create_provider(provider_name, model=None, **kwargs)`

These are convenience helpers that:

- build the provider config objects and
- pass through `**kwargs` to the provider constructor

### Cache factory

- `create_cache(backend="none", **kwargs) -> CacheCore`

Supported `backend` values:

- `"none"` -> `CacheCore(backend=None)`
- `"fs"` -> `FSCache`
- `"pg_redis"` -> `HybridRedisPostgreSQLCache`
- `"qdrant"` -> `QdrantCache`

### Agent factory

- `create_agent(provider=None, provider_name="openai", model=None, tools=None, system_message=None, **kwargs) -> Agent`

Creates an `AgentConfig` from selected kwargs and passes through remaining kwargs to `Agent(...)`.

### `Container`

An application-level container that stores a `ServiceRegistry` and caches common services:

- `Container.from_config(settings=None)` registers `Settings` (global settings by default).
- `openai_provider(...)`, `anthropic_provider(...)`, `provider(name, ...)`
- `cache(...)`
- `agent(...)`

Global helpers:

- `get_container() -> Container`
- `set_container(container: Container) -> None`

---

## Idempotency (`llm_client.idempotency`)

This module helps deduplicate identical or repeated requests at the application layer.

### `generate_idempotency_key(prefix="idem", include_timestamp=True) -> str`

Returns a unique string like:

- `idem_<unix_ts>_<random>`

### `compute_request_hash(...) -> str`

Computes a stable SHA-256-derived hash (first 32 hex chars) from:

- `messages`, `model`, `tools`, `temperature`, `max_tokens`, plus extra kwargs

Notes:

- Attempts to normalize messages/tool objects by calling `.to_dict()` / `.to_openai_format()` when available.
- Produces a JSON canonical form via `json.dumps(sort_keys=True)`.

### `IdempotencyTracker`

Tracks in-flight keys to avoid concurrent duplicate work.

Key methods:

- `can_start(key) -> bool`
- `start_request(key, request_hash=None) -> bool`
- `complete_request(key, result=None) -> None`
- `fail_request(key) -> None`
- `get_result(key) -> Any | None`
- `has_result(key) -> bool`
- `is_pending(key) -> bool`
- `clear()`

Properties:

- `pending_count`, `completed_count`

Global instance:

- `get_tracker() -> IdempotencyTracker`

---

## Validation (`llm_client.validation`)

This module validates common structures (messages, tools, schemas, request specs). Validation generally uses the
`jsonschema` library and returns a `ValidationResult`.

### Result types

- `ValidationResult(valid: bool, errors: list[str], warnings: list[str])`
  - truthy/falsey via `__bool__`
  - `raise_if_invalid()` raises `ValidationError`
- `ValidationError` (subclass of `ValueError`)
- `validate_or_raise(result: ValidationResult) -> None`

### Common validation functions

- `validate_message(message, config=None) -> ValidationResult`
- `validate_messages(messages, config=None) -> ValidationResult`
- `validate_tool_definition(tool, config=None) -> ValidationResult`
- `validate_tool_arguments(name, arguments_json, schema=None) -> ValidationResult`
- `validate_json_schema(schema_dict) -> ValidationResult`
- `validate_against_schema(data, schema_dict) -> ValidationResult`
- `validate_completion_response(result: CompletionResult) -> ValidationResult`
- `validate_spec(spec: RequestSpec) -> None` (raises on invalid specs)
- `validate_embedding_inputs(inputs: list[str]) -> None` (raises on invalid embeddings inputs)

Notes:

- Provider-facing APIs may still accept inputs that don’t conform perfectly to these validators (especially when passing
  raw dict messages for provider-specific features). Use validation when you want strict correctness guarantees.

---

## Persistence (`llm_client.persistence`)

This module isolates PostgreSQL access used by the hybrid cache backend.

### `PostgresRepository`

Constructor:

```python
PostgresRepository(pool: asyncpg.Pool, compress: bool = True, compression_level: int = 6)
```

Responsibilities:

- Ensures cache tables exist (`ensure_table`)
- Reads cached responses (`read`)
- Upserts cached responses (`upsert`)
- Deletes old entries (`delete_old`)

Important behaviors:

- Table name sanitization only allows `[a-zA-Z0-9_]` to avoid SQL injection in interpolated table identifiers.
- When `compress=True`, payloads are stored as `BYTEA` and zlib-compressed JSON.

---

## Telemetry, Hooks, and Logging

There are three related but distinct systems:

1) **Hooks** (`llm_client.hooks`) for engine-level event emission.
2) **Telemetry utilities** (`llm_client.telemetry`) for counters/gauges/histograms and usage aggregation.
3) **Structured logging** (`llm_client.logging`) for application logging formats and redaction.

### Hooks (`llm_client.hooks`)

#### `Hook` protocol

- `emit(event: str, payload: dict, context: Any) -> None`

#### `HookManager`

- `add(hook)`
- `emit(event, payload, context)` broadcasts to all hooks

#### Built-in hooks

- `InMemoryMetricsHook`: test/debug accumulator
- `OpenTelemetryHook`: emits spans using OpenTelemetry GenAI semantic conventions (requires `opentelemetry-*`)
- `PrometheusHook`: exports metrics via HTTP endpoint (requires `prometheus-client`)

### Telemetry (`llm_client.telemetry`)

Provides:

- `TelemetryConfig`
- metric types: `Counter`, `Gauge`, `Histogram`
- `MetricRegistry`, global `get_registry()`/`set_registry()`
- usage tracking: `RequestUsage`, `SessionUsage`, `UsageTracker`, global `get_usage_tracker()`
- `LatencyRecorder`

This module is library-internal friendly, but can also be used by applications to instrument usage and latency.

### Structured logging (`llm_client.logging`)

Provides:

- `StructuredLogger`, `get_logger()`, `configure_logging(...)`
- formatters: `JSONFormatter`, `TextFormatter`
- helpers:
  - IDs: `generate_trace_id()`, `generate_request_id()`
  - redaction: `redact_api_key(...)`
  - size control: `truncate_for_log(...)`
  - timing: `Timer`, `timed`, `log_timing`

Security note:

- Logging can include request/response contents. Use redaction and disable prompt logging for sensitive workloads.

---

## Errors and Exceptions

### Error reporting in results

Most operations return a `CompletionResult` (or stream `ERROR` events) that includes:

- `status: int`
- `error: str | None`

Use:

- `if result.ok: ... else: ...`

### Taxonomy (`llm_client.errors`)

`llm_client.errors` defines structured error types (e.g., `ProviderError`, `RateLimitError`, `ValidationError`, etc.) and
helpers like `is_retryable(...)`. Not all parts of the engine currently raise these structured errors; many code paths
surface errors via `CompletionResult.error` and HTTP-like status codes.

### Exceptions

- `ResponseTimeoutError` in `llm_client.exceptions` is available for compatibility.
- Provider SDKs may raise their own exceptions; provider implementations typically catch and convert many errors into
  `CompletionResult(status=..., error=...)`, but not every path is guaranteed to wrap every exception.

---

## Cancellation Support

The package provides cooperative cancellation throughout the async stack via `CancellationToken`.

### `CancellationToken` (`llm_client.cancellation`)

```python
from llm_client import CancellationToken, CancelledError, RequestContext

token = CancellationToken()
ctx = RequestContext(cancellation_token=token)

# In another task:
token.cancel()  # Signals cancellation
```

Key methods:

- `cancel()` - Signal cancellation
- `raise_if_cancelled()` - Raises `CancelledError` if cancelled
- `is_cancelled` property - Check cancellation status

Cancellation is checked in:
- `ExecutionEngine.complete()` retry loops
- `ExecutionEngine.stream()` between chunks
- `Agent.run()` between turns
- `Tool.execute_with_timeout()`

---

## Structured Output

Extract validated JSON from LLM responses with automatic repair on validation failures.

### `StructuredOutputConfig`

```python
from llm_client import StructuredOutputConfig, extract_structured

config = StructuredOutputConfig(
    schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    max_repair_attempts=2,
)
```

### `extract_structured`

```python
result = await extract_structured(provider, messages, config)
if result.valid:
    data = result.data  # Validated dict
else:
    print(result.validation_errors)
```

### Provider convenience method

```python
result = await provider.complete_structured(
    messages,
    schema={"type": "object", ...},
    max_repair_attempts=2,
)
```

---

## Tool Middleware

Composable middleware chain for tool execution with production-ready implementations.

### Available Middleware

| Middleware | Purpose |
|------------|---------|
| `LoggingMiddleware` | Execution logging |
| `TimeoutMiddleware` | Per-tool timeouts |
| `RetryMiddleware` | Automatic retries |
| `PolicyMiddleware` | Access control |
| `BudgetMiddleware` | Cost tracking |
| `ConcurrencyLimitMiddleware` | Limit parallel executions |
| `CircuitBreakerMiddleware` | Failure protection |
| `ResultSizeMiddleware` | Limit result size |
| `RedactionMiddleware` | PII redaction |
| `TelemetryMiddleware` | OpenTelemetry spans |

### Usage

```python
from llm_client.tools import MiddlewareChain, ToolRegistry

# Use production defaults
chain = MiddlewareChain.production_defaults()

# Or compose manually
chain = MiddlewareChain([
    TimeoutMiddleware(default_timeout=30.0),
    RetryMiddleware(max_attempts=3),
    LoggingMiddleware(),
])

# Execute with middleware
registry = ToolRegistry(tools=[my_tool])
result = await registry.execute_with_middleware("tool_name", '{"arg": "value"}', middleware_chain=chain)
```

---

## Summarization Interface

Pluggable conversation summarization with async-first design.

### `Summarizer` Protocol

```python
from llm_client import Summarizer, LLMSummarizer, LLMSummarizerConfig

config = LLMSummarizerConfig(max_summary_tokens=500)
summarizer = LLMSummarizer(engine=engine, config=config)
```

### Using with Conversation

```python
from llm_client import Conversation, ConversationConfig

config = ConversationConfig(truncation_strategy="summarize")
conversation = Conversation(config=config, summarizer=summarizer)

# Get messages with summarization
messages = await conversation.get_messages_async(
    model=model_profile,
    truncation="summarize",
)
```

### Sync wrappers

```python
from llm_client.sync import get_messages_sync, summarize_sync

# Note: Raises RuntimeError if called from async context
messages = get_messages_sync(conversation, model)
```

---

## Limitations and Behavioral Notes

This section calls out behaviors that are important for correct usage and expectation-setting.

### Conversation summarization

`ConversationConfig.truncation_strategy` accepts `"summarize"`. Use `Conversation.get_messages_async(truncation="summarize")` with a configured `Summarizer` (either `LLMSummarizer` or a custom implementation) to enable automatic summarization.

### Tool schema inference is intentionally minimal

- Unknown Python types default to `"string"` in JSON schema.
- `dict` parameters are treated as `"object"` with no property schema.
- Docstring parsing for parameter descriptions is best-effort (it scans for lines starting with `"{param_name}:"`), not a
  full docstring parser.
- If you need strict schemas, define `Tool.parameters` manually and set `strict=True`.

### Multimodal message content is not first-class in types

`Message.content` is annotated as `str | None`, but `Message.from_dict` will accept whatever `data["content"]` is. If you
use advanced provider features (images, file blocks, etc.), prefer passing raw dict messages to provider calls and treat
the typed `Message` object model as “best-effort”.

### Engine embedding support now includes caching

- Embeddings support caching via `cache_response=True` in `engine.embed()`.
- Router-based selection for embeddings uses the first available provider.

### Caching semantics

- Cache records are treated as “hit” only if present and (by default) `error == "OK"` in cached payloads.
- Cache read/write errors are swallowed (counted in stats) so cache failures don’t crash requests by default.

### Optional dependencies

Some features require extras:

- Anthropic provider: `anthropic`
- Google provider: `google-genai`
- YAML configs: `pyyaml`
- TOML configs on older Python: `tomli`
- Telemetry hooks: `prometheus-client` / `opentelemetry-*`
- Faster JSON: `orjson`

### API pass-through and vendor evolution

Provider `**kwargs` are forwarded to underlying SDK calls in many cases. This means:

- You can use vendor-specific parameters not modeled in this library.
- You are responsible for keeping those kwargs compatible with the installed SDK versions.

---

## Public API Index (`llm_client.__init__` exports)

The top-level `llm_client` package re-exports the most important symbols.

### Agent layer

- `Agent`, `AgentConfig`, `AgentResult`, `TurnResult`, `quick_agent`

### Batch processing

- `BatchManager`, `RequestManager`

### Caching

- `CacheStats`, `FSCache`, `HybridRedisPostgreSQLCache`, `QdrantCache`

### Backward-compatible OpenAI facade

- `OpenAIClient`

### Configuration

- `load_env`

### Conversation

- `Conversation`, `ConversationConfig`

### Execution engine

- `ExecutionEngine`, `RetryConfig`

### Hooks

- `Hook`, `HookManager`, `InMemoryMetricsHook`, `OpenTelemetryHook`, `PrometheusHook`

### Model profiles

- `ModelProfile` and named profiles such as `GPT5`, `GPT5Mini`, `GPT5Nano`, `TextEmbedding3Large`, etc.

### Provider layer

- `Provider`, `BaseProvider`
- `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`
- `ANTHROPIC_AVAILABLE`, `GOOGLE_AVAILABLE`
- Types: `CompletionResult`, `EmbeddingResult`, `Message`, `MessageInput`, `Role`, `StreamEvent`, `StreamEventType`,
  `ToolCall`, `ToolCallDelta`, `Usage`, `normalize_messages`

### Rate limiting and resilience

- `Limiter`, `TokenBucket`
- `CircuitBreaker`, `CircuitBreakerConfig`

### Routing

- `ProviderRouter`, `StaticRouter`

### Hashing and serialization

- `compute_hash`, `content_hash`, `cache_key`, `int_hash`
- `canonicalize`, `fast_json_dumps`, `fast_json_loads`, `stable_json_dumps`
- `fingerprint`, `fingerprint_messages`, `get_fingerprint`, `clear_fingerprint_cache`, `FingerprintCache`

### Request spec

- `RequestContext`, `RequestSpec`

### Streaming

- `SSEAdapter`, `CallbackAdapter`, `BufferingAdapter`, `PusherStreamer`
- `collect_stream`, `format_sse_event`, `stream_to_string`

### Telemetry

- `TelemetryConfig`, `MetricRegistry`, `UsageTracker`, `LatencyRecorder`, and metric primitives (`Counter`, `Gauge`,
  `Histogram`) plus accessors (`get_registry`, `set_registry`, `get_usage_tracker`).

### Tools

- `Tool`, `ToolRegistry`, `ToolResult`, `tool`, `sync_tool`, `tool_from_function`

### Additional modules (not re-exported, but part of the package)

- `llm_client.container`: DI container + factories (`Container`, `ServiceRegistry`, `create_*`)
- `llm_client.idempotency`: request deduplication helpers (`IdempotencyTracker`, `compute_request_hash`, etc.)
- `llm_client.validation`: message/tool/spec validation (`validate_spec`, `validate_tool_definition`, etc.)
- `llm_client.persistence`: PostgreSQL repository (primarily used by the hybrid cache backend)


# Configuration Reference

`llm-client` exposes a hierarchical configuration system in `llm_client.config`. You can configure it via:

- environment variables (optionally loaded from `.env`)
- YAML/TOML config files
- programmatic `Settings(...)` objects

Important: configuration loading is **explicit**. If you rely on `.env`, call `llm_client.load_env(...)` yourself.

---

## Configuration Methods

### 0) Load `.env` (explicit)

```python
from llm_client import load_env

load_env()                # finds a .env file from current working directory
load_env(override=True)   # optionally override existing env vars
```

### 1) Environment Variables (`Settings.from_env`)

`Settings.from_env(prefix="LLM_")` reads a subset of env vars with the `LLM_` prefix and maps them into a `Settings`
object.

Example:

```bash
LLM_OPENAI_API_KEY=sk-...
LLM_OPENAI_MODEL=gpt-5
LLM_CACHE_BACKEND=fs
LLM_CACHE_DIR=./cache
LLM_LOG_LEVEL=DEBUG
LLM_METRICS_ENABLED=true
LLM_METRICS_PROVIDER=prometheus
LLM_METRICS_PROMETHEUS_PORT=8000
```

### 2) Configuration File (`Settings.from_file`)

Load from YAML or TOML:

```python
from llm_client import Settings

settings = Settings.from_file("config.yaml")
```

Notes:

- YAML requires `pyyaml` (`pip install pyyaml`).
- TOML uses stdlib `tomllib` when available; otherwise requires `tomli` (`pip install tomli`).
- The file is validated against `llm_client.config_schema.CONFIG_SCHEMA` using `jsonschema` (if available).

### 3) Programmatic (`configure`)

You can set global settings used by helpers like `get_settings()` and some legacy APIs:

```python
from llm_client import configure
from llm_client.config import LoggingConfig, OpenAIConfig, Settings

configure(
    Settings(
        openai=OpenAIConfig(api_key="..."),
        logging=LoggingConfig(level="DEBUG"),
    )
)
```

---

## What is “global configuration” used for?

Not every part of the library automatically reads `Settings`. In the current codebase:

- `llm_client.config.get_settings()` returns a global `Settings` instance (defaulting to `Settings.from_env()`).
- `OpenAIClient` (legacy facade) reads:
  - `settings.openai.default_model` / `settings.openai.api_key`
  - `settings.agent.batch_concurrency`
  - `settings.metrics.*` (to auto-attach metrics hooks)
- `Container.from_config(...)` stores settings and uses some defaults when creating providers.

Many `Settings.*` fields are “configuration data” that you can use in your own factory code, but are not automatically
plumbed into every provider/engine call unless you pass them explicitly.

---

## Environment Variable Reference

### `LLM_` variables supported by `Settings.from_env`

Provider:

- `LLM_OPENAI_API_KEY`
- `LLM_OPENAI_BASE_URL`
- `LLM_OPENAI_MODEL` (maps to `settings.openai.default_model`)
- `LLM_ANTHROPIC_API_KEY`
- `LLM_ANTHROPIC_MODEL` (maps to `settings.anthropic.default_model`)
- `LLM_GOOGLE_API_KEY`
- `LLM_GOOGLE_MODEL` (maps to `settings.google.default_model`)

Cache:

- `LLM_CACHE_BACKEND` (`none|fs|pg_redis|qdrant`)
- `LLM_CACHE_DIR` (if set, `Settings.from_env` switches `settings.cache` to the filesystem cache config)

Agent:

- `LLM_AGENT_MAX_TURNS`
- `LLM_AGENT_TOOL_TIMEOUT`
- `LLM_AGENT_BATCH_CONCURRENCY`

Logging:

- `LLM_LOG_LEVEL` (`DEBUG|INFO|WARNING|ERROR|CRITICAL`)
- `LLM_LOG_FORMAT` (`text|json`)

Metrics:

- `LLM_METRICS_ENABLED` (`true|false`)
- `LLM_METRICS_PROVIDER` (`none|prometheus|otel`)
- `LLM_METRICS_PROMETHEUS_PORT`
- `LLM_METRICS_OTEL_ENDPOINT`

### Provider SDK env vars (used by provider configs / SDKs)

Even if you do not use `Settings.from_env`, provider configs default to standard env vars:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY` (the Google provider also checks `GEMINI_API_KEY`)

Cache backends may also read:

- PostgreSQL DSN: `POSTGRES_DSN`
- Redis URL: `REDIS_URL`
- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`

---

## Configuration File Reference (YAML/TOML)

Top-level keys accepted by `Settings.from_file(...)`:

- `openai`
- `anthropic`
- `google`
- `cache`
- `agent`
- `logging`
- `metrics`
- `rate_limit`

Example (YAML):

```yaml
openai:
  api_key: "sk-..."
  default_model: "gpt-5"
  use_responses_api: false

anthropic:
  api_key: "..."
  default_model: "claude-sonnet-4-20250514"

google:
  api_key: "..."
  default_model: "gemini-2.0-flash"

cache:
  backend: "fs"
  cache_dir: "./cache"
  default_collection: "default"

agent:
  max_turns: 10
  tool_timeout: 30.0
  parallel_tool_execution: true

logging:
  level: "INFO"
  format: "text"
  log_requests: true

metrics:
  enabled: false
  provider: "none"

rate_limit:
  enabled: true
  requests_per_minute: 60
  tokens_per_minute: 100000
  wait_on_limit: true
  max_wait_seconds: 60.0
```

---

## Settings Reference (complete)

This section documents every field in the config dataclasses under `llm_client.config.*`.

### Provider Configuration (`openai`, `anthropic`, `google`)

All providers inherit common fields from `ProviderConfig` (`llm_client.config.provider.ProviderConfig`):

| Field                 | Type          | Default                       | Description                                                      |
|-----------------------|---------------|-------------------------------|------------------------------------------------------------------|
| `api_key`             | `str \| None` | provider-specific env default | API key                                                          |
| `base_url`            | `str \| None` | `None`                        | Custom API base URL / endpoint                                   |
| `organization`        | `str \| None` | `None`                        | Organization / project identifier (provider-specific meaning)    |
| `timeout`             | `float`       | `60.0`                        | Request timeout seconds (not automatically applied everywhere)   |
| `max_retries`         | `int`         | `3`                           | Retry attempts (not automatically applied everywhere)            |
| `retry_backoff`       | `float`       | `1.0`                         | Retry backoff seconds (not automatically applied everywhere)     |
| `default_model`       | `str \| None` | provider-specific             | Default model key/name                                           |
| `default_temperature` | `float`       | `0.7`                         | Default temperature (not automatically applied everywhere)       |
| `default_max_tokens`  | `int \| None` | `None`                        | Default max output tokens (not automatically applied everywhere) |

#### OpenAI (`openai`)

Defined in `llm_client.config.provider.OpenAIConfig`:

| Field               | Type          | Default              | Description                                      |
|---------------------|---------------|----------------------|--------------------------------------------------|
| `api_key`           | `str \| None` | `env:OPENAI_API_KEY` | OpenAI API key                                   |
| `default_model`     | `str`         | `"gpt-4o"`           | Default model                                    |
| `use_responses_api` | `bool`        | `False`              | Prefer responses-style API paths where supported |

#### Anthropic (`anthropic`)

Defined in `llm_client.config.provider.AnthropicConfig`:

| Field                 | Type          | Default                      | Description                                        |
|-----------------------|---------------|------------------------------|----------------------------------------------------|
| `api_key`             | `str \| None` | `env:ANTHROPIC_API_KEY`      | Anthropic API key                                  |
| `default_model`       | `str`         | `"claude-sonnet-4-20250514"` | Default model                                      |
| `max_thinking_tokens` | `int \| None` | `None`                       | Extended thinking token budget (provider-specific) |

#### Google (`google`)

Defined in `llm_client.config.provider.GoogleConfig`:

| Field           | Type          | Default              | Description    |
|-----------------|---------------|----------------------|----------------|
| `api_key`       | `str \| None` | `env:GOOGLE_API_KEY` | Google API key |
| `default_model` | `str`         | `"gemini-2.0-flash"` | Default model  |

### Cache Configuration (`cache`)

Defined in `llm_client.config.cache`.

Common fields (`CacheConfig`):

| Field                | Type                         | Default  | Description                                                    |
|----------------------|------------------------------|----------|----------------------------------------------------------------|
| `backend`            | `none\|fs\|pg_redis\|qdrant` | `"none"` | Cache backend selection                                        |
| `enabled`            | `bool`                       | `True`   | Enable/disable caching (not automatically enforced everywhere) |
| `default_collection` | `str \| None`                | `None`   | Cache namespace / collection name                              |
| `ttl_seconds`        | `int \| None`                | `None`   | TTL in seconds (backend-specific support)                      |
| `cache_errors`       | `bool`                       | `False`  | Whether to cache error responses                               |
| `only_cache_ok`      | `bool`                       | `True`   | Whether cache reads should return only `"OK"` results          |

#### Filesystem cache (`backend: "fs"`) — `FSCacheConfig`

| Field       | Type   | Default   | Description                   |
|-------------|--------|-----------|-------------------------------|
| `cache_dir` | `Path` | `./cache` | Directory storing cache files |

#### Postgres + Redis hybrid (`backend: "pg_redis"`) — `RedisPGCacheConfig`

| Field               | Type   | Default            | Description                   |
|---------------------|--------|--------------------|-------------------------------|
| `pg_dsn`            | `str`  | `env:POSTGRES_DSN` | PostgreSQL DSN                |
| `redis_url`         | `str`  | `env:REDIS_URL`    | Redis URL                     |
| `redis_ttl_seconds` | `int`  | `86400`            | Redis TTL seconds             |
| `compress`          | `bool` | `True`             | Compress stored payloads      |
| `compression_level` | `int`  | `6`                | zlib compression level `0..9` |

#### Qdrant (`backend: "qdrant"`) — `QdrantCacheConfig`

| Field            | Type          | Default                                     | Description     |
|------------------|---------------|---------------------------------------------|-----------------|
| `qdrant_url`     | `str`         | `env:QDRANT_URL` or `http://localhost:6333` | Qdrant base URL |
| `qdrant_api_key` | `str \| None` | `env:QDRANT_API_KEY`                        | Qdrant API key  |

### Agent Configuration (`agent`)

Defined in `llm_client.config.agent.AgentConfig`.

| Field                            | Type          | Default | Description                                           |
|----------------------------------|---------------|---------|-------------------------------------------------------|
| `max_turns`                      | `int`         | `10`    | Maximum agent turns                                   |
| `max_tool_calls_per_turn`        | `int`         | `10`    | Cap tool calls executed per turn                      |
| `parallel_tool_execution`        | `bool`        | `True`  | Execute tools in parallel when possible               |
| `tool_timeout`                   | `float`       | `30.0`  | Tool execution timeout seconds                        |
| `max_tool_output_chars`          | `int \| None` | `None`  | Truncate tool output before adding to context         |
| `tool_retry_attempts`            | `int`         | `0`     | Retry count per tool call                             |
| `max_tokens`                     | `int \| None` | `None`  | Conversation token limit (enables truncation)         |
| `reserve_tokens`                 | `int`         | `2000`  | Reserve for model output when truncating              |
| `stop_on_tool_error`             | `bool`        | `False` | Stop agent if a tool fails                            |
| `include_tool_errors_in_context` | `bool`        | `True`  | Add tool errors to the conversation                   |
| `stream_tool_calls`              | `bool`        | `True`  | Whether agent streaming includes tool call events     |
| `batch_concurrency`              | `int`         | `20`    | Default concurrency used by some batch/engine helpers |

### Logging Configuration (`logging`)

Defined in `llm_client.config.logging.LoggingConfig`.

| Field               | Type                                    | Default  | Description                                           |
|---------------------|-----------------------------------------|----------|-------------------------------------------------------|
| `level`             | `DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL` | `"INFO"` | Log level                                             |
| `format`            | `text\|json`                            | `"text"` | Log format                                            |
| `log_file`          | `Path \| None`                          | `None`   | Optional log file path                                |
| `include_timestamp` | `bool`                                  | `True`   | Include timestamps                                    |
| `include_trace_id`  | `bool`                                  | `True`   | Include trace/request IDs where available             |
| `log_requests`      | `bool`                                  | `True`   | Log requests/prompts (be careful with sensitive data) |
| `log_responses`     | `bool`                                  | `True`   | Log responses                                         |
| `log_tool_calls`    | `bool`                                  | `True`   | Log tool calls                                        |
| `log_usage`         | `bool`                                  | `True`   | Log token usage/cost                                  |
| `redact_api_keys`   | `bool`                                  | `True`   | Redact API keys in logs                               |

### Metrics Configuration (`metrics`)

Defined in `llm_client.config.logging.MetricsConfig`.

| Field               | Type                     | Default        | Description                      |
|---------------------|--------------------------|----------------|----------------------------------|
| `enabled`           | `bool`                   | `False`        | Enable metrics collection        |
| `provider`          | `none\|prometheus\|otel` | `"none"`       | Metrics backend                  |
| `prometheus_port`   | `int`                    | `8000`         | Prometheus scrape port           |
| `otel_endpoint`     | `str \| None`            | `None`         | OpenTelemetry collector endpoint |
| `otel_service_name` | `str`                    | `"llm-client"` | OpenTelemetry service name       |

### Rate limit Configuration (`rate_limit`)

Defined in `llm_client.config.logging.RateLimitConfig`.

| Field                 | Type    | Default  | Description                                                          |
|-----------------------|---------|----------|----------------------------------------------------------------------|
| `enabled`             | `bool`  | `True`   | Enable/disable rate limiting (not automatically enforced everywhere) |
| `requests_per_minute` | `int`   | `60`     | Request budget per minute                                            |
| `tokens_per_minute`   | `int`   | `100000` | Token budget per minute                                              |
| `wait_on_limit`       | `bool`  | `True`   | Wait (vs fail fast) when the limit is reached                        |
| `max_wait_seconds`    | `float` | `60.0`   | Maximum wait time                                                    |

---

## Known gaps / limitations of the current configuration wiring

- `Settings.from_env(...)` only maps a subset of fields. Any field not explicitly handled there must be set via
  `Settings.from_file(...)` or programmatically.
- Many config fields (timeouts/retry defaults, rate-limit budgets, cache TTLs) are defined but are not automatically
  applied to every provider/engine call unless you pass them explicitly.
