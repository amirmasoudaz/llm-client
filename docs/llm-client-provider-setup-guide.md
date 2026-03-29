# llm-client Provider Setup Guide

This guide covers the provider side of `llm_client`: direct provider usage,
provider registry setup, and the practical environment setup for the live
cookbook examples.

Runnable examples:

- [01_one_shot_completion.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/01_one_shot_completion.py)
- [02_streaming.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/02_streaming.py)
- [03_embeddings.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/03_embeddings.py)
- [06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/06_provider_registry_and_routing.py)

## Stable imports

For new projects, prefer:

```python
from llm_client.providers import OpenAIProvider, AnthropicProvider, GoogleProvider
from llm_client.providers.types import Message, StreamEventType
from llm_client.engine import ExecutionEngine
```

## Direct provider usage

Use a provider directly when you want the thinnest path:

- one-shot completion
- raw streaming events
- embeddings
- provider-specific debugging

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
[`load_env()`](../llm_client/README.md) explicitly first.

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

The runnable reference is
[06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/06_provider_registry_and_routing.py).

## Practical recommendation

Default application pattern:

1. construct providers through the registry or directly
2. run requests through `ExecutionEngine`
3. keep direct-provider usage for low-level tests and thin scripts

That keeps the package value concentrated in one path instead of bypassing the
engine and reimplementing reliability features at the call site.
