# llm-client Routing and Failover Guide

This guide covers provider selection, failover, and the policy knobs that sit
between a request and the selected provider.

Runnable examples:

- [06_provider_registry_and_routing.py](/home/namiral/Projects/Packages/llm-client-v1/examples/06_provider_registry_and_routing.py)
- [07_engine_cache_retry_idempotency.py](/home/namiral/Projects/Packages/llm-client-v1/examples/07_engine_cache_retry_idempotency.py)

## The routing stack

There are three layers:

1. `ProviderRegistry`
2. `RegistryRouter`
3. `ExecutionEngine`

Responsibilities:

- registry: what providers exist and what they can do
- router: which providers are candidates for one request
- engine: how to execute across that ordered candidate list

## What the router considers

`RegistryRouter` can filter/rank providers by:

- explicit provider request
- model/provider compatibility
- completions/streaming/embeddings/tool-calling/structured-output capability
- ordered provider overrides
- latency tier preference
- cost tier preference
- required compliance tags
- provider health/degradation state

## What the engine adds

The engine adds execution policy on top of routing:

- retries
- fallback on configured statuses
- fallback on provider exceptions
- circuit breakers
- lifecycle diagnostics

So the router answers “who should I try,” and the engine answers “what do I do
when the first one fails?”

## Minimal failover shape

```python
router = RegistryRouter(registry=registry)
engine = ExecutionEngine(router=router)
result = await engine.complete(spec)
```

If the first provider fails with a fallback-eligible status, the engine moves
to the next selected provider.

## Ordered provider overrides

If your application wants to constrain the candidate list for one request, use
ordered overrides in request metadata instead of hard-coding a router branch.

That is the clean way to express:

- “try OpenAI, then Anthropic”
- “use only Google for this task”
- “prefer low-cost providers but stay within these two names”

## Practical patterns

Use routing when you need:

- resilience
- cost control
- provider A/B comparisons
- staged migrations
- compliance-aware provider selection

Do not use routing when you know there is exactly one valid provider and you
want the smallest possible call path.

## Recommended rollout path

1. start with a direct provider in `ExecutionEngine`
2. introduce a registry and router once you need fallback or policy selection
3. keep routing preferences declarative in request specs or engine wiring, not
   buried in business logic
