# llm-client Observability and Redaction Guide

This guide covers hooks, lifecycle reports, metrics, replay/event surfaces, and
the redaction controls that keep those outputs safe by default.

Runnable examples:

- [11_observability_and_redaction.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/11_observability_and_redaction.py)
- [12_benchmarks.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/12_benchmarks.py)

Security background:

- [llm-client-threat-model.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-threat-model.md)
- [llm-client-secure-deployment-defaults.md](/home/namiral/Projects/Packages/intelligence-layer-bif/docs/llm-client-secure-deployment-defaults.md)

## Observability layers

The package observability surface is intentionally layered:

- hooks
- diagnostics recorders
- lifecycle reports
- runtime events
- replay
- telemetry/metrics

This lets applications consume just the level they need.

## Hooks

`HookManager` is the lowest-friction integration point. It is how the engine,
planner, benchmark harness, and other runtime surfaces emit operational events.

Use hooks when you want:

- in-memory diagnostics in tests
- custom logging
- tracing/metrics adapters
- domain-specific event forwarding

## Lifecycle reports

Use lifecycle reports when you want normalized request/session summaries rather
than raw event streams.

They are the right abstraction for:

- request dashboards
- support debugging
- usage/cost accounting
- session-level health views

## Redaction

Redaction is not an afterthought in this package. It is a first-class policy.

Core controls include:

- field classification
- provider payload capture modes
- preview modes
- payload sanitization
- tool output sanitization

The practical rule is simple: never build your own observability payloads from
raw model/tool/provider data if the package already exposes a sanitized path.

## Benchmark instrumentation

Benchmarks emit the same style of operational signals as the rest of the
runtime. That means you can treat benchmark runs as operational data, not as a
special case that bypasses your reporting stack.

## Practical recommendation

For most services:

1. attach a `LifecycleRecorder`
2. attach an engine diagnostics recorder
3. use `RedactionPolicy` everywhere observability payloads leave process
   boundaries
4. add OTel/Prometheus adapters only after the in-process lifecycle model is
   stable
