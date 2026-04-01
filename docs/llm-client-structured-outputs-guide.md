# llm-client Structured Outputs Guide

This guide covers schema-driven extraction, repair loops, structured streaming,
and provider-aware structured transport.

Runnable examples:

- [05_structured_extraction.py](/home/namiral/Projects/Packages/llm-client-v1/examples/05_structured_extraction.py)
- [12_benchmarks.py](/home/namiral/Projects/Packages/llm-client-v1/examples/12_benchmarks.py)

## What the package gives you

`llm_client.structured` is not just “parse JSON and hope.” It provides:

- schema validation
- response-format selection
- schema normalization
- bounded repair loops
- structured diagnostics
- structured streaming
- normalized structured result envelopes

## The normal extraction path

Use `extract_structured(...)` when you want the package to:

1. request structured output
2. validate it
3. repair it if necessary
4. return parsed data plus diagnostics

Minimal shape:

```python
result = await extract_structured(provider, messages, config)
```

## Why diagnostics matter

The structured runtime returns more than “valid or invalid.” It also tracks:

- repair attempts
- validation errors
- response kind
- structured attempt traces

That is critical in production because structured-output failures are often
prompt/schema/provider transport problems, not random parsing accidents.

## Provider-aware transport

The package chooses structured transport based on provider/model capabilities.
That means it can distinguish between:

- strict schema-capable paths
- JSON-object transport
- prompt-only structured fallback

You should rely on the package to choose this rather than hard-coding
provider-specific response-format payloads in application code.

## When to benchmark structured output

If structured output is on a user-facing or operationally critical path, do not
stop at unit tests. Measure:

- first-pass success rate
- repair success rate
- repair-attempt distribution

The benchmark reference is
[12_benchmarks.py](/home/namiral/Projects/Packages/llm-client-v1/examples/12_benchmarks.py).

## Practical recommendation

For standalone applications:

1. model the desired output as a real JSON schema
2. use `extract_structured(...)`
3. inspect diagnostics, not just `.data`
4. benchmark repair frequency before treating a schema prompt as stable
