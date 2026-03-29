# llm-client Examples Guide

This guide explains how to use the cookbook under
[examples/README.md](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/README.md)
as a package adoption surface.

## What The Cookbook Is For

The cookbook is not just a smoke-test folder. It serves three purposes:

- prove real package behavior against real providers and optional infra
- show the recommended way to compose package capabilities
- provide application-shaped references without promoting all workflows into
  the core package surface

## Prerequisites

Baseline:

- install the package and any needed extras
- load environment variables explicitly
- provide provider credentials for the examples you want to run

Common environment variables:

- `OPENAI_API_KEY`
- `LLM_CLIENT_EXAMPLE_PROVIDER`
- `LLM_CLIENT_EXAMPLE_MODEL`
- `LLM_CLIENT_EXAMPLE_SECONDARY_PROVIDER`
- `LLM_CLIENT_EXAMPLE_SECONDARY_MODEL`
- `LLM_CLIENT_EXAMPLE_EMBEDDINGS_PROVIDER`
- `LLM_CLIENT_EXAMPLE_EMBEDDINGS_MODEL`

Optional infrastructure:

- `LLM_CLIENT_EXAMPLE_PG_DSN` for persistence examples
- `QDRANT_URL` and optionally `QDRANT_API_KEY` for retrieval/cache examples

## Example Categories

### Core capability examples

These demonstrate stable package capabilities directly:

- `01` one-shot completion
- `02` streaming
- `03` embeddings
- `04` content blocks and envelopes
- `05` structured extraction
- `06` provider registry and routing
- `07` engine cache/retry/idempotency
- `08` tool execution modes
- `09` tool-calling agent
- `10` context and memory planning
- `11` observability and redaction
- `12` benchmarks
- `13` batch processing
- `14` sync wrappers
- `15` rate limiting
- `35` file block transport

### Application-shaped examples

These show realistic compositions of stable package primitives:

- `16` FastAPI SSE
- `17` persistence repository
- `18` memory-backed assistant
- `19` multi-provider failover gateway
- `20` RAG with citations
- `21` document review diff
- `22` human-in-the-loop approvals
- `23` async job queue + SSE
- `24` customer support copilot
- `25` incident war room assistant
- `26` research briefing agent
- `27` SQL analytics assistant
- `28` release readiness control plane
- `29` multimodal intake pipeline
- `30` eval and regression gate
- `31` partial tool failures
- `32` cache strategy showdown
- `33` compliance redaction pipeline
- `34` end-to-end mission control
- `36` direct PostgreSQL adaptor usage
- `37` tool-wrapped SQL adaptor agent

Current experimental classification:

- none of the canonical cookbook entries are marked experimental
- some are infra-heavy and application-shaped, but they are still part of the
  documented cookbook contract

## How To Interpret Success

Expected outcomes:

- core examples should demonstrate one major package capability clearly
- application-shaped examples should demonstrate composition, not just isolated
  API calls
- examples that need unavailable optional services should fail fast with clear
  setup guidance rather than silently degrade

Typical failure classes:

- missing provider credentials
- optional provider package not installed
- optional infrastructure not running
- provider quota/rate limits

These failures should be treated differently from package regressions.

## How Examples Map To Stable APIs

Examples are intended to map back to the stable package contract:

- provider usage maps to `llm_client.providers`
- engine/routing/failover maps to `llm_client.engine`
- tool examples map to `llm_client.tools`
- agent examples map to `llm_client.agent`
- memory and context examples map to `llm_client.context`,
  `llm_client.context_assembly`, and `llm_client.memory`
- observability examples map to `llm_client.observability`
- benchmark examples map to `llm_client.benchmarks`
- file transport examples map to `llm_client.content` and provider translation
  through the engine/provider surface
- service adaptor examples map to `llm_client.adapters` and
  `llm_client.adapters.tools`

If an example demonstrates an application architecture pattern, treat the
underlying stable primitive as the real package API, not the full surrounding
workflow.

## Recommended Reading Order

For a new adopter:

1. `01`, `02`, `05`
2. `07`, `08`, `09`
3. `10`, `11`, `35`
4. `36`, `37` if you plan to expose controlled service access through tools
5. application-shaped examples that match your target use case

## Per-Example Purpose Notes

The cookbook README already contains one short purpose note for every example.
Use this guide alongside that index:

- the README is the quick inventory
- this document explains how to interpret the inventory
- the package API and usage guides explain which package surface each example
  is actually demonstrating
