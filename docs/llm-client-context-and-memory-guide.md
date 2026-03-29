# llm-client Context and Memory Guide

This guide covers conversation state, context planning, summaries, memory
stores, and multi-source context assembly.

Runnable examples:

- [10_context_memory_planning.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/10_context_memory_planning.py)
- [14_sync_wrappers.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/14_sync_wrappers.py)
- [18_memory_backed_assistant.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/18_memory_backed_assistant.py)

## The layers

The package exposes several related but distinct context layers:

- `Conversation`
- context planner
- memory interfaces/stores
- summary stores
- context assembly contracts

Keep them separate mentally:

- conversation tracks turns
- planner decides what to keep
- memory stores durable/reusable facts
- summaries compress older context
- context assembly joins multiple sources into one request-ready payload

## Conversation state

Use `Conversation` when you want simple message history with truncation and
optional summarization behavior.

Use the sync wrappers only in non-async environments; the package is async
first.

## Context planning

Use `HeuristicContextPlanner` when you need a reusable context policy without
building your own planner from scratch.

It can combine:

- recent entries
- memory retrieval
- optional summarization
- persistent summaries

The runnable reference is
[10_context_memory_planning.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/10_context_memory_planning.py).

## Memory

The package gives you interfaces first, then minimal in-memory
implementations.

Out of the box:

- short-term memory store
- summary store
- retrieval strategy hook
- semantic relevance selection hook

That is enough to prototype memory-backed behavior without forcing a database
or vector engine into the base package.

## Combined assistant flow

The memory-backed assistant example shows the intended composition:

1. retrieve memory
2. plan context
3. optionally summarize/truncate
4. turn the resulting context into a prompt
5. execute through the engine

Reference:
[18_memory_backed_assistant.py](/home/namiral/Projects/Packages/intelligence-layer-bif/examples/18_memory_backed_assistant.py)

## Practical recommendation

For most projects:

1. start with `Conversation`
2. add `HeuristicContextPlanner` when prompt size matters
3. add `ShortTermMemoryStore` and `InMemorySummaryStore` for early prototypes
4. replace those stores with project-specific persistence later without
   changing the planner contract
