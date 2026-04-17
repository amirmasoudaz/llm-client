# llm-client Extraction Matrix

This matrix records what generic runtime capabilities belong in `llm_client`
 versus what should remain in higher-layer consumers.

## Extraction Decision Rule

Move code into `llm_client` when it is:

- provider-agnostic or provider-normalizing
- generic across products or domains
- part of execution, tool, agent, context, observability, or model-runtime
  infrastructure
- reusable without application-specific policy

Keep code outside `llm_client` when it is:

- business-domain logic
- repo-specific orchestration or workflow policy
- UI, API-server, or transport glue
- tenant-specific persistence or billing logic
- product prompts, manifests, or operator policies

## Current Matrix

| Area | Belongs in `llm_client` | Keep outside `llm_client` |
| --- | --- | --- |
| Provider abstraction | OpenAI/Anthropic/Google provider interfaces, request translation, capability normalization | Product-specific provider routing rules tied to business policy |
| Model metadata | Model profiles, asset-backed catalog, provider inference, capability flags | Product-default model selection rules derived from tenant or workflow policy |
| Request/result types | Messages, content blocks, usage, stream events, background/conversation results | Product DTOs shaped for a specific API or UI |
| Execution engine | Retry, timeout, failover, idempotency, hooks, diagnostics | Product workflow sequencing and approval policy |
| Tools | Generic tool schema, execution runtime, middleware, hosted-tool descriptors | Domain tool implementations and business allowlists |
| Agents | Generic tool-loop runtime, memory/context integration, streaming support | Product-specific agent personas and orchestration rules |
| Context and memory | Token-aware context assembly, summarization interfaces, reusable memory primitives | Business-domain memory content and tenant-specific state |
| Observability | Request reports, lifecycle hooks, redaction helpers, replay/runtime events | Product dashboards, incident policy, business SLIs |
| Caching | Generic cache backends and storage-agnostic cache policies | Application-specific cache invalidation rules tied to product semantics |
| Persistence | Reusable generic persistence helpers for package workflows | Product job tables, tenant state, billing ledgers |
| API/web integration | None by default beyond generic helpers | FastAPI/HTTP/SSE endpoints and app-specific transport contracts |

## Agent Runtime Extraction Notes

The package should own:

- generic agent loop mechanics
- reusable tool execution behavior
- shared context assembly and truncation logic
- provider-agnostic execution policies

The package should not own:

- repo-specific agent orchestration trees
- domain task policies
- product-specific response shaping

## Intelligence Layer Extraction Notes

The package should own:

- LLM runtime kernels
- provider and engine abstractions
- reusable tooling and observability primitives

Higher layers should own:

- application semantics
- domain models
- workflow/product policies
- persistence and transport choices tied to one system

## Practical Use

Use this matrix when deciding whether a new feature belongs in the package:

1. Is it generic across domains?
2. Is it reusable without product policy?
3. Does it strengthen the runtime contract rather than a single app?

If the answer is mostly yes, it likely belongs in `llm_client`.
