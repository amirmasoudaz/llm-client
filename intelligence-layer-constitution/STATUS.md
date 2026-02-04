# Repo Status (What’s covered so far)

This repository is a **constitution + specs + schemas + plan templates + runnable demos** for the CanApply Intelligence Layer (FastAPI/Python).

## What exists (high-level)

- **System “law” + architecture RFCs**
  - `CONSTITUTION.md`, `KERNEL-RFC-V1.md`, `LEDGERS-RFC-V1.md`, `EXECUTION-RFC-V1.md`
  - Deep references: `0-introduction.md` … `9-capabilities.md`
- **Master build plans**
  - `PLAN.md` (principal-level implementation plan)
  - `DESIGN-AND-IMPLEMENTATION.md` (phased blueprint / green-light criteria)
  - `IMPLEMENTATION-PLAYBOOK.md` (practical build order + checklists)
- **Data + runtime design**
  - `DATA-STRUCTURE.md` (Postgres schemas/DDLs, caches, S3, idempotency/outbox notes)
  - `API-RUNTIME-DESIGN.md` (FastAPI endpoints, SSE/webhooks, queues/workers/brokers, observability)
  - `TECHNICAL-SPECS.md` (stack constraints, infra choices)
  - `version-messy.txt`, `PREREQUISITE.txt` (historical notes / sketches; not canonical)
- **Contracts (JSON Schemas)**
  - `schemas/` defines versioned schemas for intents, plans, operators, outcomes, and SSE events.
  - Index: `schemas/INDEX.md`
- **Discovery + workflow templates**
  - `manifests/` registers intent schemas, capabilities, and operator plugin manifests.
  - `plan-templates/` defines deterministic step programs per intent.
- **Working demos (reference implementations)**
  - `demo/run_email_review_demo.py` (deterministic: schema + plan template + policy gate)
  - `demo/run_ai_outreach_chat.py` (LLM switchboard + LLM operators + token streaming + apply gate)
  - `demo/fastapi_email_review_app.py` (optional FastAPI + SSE wrapper)
- **LLM substrate**
  - `llm-client/` provides provider abstraction, streaming, tool calling, retries/backoff, hooks, caching scaffolding.

## What we proved end-to-end (demo)

- User sends **one query** → switchboard chooses `intent_type` → kernel validates intent + plan template.
- Execution emits typed **SSE-style events**: `progress`, `token_delta`, `action_required`, `final`, `error`.
- Email optimize/review can pause at `apply_platform_patch` and resume via gate resolution.

## How to run

Deterministic (no LLM):

```bash
python demo/run_email_review_demo.py
```

LLM-backed demo (offline MockLLM):

```bash
python demo/run_ai_outreach_chat.py
```

LLM-backed demo (real OpenAI provider via `llm-client`, requires deps + API key + network):

```bash
python demo/run_ai_outreach_chat.py --real --model gpt-5-nano
```

## How to add features (walkthrough)

- `FEATURE-DEVELOPMENT-WALKTHROUGH.md` explains how to add an intent/operator end-to-end using:
  - `schemas/` → `manifests/` → `plan-templates/` → operator implementation + switchboard wiring.

## What is *not* built yet (still planning / blueprint level)

- Production Kernel service (FastAPI API surface, persistence, workers, brokers) — only demo scaffolding exists here.
- Real ledgers in Postgres (append-only event/job/outcome tables) and migrations.
- Full policy engine + enforcement + audit (`PolicyDecision` ledger).
- Real platform adapters (context loader + patch apply executor), real S3/Qdrant integration, real credits/budget enforcement.

