# Intelligence Layer Implementation Plan (V1)

**Repo context:** `/home/namiral/Projects/Packages/llm-client` (layers 0–2 in one repo)  
**Date:** 2026-02-05  
**Goal:** Production-grade Intelligence Layer (Layer 2: “intelligence”) built as vertical slices on top of `llm-client` (Layer 0) + `agent-runtime` (Layer 1), constrained by `intelligence-layer-constitution/` (schemas/manifests/plan-templates + laws/RFCs), and integrated as a **Funding Outreach sidebar copilot** for CanApply.

---

## Executive Summary (10–20 bullets)

- Implement the constitution’s core runtime invariant: **every request becomes `Intent → Plan → Actions(Operators) → Outcomes → Events`**, auditable without re-running an LLM (`intelligence-layer-constitution/CONSTITUTION.md`).
- Make **Postgres the system-of-record** for Intelligence Layer ledgers (`ledger.*`) and runtime control-plane tables (`runtime.*`) using the DDL patterns in `intelligence-layer-constitution/DATA-STRUCTURE.md`.
- Treat the Funding Outreach copilot as **funding-request-scoped**:
  - A thread is bound to exactly one `funding_request_id`.
  - All context derives from that request: student + professor + institution + templates + attachments + replies.
- Keep **platform MariaDB/MySQL** as domain state; Intelligence Layer reads it for context and performs platform writes only through **pre-tested, scope-limited Operators** behind a **human approval gate** (“Apply to Request”).
- Use the existing “contract artifacts” as the baseline product surface:
  - 14 intent schemas (`intelligence-layer-constitution/schemas/intents/*`)
  - 14 plan templates (`intelligence-layer-constitution/plan-templates/*`)
  - 19 operator manifests (`intelligence-layer-constitution/manifests/plugins/operators/*`)
  - 3 capability manifests (`intelligence-layer-constitution/manifests/capabilities/*`)
  - plus v1 additions for: student profile onboarding, reply interpretation, and post-turn suggestions (new schemas/templates/manifests, versioned like the baseline).
- Build a production Kernel that:
  - validates intent/plan/operator I/O with JSON Schema draft 2020-12,
  - resolves plan-template bindings (`from|const|template|computed.*`),
  - enforces **policy at intake/plan/action/outcome/apply**,
  - enforces **request-level dedupe** and **operator-level idempotency**,
  - supports **pause/resume** via a first-class **gate** protocol.
- Implement **DB-leased workers** for step execution (Postgres leasing: `FOR UPDATE SKIP LOCKED`) with optional “kick” signals via Redis/outbox.
- Treat LLM calls as operators, routed through `llm-client` for provider abstraction, caching hooks, retries/backoff, structured outputs, and telemetry normalization.
- Implement **AI credits** per constitution (deterministic + auditable + reservable):
  - charge credits from `llm-client` usage **cost_usd** using the normative rule `credits = ceil(effective_cost_usd × R)` (`R = credits_per_usd`, versioned),
  - enforce a **credit reservation (hold)** before starting any billable work (streaming-safe),
  - settle at workflow end (capture actual, release unused), idempotent by `request_key`,
  - emit consumption events/receipts so the platform backend can mirror remaining credits.
- Provide a small adapter API surface for the platform (already prototyped in `src/intelligence_layer_api/app.py`): `threads/init`, `threads/{id}/queries`, SSE `queries/{id}/events`, `actions/{id}/resolve`, cancel.
- Use **Jinja2 `.j2` prompt templates** for all LLM-backed operators (versioned, testable, auditable).
- Deliver v1 as vertical slices that directly improve Funding Outreach:
  1) Core runtime + SSE + gates (including “Apply to Request”),
  2) Student intelligence profile (incremental, schema-validated) + memory,
  3) Funding request completion (fill/update fields) with human-gated DB apply,
  4) Email review + iterative optimize loop (human-gated apply to `funding_requests`/`funding_emails`),
  5) Professor match review (alignment score + evidence),
  6) CV/SOP review from existing request attachments (no chat upload in v1),
  7) Reply interpretation + follow-up drafting (no DB writes),
  8) Next-prompt suggestions after each turn.
- Hardening is non-optional: security boundaries, idempotency, failure taxonomy, observability, reconciliation, cost controls, and test harnesses.

---

## Scope and Non-Goals

### In scope (v1)
- A production Kernel implementing the constitutional primitives and execution semantics (intent/plan/steps/operators/outcomes/events/gates).
- Postgres-backed ledgers + runtime control plane (leases, retries, outbox) aligned to `intelligence-layer-constitution/DATA-STRUCTURE.md`.
- Thread-centric platform adapter API + SSE streaming derived from ledger/events.
- Funding Outreach copilot capabilities for a single `funding_request_id`:
  - student intelligence profile onboarding/gating (incremental),
  - funding request field completion (research interest + paper metadata + connection),
  - professor match review (alignment score + evidence),
  - email review + iterative optimization loop,
  - CV/SOP/cover-letter review using existing platform attachments,
  - reply interpretation + follow-up drafting,
  - next-prompt suggestions + minimal long-term memory.
- Credits enforcement per constitution: reservation (hold) → usage events → settlement → immutable credit ledger (`billing.*`), with optional mirroring to platform backend for display/billing.
- Platform integration boundaries: context loading from platform DB, **proposal → human gate → scoped operator apply**, then UI refresh hint.
- Storage lifecycle for derived artifacts (parsed text, extracted structured JSON) with content hashing (blake3); **no chat upload** in v1.
- Observability and security baseline: structured logs, correlation IDs, metrics, traces, and egress/PII policy gates.

### Out of scope (explicitly deferred unless you confirm)
- Replacing the Funding Outreach sender/reminders pipeline (`canapply-funding` keeps Gmail send + reminders cron + reply digestion in v1).
- Chat-based document upload (v1 uses documents already attached to the funding request).
- Automatic paper URL/PDF loading for metadata extraction (v1 uses user-provided title/abstract/journal/year).
- Direct “send email” and “send follow-up” from the copilot (planned future; high-risk `external_send`).
- Email template optimization/editing (planned v2).
- A full enterprise multi-tenant control plane UI (capability pinning UI, tenant policy editors).
- Full adapter ecosystem beyond CanApply platform (Salesforce/SIS/etc).
- Kafka/Temporal/microservice mesh migration (v1 uses Postgres leasing + optional Redis/outbox).
- Complex automated evaluation harness beyond basic regression tests (can be a later “SOTA upgrades” phase).
 

---

## Inputs Reviewed (Mandatory Checklist)

### Constitution + contracts (completed)
- [x] All files under `intelligence-layer-constitution/` (158 files total)
  - [x] `intelligence-layer-constitution/HISTORICAL-NOTES.txt`
  - [x] `intelligence-layer-constitution/DANA_AI_COPILOT.md`
  - [x] `intelligence-layer-constitution/TECHNICAL-SPECS.md`
  - [x] Manifests: `intelligence-layer-constitution/manifests/**`
  - [x] Schemas: `intelligence-layer-constitution/schemas/**`
  - [x] Plan templates: `intelligence-layer-constitution/plan-templates/**`
  - [x] RFCs / playbook / blueprint:
    - `intelligence-layer-constitution/CONSTITUTION.md`
    - `intelligence-layer-constitution/KERNEL-RFC-V1.md`
    - `intelligence-layer-constitution/LEDGERS-RFC-V1.md`
    - `intelligence-layer-constitution/EXECUTION-RFC-V1.md`
    - `intelligence-layer-constitution/API-RUNTIME-DESIGN.md`
    - `intelligence-layer-constitution/DATA-STRUCTURE.md`
    - `intelligence-layer-constitution/DESIGN-AND-IMPLEMENTATION.md`
    - `intelligence-layer-constitution/IMPLEMENTATION-PLAYBOOK.md`
    - `intelligence-layer-constitution/FEATURE-DEVELOPMENT-WALKTHROUGH.md`
  - [x] Demos: `intelligence-layer-constitution/demo/**`

### Current platform/kernel/runtime repos (completed)
- [x] Layer 0: `src/llm_client/**` + `docs/ARCHITECTURE.md` + `docs/IMPLEMENTATION.md`
- [x] Layer 1: `src/agent_runtime/**` + storage/event bus + replay + policy + plugins
- [x] Layer 2 prototypes:
  - [x] `src/intelligence_layer_api/**` (FastAPI + SSE + Postgres)
  - [x] `src/intelligence_layer_ops/**` (platform DB access + tool plugin)
  - [x] `src/stuff/student_profile.json` (StudentProfileV2 schema for incremental onboarding)
  - [x] `test_intelligence_layer.py` (API contract runner)
- [x] Cross-repos in this workspace:
  - [x] `canapply-funding/**` (current production flows: email generation, reminders, reply digestion)
  - [x] `dana-prototype/**` (older system patterns: S3 lifecycle, LaTeX compile, usage/credits integration)
 - [x] Funding Outreach product behavior + platform DDL excerpts (provided in this conversation, 2026-02-05)

### Missing references noted (not blocking, but tracked)
- `intelligence-layer-constitution/README.md` + `STATUS.md` mention files not present in this folder: `PREREQUISITE.txt`, `PLAN.md`, `version-messy.txt`. Treat as **UNKNOWN** inputs (see Open Questions).

---

## Assumptions and Constraints

### Assumptions (DEFAULTS unless you override)
- **Single tenant v1:** `tenant_id = 1` everywhere (but still stored/required).
- **Funding-request-scoped threads:** `threads/init` accepts `funding_request_id`; `student_id`/`professor_id` are derived from platform context. When session validation is implemented, enforce `funding_requests.student_id == principal_id` before any action/apply.
- **Session validation is deferred (TODO):** the frontend will send the platform session cookie; v1 plan leaves a placeholder to validate it against platform backend/DB. Until implemented, only enable cookie-bypass in local dev.
- **Credits are enforced in IL Postgres per constitution:** IL maintains balances/reservations/usage events/credit ledger in `billing.*` and can optionally report deltas to platform backend for display/billing.
- **Credit mapping (DEFAULT):** `R = 100 credits/USD` (1 credit = $0.01) and `credits = ceil(effective_cost_usd × R)` (constitution §8.4).
- **Platform writes are operator-only and human-gated:** IL proposes concrete, typed changes and applies them via allowlisted DB-write MySQL operators only after explicit UI approval (“Apply to Request”).
- **Email sending/reminders remain in Funding Outreach** (`canapply-funding`) for v1; IL does not send emails/reminders/replies directly.
- **LLM provider:** OpenAI primary via `llm-client`, with optional fallback providers enabled later.
- **Dev storage:** local/minio-like S3-compatible store is acceptable; stage/prod uses AWS S3.
- **No chat document upload in v1:** document review flows ingest documents already attached to the funding request via `canapply_api.attachments` (S3). (Optional legacy fallback: `funding_requests.attachments` JSON during migration only.)
- **Prompts are stored as Jinja2 templates:** `.j2` files checked into repo, versioned, and referenced by operators.

### Constraints (non-negotiable)
- Constitutional laws in `intelligence-layer-constitution/CONSTITUTION.md` (workflow-first, ledgers as truth, operators-only effects, policy-first, tenancy everywhere, explicit replay semantics, capability admission contract).
- Execution semantics in `intelligence-layer-constitution/7-execution-semantics.md` (resumability, idempotency, safe partial results).
- Credits enforcement requirements in `intelligence-layer-constitution/8-ai-credits-and-budget-enforcement.md` (deterministic costing, idempotent debit, reservation strongly recommended).
- Binding resolution model in `intelligence-layer-constitution/plan-templates/README.md` must be implemented exactly (`from|const|template|computed.*`).
- Platform DB schema changes must be explicitly approved (per `intelligence-layer-constitution/HISTORICAL-NOTES.txt`).

---

## Phase 0 — Input Triage (As Required)

### 0.1 What was provided
- A working `llm-client` (Layer 0) + `agent-runtime` (Layer 1) codebase with tests.
- A complete “constitution pack” (`intelligence-layer-constitution/`) defining:
  - non-negotiable laws,
  - versioned schemas (intents/operators/outcomes/plans/SSE),
  - manifests (capabilities/operators/intents),
  - deterministic plan templates,
  - runnable demos.
- Layer 2 prototypes for FastAPI + Postgres + platform context loader.
- Two adjacent systems for cross-reference:
  - `canapply-funding` (current production funding outreach workflows, reminders, Gmail send, reply digestion).
  - `dana-prototype` (older “Dana” system with S3 lifecycle, LaTeX compilation, usage/credits patterns).

### 0.2 Codebase state (not greenfield)
- **Large/production-adjacent** for Layers 0–1 (tests, optional adapters, careful boundaries).
- **Partial** for Layer 2 (prototype endpoints + minimal IL DB tables; constitution-level kernel/ledgers/workers not implemented yet).

### 0.3 Explicit requirements (verbatim / near-verbatim)
From `intelligence-layer-constitution/CONSTITUTION.md`:
- “Every request MUST normalize into a typed **Intent**.”
- “Every Intent MUST execute as a workflow: **Intent → Plan → Actions → Outcomes → Events**.”
- “Agents MUST be pure decision functions… Any persistent change… MUST be produced via an **Action** that invokes an **Operator**, and MUST pass policy.”
- “Any side effect MUST happen only via Operators… Every operator invocation MUST be idempotent under an `idempotency_key`.”
- “All records and queries MUST be tenant-scoped (`tenant_id` required everywhere).”
- “The system MUST distinguish: **Reproduce** vs **Replay** vs **Regenerate**; UI defaults SHOULD be reproduce.”

From `intelligence-layer-constitution/7-execution-semantics.md`:
- “Resumability… Idempotency… Best safe partial value…”

From `intelligence-layer-constitution/8-ai-credits-and-budget-enforcement.md`:
- “Credits MUST be computed deterministically… Debiting MUST be idempotent… request MUST acquire a credit reservation before initiating billable operations (recommended).”

From `intelligence-layer-constitution/HISTORICAL-NOTES.txt` (platform surface + workflow expectations):
- Thread init must be idempotent by the funding-request identity (constitution notes `(student_id, funding_request_id)`; v1 API accepts `funding_request_id` and derives `student_id` from platform DB after session validation).
- Submit query returns an immediate identifier used for SSE streaming.
- All apply steps become `action_required(apply_platform_patch)` or `artifact_ready` plus an apply step.
- S3 lifecycle: temps per query, sandbox drafts per thread, finals content-addressed; blake3 hash drives dedupe/caching.

From clarified Funding Outreach copilot requirements (this conversation, 2026-02-05):
- The copilot thread is **scoped to a single `funding_request_id`**; user is not expected to switch professors within the same thread.
- Any platform DB write requested by the copilot must be **human-approved** in the UI via “Apply to Request”, then executed by a **scoped operator** with pre-tested SQL.
- V1 must collect and persist a **student intelligence profile** incrementally, validated against `src/stuff/student_profile.json` (StudentProfileV2).
- V1 has **no chat document upload**; CV/SOP review must load documents already attached to the funding request.
- Frontend expects **SSE streaming of progress stages** (not just tokens) and explicit events to show “Apply to Request”/refresh hints.
- Backend expects preflight credits remaining check and post-response consumed-credits reporting.

### 0.4 Constraints / policies (from docs + current codebase)
- **Multi-tenancy fielding is mandatory** even if single-tenant today (tenant_id everywhere).
- **Platform DB is MySQL/MariaDB**; IL must not become a second “domain truth” for the same entity.
- **Plan templates** already codify step tags/effects/cache/idempotency templates; production executor must implement binding resolution + computed hashes (`plan-templates/README.md`).
- **Operator manifests** codify policy_tags + effects; policy engine must gate on these declared properties.
- **Existing layer2 prototype** uses:
  - Postgres for runtime threads/queries + persisted event stream,
  - MySQL query via `aiomysql` for platform context,
  - `agent-runtime` RuntimeKernel for orchestration (not constitution plan templates yet).

### 0.5 Inferred requirements (clearly labeled as inferred)
- **Inferred:** We need a durable, queryable “workflow run” + “step state” model to support cross-process resume; in-process-only execution is insufficient for production crash recovery.
- **Inferred:** We need a clear integration contract for “apply patch” so platform DB writes remain **allowlisted, human-gated, idempotent, and auditable** (regardless of whether execution happens in IL or platform backend).
- **Inferred:** We need a consistent, deterministic “context hashing” strategy because plan templates rely on `computed.*` hashes and caching/idempotency depend on stable hashes.

### 0.6 Success criteria for vertical slices
A vertical slice is “done” only when it meets **all**:
- **API contract:** thread/query endpoints + SSE events reach terminal states; `test_intelligence_layer.py`-style contract passes (or an updated equivalent).
- **Ledger completeness:** you can answer “what happened?” from Postgres ledgers without rerunning LLMs (intent/plan/actions/outcomes/events).
- **Idempotency:** retries do not duplicate side effects; request-level dedupe returns/attaches to existing workflow.
- **Policy enforcement:** at least intake + per-step action policy is enforced and recorded as a first-class decision.
- **Credits enforcement:** every billable operation yields usage events and credits settlement is idempotent per request/workflow.
- **Observability:** correlation/workflow/step IDs appear in logs; key metrics emitted; trace context is propagated.
- **Failure behavior:** partial outcomes returned with explicit blocked step + reason codes; no silent success.

---

## Phase 1 — Mandatory Reading and System Recon (Completion Confirmation)

Completed as per checklist in “Inputs Reviewed”. Recon summary below is the “current platform shape” this plan must fit.

### Current Platform Recon (in this workspace)

1) **Entry points (API/CLI/workers)**
   - Layer 2 prototype API: `src/intelligence_layer_api/app.py` (FastAPI; endpoints: init/query/events/action-resolve/cancel).
   - Layer 1 runtime kernel: `src/agent_runtime/runtime.py` (in-process orchestration; persisted events optional).
   - Production funding service: `canapply-funding/src/api/routers/outreach.py` + `canapply-funding/src/reminders_worker.py`.
2) **Runtime model**
   - Async-first (FastAPI + asyncio).
   - SSE is currently implemented by polling a Postgres table (`runtime.runtime_events`) in `src/intelligence_layer_api/app.py`.
   - No constitution-grade DB-leased worker pool yet; execution is primarily in-process/prototype.
3) **Data layer**
   - IL prototype Postgres tables: `runtime.threads`, `runtime.queries` (`src/intelligence_layer_api/il_db.py`).
   - agent-runtime Postgres tables (if enabled): `runtime.runtime_events`, plus `runtime_jobs`/`runtime_actions` etc (`src/agent_runtime/storage/postgres.py`).
   - Platform DB: MariaDB/MySQL accessed via `aiomysql` (`src/intelligence_layer_ops/platform_db.py`).
4) **File/storage**
   - Layer 2 prototype: no S3/MinIO integration yet.
   - `dana-prototype` contains production-grade S3 patterns and a full temp/sandbox/final lifecycle (`dana-prototype/src/services/storage.py`).
5) **Agent framework/orchestration**
   - Layer 0 provides `ExecutionEngine` + `Agent` + tool calling + caching middleware (`src/llm_client/**`).
   - Layer 1 provides job/action/event/ledger primitives + replay + graph execution (`src/agent_runtime/**`).
   - Constitution plan-template execution exists as a demo harness (not yet wired into Layer 2): `intelligence-layer-constitution/demo/kernel_email_review.py`.
6) **Config/secrets management**
   - `.env` keys in this repo include: `IL_PG_DSN`, platform DB vars, OpenAI vars, debug flag (`docs/LAYER2.md`, `src/intelligence_layer_api/settings.py`).
   - Prototype loads `.env` automatically in API process via `llm_client.load_env()`; production should avoid implicit loading.
7) **Observability**
   - Persisted runtime events (Postgres) exist; OTel adapters exist but are not fully integrated at Layer 2 yet (`src/agent_runtime/observability/*`).
   - Contract test script validates cancellation + policy/budget deny streaming behavior (`test_intelligence_layer.py`).
8) **Existing policies**
   - agent-runtime has a baseline PolicyEngine (tool/model/constraints/redaction) but not constitution-grade effects/data-class gating (`src/agent_runtime/policy/*`).
   - Constitution requires multi-stage policy with deterministic decisions recorded as ledgers.

### Cross-Repo Recon (compatibility + reuse)

**canapply-funding (current production)**
- What exists:
  - Main email generation/paraphrasing agents using `llm-client` (`canapply-funding/src/agents/*`).
  - Draft preview generation via `/review`:
    - `OutreachLogic.get_review()` composes the main email, upserts `funding_emails` (`UPSERT_DRAFT`) and updates `funding_requests.email_subject/email_content` (`UPDATE_REQUEST`) (`canapply-funding/src/outreach/logic.py`, `canapply-funding/src/db/queries.py`).
    - `match_status` is mapped `{0:"low",1:"medium",2:"high"}`; `"medium"` follows the standard template path and `"high"` uses the detailed generation agent (`canapply-funding/src/outreach/logic.py`).
  - Gmail send + reminders/reply checks + reply digestion pipeline (`canapply-funding/src/outreach/logic.py`, `canapply-funding/src/reminders_worker.py`).
  - Direct writes to platform DB tables `funding_requests`/`funding_emails`/`funding_replies`.
- Compatibility constraints for Intelligence Layer:
  - Avoid double-send/double-reminder: any new IL `external_send` operator must be idempotent against existing platform rows and Gmail thread IDs.
  - Prefer IL generating drafts/reviews + patch proposals first; let existing pipeline remain sender until IL “send” is explicitly introduced and proven safe.
  - **Attachments (source of truth):** use `canapply_api.attachments` with `disk="s3"` and `file_path` (S3). The FTP-based path in this repo’s `canapply-funding/src/outreach/gmail.py` is legacy/outdated and should not drive IL design; keep a temporary fallback only if needed during migration.

**dana-prototype (inspiration, not a template)**
- Reusable ideas:
  - S3 temp/sandbox/final lifecycle + content-hash naming (`dana-prototype/src/services/storage.py`).
  - Async S3 wrapper with rate limits, concurrency limiting, retries, multipart upload (`dana-prototype/src/tools/async_s3.py`).
  - Document parsing/conversion engine for CV/SOP ingestion (`dana-prototype/src/agents/converter/engine.py`).
  - Expert review agents (schemas + prompts + response formats are useful baselines):
    - Email review: `dana-prototype/src/agents/email/engine.py`
    - Resume review: `dana-prototype/src/agents/resume/engine.py`
    - Letter/SOP review: `dana-prototype/src/agents/letter/engine.py`
    - Professor alignment: `dana-prototype/src/agents/alignment/engine.py`
  - Follow-up suggestion patterns and post-turn “next prompt” helpers (`dana-prototype/src/agents/orchestrator/helpers.py`).
  - LaTeX compilation pipeline for PDF exports with fallbacks (`dana-prototype/src/agents/resume/engine.py`, `dana-prototype/src/agents/letter/engine.py`).
  - Credits/usage patterns (platform backend billing RPC) as one option (`dana-prototype/src/services/usage.py`).
- What not to replicate directly:
  - Conversation-first persistence as truth; the constitution requires ledgers + explicit plans/gates/outcomes.

---

## Open Questions

### Deferred / TODO (explicitly postponed)
1) **Session validation contract (TODO):** implement cookie/session validation with the platform engineer.
   - Why it matters: production authz boundary (prevents IDOR) and determines scopes/trust-level for policy.
   - Placeholder in plan: `AuthAdapter` interface + dev-only bypass flag, stamped into ledgers/events when enabled.

### Remaining important (can be defaulted)
2) **Email draft lifecycle edge cases:** What should the copilot do when:
   - `funding_emails` row does not exist yet (draft never generated), or
   - `funding_emails.main_sent=1` (email already sent), or
   - `funding_emails.no_more_checks=1` (lifecycle ended)?
   - Default: if draft missing, instruct user to click “Generate draft” in platform UI (or offer a gated “Generate Draft Preview” action); if already sent, block “apply optimized draft” and switch to follow-up drafting mode.
3) **Tenancy model v1:** Is `tenant_id` always `1` for now, or do you already have multiple tenants/orgs?
   - Default: single tenant (`tenant_id=1`) everywhere, but keep fields required.
4) **Storage backend:** S3 vs MinIO for dev/stage; bucket/prefix conventions; encryption requirements.
   - Default: MinIO in dev; S3 in stage/prod; key scheme per `HISTORICAL-NOTES.txt`.
5) **LLM provider + model routing:** Do we pin to OpenAI only for v1 or require Anthropic fallback?
   - Default: OpenAI primary with optional Anthropic fallback via `llm-client` provider abstraction.

### Nice-to-have (can be layered later)
6) **Qdrant usage in v1:** Is retrieval required for v1 flows (memory retrieval and preference search), or is simple DB-backed memory enough for v1?
   - Default: ship v1 without Qdrant dependency; keep interfaces to add later.

If you cannot answer now, this plan proceeds with **DEFAULTS** as stated above and flags the rest as **UNKNOWN** in the relevant sections.

---

## Target Architecture

### 3.1 System map (components + responsibilities)

```
Platform Frontend/Backend (“Client”)
  |  REST: threads/init, threads/{thread}/queries, actions/{id}/resolve, cancels
  |  SSE: queries/{workflow_id}/events  (derived projection)
  v
Intelligence Layer API (FastAPI adapter)
  - Session validation (TODO; dev-only bypass toggle)
  - Normalize request -> Intent (schema-validated)
  - Request-level idempotency (request_key)
  - Credits preflight + reservation (IL `billing.*`)
  - Create workflow run + plan + initial steps
  - Enqueue runnable steps (DB-leased)
  - SSE endpoint tails ledger/events projection
  v
Intelligence Layer Postgres (system of record)
  runtime.*  : threads, workflow_runs, workflow_steps, scheduled_jobs, outbox
  ledger.*   : intents, plans, actions/jobs, outcomes, events, policy_decisions, gates, documents, memory, entities
  cache.*    : cold cache entries
  billing.*  : credit balances + reservations + usage events + credit ledger + pricing snapshots (authoritative)
  registry.* : (optional) capability pins + plugin registry (can be FS-managed in v1)
  ^
  |  lease runnable steps (FOR UPDATE SKIP LOCKED)
  |
Workers (one or more processes)
  - Lease READY steps
  - Evaluate action policy (stage: action/apply)
  - Invoke operator implementation (idempotent)
  - Write Action/Job ledger + Outcomes + Events
  - Emit outbox/webhook signals (best-effort; outbox is durable)
  |
  +--> llm-client (provider gateway, streaming, structured outputs, caching hooks)
  +--> Platform Adapter
  |      - Platform context reads (MySQL/MariaDB) OR platform backend RPC
  |      - Human-gated apply: run allowlisted DB-write operators (MySQL) after explicit approval
  |      - (Optional) Report credits deltas/receipts to platform backend (mirror/display)
  +--> Object storage (S3/MinIO/other) for platform attachments + derived artifacts
  +--> (Optional) Qdrant for retrieval/memory embeddings

Truth boundary: Postgres ledgers + platform domain DB; all platform writes are human-gated and operator-only.
Projection boundary: SSE streams, chat transcript, “assistant narrative”.
```

### 3.2 Data flow (request → orchestration → tools → storage → outputs)
1) `POST /v1/threads/init`:
   - Input: `funding_request_id` (+ session cookie); **student_id is derived** from platform DB (auth validation is TODO)
   - Normalize to `Thread.Init` intent
   - Execute `Thread.Init@1.0.0` plan template → `Thread.CreateOrLoad` operator
   - Preload platform context snapshot + ensure student profile record exists
   - Return `thread_id` + onboarding/profile gate summary
2) `POST /v1/threads/{thread_id}/queries`:
   - Normalize to an Intent (via switchboard OR explicit hint)
   - Compute `request_key` (dedupe)
   - Credits preflight check + **reserve credits (hold)** in `billing.credit_reservations`
   - Persist intent + plan + initial workflow_steps
   - Return `workflow_id` immediately and SSE URL
3) Worker executes steps:
   - Load context (platform context operator)
   - Run policy checks (may create a Gate + emit `action_required`)
   - Run LLM-backed operators via `llm-client` (usage recorded)
   - Produce Outcomes and potentially patch proposals
   - On human gate: pause workflow
4) `POST /v1/actions/{action_id}/resolve`:
   - Records gate decision
   - If accepted, executes the scoped platform-write operator(s) (allowlisted MySQL operators)
   - Writes `PlatformPatch.Receipt` outcome and emits a UI refresh hint
   - Workflow resumes (or completes)
5) SSE `GET /v1/queries/{workflow_id}/events`:
   - Streams ordered events derived from ledger/event store until terminal.
6) End of workflow:
   - Compute deterministic credit usage from recorded LLM usage events
   - Settle reservation: capture actual credits, release unused; write `billing.credit_ledger` (idempotent by `request_key`)
   - (Optional) Report consumption delta/receipt to platform backend; emit `credits_settled`

### 3.3 Trust boundaries and policy enforcement points
- **Intake policy:** auth scopes, quotas, credit availability, allowed intent types.
- **Plan policy:** validate template + step effects/tags/gates; deny/transform if illegal.
- **Action policy (per step):** enforce `effects[]` + `policy_tags[]` + data classes + trust level before operator invocation.
- **Outcome policy:** enforce redaction/egress rules before returning/streaming outcomes.
- **Apply policy:** any `db_write_platform` or `external_send` requires explicit checks + approval as configured.

### 3.4 Where vertical slices plug in
- Vertical slices correspond to **capabilities** and their intent/plan templates:
  - baseline: `CoreRuntime`, `FundingOutreach`, `Documents`
  - v1 additions: `StudentProfile`, `Memory`, `Conversation` (suggestions/replies/follow-ups)
- Each new slice must ship:
  - intent schema(s),
  - operator manifest(s) + input/output schemas,
  - plan template(s),
  - operator implementations,
  - policy rules + tests.

### 3.5 Frontend event contract (SSE) (v1)
Frontend requires more than token streaming; it needs **progress + gates + UI instructions**.

Minimum v1 event types (names are illustrative; keep them versioned and stable):
- `progress`: `{stage, detail?, operator_name?, attempt?}`
- `credits_preflight`: `{estimate, remaining, decision}`
- `credits_reserved`: `{reservation_id, reserved_credits, expires_at}`
- `model_token`: streamed token fragments (existing behavior)
- `action_required`: `{action_id, action_type:"apply_platform_patch"|"collect_profile_fields", proposal?, ui_hint?}`
- `ui_refresh_required`: `{reason:"platform_state_changed"}`
- `credits_settled`: `{credits_used, credit_ledger_id, balance_after, status}`
- Terminal: `final_result` / `final_error` / `job_cancelled`

Guidelines:
- Every event must include correlation identifiers: `thread_id`, `workflow_id/query_id`, `tenant_id`, `principal_id`, `trace_id`.
- Progress stages must match the product flow (examples from your requirement): `reserving_credits`, `classifying_intent`, `parsing_cv`, `reviewing_cv`, `optimizing_email`, etc.
- Any platform write must emit an `action_required` event before apply; apply execution must emit a receipt + refresh hint.

---

## Vertical Slice Definitions

These are the “ship units” for v1, grounded in `intelligence-layer-constitution/HISTORICAL-NOTES.txt`, existing manifests/plan templates, and the clarified Funding Outreach product flow (this conversation, 2026-02-05).

### Slice 1 — Core runtime adapter API + SSE + gates (CoreRuntime@1.0.0)
- **Primary intents:** `Thread.Init`, `Workflow.Gate.Resolve`
- **User-visible behavior:** create/load thread; submit query gets `workflow_id`; SSE streams progress + tokens; UI can resolve gates (“Apply to Request”).
- **Required operators:** `Thread.CreateOrLoad`, `Workflow.Gate.Resolve`
- **Key outcomes/events:** thread status + gate summaries; `action_required` and terminal `final`.

### Slice 2 — Funding-request-scoped context + auth/credits handshake
- **Primary intents:** (adapter-level preflight) `Thread.Init`, any chat intent
- **Required operators:** `Platform.Context.Load` (must load `funding_requests` + related tables)
- **Kernel subsystems:** `AuthAdapter` (TODO; session cookie validation), `CreditsManager` (reserve→settle in `billing.*`)
- **Policy checks:** `EnsureFundingRequestOwnedBySession` (TODO), `EnsureCreditsSufficient(reserve_estimate)`
- **Output:** a stable `ContextBundle` snapshot, credit reservation/preflight events, and early-deny behavior that streams **no** model tokens.

### Slice 3 — Student intelligence profile + memory (new capability for v1)
- **Primary intents (new):** `Student.Profile.Ensure`, `Student.Profile.Collect`
- **Required operators (new):** `StudentProfile.LoadOrCreate`, `StudentProfile.Update`, `StudentProfile.Requirements.Evaluate`, `Memory.Upsert`, `Memory.Retrieve`
- **Schema:** `src/stuff/student_profile.json` (StudentProfileV2; strict incremental completion)
- **Gate:** `collect_profile_fields` (human input gate, not DB-write gate)
- **Output:** schema-valid profile stored per `student_id`; minimal “do/don’t + tone” memory retrieved into copilot context.

### Slice 4 — Funding request completion (fields update with human-gated apply)
- **Primary intent:** `Funding.Request.Fields.Update` (`intelligence-layer-constitution/schemas/intents/funding_request_fields_update.v1.json`)
- **Required operators:** `Platform.Context.Load`, `FundingRequest.Fields.Update.Propose`, `FundingRequest.Fields.Update.Apply`
- **Gate:** `apply_platform_patch` (“Apply to Request”)
- **Output:** proposal showing exact column updates (e.g., `research_interest`, `paper_title`, `journal`, `year`, `research_connection`) + apply receipt; emit a UI refresh hint.

### Slice 5 — Email review + iterative optimize loop (human-gated apply)
- **Primary intents:** `Funding.Outreach.Email.Review`, `Funding.Outreach.Email.Optimize`
- **Required operators:** `Platform.Context.Load`, `Email.ReviewDraft`, `Email.OptimizeDraft`, `FundingEmail.Draft.Update.Propose`, `FundingEmail.Draft.Update.Apply`
- **Policy checks:** `EnsureEmailPresent` (if absent, instruct user to generate via platform UI or run a gated “Generate Draft Preview” operator)
- **Gate:** `apply_platform_patch`
- **Output:** review report + optimized draft + patch proposal to update both `funding_requests` and `funding_emails` consistently; support repeated optimize→apply cycles.

### Slice 6 — Professor match review (alignment score + evidence)
- **Primary intent:** `Funding.Outreach.Alignment.Score` (`intelligence-layer-constitution/schemas/intents/funding_outreach_alignment_score.v1.json`)
- **Required operators:** `Platform.Context.Load`, `Professor.Alignment.Score` (+ optional `Funding.Outreach.Professor.Summarize`)
- **Output:** evidence-based score and rationale grounded in professor + student profile; no platform writes.

### Slice 7 — Document review from existing request attachments (CV/SOP/cover-letter)
- **Primary intents:** `Documents.Process`, `Documents.Review` (adapter maps “review my CV” to these)
- **Required operators:** `Platform.Attachments.List`, `Documents.ImportFromPlatformAttachment`, `Documents.Process`, `Documents.Review`
- **Constraints:** no chat upload in v1; attachments must be loaded from the platform request and fetched from storage (AWS S3; MinIO in dev).
- **Output:** extracted text + structured JSON + review report; optionally hydrate missing student-profile fields (non-platform write).

### Slice 8 — Replies + follow-ups + next-prompt suggestions
- **Primary intents (new):** `Funding.Outreach.Reply.Interpret`, `Funding.Outreach.FollowUp.Draft`, `Conversation.Suggestions.Generate`
- **Required operators (new):** `FundingReply.Load`, `Reply.Interpret`, `FollowUp.Draft`, `Suggestions.Generate`
- **Output:** interpretation of `funding_replies` + follow-up suggestions/drafts; next-prompt suggestions computed after each assistant turn.

---

## Plugin System Design

### 4.1 Plugin types (constitution-aligned)
- **Operators:** effectful or expensive units; only place credentials live; idempotent under `idempotency_key`.
- **Agents:** stateless decision functions (e.g., Switchboard/Planner); outputs are structured proposals.
- **Tools:** read-only helpers (parsing/scoring/retrieval); can be wrapped as operators if needed for auditability.
- **Adapters:** platform integration glue (session validation, apply patch, webhooks).

### 4.2 Registration/discovery
Baseline v1: filesystem-managed manifests under `intelligence-layer-constitution/manifests/**`.
- `intent-registry.v1.json` declares supported intent types + schema refs.
- capability manifests map intent types → plan templates.
- operator plugin manifests map operator name/version → input/output schemas + tags/effects.

Implementation plan:
- Build a `ContractRegistry` that loads:
  - schemas (`schemas/**`),
  - manifests (`manifests/**`),
  - plan templates (`plan-templates/**`),
  validates internal references, and exposes:
  - `get_intent_schema(intent_type)`
  - `get_plan_template(intent_type)`
  - `get_operator_manifest(operator_name, operator_version)`
- Build an `OperatorRegistry` that binds manifest entries to Python implementations:
  - v1: explicit mapping table in code (deterministic, auditable)
  - v1.1+: optional setuptools entrypoints for dynamic discovery

### 4.3 Permissioning / policy gates
- Operator manifests already provide `effects[]` and `policy_tags[]`; policy engine uses these as primary gating signals.
- Operators must declare a normalized `Effect` vocabulary; for v1 adopt the manifest effect strings and standardize to a canonical enum internally.

### 4.4 Versioning and compatibility
- Treat manifest version as a semver-like contract:
  - patch: implementation fixes; schema-compatible
  - minor: additive fields allowed
  - major: breaking schema changes; must ship side-by-side
- Store in ledgers:
  - operator name + version,
  - schema versions,
  - capability version,
  - policy snapshot hash,
  - model version and parameters.

---

## Credits + Ledger + Policy Enforcement Design

### 5.1 Data model (authoritative; constitution-aligned)
Adopt `intelligence-layer-constitution/DATA-STRUCTURE.md` billing schema **as-is**:
- `billing.credit_balances` (per-tenant pool and/or per-user `principal_id=student_id`)
- `billing.credit_reservations` (pre-auth holds; unique per `request_key`)
- `billing.pricing_versions` (provider pricing snapshots)
- `billing.credit_rate_versions` (credits/USD conversion rate **R**, versioned)
- `billing.usage_events` (one row per billable AI operation with provider usage + `cost_usd`)
- `billing.credit_ledger` (immutable debits/grants; unique per `request_key` for idempotent settlement)

v1 defaults (from constitution §8):
- **Credit rate:** `R = 100 credits/USD` (1 credit = $0.01), stored in `billing.credit_rate_versions`.
- **Conversion rule:** `credits = ceil(effective_cost_usd × R)` (deterministic; store both USD + credits).
- **Overhead/margin:** default `effective_cost_usd = cost_usd` (multiplier = 1.0); if you add overhead, it must be explicit and versioned.

Sanity check with your example tier:
- If a plan grants `G_credits = 500` and `R = 100 credits/USD`, the implied AI spend cap is `S_usd = G/R = $5.00`.
- For `P_usd = $100`, that corresponds to a spend coefficient `α = S/P = 0.05` (5% of revenue allocated to AI spend). If you want a different spend ceiling, change `G_credits`, `R`, and/or introduce an overhead multiplier.

### 5.2 Reservation + settlement protocol (two-phase; streaming-safe)
The constitution strongly recommends reservations for streamed responses. v1 should implement **reserve → execute → settle**:
1) **Reserve (pre-auth hold):** before any billable operator runs:
   - compute `reserve_credits` as a conservative upper bound (see §5.3),
   - attempt to create `billing.credit_reservations` for `(tenant_id, principal_id, request_key)` inside a transaction that prevents oversubscription,
   - if insufficient credits: deny early (no model tokens) or downgrade behavior if policy allows.
2) **Execute:** run workflow steps; ensure runtime budgets cannot exceed reserved envelope:
   - set `BudgetSpec.max_cost` (USD) derived from `reserve_credits / R`,
   - stop initiating new billable operations when approaching the envelope (safety valve).
3) **Settle (capture):** at workflow terminal:
   - compute `actual_credits = Σ usage_events.credits_charged` for the request/workflow,
   - atomically debit `billing.credit_balances.balance_credits -= actual_credits`,
   - append a single `billing.credit_ledger` entry with `request_key` (idempotent unique index),
   - mark reservation `captured` (or `released` if the workflow never executed billable steps / was cancelled).
4) **Expiry/recovery:** if a worker crashes mid-run:
   - reservations expire (`expires_at`) and are released by a reaper,
   - workflow can be resumed/replayed without double-debit due to `credit_ledger_request_uq`.

### 5.3 Reservation estimation (deterministic upper bound)
`reserve_credits` must be conservative and reproducible:
- Derive an **envelope** from the plan template + operator budgets:
  - per-operator `max_tokens` / `max_output_tokens`,
  - maximum retry count for billable steps,
  - maximum optimization iterations (email optimize loop),
  - fixed overhead for context packaging and safety margin.
- Convert worst-case tokens → `reserve_cost_usd` using the **current pricing snapshot** (`billing.pricing_versions`), then:
  - `reserve_credits = ceil(reserve_cost_usd × R)` and apply a small safety multiplier (e.g., +10–20%).
- Emit SSE events:
  - `credits_preflight` (estimate + decision),
  - `credits_reserved` (reservation_id + reserved_credits) once the hold is acquired.

### 5.4 Usage events and credit computation (llm-client integration)
- For every provider call, record a `billing.usage_events` row:
  - provider usage metrics (prompt/output/cached tokens),
  - `cost_usd` computed by `llm-client` (prefer provider-reported usage; conservative estimate only when missing),
  - `credits_charged = ceil(effective_cost_usd × R)`,
  - prompt template hash + model key + operator name/version for determinism.

### 5.5 Grants, expiry, and platform mirroring
- **Credit grants:** platform subscription/billing system must create grants as `billing.credit_ledger` entries (`reason_code='grant'`) and update `billing.credit_balances`.
  - Recommended integration: platform backend calls an IL admin endpoint/operator “GrantCredits” on subscription purchase/renewal.
  - Dev fallback: seed credits via a CLI/admin script.
- **Mirroring to platform backend:** after settlement, emit an outbox event (or webhook) with `{student_id, delta_credits, balance_after, request_key}` so the platform can display remaining credits without reimplementing billing logic.

### 5.6 Policy engine responsibilities (constitution-aligned)
Policy evaluation stages:
- Intake: allow intent type? quota? credits?
- Plan: are step effects/tags legal? require extra gates?
- Action: can this operator run now with these data classes/effects?
- Outcome: redact/transform before returning? citations required?
- Apply: allow platform patch / external send?

Minimum v1 policy rules:
- Deny `external_send` when data class includes `Regulated`.
- Require approval for `db_write_platform` and `external_send` unless trust level is high enough.
- Enforce per-tenant rate limits (emails/day, workflows/min, etc.) as policy outputs.
- Record every decision as a first-class `PolicyDecision` ledger record.

---

## Data Model and Migration Plan

### 6.1 Postgres schemas (target)
Adopt the logical schemas from `intelligence-layer-constitution/DATA-STRUCTURE.md`:
- `runtime.*` (threads/workflow_runs/workflow_steps/scheduled_jobs/outbox)
- `ledger.*` (intents/plans/actions/jobs/outcomes/events/documents/memory/entities/policy_decisions/gates)
- `cache.*` (cold cache entries)
- `billing.*` (credit balances + reservations + usage events + credit ledger + pricing snapshots)
- `registry.*` (optional: capability pins + plugin registry)

Additional v1 tables (needed for Funding Outreach copilot):
- `profile.student_profiles` (or `ledger.entities` specialization) keyed by `(tenant_id, student_id)` storing:
  - `profile_json` (JSONB validated against `src/stuff/student_profile.json`)
  - `schema_version`, `created_at`, `updated_at`
  - `completeness_state` (what’s missing for common intents)
 

### 6.2 Migration mechanism (decision point)
**DEFAULT:** SQL migrations checked into repo and run at startup (idempotent DDL) for v1, with a clear later path to Alembic.
- Rationale: the repo currently uses “ensure_schema” style in prototypes (`src/intelligence_layer_api/il_db.py`) and `agent-runtime` stores self-ensure tables.
- Upgrade path: once stable, introduce Alembic to manage versioned migrations and rollback plans.

### 6.3 Coexistence with current prototype tables
Current prototype tables:
- `runtime.threads`, `runtime.queries` (`src/intelligence_layer_api/il_db.py`)
- `runtime.runtime_events` (persisted events from `agent-runtime` Postgres bus)
- `runtime_jobs`, `runtime_actions`, ledger event tables from `agent-runtime` Postgres storage

Plan:
- v1 foundation introduces the constitution-aligned tables; prototype tables can be:
  - migrated into the target shape, or
  - kept temporarily with a deprecation plan.
**Avoid:** having two “event truths”. Decide a single authoritative event ledger (recommended: `ledger.events`).

### 6.4 Platform DB (MySQL/MariaDB)
- Reads:
  - Use a single, versioned `PlatformContextLoader` (current `SELECT_FUNDING_THREAD_CONTEXT` is the baseline).
- Writes:
  - v1: only via **human-gated, allowlisted operators** with fixed SQL and narrow scope (no arbitrary SQL; no agent-written queries).
  - Apply operators must use:
    - optimistic concurrency where feasible (`WHERE id=? AND updated_at=?`) to avoid clobbering user edits,
    - explicit column allowlists per operator (e.g., only `funding_requests.research_interest/paper_title/journal/year/research_connection`, and `funding_requests.email_subject/email_content` + matching `funding_emails.main_email_subject/main_email_body`),
    - idempotency keys derived from `(funding_request_id, patch_hash)` so retries don’t double-apply.

**Constraint:** `HISTORICAL-NOTES.txt` says platform DB schema changes must be checked with you. This plan does not assume any unapproved platform migrations.

### 6.5 Object storage
Use the path scheme from `HISTORICAL-NOTES.txt` (aligns with `dana-prototype` patterns):
- Temps: `.../intelligence_layer/{student_id}/temporary/{thread_id}/{workflow_id}/...`
- Sandbox: `.../intelligence_layer/{student_id}/sandbox/{thread_id}/...`
- Finals: `.../intelligence_layer/{student_id}/documents/{content_hash}.{ext}`
Rules:
- Every artifact has blake3 hash (dedupe + idempotency + caching).
- Sandbox → finals promotion is policy-gated and auditable.

v1 note: user-facing chat upload is deferred, but IL still needs storage access to:
- fetch platform attachments (CV/SOP/etc) for parsing/review, and
- store derived extracted text/JSON artifacts for reproducibility and caching.

---

## Agent Orchestration and Doc Pipeline Design

### 7.1 Orchestration model (v1)
- **Switchboard** (agent): maps user query → `intent_type` + typed `inputs` (JSON-schema validated).
  - v1: hybrid router (deterministic heuristics first, LLM JSON mode fallback), inspired by:
    - `intelligence-layer-constitution/demo/kernel_ai_outreach.py`
    - `dana-prototype` DIRECT/GUIDED/AGENTIC routing patterns
- **Planner** (deterministic): chooses the plan template for intent type and resolves it to a concrete plan instance.
- **Executor** (state machine): runs steps; supports pause/resume; writes ledgers; ensures idempotency.
- **Prompt templating (required by stack):** all LLM-backed operators render prompts from versioned Jinja2 `.j2` files with explicit input contracts (no inline prompt strings).

### 7.2 Binding resolution + computed hashes (non-negotiable)
Implement the plan-template binding model from `plan-templates/README.md`:
- Dotted-path reads: `{ "from": "context.platform.funding_request.id" }`
- Constants: `{ "const": ... }`
- Template interpolation: `{ "template": "email_draft:{tenant_id}:{thread_id}:{computed.email_body_hash}" }`
- Computed deterministic values under `computed.*`:
  - `computed.email_body_hash`
  - `computed.requested_edits_hash`
  - `computed.fields_hash`
  - etc.

Acceptance requirement: before operator invocation, the resolved payload **must** validate against the operator input schema.

### 7.3 Document pipeline (v1)
v1 document flows are **attachment-first** (no chat upload):
- Import: resolve a Funding Outreach request attachment (CV/SOP/transcript) into a document reference:
  - `Platform.Attachments.List` → find the right attachment(s)
  - `Documents.ImportFromPlatformAttachment` → create a `ledger.documents` entry pointing at the platform storage location
- Process: parse PDF/DOCX/text into canonical extracted text + structured JSON; record parser version + hashes.
- Review: run expert review (resume/letter) and emit typed review outcomes.

Deferred to v2+: optimize/compose/export/pdf generation and “apply back to platform docs”.

Reuse/inspiration:
- S3 lifecycle and key scheme: `dana-prototype/src/services/storage.py`
- Async S3 wrapper with retry/concurrency: `dana-prototype/src/tools/async_s3.py`
- Converter + review agent baselines: `dana-prototype/src/agents/converter/engine.py`, `dana-prototype/src/agents/resume/engine.py`, `dana-prototype/src/agents/letter/engine.py`

### 7.4 Memory + context handling (v1 baseline)
- ContextBundle is built by the Kernel from:
  - platform context snapshot (hashable)
  - student intelligence profile snapshot (schema-validated)
  - latest outcomes relevant to the thread
  - minimal memory preferences (if implemented)
- v1 memory baseline (DB-backed, no Qdrant required):
  - store “Do/Don’t” guardrails, tone/style, and persistent preferences keyed by `student_id`
  - retrieve top-N most relevant memories by simple tags + recency (upgrade to embeddings later)
- After each assistant turn, run `Conversation.Suggestions.Generate` to produce next-prompt suggestions for the UI.
- Retrieval (Qdrant) is optional; keep interfaces so it can be added without rewriting operators.

---

## Detailed Phased Plan (Phased, Vertical Slices)

This section is written as atomic, checkable tasks with acceptance criteria.

### Build order (required)
1) Constitution-aligned ledgers + SSE projection
2) Operator registry + policy + idempotency core
3) Platform adapter: auth + credits + context loader
4) Student profile + memory baseline (schema-validated, incremental)
5) MVP Funding Outreach copilot: email review + next-prompt suggestions
6) Platform-write slices: request completion + email optimize (human-gated)
7) Professor match + document review + replies/follow-ups
8) Production hardening + rollout + optional SOTA upgrades

---

### Phase A — Foundations (Kernel + Postgres ledgers + contracts)

**Goal:** A constitution-aligned Kernel skeleton with Postgres persistence and contract validation.

**Scope**
- Implement contract registries (schemas/manifests/plan templates).
- Implement Postgres schemas/tables (runtime + ledger + profile + billing credits/reservations/usage/ledger).
- Implement event types + SSE projection pipeline.
- Establish Jinja2 prompt template conventions and loaders.

**Exact tasks**
1) Implement `ContractRegistry` that loads and validates:
   - `intelligence-layer-constitution/schemas/**`
   - `intelligence-layer-constitution/manifests/**`
   - `intelligence-layer-constitution/plan-templates/**`
   Acceptance: a single command/test validates the baseline contracts (14 intents, 14 plan templates, 19 operator manifests) **plus any v1-added contracts** against schema and checks that all `$ref`/file refs resolve.
2) Implement Postgres DDL/migrations for:
   - `runtime.threads`, `runtime.workflow_runs`, `runtime.workflow_steps`
   - `ledger.intents`, `ledger.plans`, `ledger.events`, `ledger.outcomes`, `ledger.policy_decisions`, `ledger.gates`
   - `profile.student_profiles` (StudentProfileV2 JSONB + validation metadata)
   - `billing.credit_balances`, `billing.credit_reservations`, `billing.credit_ledger`
   - `billing.usage_events`, `billing.pricing_versions`, `billing.credit_rate_versions`
   Acceptance: idempotent migration run; schema present; basic CRUD smoke tests.
3) Define canonical IDs and trace propagation:
   - `tenant_id`, `thread_id`, `workflow_id`, `intent_id`, `plan_id`, `step_id`, `job_id` (attempt)
   Acceptance: every ledger/event row includes these fields where applicable.
4) Implement `EventWriter` and `SSEProjector`:
   - SSE stream is a projection of ledger events (or a dedicated SSE table derived from ledger events).
   - Ensure events include: `progress`, `model_token`, `action_required`, `final_result|final_error`.
   Acceptance: SSE endpoint can stream progress + tokens + final from persisted events (cross-process).
5) Implement prompt template loader:
   - Repo layout: `src/intelligence_layer_prompts/{operator_name}/{version}/*.j2`
   - Render with strict undefined variables; record template hash in usage events/outcomes.
   Acceptance: operators can reference a prompt template by a stable identifier and render deterministically.

**Files/modules to create/change (proposed)**
- New: `src/intelligence_layer_kernel/contracts/*` (registry, schema validator, loaders)
- New: `src/intelligence_layer_kernel/db/*` (migrations, connection, repositories)
- Update: `src/intelligence_layer_api/app.py` to stream from the new event store (or keep current polling with a bridge).

**Acceptance criteria**
- Contract validation suite passes.
- A “hello workflow” can:
  - write intent + plan + events
  - stream SSE
  - end in a terminal status
  - be reproduced without rerun.

**Risks + mitigations**
- Risk: duplicate event stores (agent-runtime persisted bus vs ledger.events).  
  Mitigation: choose a single source-of-truth early; provide a one-time migration/bridge if needed.

**Dependencies**
- Postgres available (`docker-compose.yml`).

---

### Phase B — Plugin framework + operator execution core

**Goal:** Operators are invokable by `(name, version)` with schema validation + idempotency + policy hooks.

**Scope**
- Operator interface + registry.
- Idempotent execution wrapper.
- Minimal policy engine skeleton with recorded decisions.

**Exact tasks**
1) Implement `Operator` base contract:
   - input: `{payload, idempotency_key, auth_context, trace_context}`
   - output: `{status, result|error, metrics, artifacts}`
   Validate I/O against schemas in `intelligence-layer-constitution/schemas/operators/*`.
2) Implement `OperatorRegistry`:
   - resolves operator manifest by name/version
   - resolves Python implementation
   - enforces allow/deny list by capability/policy.
3) Implement operator idempotency store:
   - unique key `(tenant_id, operator_name, idempotency_key)` → prior result/receipt
   - safe replay: return prior result without repeating side effects.
4) Implement policy evaluation hooks:
   - stage: plan validation
   - stage: action pre-invoke
   - stage: outcome pre-store/pre-return
   Every decision is written as `ledger.policy_decisions` + event.
5) Add prompt template binding convention for LLM-backed operators:
   - operator manifest declares prompt template id(s)
   - executor renders templates with validated inputs and records template hash in ledgers.

**Files/modules to create/change (proposed)**
- New: `src/intelligence_layer_kernel/operators/*`
- New: `src/intelligence_layer_kernel/policy/*`
- Update: integrate with `llm-client` tool middleware boundaries (avoid double-enforcement; follow `docs/FEEDBACK_IMPLEMENTATION.md` guidance).

**Acceptance criteria**
- A stub operator can be executed end-to-end with:
  - payload validation,
  - idempotency key enforcement,
  - recorded policy decision,
  - recorded job/action attempt + events.

**Risks + mitigations**
- Risk: mixing “tool” vs “operator” semantics.  
  Mitigation: treat anything that touches external systems or reads platform DB as an operator for auditability; tools remain pure helpers.

---

### Phase C — Kernel runtime (Intent → Plan → Steps) + gates

**Goal:** A production Kernel that compiles intents into concrete plans and executes/resumes steps with gates.

**Scope**
- Intent normalization + switchboard.
- Plan template resolution + binding context.
- Step state machine persistence (`runtime.workflow_steps`).
- Gates (`action_required`) + resolution (`Workflow.Gate.Resolve` intent), including:
  - `apply_platform_patch` (human approval for DB writes),
  - `collect_profile_fields` (human input gate for onboarding/profile completion).

**Exact tasks**
1) Implement switchboard:
   - deterministic router for common intents
   - LLM JSON fallback restricted to the allowlisted intent registry (baseline 14 + v1 additions)
   - validation against per-intent schema (`schemas/intents/*`)
2) Implement `Planner`:
   - selects plan template by intent type
   - resolves bindings
   - computes `computed.*` hashes deterministically
   - produces a stored `Plan` record
3) Implement `Executor`:
   - persists workflow_run + workflow_steps projection
   - executes READY steps
   - supports `WAITING_APPROVAL` for:
     - human_gate steps (`apply_platform_patch`)
     - policy_check steps that emit required inputs (`collect_fields`)
4) Implement gate resolution:
   - `POST /v1/actions/{action_id}/resolve` creates a `Workflow.Gate.Resolve` intent
   - executor resumes from gated step with original idempotency keys

**Acceptance criteria**
- A workflow can pause/resume through both gate types:
  - onboarding/profile collection gates, and
  - apply-to-platform gates.
- Reproduce/replay/regenerate semantics are explicitly represented and testable.

---

### Phase D — MVP vertical slice: Funding Outreach copilot (thread init + auth + credits + email review)

**Goal:** First shippable slice that matches the real Funding Outreach UX: a sidebar copilot scoped to one `funding_request_id`, streaming progress + tokens, and producing actionable, safe outputs.

**Scope**
- Update API contract for `threads/init` to be funding-request-scoped.
- Implement auth placeholder (TODO) + credits reservation/settlement + context load.
- Implement `Funding.Outreach.Email.Review` and post-turn suggestions.
- Implement minimal student-profile “ensure record exists” (full onboarding is Phase E).

**Exact tasks**
1) Update adapter API (`src/intelligence_layer_api/app.py`) contract:
   - `POST /v1/threads/init` accepts `funding_request_id` (keep `student_id` optional only for backward compatibility in dev).
   - Derive `student_id` (principal) from platform DB context for `funding_request_id`.
   - TODO (production): validate session cookie → verify `funding_requests.student_id == principal_id` (IDOR prevention).
   Acceptance (dev): thread init is idempotent and returns a stable `thread_id` for the request.
   Acceptance (prod): IDOR is impossible; invalid ownership returns 403 before any context is returned.
2) Implement auth/session adapter:
   - Define `AuthAdapter` interface with a dev-only bypass implementation.
   - TODO (production): implement cookie/session validation with platform engineer; propagate `{student_id, scopes, trust_level}` into `AuthContext`.
   - Debug bypass flag is default-off and stamped into ledgers/events when enabled.
3) Implement credits reservation/settlement (constitution §8):
   - Preflight + reserve: compute `reserve_credits` estimate and create `billing.credit_reservations` (idempotent by `request_key`).
   - Enforce `BudgetSpec.max_cost` derived from reserved credits so runtime cannot exceed the hold.
   - Settle: compute `actual_credits` from `billing.usage_events`, debit `billing.credit_balances`, append `billing.credit_ledger` row (idempotent by `request_key`).
   - Dev bootstrap: seed `billing.credit_balances` for test students; production bootstrap via a “GrantCredits” integration from platform billing/subscription.
4) Platform context loader operator:
   - Extend `Platform.Context.Load@1.0.0` to load full Funding Outreach context for the `funding_request_id` (baseline query: `src/intelligence_layer_ops/platform_queries.py`).
   - Include enough data for email review + professor match + request completion decisions.
5) Email review operator:
   - `Email.ReviewDraft@1.0.0` uses a Jinja2 `.j2` prompt template and produces structured output validated against `schemas/operators/email_review_draft.output.v1.json`.
6) Post-turn suggestions:
   - Implement `Conversation.Suggestions.Generate` (new intent/operator) to generate next prompts for the UI based on the conversation + current workflow outcomes.
7) SSE progress states:
   - Emit explicit `progress` events: `query_received → checking_auth → reserving_credits → loading_context → classifying_intent → running_operator:<name> → awaiting_approval → completed`.

**Acceptance criteria**
- Updated contract suite passes with:
  - idempotent thread init by `funding_request_id`,
  - SSE streams progress + tokens + terminal event.
- Credits insufficient / auth denied requests stream **no** `model_token` events.
- `Funding.Outreach.Email.Review` produces schema-valid outcomes and is reproducible.

**Risks + mitigations**
- Risk: platform session validation is not stable/available.  
  Mitigation: implement an adapter that can swap between backend RPC and direct DB reads once the session table is known.

---

### Phase E — Vertical slice: Student intelligence profile onboarding + memory

**Goal:** Collect and persist StudentProfileV2 incrementally; gate expert actions until required profile fields are present.

**Scope**
- Create/maintain `profile.student_profiles` keyed by `(tenant_id, student_id)` with JSON Schema validation against `src/stuff/student_profile.json`.
- Implement incremental collection workflow (“don’t ask everything at once”).
- Implement minimal long-term memory (preferences/guardrails) and retrieval.

**Exact tasks**
1) Implement `StudentProfile.LoadOrCreate` and `StudentProfile.Update` operators (internal DB writes only).
2) Implement `StudentProfile.Requirements.Evaluate`:
   - deterministically compute missing fields per intent type (e.g., email optimization vs professor match vs resume review).
3) Implement `Student.Profile.Collect` intent:
   - ask 1–3 targeted questions per turn,
   - write incremental updates,
   - re-run requirements evaluation until satisfied.
4) Prefill from platform data where available:
   - `students` basic fields (name/email/etc),
   - `metas.key='funding_template_initial_data'` onboarding JSON.
5) Implement minimal memory:
   - `Memory.Upsert` (write) and `Memory.Retrieve` (read) keyed by `student_id`, with explicit types: tone/style, do/don’t, long-term goals.

**Acceptance criteria**
- New student entering the copilot gets a profile record created automatically.
- When required fields are missing, the workflow pauses into a `collect_profile_fields` gate and resumes after the user answers.
- Profile remains schema-valid after every update; invalid updates are rejected with a user-facing explanation.

---

### Phase F — Vertical slice: Funding request completion (fields update with Apply to Request)

**Goal:** Let the copilot propose and apply updates to `funding_requests` fields safely and repeatably.

**Scope**
- Implement `Funding.Request.Fields.Update` end-to-end:
  - propose an update,
  - emit `action_required` (“Apply to Request”),
  - apply via allowlisted SQL operator on approval,
  - emit UI refresh hint.

**Exact tasks**
1) Implement `FundingRequest.Fields.Update.Propose`:
   - allowlist columns: `research_interest`, `paper_title`, `journal`, `year`, `research_connection`,
   - validate types and max lengths,
   - return a typed proposal with before/after values.
2) Implement `FundingRequest.Fields.Update.Apply`:
   - optimistic concurrency check on `funding_requests.updated_at` where feasible,
   - return receipt `{rows_affected, request_id, applied_at}`.
3) UX integration:
   - emit `action_required` with proposal preview and `apply_action_id`,
   - on apply, emit `ui.refresh_required` (or equivalent) so the Funding Outreach page reloads updated fields.

**Acceptance criteria**
- User can ask: “Set my research interest to X and paper title to Y”; IL proposes, user approves, platform row updates.
- Repeated apply attempts with same idempotency key do not double-apply.

---

### Phase G — Vertical slice: Email optimize loop (review → optimize → apply; repeatable)

**Goal:** Support iterative email optimization with human-gated apply to platform fields.

**Scope**
- Implement `Funding.Outreach.Email.Optimize`:
  - optimize subject/body based on current draft + review + user feedback,
  - propose/apply updates to both `funding_requests` and `funding_emails`,
  - support multiple optimization loops in one thread.

**Exact tasks**
1) Implement `Email.OptimizeDraft@1.0.0`:
   - inputs: current draft, professor context, student profile, user requested edits, optional prior review report,
   - output: optimized `{subject, body}` + rationale + diff summary.
2) Implement platform apply operators:
   - `FundingEmail.Draft.Update.Propose` (updates `funding_requests.email_subject/email_content` and `funding_emails.main_email_subject/main_email_body` together),
   - `FundingEmail.Draft.Update.Apply` (block if `funding_emails.main_sent=1`).
3) Missing draft handling:
   - If no draft exists, emit a gate instructing the user to generate draft in UI; optionally offer a gated “Generate Draft Preview” action that calls `canapply-funding` `/review`.
4) Versioning:
   - store each optimized draft as an Outcome with a version id; allow user to refer back (“use version 2 but make it shorter”).

**Acceptance criteria**
- User can iteratively optimize and apply multiple times; each apply is auditable and idempotent.
- If email already sent, the system blocks apply and pivots to follow-up drafting.

---

### Phase H — Vertical slice: Professor match review (alignment score + evidence)

**Goal:** Provide an evidence-based match score (“should I reach out?”) grounded in professor + student profile context.

**Scope**
- Implement `Funding.Outreach.Alignment.Score` + optional `Funding.Outreach.Professor.Summarize`.
- No platform writes.

**Exact tasks**
1) Implement `Professor.Alignment.Score` using `dana-prototype/src/agents/alignment/engine.py` as inspiration/baseline (or an IL-native operator that matches its schema and evidence style).
2) Ensure operator uses:
   - professor record (`funding_professors` + `funding_institutes`),
   - student profile,
   - request context (research_interest/paper metadata if present).
3) Emit outcome and next-prompt suggestions (“do you want to adjust your research interest to improve fit?”).

**Acceptance criteria**
- Alignment score output is schema-valid, evidence-based, and reproducible.

---

### Phase I — Vertical slice: CV/SOP/cover-letter review from request attachments

**Goal:** Review student documents without chat uploads by ingesting platform attachments, parsing them, and running expert review.

**Scope**
- Attachment listing + secure fetch + parsing/conversion + review.
- Optional hydration of student profile from parsed CV/SOP.

**Exact tasks**
1) Implement `Platform.Attachments.List`:
   - query `canapply_api.attachments` where `attachable_type='funding_request'` and `attachable_id=<funding_request_id>`
   - filter by attachment kind (`collection` and/or `metadata.type`), e.g. `cv|sop|transcript|portfolio`
   - enforce `student_id` matches the request’s student for defense-in-depth.
2) Implement `Documents.ImportFromPlatformAttachment`:
   - create `ledger.documents` entry referencing the platform object location and content hash.
3) Implement storage adapter for fetching bytes (AWS S3; MinIO for dev) with:
   - content hashing (blake3),
   - size limits and MIME allowlists,
   - streaming download to avoid memory spikes.
4) Implement `Documents.Process` using converter patterns (`dana-prototype/src/agents/converter/engine.py`):
   - extract text + structured JSON,
   - store derived artifacts for reproduce.
5) Implement `Documents.Review` specializations:
   - resume review and letter/SOP review (prompt templates + typed outputs).
6) (Optional) hydrate profile fields from parsed resume (internal DB write only) and re-run requirements evaluation.

**Acceptance criteria**
- “Review my CV” pulls the CV from the request, parses it, returns a structured review report, and streams progress stages.
- Missing attachment yields a helpful gate (“please upload a CV to the request first”).

---

### Phase J — Replies + follow-ups + production hardening + rollout

**Goal:** Close the loop on professor replies and ship v1 safely to production.

**Scope**
- Reply interpretation (`funding_replies`) and follow-up drafting (no DB writes in v1).
- Hardening: idempotency, retries, observability, security, and rollout controls.

**Exact tasks**
1) Implement reply loading:
   - `FundingReply.Load` reads `canapply_api.funding_replies` (by `funding_request_id`).
2) Implement `Reply.Interpret`:
   - summarizes reply intent (interview / rejection / auto-generated / needs human review),
   - aligns with platform’s existing reply digestion outputs when present.
3) Implement `FollowUp.Draft`:
   - generate suggested follow-up email text based on reply + student profile + prior thread email.
4) Implement hardening:
   - rate limiting and backpressure,
   - abuse/prompt-injection checks on untrusted inputs,
   - credit settlement retry via outbox,
   - operator timeouts, retries with jitter, circuit breakers for platform dependencies.
5) Rollout controls:
   - feature flags per slice,
   - shadow mode for apply steps,
   - canary cohorts and rollback procedures.

**Acceptance criteria**
- Reply interpretation + follow-up drafting works end-to-end and is reproducible.
- Apply steps can be fully disabled via feature flag without breaking the copilot.

---

### Phase K — SOTA upgrades (optional; clearly separated)

**Goal:** Improve quality/cost/latency without changing v1 safety contracts.

**Candidate upgrades**
- Retrieval (Qdrant) for memory/preferences and optionally professor enrichment.
- Automated eval harness + golden tests for key operators (email review/optimize, alignment, resume review).
- Token/cost optimization: context compaction, caching, speculative decoding where supported.
- Better credit estimation via learned predictors (still deterministic settlement).

---

## Phase 5 — Snippets and Examples (Illustrative Only)

### 10.1 Orchestration pseudocode (intent → plan → steps)
```text
submit_query(thread_id, message):
  auth = platform_auth(cookie) -> AuthContext(tenant_id, principal_id, scopes, trust_level)
  intent = switchboard(message, allowed_intents) -> Intent (schema validate)
  request_key = hash(intent + context_hashes + policy_snapshot)
  if workflow exists for request_key: return attach/reproduce
  reserve_credits = estimate(intent)
  if not credits.reserve(request_key, principal_id, reserve_credits): deny (no model tokens)
  set_runtime_budget(max_cost_usd = reserve_credits / R)
  plan = resolve_plan_template(intent_type) + resolve_bindings(context) + compute_hashes
  persist intent+plan+workflow_steps(status=READY/PENDING)
  enqueue worker kick (outbox)
  return workflow_id + sse_url
```

### 10.2 Idempotent operator invocation pattern
```text
invoke_operator(step):
  id_key = interpolate(step.idempotency_template, binding_ctx)
  if job exists (tenant_id, operator_name, id_key) with success:
    return stored result (no re-effect)
  policy = evaluate(action_policy, effects=step.effects, tags=step.policy_tags, data_classes=context.data_classes)
  if deny: record policy decision + fail step (non-retryable)
  run operator impl -> result
  validate output schema
  write job ledger + outcome + events
```

### 10.3 Apply gate handshake
```text
human_gate(apply_platform_patch):
  emit action_required(gate_id, proposal_outcome_ref)
  wait

resolve_action(gate_id, accepted):
  if accepted:
    platform_write_ops.apply(proposal)  # allowlisted SQL operators OR backend "apply" endpoint
    store PlatformPatch.Receipt outcome + emit ui_refresh_hint + events
    resume execution
  else:
    store gate declined + emit events + complete workflow (partial)
```

---

## Observability Plan

### Metrics (minimum)
- Workflow: count, success rate, p50/p95 latency, retries, stuck leases
- Steps/operators: success rate, latency, idempotency hit rate, cache hit rate
- Policy: denials by reason_code, approvals required count
- Billing: reservation attempts/denies, reservation expiries, usage events/day, settlement success/fail, reconciliation drift
- SSE: active streams, lag between event write and stream delivery

### Logs
- Structured logs with: `tenant_id`, `workflow_id`, `thread_id`, `intent_id`, `plan_id`, `step_id`, `job_id`, `correlation_id`.
- No raw PII in logs; log redacted views only.

### Tracing
- OpenTelemetry spans:
  - API intake
  - context building
  - plan compilation
  - each operator invocation (LLM provider spans nested)
  - platform adapter calls
  - S3/Qdrant calls

---

## Security Plan

### Authn/authz
- Platform-backed session validation; never trust frontend directly.
- `AuthContext` passed to all operators; operators enforce scopes (defense-in-depth).
- Dev-only bypass: a configuration flag may disable auth for local debugging, but must be:
  - default-off,
  - impossible to enable in prod via runtime config drift (fail closed),
  - visibly stamped into every workflow/event for audit (`auth_mode=debug_bypass`).
- Authorization invariant: `funding_request_id` must belong to the session principal (`funding_requests.student_id == principal_id`) before any context load or apply.
- Until session validation is implemented, treat the IL API as **non-production** and run it only behind trusted internal networking (no public ingress).

### Secrets handling
- No secrets in agent prompts or logs.
- Operator credentials only in worker environment; avoid loading provider keys in API if API isn’t executing LLM steps.

### Prompt injection defenses (LLM involvement)
- Strict schema validation for all LLM structured outputs.
- ContextBundle built by Kernel (no arbitrary tool reads by agents).
- For web/profile retrieval: treat external content as untrusted; apply redaction and safe quoting rules; store provenance.

### Data egress controls
- DataClass tagging (`Public/Internal/Confidential/Regulated`) and policy rules:
  - deny external_send for regulated
  - require approval for confidential
  - redact before streaming.

---

## Rollout Plan

1) **Shadow mode (recommended first):**
   - IL runs workflows and produces outcomes + proposals but does not apply platform writes.
   - Compare IL results with current canapply-funding outputs (offline evaluation).
2) **Limited apply mode:**
   - enable `apply_platform_patch` for a small cohort; require explicit approval always.
3) **Incremental capability flags:**
   - ship: auth + credits preflight + context load + email review + next-prompt suggestions
   - then: student profile onboarding gates + memory
   - then: funding request completion apply (`Funding.Request.Fields.Update`)
   - then: email optimize apply loop
   - then: professor match review
   - then: CV/SOP review from attachments
   - then: replies + follow-up drafting
4) **Canary + rollback:**
   - canary by tenant_id / student_id hash bucket
   - rollback by disabling capability pins and preserving ledgers for audit.

---

## Risks and Mitigations

- **Risk: Dual sources of truth (platform DB vs IL ledgers).**  
  Mitigation: platform DB remains materialized downstream state; all IL writes are auditable outcomes + patch receipts.
- **Risk: Missing session validation (IDOR / data leak).**  
  Mitigation: ship behind trusted internal networking only until `AuthAdapter` is implemented; require `funding_request_id` ownership checks before any context load/apply in production.
- **Risk: Double-charging or missed charging.**  
  Mitigation: reservation+settlement are idempotent by `request_key` (`billing.credit_reservations` + `billing.credit_ledger_request_uq`); usage events store USD+credits; settlement retries are safe; reconciliation job alerts on drift; platform mirroring events are idempotent by `request_key`.
- **Risk: Duplicate side effects on retries (emails, patches).**  
  Mitigation: operator idempotency keys; platform apply uses expected-constraints; store receipts; never re-send under same key.
- **Risk: PII leakage through logs/SSE.**  
  Mitigation: outcome policy stage redaction; redacted event views; structured logging hygiene.
- **Risk: Stuck leases / zombie workflows.**  
  Mitigation: lease expiry + reclaimer; scheduled_jobs/backoff; dead-letter policies.
- **Risk: Prompt drift / nondeterminism breaks reproducibility.**  
  Mitigation: store model + params + prompt versions; reproduce default; explicit regenerate semantics.

---

## Appendix — Evidence Map (File Paths)

### Constitutional law + deep specs
- `intelligence-layer-constitution/CONSTITUTION.md`
- `intelligence-layer-constitution/1-non-negotiable-principles.md`
- `intelligence-layer-constitution/2-canonical-primitives.md`
- `intelligence-layer-constitution/3-the-kernels.md`
- `intelligence-layer-constitution/4-ledgers-sources-of-truth.md`
- `intelligence-layer-constitution/5-plugin-contracts.md`
- `intelligence-layer-constitution/6-policy-engine.md`
- `intelligence-layer-constitution/7-execution-semantics.md`
- `intelligence-layer-constitution/8-ai-credits-and-budget-enforcement.md`
- `intelligence-layer-constitution/9-capabilities.md`

### Implementation blueprints / RFCs
- `intelligence-layer-constitution/DESIGN-AND-IMPLEMENTATION.md`
- `intelligence-layer-constitution/API-RUNTIME-DESIGN.md`
- `intelligence-layer-constitution/DATA-STRUCTURE.md`
- `intelligence-layer-constitution/KERNEL-RFC-V1.md`
- `intelligence-layer-constitution/LEDGERS-RFC-V1.md`
- `intelligence-layer-constitution/EXECUTION-RFC-V1.md`
- `intelligence-layer-constitution/IMPLEMENTATION-PLAYBOOK.md`
- `intelligence-layer-constitution/FEATURE-DEVELOPMENT-WALKTHROUGH.md`
- `intelligence-layer-constitution/HISTORICAL-NOTES.txt`

### Contracts: schemas/manifests/plan templates
- `intelligence-layer-constitution/schemas/INDEX.md`
- `intelligence-layer-constitution/manifests/intent-registry.v1.json`
- `intelligence-layer-constitution/manifests/capabilities/*.json`
- `intelligence-layer-constitution/manifests/plugins/operators/*.json`
- `intelligence-layer-constitution/plan-templates/*.plan.v1.json`

### Runnable demos (reference)
- `intelligence-layer-constitution/demo/kernel_email_review.py`
- `intelligence-layer-constitution/demo/kernel_ai_outreach.py`
- `intelligence-layer-constitution/demo/fastapi_email_review_app.py`

### Layer 2 prototype in this repo
- `src/intelligence_layer_api/app.py`
- `src/intelligence_layer_api/runtime_kernel.py`
- `src/intelligence_layer_api/il_db.py`
- `src/intelligence_layer_ops/platform_tools.py`
- `src/intelligence_layer_ops/platform_db.py`
- `src/intelligence_layer_ops/platform_queries.py`
- `src/stuff/student_profile.json` (StudentProfileV2 schema)
- `test_intelligence_layer.py`

### Layer 0/1 foundations in this repo
- `src/llm_client/**`
- `src/agent_runtime/**`
- `docs/ARCHITECTURE.md`
- `docs/IMPLEMENTATION.md`
- `docs/LAYER2.md`

### Cross-repo patterns
- `canapply-funding/src/outreach/logic.py` (production reminders/send flow)
- `canapply-funding/src/db/queries.py` (review/send SQL; `match_status` mapping usage)
- `canapply-funding/src/outreach/gmail.py` (legacy FTP attachment fetch path in this repo; reference only—source of truth is S3 + `canapply_api.attachments`)
- `canapply-funding/src/agents/*` (email generation/paraphrase/reply digestion patterns)
- `dana-prototype/src/services/storage.py` (S3 lifecycle)
- `dana-prototype/src/tools/async_s3.py` (S3 retry/concurrency)
- `dana-prototype/src/services/usage.py` (credits integration patterns)
- `dana-prototype/src/agents/converter/engine.py` (CV/SOP parsing/conversion baseline)
- `dana-prototype/src/agents/email/engine.py` (email review baseline)
- `dana-prototype/src/agents/resume/engine.py` and `dana-prototype/src/agents/letter/engine.py` (resume/letter review + LaTeX compile pipeline)
- `dana-prototype/src/agents/alignment/engine.py` (professor alignment baseline)
- `dana-prototype/src/agents/orchestrator/helpers.py` (follow-up + suggestion patterns)
