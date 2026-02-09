# CanApply Intelligence Layer — Design & Implementation Blueprint (v1)

This is the **master build document** for implementing the CanApply Intelligence Layer (“the Layer”) end‑to‑end.

It is written to be executable by an engineer/AI builder: clear phases, concrete deliverables, and **green‑light criteria** to proceed.

---

## 0) How to use this document

### 0.1 What you should read first (required)

1. `CONSTITUTION.md` — non-negotiable laws (must not break).
2. `TECHNICAL-SPECS.md` — assumed v1 stack (FastAPI + SSE, Redis, Postgres, Qdrant, S3/MinIO).
3. `DATA-STRUCTURE.md` — the data layer (DDL + ledgers + cache + billing tables).
4. `API-RUNTIME-DESIGN.md` — endpoints, SSE, webhooks, workers, observability.
5. `PREREQUISITE.txt` — practical flows and the funding outreach scenario (reference, not law).
6. Deep references as needed:
   - `2-canonical-primitives.md`
   - `4-ledgers-sources-of-truth.md`
   - `5-plugin-contracts.md`
   - `6-policy-engine.md`
   - `7-execution-semantics.md`
   - `9-capabilities.md`

### 0.2 What this document adds

This doc:

- consolidates the system behavior described in the docs + your scenario
- fixes the “what builds first?” question with explicit phases
- defines concrete module boundaries and integration contracts
- defines **green-light gates** that prevent premature scaling/complexity

---

## 1) Executive summary (v1)

- Build a **workflow runtime**: each user query becomes `Intent → Plan → Actions(Operators) → Outcomes → Events`.
- Store truth in **Postgres ledgers** (events/jobs/outcomes/documents/memory/entities/policy/gates); chat is a projection.
- Expose a **thread-centric API** for the platform:
  - `POST /v1/threads/init`
  - `POST /v1/threads/{thread_id}/queries` → returns `query_id` immediately
  - `GET /v1/queries/{query_id}/events` (SSE) → progress, tokens, action_required, artifacts, final
  - `POST /v1/actions/{action_id}/resolve` (accept/decline/provide input)
- Integrate with the platform backend for:
  - **session validation** (cookie → principal)
  - **credits/quota** (green-light to run, plus usage settlement)
  - **platform patches** (IL proposes; platform applies after user approval)
  - **webhook notifications** (IL → platform backend only, v1)
- Implement **DB-leased workers** (Postgres as the truth queue) with optional Redis Streams “kick”.
- Ship a tight v1 capability set focused on funding outreach and document workflows, with deterministic gating (onboarding, missing data, apply steps).

---

## 2) Decisions captured (v1 planning defaults)

### 2.1 Tenancy and identity

- **Single-tenant v1/dev**: `tenant_id = 1` everywhere.
- **Principal identity** comes from the platform session:
  - end-user: `principal.type = student`, `principal.id = students.id`
  - internal calls: `principal.type = service`
- **No Postgres RLS in v1**; enforce `tenant_id` scoping in application code (add RLS later if needed).

### 2.2 Who calls the Intelligence Layer

- The “client” can be platform frontend or backend. The Layer answers regardless.
- In production, the Layer validates the cookie/session via the platform backend (server-side authority).
- In dev, allow a bypass mode (feature flag) so development is not blocked.

### 2.3 Platform DB mutation rule

- The Intelligence Layer **does not directly mutate** platform tables.
- Instead it emits `action_required(apply_platform_patch)` and the platform backend executes the patch after user approval.
- The platform then acknowledges/returns a receipt to the Layer so the Layer can record the final outcome and continue execution.

### 2.4 Webhooks scope

- v1 outbound webhooks go **to the platform backend only** (no tenant-configurable endpoints, no frontend webhooks).
- We keep webhook audit tables in the Layer DB now; tenant UI and third-party endpoints are deferred.

### 2.5 Scheduled workflows scope

- v1 supports **basic scheduled jobs** (`runtime.scheduled_jobs`) for delayed retries/backoffs and internal housekeeping.
- Reply checks and reminders are out-of-scope here (implemented elsewhere; to be merged later).

### 2.6 Broker direction

- v1: DB-leased executor + optional Redis Streams “kick”.
- v1.1+: prefer Redis Streams → NATS JetStream (managed alternative: SQS).
- Implement `BrokerAdapter` + **DB outbox** now to make broker swaps non-disruptive.

---

## 3) Goals, non-goals, and v1 scope

### 3.1 v1 goals (must ship)

**Runtime**

- thread-centric UX API; each query is a resumable workflow
- ledgers are queryable; SSE streams are derived from ledgers
- pause/resume and human gates (`action_required`) are first-class
- idempotency at request and operator levels

**Funding outreach capability (minimum)**

- professor profile summarization from platform professor info + optional web URL
- alignment score between student profile/background and professor
- email draft generation, review, and optimization
- request field updates (research_interest/paper metadata/etc.) as **proposed patches** requiring approval

**Document workflows (minimum)**

- upload + dedupe by blake3 hash
- parse/digest (extract text/structured fields)
- review + optimize + compose (SOP/resume) with artifact generation
- “apply to documents” as action_required (platform executes if needed)

**Onboarding**

- deterministic gating: base profile required, background data completeness, templates finalized (later), gmail connected (later), composer prerequisites when SOP requested
- step-by-step prompts/actions to collect missing info

**Platform integration**

- session validation via cookie + platform backend call (or dev bypass)
- credits/quota check at start + usage settlement at end
- apply patch handshake for platform mutations
- platform-only outbound webhook notifications for workflow status and action prompts

### 3.2 v1 non-goals (explicitly deferred)

- tenant-managed webhook endpoints and UI
- full enterprise multi-tenancy controls (RLS, per-tenant capability pins UI)
- a second workflow engine or DAG scheduler
- full CRM/SIS adapter ecosystem
- reply checks/reminders (merge later)
- Kafka-scale event bus; Temporal; complex microservices mesh

---

## 4) Target architecture (v1)

### 4.1 High-level component diagram

```
Client (frontend or platform backend)
  |
  |  REST: threads/init, queries, actions/resolve
  |  SSE: /v1/queries/{query_id}/events
  v
FastAPI Kernel/API (the Layer)
  - intake adapter (thread/query)
  - normalize intent + build plan
  - policy evaluation + gate creation
  - append events; expose SSE streams
  - writes runtime projections (workflow_steps)
  |
  |  Postgres (truth + runtime)
  v
Postgres
  - runtime.* (threads/workflow_runs/workflow_steps/scheduled_jobs/outbox)
  - ledger.*  (events/jobs/outcomes/documents/memory/entities/policy/gates/webhooks)
  - cache.*   (cold cache)
  - billing.* (usage events; optional credit ledger for audit)
  |
  | DB-leased runnable steps
  v
Workers (operator runtime)
  - lease steps, run operators, write jobs/outcomes/events
  - store artifacts in S3/MinIO
  - optional Redis Streams “kick” for latency
  |
  +--> Redis (hot cache, rate limits, broker kick)
  +--> S3/MinIO (files: temp/sandbox/finals)
  +--> Qdrant (embeddings, if/when retrieval is used)
  |
  +--> Platform backend (session validate, credits/quota, apply patches)
       ^ outbound webhooks (platform-only)
```

### 4.2 Repository structure (recommended)

Follow `TECHNICAL-SPECS.md` suggestion:

- `apps/api/` — FastAPI app (HTTP + SSE)
- `apps/worker/` — worker entrypoints (pool configs)
- `packages/common/` — shared types, hashing, tracing, error taxonomy
- `packages/kernel/` — intent/plan/policy/executor runtime (library)
- `packages/operators/` — operator implementations + manifests
- `packages/llm_client/` — provider gateway + caching + telemetry normalization
- `packages/platform_adapter/` — platform auth/context/apply/usage reporting clients
- `packages/documents/` — parsing, chunking, export
- `packages/retrieval/` — Qdrant integration (optional v1)

---

## 5) Data architecture (authoritative)

### 5.1 Sources of truth

Truth is reconstructible from:

- `ledger.events` (append-only timeline)
- `ledger.jobs` + `ledger.job_attempts` (operator attempts; idempotency)
- `ledger.outcomes` (typed artifacts; versioned lineage)
- `ledger.policy_decisions` (deterministic governance record)
- `ledger.gates` + `ledger.gate_decisions` (action_required / approvals / user input)
- `ledger.documents` + revisions (uploads + transformations)
- `ledger.memory_facts` (user memory; typed; TTL; append + active pointer)
- `ledger.entities` (normalized entities; professor/institute if needed in IL)
- `runtime.*` projections (workflow state/leasing, threads, scheduled jobs, outbox)

See: `DATA-STRUCTURE.md`.

### 5.2 Identifiers and mappings (v1)

- `thread_id`: `runtime.threads.thread_id` (BIGINT identity), unique per `(student_id, funding_request_id)` (v1 scope).
- `query_id`: UUID; **recommend: same as `workflow_id`** for simplicity.
- `action_id`: UUID; **use `ledger.gates.gate_id`** as the canonical ID for `action_required`.

---

## 6) External API contract (v1)

This is the **platform-facing** API. Internally, it maps to kernel primitives.

Reference: `API-RUNTIME-DESIGN.md` (thread adapter + kernel submit + SSE + webhooks).

### 6.1 Thread init

`POST /v1/threads/init`

Inputs:

- `student_id: int`
- `funding_request_id: int`
- cookie header (optional in dev bypass; required in prod)

Behaviors:

- idempotent find-or-create thread by `(tenant_id, student_id, funding_request_id)`
- loads minimal platform context (base fields + meta) to decide onboarding gating
- returns thread metadata + missing requirements list (fast path)

### 6.2 Submit query

`POST /v1/threads/{thread_id}/queries`

Inputs:

- `message: str`
- optional attachments refs (S3 object_uri, document_id, or URL)

Output immediately:

- `query_id` (UUID; equals workflow_id)
- `sse_url` (query events stream)

### 6.3 Query events stream (SSE)

`GET /v1/queries/{query_id}/events`

The SSE stream emits typed events, including:

- `progress`
- `token_delta`
- `action_required`
- `artifact_ready`
- `final`
- `error`

Important:

- SSE is a projection: truth lives in ledgers.
- support reconnect with cursor (`Last-Event-ID` / `after_event_no`).

### 6.4 Resolve action (user approval / user input)

`POST /v1/actions/{action_id}/resolve`

Input:

- `status: accepted|declined`
- optional `payload`:
  - user-provided fields
  - selected option
  - platform patch application receipt

Behavior:

- records gate decision in `ledger.gate_decisions` and emits events
- resumes the blocked step if accepted

### 6.5 Documents API (Layer-owned storage)

Minimal set:

- `POST /v1/documents/upload`
- `GET /v1/documents/{document_id}`
- `GET /v1/documents/{document_id}/download` (signed URL)

### 6.6 History + usage API (nice-to-have v1)

- `GET /v1/threads/{thread_id}/history` — projection for UI
- `GET /v1/users/{student_id}/usage?from=...&to=...` — usage summaries derived from job ledger and usage events

---

## 7) Event protocol (single multiplexed SSE stream)

The stream is designed so the client can:

- show progress in real-time
- render assistant tokens
- prompt user actions with buttons/forms
- show artifacts and links
- receive a final summary and usage report

### 7.1 Common envelope

All SSE events:

- have a stable `schema_version`
- include `thread_id`, `query_id`, timestamps, and a small payload
- are replayable from ledger cursor

Recommended event types and payloads:

1) `progress`

- `percent: 0..100`
- `stage: str` (loading_context, planning, drafting, finalizing…)
- `message: str`

2) `token_delta`

- `delta: str`
- `role: assistant`

3) `action_required`

- `action_id`
- `action_type` (apply_platform_patch, upload_required, collect_fields, select_option, confirm, redirect, refresh_ui)
- `title`, `description`
- `requires_user_input: bool`
- `ui_hints` (button labels, form schema)
- `proposed_changes` (for apply_platform_patch)
- `expires_at` (optional)

4) `artifact_ready`

- `artifact_id` (can be a document revision or outcome ID)
- `kind` (pdf/docx/json/markdown)
- `storage_url` (pre-signed)
- `hash` (blake3/sha256)
- `metadata` (filename, provenance)

5) `final`

- `final_text` (optional if you stream tokens)
- `summary`
- `outcomes[]` (IDs + types)
- `usage_report` (tokens/cost/credits)
- `next_suggested_prompts` (optional)

6) `error`

- typed error object (code, category, retryable, remediation)

---

## 8) Runtime execution model (Kernel + Workers)

### 8.1 Core primitives

- **Intent**: normalized request (`ledger.intents`)
- **Plan**: explicit step list with effects/tags/cache/idempotency (`ledger.plans`)
- **Action/Job**: operator invocation attempts (`ledger.jobs`, `ledger.job_attempts`)
- **Outcome**: versioned artifacts (`ledger.outcomes`)
- **Event**: append-only truth timeline (`ledger.events`)
- **PolicyDecision**: deterministic governance (`ledger.policy_decisions`)
- **Gate**: user approval / input wait (`ledger.gates`, `ledger.gate_decisions`)

### 8.2 State machine (step-level)

Implement the states from `EXECUTION-RFC-V1.md` / `7-execution-semantics.md`:

- `PENDING → READY → RUNNING → SUCCEEDED|FAILED_*|WAITING_APPROVAL`

Rules:

- effects only happen in operators
- idempotency always enforced at operator boundary
- resumption is derived from ledgers and runtime projections

### 8.3 Idempotency (two layers)

1) **Request-level**: stable `request_key → workflow_id`
   - used to dedupe identical queries and attach to running/completed workflows
2) **Action-level**: `(tenant_id, operator_name, idempotency_key)` unique in `ledger.jobs`
   - prevents duplicate side effects on retries/crashes

### 8.4 Caching

Follow `TECHNICAL-SPECS.md`:

- Redis hot cache for ultra-fast reuse
- Postgres cold cache for durable reuse + analytics
- cache keys include: model+params, prompt version, tool schema version, normalized input, retrieval snapshot id (if used)

### 8.5 Determinism categories

- deterministic tools: cache aggressively
- “stable nondeterministic” LLM drafts: cache by input hash when policy allows; always store outputs
- external nondeterministic: cache only with TTL + provenance

---

## 9) Platform integration contracts (server-to-server)

### 9.1 Session validation / auth

**Goal:** accept calls from frontend or backend; validate session cookie via platform backend.

Recommended pattern:

- IL receives request with cookie header.
- IL calls platform backend:
  - `POST /internal/intelligence-layer/session/validate`
  - includes raw cookie header
- platform responds:
  - `ok`, `student_id`, `scopes[]`, `trust_level`, and optionally `credits/quota`

Dev bypass:

- feature flag `IL_AUTH_BYPASS=true` allows trusting `student_id` from request (dev only).

### 9.2 PlatformContextLoader (read-only)

Implement the “context loader SQL” as a platform-owned service:

- IL calls platform backend `GET /internal/intelligence-layer/context?funding_request_id=...`
- platform backend reads platform DB, returns a typed payload (student, request, professor, institute, templates, email status, metas).

This keeps platform DB access centralized and avoids coupling IL to platform schema drift.

### 9.3 Apply patch handshake (the key integration)

Flow:

1. IL proposes a patch (never writes platform DB).
2. IL emits `action_required(action_type=apply_platform_patch)` with:
   - patch target (table/entity)
   - proposed changes (field-level)
   - human-readable summary
3. Platform UI asks user for approval.
4. Platform backend applies the patch to platform DB (idempotent).
5. Platform backend calls:
   - `POST /v1/actions/{action_id}/resolve` with `accepted` and includes:
     - `platform_patch_receipt` (what was applied, timestamps, ids)
6. IL records apply outcome and resumes workflow.

Idempotency:

- platform apply endpoint must dedupe by `action_id` (gate_id) or by a stable patch hash.

### 9.4 Credits/quota enforcement (v1)

At the start of a query:

- IL requests a quota/credits “green light” from platform backend (dummy stub allowed in dev).
- IL optionally computes an estimated reserve (best-effort) and downgrades/refuses if insufficient.

During/after execution:

- IL records per-operation usage (tokens/cost) in its job ledger.
- At workflow end, IL sends a usage report to platform backend for debit:
  - idempotent by `query_id` / `request_key`

Reference: `8-ai-credits-and-budget-enforcement.md`.

### 9.5 Outbound webhooks to platform backend (v1)

The Layer may notify the platform backend of:

- workflow accepted/started/completed/failed
- gate requested (action_required)
- artifact ready
- credits low/exhausted (if platform wants proactive notifications)

Reference: `API-RUNTIME-DESIGN.md` and webhook audit tables in `DATA-STRUCTURE.md`.

---

## 10) Onboarding and user state (deterministic gating)

### 10.1 What “user state” means in v1

The Layer must track:

- what base profile fields are missing (from platform `students` row)
- what intelligence data has been collected (preferences, background, composer prereqs)
- completion flags and missing_fields per category

#### 10.1.1 Required data (v1 checklist)

This is the concrete v1 checklist derived from your scenario. The onboarding controller should treat these as requirements that can be satisfied by:

- platform base fields (authoritative in platform DB), and/or
- IL-collected structured state (memory + onboarding records), and/or
- uploaded documents (document ledger + processing outcomes)

**A) General (base profile) — required**

- `first_name`
- `last_name`
- `email`
- `mobile_number`
- `date_of_birth`
- `gender`
- `country_of_citizenship`

**B) Intelligence — personalization preferences**

- preferred name to use in documents/emails
- contact info to include in letter headers/email signature (email, phone, LinkedIn, GitHub, portfolio, ORCID, etc.)
- affiliation/titles to use (if any)

**C) Intelligence — background data (primary)**

Either:

- upload/confirm a current resume (recommended) and let the Layer extract/derive structured background, then collect missing fields, **or**
- enter manually (structured):
  - degrees (GPA, achievements, activities, thesis/awards, ranks/accreditations)
  - research interests (topics + why + reasoning)
  - work experience (role, org, location, timeline, achievements)
  - selected coursework (optionally via transcript upload and digest)
  - publications/presentations (authors, title, venue, year, type)
  - workshops/certificates (name minimum)
  - skills (list; Layer can suggest; user can edit)
  - language competence + test scores (or estimates + planned exam date)
  - references (name, email, position/organization)

**D) Composer prerequisites (SOP-specific; collected on first SOP intent)**

Only required when the user requests SOP generation. Captured once and reused via memory:

- concrete research direction (problem/method/context)
- motivation with causality (not emotion)
- academic/technical preparation
- research experience evidence and output
- fit with supervisor/group
- why institution (functional)
- career trajectory and ROI
- resilience/research temperament
- professional closing
- “avoid” list (guardrails)

**E) Templates (deferred behavior; minimal gating only)**

- `templates_finalized` boolean (how it’s produced is deferred; gate can be used later)

**F) Drivers (Gmail integration)**

- `gmail_connected` boolean:
  - in v1, derived from platform `funding_credentials` existence for `user_id=student_id`
  - onboarding cycle steps are defined, but execution can be feature-flagged

#### 10.1.2 State record shape (what to store)

Model each “requirement category” as:

- `is_complete: bool`
- `missing_fields: string[]`
- `last_updated_at`
- `source: platform|user|inferred`

And maintain top-level gates:

- `base_profile_complete`
- `background_data_complete`
- `templates_finalized`
- `gmail_connected`
- `composer_prereqs_complete` (SOP-only)

Top-level gates are listed in §10.1.2 and are evaluated at `threads/init` and per query.

### 10.2 Where the state lives

- Base profile fields live in platform DB (students table).
- “Intelligence” and onboarding state lives in IL (memory + a small derived state record).
- When IL needs to update platform fields, it proposes patches (apply_platform_patch).

### 10.3 Onboarding controller (runtime behavior)

On `threads/init` and on each query:

1. Load platform base context.
2. Load IL memory/state for the student.
3. Determine required gates for the requested intent:
   - if missing, emit `action_required(collect_fields|upload_required|select_option)` for the smallest next step
   - do not run heavy orchestration until satisfied
4. When user supplies missing data:
   - record in IL memory/state (and propose platform patch if relevant)
5. Once gates satisfied:
   - proceed to switchboard + capability workflows

This makes onboarding deterministic, resumable, and cheap.

---

## 11) Capability architecture (FundingOutreach v1)

Reference: `9-capabilities.md` + `PREREQUISITE.txt` scenarios.

### 11.1 Capability manifest (v1)

Create a `FundingOutreach@1.0.0` capability with:

- supported intents
- required entities and docs
- operator dependencies
- outcome schemas
- verification checks (preconditions and pre-apply)

#### 11.1.1 Supported intents (v1 minimum)

- `Thread.Init` (adapter-level; produces onboarding gate status)
- `Funding.Outreach.Professor.Summarize`
- `Funding.Outreach.Alignment.Score`
- `Funding.Outreach.Email.Generate`
- `Funding.Outreach.Email.Review`
- `Funding.Outreach.Email.Optimize`
- `Funding.Request.Fields.Update` (propose patch; apply is a separate action)
- `Funding.Paper.Metadata.Extract` (URL/PDF → request fields; propose patch)
- `Documents.Upload`
- `Documents.Process`
- `Documents.Review`
- `Documents.Optimize`
- `Documents.Compose.SOP`
- `Workflow.ApplyPlatformPatch` (gate resolution; records receipt)

#### 11.1.2 Core outcomes (v1 minimum)

- `Professor.Summary` (typed summary + hooks + provenance hashes)
- `Alignment.Score` (score + rationale + key matches)
- `Email.Draft` (subject/body + personalization hooks + suggested attachments)
- `Email.Review` (rubric score + issues + suggestions)
- `Email.Draft.Optimized` (new version in same lineage)
- `PlatformPatch.Proposal` (target + field diffs + human summary)
- `PlatformPatch.Receipt` (what was applied; by whom; timestamps; idempotency key)
- `Document.Processed` (missing_fields + extracted structured fields refs)
- `Artifact.PDF` (signed URL + hash + provenance)

### 11.2 Minimum operator set (v1)

All operators must:

- have typed input/output schemas
- declare effects + policy tags
- enforce idempotency
- write job/outcome records and events

Recommended v1 operators (names can be refined):

Context / platform:

1. `Platform.Context.Load`
2. `Platform.Patch.Propose` (produces patch object/outcome)

Professor:

3. `Professor.Profile.Retrieve` (from platform professor data + optional URL)
4. `Professor.Summarize` (LLM)

Alignment:

5. `Professor.Alignment.Score` (LLM/tool + rubric)

Funding request updates:

6. `Paper.Metadata.Extract` (URL/PDF → {title, year, journal, abstract})
7. `FundingRequest.Fields.Update.Propose` (creates patch for request fields)

Email:

8. `Email.GenerateDraft`
9. `Email.ReviewDraft`
10. `Email.OptimizeDraft`
11. `Email.ApplyToPlatform.Propose` (patch to funding_requests/funding_emails)

Documents:

12. `Documents.Upload`
13. `Documents.Process` (parse/digest/missing fields)
14. `Documents.Review`
15. `Documents.Optimize`
16. `Documents.Compose.SOP`
17. `Documents.Export` (pdf)
18. `Documents.ApplyToPlatform.Propose` (patch to attachments/metas if needed)

Memory:

19. `Memory.Write` (writes `ledger.memory_facts`)
20. `Memory.Read` (read-only tool/service)

#### 11.2.1 Mapping to the scenario’s named operators (alias table)

To align implementation naming with your scenario, you can treat the following as aliases (wire names) for the same operator concepts:

- `professor_profile_retrieve` → `Professor.Profile.Retrieve`
- `professor_alignment_score` → `Professor.Alignment.Score`
- `paper_metadata_extract` → `Paper.Metadata.Extract`
- `funding_request_update_fields` → `FundingRequest.Fields.Update.Propose`
- `apply_platform_patch` (action_required type) → `Email.ApplyToPlatform.Propose` / `Documents.ApplyToPlatform.Propose` (plus `Platform.Patch.Apply` on the platform side)

### 11.3 Switchboard (planner) and orchestration

Implement a deterministic “router” with an LLM assist:

- Inputs:
  - user message
  - platform context snapshot
  - IL memory snapshot
  - tool/operator registry (allowed ops)
- Output:
  - intent_type + minimal inputs
  - a plan template reference (or fully expanded plan)
  - required gates (missing data) if any

Hard rules:

- switchboard does not perform side effects
- plans must be machine-checkable; steps include effects/tags/cache/idempotency

---

## 12) Policy (v1 baseline)

Reference: `6-policy-engine.md` and constitutional laws.

### 12.1 Policy stages implemented

- intake policy
- plan policy
- action policy
- outcome policy
- apply policy

### 12.2 v1 baseline rules (defaults)

- allow read-only operators (context load, professor retrieval) under scopes
- allow draft generation/review/optimization
- require action_required for any platform mutation (apply_platform_patch)
- deny external_send by default (unless explicitly enabled later)
- enforce data-class egress rules (Regulated blocks external send)

Each policy decision is recorded in `ledger.policy_decisions` and referenced in `ledger.events`.

---

## 13) Observability and operations (v1)

Reference: `API-RUNTIME-DESIGN.md` §8.

Must ship:

- OpenTelemetry instrumentation (API + workers)
- structured logs with `{tenant_id, thread_id, workflow_id/query_id, step_id, job_id, correlation_id}`
- metrics for:
  - workflow lifecycle
  - step leasing
  - operator success/failure and latency
  - SSE connections + disconnect reasons
  - cache hits/misses
  - policy decision counts by reason_code
  - usage/cost accounting

Must ship runbooks:

- “stuck workflow” investigation (read events/jobs/outcomes)
- “duplicate side effect” prevention (idempotency checks)
- “credits settlement mismatch” resolution (idempotent settlement by query_id)

---

## 14) Testing strategy (v1)

Minimum tests to avoid shipping chaos:

- Unit tests:
  - stable hashing / cache keys
  - schema validation for intents/plans/operators/outcomes
  - policy determinism (same inputs → same decision)
- Integration tests:
  - DB leasing correctness under concurrency
  - operator idempotency enforcement (unique constraint + behavior)
  - gate pause/resume behavior
  - SSE cursor resume correctness
- End-to-end tests:
  - init thread → submit query → stream → action_required → resolve → complete
  - reproduce vs replay vs regenerate semantics on outcomes

---

## 15) Phased implementation plan (with green-light criteria)

> Each phase produces concrete artifacts. Do not start the next phase until green-light criteria are met.

### Phase 0 — Project bootstrap (S)

**Goal:** runnable skeleton with consistent conventions.

**Deliverables**

- Repo structure (`apps/`, `packages/`) per §4.2
- Docker Compose for: `postgres`, `redis`, `minio`, optional `qdrant`
- Config system: `.env` in dev, secrets placeholders for prod
- CI job skeleton (lint/test placeholder)
- Code conventions: typing (pyright), formatting (ruff/black or chosen standard)

**Green-light criteria**

- `docker compose up` brings up dependencies
- API container starts and serves `/healthz`
- Worker container starts and idles without errors

---

### Phase 1 — Data layer + migrations (M)

**Goal:** make the ledger and runtime DB real.

**Deliverables**

- Postgres migrations implementing `DATA-STRUCTURE.md` tables:
  - `runtime.threads`, `runtime.workflow_runs`, `runtime.workflow_steps`, `runtime.scheduled_jobs`, `runtime.outbox`
  - `ledger.*` core ledgers and webhook audit tables
  - `cache.entries`
  - optional `billing.*` (at least usage events)
- Minimal DB access layer (repositories)
- Hashing utility (blake3) and canonical serialization rules

**Green-light criteria**

- migrations apply cleanly on empty DB
- can insert and query events/jobs/outcomes in a dev script
- idempotency unique constraints behave as expected

---

### Phase 2 — Thread + query adapter API + SSE skeleton (M)

**Goal:** ship the platform-facing API shape early.

**Deliverables**

- `POST /v1/threads/init`:
  - find/create thread
  - return onboarding gate + missing requirements (mocked initially)
- `POST /v1/threads/{thread_id}/queries`:
  - create a workflow run (`query_id`)
  - append initial ledger events (`INTENT_RECEIVED`, `PLAN_CREATED` placeholder)
  - return `query_id` immediately
- `GET /v1/queries/{query_id}/events`:
  - streams events from `ledger.events` with cursoring
  - emits keepalives
- `POST /v1/actions/{action_id}/resolve`:
  - records gate decision and emits events (mocked gating initially)

**Green-light criteria**

- a client can:
  - init thread
  - submit query and receive query_id immediately
  - connect SSE and see ordered events
  - reconnect with cursor and not miss/duplicate events

---

### Phase 3 — Kernel primitives + worker leasing executor (L)

**Goal:** real resumable workflows with idempotent operators.

**Deliverables**

- Intent normalization (minimal v1 intent types)
- Plan schema + validation (steps must declare effects/tags/cache/idempotency)
- Runtime projections:
  - create `workflow_steps` rows per plan
  - leasing loop with `FOR UPDATE SKIP LOCKED`
- Worker runtime:
  - leases steps, executes operators, writes jobs/outcomes/events
  - retry taxonomy and backoff scheduling (basic)
- Gate support:
  - step transitions to `WAITING_APPROVAL` and creates `ledger.gates`
  - resume on gate decision

**Green-light criteria**

- crash/resume demo:
  - kill worker mid-step, restart, no duplicate side effects
- idempotency demo:
  - same operator idempotency key returns same receipt/result
- gate demo:
  - workflow pauses and resumes correctly with `actions/{action_id}/resolve`

---

### Phase 4 — Platform integration (auth + context + apply patch) (L)

**Goal:** connect the Layer to the real platform environment safely.

**Deliverables**

- `PlatformAuthAdapter`:
  - validate cookie/session via platform backend (dev stub allowed)
- `PlatformContextLoader`:
  - fetch the typed context for `(student_id, funding_request_id)` via platform backend
- Apply patch handshake:
  - operators that generate `apply_platform_patch` payloads
  - action_required emitted; platform applies; platform resolves action with receipt
- Outbound webhook to platform backend:
  - workflow status + action_required notifications (at least-once, signed)

**Green-light criteria**

- thread init loads platform context successfully
- a query can propose a patch and pause
- after platform applies patch and resolves action, the workflow continues and completes
- webhook audit trail exists in Layer DB

---

### Phase 5 — LLM client, caching, and usage reporting (M/L)

**Goal:** make LLM operations reliable, cheap, and auditable.

**Deliverables**

- `packages/llm_client`:
  - provider calls, retries, timeouts
  - request normalization, cache keys
  - Redis hot cache + Postgres cold cache
  - telemetry normalization (tokens/cost/provider trace id)
- Usage reporting integration:
  - request start: call platform backend for credits/quota “green light”
  - request end: send idempotent usage report keyed by `query_id`

**Green-light criteria**

- caching demo: repeated identical LLM calls hit cache
- usage report is emitted once even on retries/replays
- budget denial returns a clean error outcome (no partial side effects)

---

### Phase 6 — User state, onboarding controller, and memory (L)

**Goal:** deterministic “collect missing info” behavior and durable personalization.

**Deliverables**

- User onboarding/state model:
  - gates and missing_fields per category
  - fast evaluation at thread init and per query
- Action_required types:
  - `collect_fields`, `upload_required`, `select_option`, `confirm`, `redirect`
- Memory operators:
  - write typed memory facts with source/confidence/ttl
  - read active memory facts for context

**Green-light criteria**

- a new/missing-profile user triggers onboarding gates instead of running heavy workflows
- after the client supplies required fields, gates flip and workflows proceed
- memory is recorded and influences subsequent drafts (observable via context hash and outcomes)

---

### Phase 7 — FundingOutreach capability (core v1 value) (L)

**Goal:** ship the end-to-end funding outreach experience.

**Deliverables**

- Switchboard planner that routes:
  - professor questions, alignment requests, email compose/review/optimize, request field edits, paper metadata extraction
- Operators for:
  - professor summarize
  - alignment score
  - email generate/review/optimize
  - paper metadata extraction and request field update proposal
  - apply patch proposal for funding_requests/funding_emails fields
- Outcome schemas:
  - professor summary, alignment report, email draft/review, patch proposal receipts

**Green-light criteria**

- demo scenario works end-to-end with ledgers:
  - ask about professor → summary outcome
  - ask alignment → alignment outcome
  - generate/optimize email → email draft outcome
  - propose platform updates → action_required → resolve → applied outcome
- “What happened?” is answerable by reading ledgers (no LLM rerun)

---

### Phase 8 — Documents capability (upload/review/optimize/compose/export) (L)

**Goal:** make document handling production-grade.

**Deliverables**

- S3 path scheme + lifecycle enforcement:
  - temp per query
  - sandbox per thread
  - final per content hash
- Document dedupe by blake3 hash (avoid re-processing re-uploads)
- Operators:
  - upload, process, review, optimize, compose SOP, export PDF
  - apply patch proposal for platform attachments if required
- Artifacts:
  - `artifact_ready` events with signed URLs and provenance hashes

**Green-light criteria**

- upload same file twice → second run short-circuits processing
- optimize resume/SOP produces sandbox and final artifacts with correct lineage
- artifact URLs are short-lived and safe (no raw PII in logs/events)

---

### Phase 9 — Hardening and production readiness (L)

**Goal:** make the system boring in production.

**Deliverables**

- Observability full rollout:
  - dashboards, alerts, sampling policies, error reporting (Sentry optional)
- Load tests:
  - SSE connections, worker concurrency, DB leasing contention
- Operational jobs:
  - outbox dispatcher
  - cache eviction
  - temp object TTL cleanup
  - ledger archival policy (if needed)
- Security review:
  - data class enforcement
  - redaction for webhooks/logs
  - rate limits and abuse safeguards

**Green-light criteria**

- p95 latency within v1 targets for key flows
- no stuck workflow regressions under chaos tests
- incident playbooks exist and are validated on staging

---

### Phase 10 — Release and future merges (post-v1)

**Goal:** controlled rollout and integration of existing sub-systems (reply checks/reminders project).

**Deliverables**

- Staging environment with canary rollout
- Feature flags for new operators/capability versions
- Merge plan for the external reminders/reply-check project:
  - integrate as scheduled jobs/operators
  - ensure idempotency and ledger compliance

**Green-light criteria**

- production rollout checklist complete
- clear rollback plan (disable new workflows; reproduce remains available)

---

## 16) Builder quick-start checklist (what to implement first)

If you only do one thing first, do this:

1. Implement ledgers + SSE events derived from `ledger.events`.
2. Implement DB-leased worker executor with idempotent jobs.
3. Implement `action_required` gates and `actions/{id}/resolve`.
4. Only then build the FundingOutreach operators.

This matches the constitution’s “make the spine boring and stable” doctrine.

---

## 17) References (living documents)

- Data layer: `DATA-STRUCTURE.md`
- API/runtime and observability: `API-RUNTIME-DESIGN.md`
- Data layer implementation plan: `PLAN.md`
- Canonical laws and primitives: `CONSTITUTION.md`, `2-canonical-primitives.md`, `4-ledgers-sources-of-truth.md`
- Execution and plugins: `EXECUTION-RFC-V1.md`, `5-plugin-contracts.md`, `7-execution-semantics.md`
- Policy: `6-policy-engine.md`
- Capabilities: `9-capabilities.md`
- Historical flows: `PREREQUISITE.txt`
