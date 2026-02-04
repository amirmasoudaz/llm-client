# CanApply Intelligence Layer — API, Runtime, Streaming, Webhooks, Queues, Observability (v1)

This document specifies the **implementation-facing** design for:

- HTTP APIs and endpoint behaviors
- request/response **schemas** (versioned)
- SSE streams (events/progress/assistant projections)
- outbound Webhooks (platform + integrations)
- job queuing, worker pools, and message broker choices
- observability (logs/metrics/traces), SLOs, and operational safeguards

It is written to be consistent with:

- `CONSTITUTION.md` (laws, primitives, ledger doctrine)
- `KERNEL-RFC-V1.md`, `LEDGERS-RFC-V1.md`, `EXECUTION-RFC-V1.md`
- `TECHNICAL-SPECS.md` (FastAPI + SSE, Redis, Postgres, Qdrant, S3)
- `8-ai-credits-and-budget-enforcement.md` (credits reservation + settlement)

---

## 0) Non-negotiable runtime guarantees (API-visible)

1. **Workflow-first**: every intake becomes a stored `Intent` and (internal) `Plan`, and execution emits ledger `Events`, `Jobs`, and `Outcomes`.
2. **Reproduce by default**: fetching outcomes returns stored artifacts without rerunning models unless explicitly asked (`replay_mode`).
3. **Idempotency everywhere**:
   - request-level dedupe (same normalized input + policy snapshot → same workflow)
   - action-level idempotency (same `(tenant_id, operator_name, idempotency_key)` → same result/receipt; never repeat side effects)
4. **Policy-first**: policy evaluates intake, plan, action, outcome, apply; each evaluation is recorded as a first-class `PolicyDecision`.
5. **Tenancy mandatory**: `tenant_id` scopes every record and every query.
6. **Interfaces are projections**: chat messages and “assistant stream” are not required to audit truth; they are derived from ledgers.

---

## 1) Service roles and boundaries (v1)

### 1.1 Services/modules (logical)

- **Kernel API (FastAPI)**:
  - intake normalization → `Intent`
  - create/validate `Plan`
  - schedule execution (create workflow + steps)
  - expose read APIs and SSE streams derived from ledgers
  - record policy decisions and gate requests
- **Worker(s)**:
  - lease runnable steps
  - invoke operators
  - write jobs/outcomes/events
- **Policy Engine**:
  - deterministic evaluation given normalized inputs
- **LLM Client**:
  - provider routing, retries, caching, telemetry normalization
- **Webhook Dispatcher**:
  - deliver outbound events reliably (with retry + signatures)

### 1.2 “Truth” boundaries

- **Truth**: ledgers (events/jobs/outcomes/documents/memory/entities/policy decisions) + downstream applied writes.
- **Projection**: SSE streams, assistant narrative, UI chat transcript.
- **Side-effects**: operators only (and must be idempotent and policy-gated).

---

## 2) API design principles (how endpoints behave)

### 2.1 Versioning

- Public API version prefix: `/v1/...`
- Every typed object includes `schema_version`.
- Backward compatibility: additive changes only within major version.

### 2.2 Idempotency and dedupe

- Clients MAY provide `Idempotency-Key` header for intake.
- Kernel computes a deterministic `request_key` from:
  - `tenant_id`, principal identity
  - normalized intent payload
  - context refs hashes (document hashes, professor digest hash, template hash)
  - policy snapshot hash/version
  - execution mode and replay mode
- If an equivalent workflow exists:
  - **completed** → return `workflow_id` and stored outcomes (reproduce)
  - **running** → return `workflow_id` and stream links (attach)
  - **failed/partial** → return `workflow_id` with blocked step + allow resume

### 2.3 Consistent envelopes

All non-stream endpoints return:

```json
{
  "ok": true,
  "data": { },
  "meta": {
    "request_id": "uuid",
    "tenant_id": 1,
    "correlation_id": "uuid"
  }
}
```

Errors return:

```json
{
  "ok": false,
  "error": {
    "code": "POLICY_DENIED",
    "message": "External send denied for DataClass.Regulated.",
    "details": { "reason_code": "SENSITIVE_ID_BLOCKED" }
  },
  "meta": { "request_id": "uuid", "tenant_id": 1, "correlation_id": "uuid" }
}
```

### 2.4 Auth model (API-facing)

Requests carry:

- `Authorization: Bearer <token>`
- Token resolves:
  - `tenant_id`
  - `principal` (type/id)
  - `role`
  - `trust_level`
  - `scopes[]` (fine-grained, operator-relevant)

Kernel passes an `AuthContext` to operators and records it (redacted) in ledgers.

### 2.5 Tenancy + principal mapping (planning defaults)

You asked “What’s the authoritative `tenant_id`/principal mapping?”—this is simply the **source and meaning** of:

- `tenant_id`: which organization/account owns the data and policies
- `principal`: the actor initiating the request (student/admin/service), including their scopes and trust level

Even if CanApply is single-tenant today, we keep these fields from day 1 because the constitution requires tenant scoping everywhere.

**Recommended v1 default (planning / dev mode):**

- Treat CanApply as **single-tenant**:
  - `tenant_id = 1` for all requests.
- Treat “principal” as the platform identity:
  - `principal.type = "student"` and `principal.id = students.id` for end-users
  - `principal.type = "service"` for internal system calls (platform backend, schedulers)
- Scopes come from the platform backend (auth token) and are mapped into `AuthContext.scopes[]` (e.g., `intent:submit`, `gate:approve`, `email:generate`, `email:send`).

**Postgres RLS (Row Level Security) decision for v1:**

- Do **not** enable RLS in v1 by default.
- Enforce tenant isolation in the application layer (every query includes `tenant_id`) and add RLS later if compliance or multi-tenant risk requires it.

---

## 3) Core endpoints (Kernel API)

### 3.0 Public adapter API (recommended v1 surface)

Your v1 “super simple” API is thread-centric. Treat it as an **adapter layer** over the internal workflow primitives:

- `thread_id` = UX grouping (stable)
- `query_id` = workflow execution identity (recommended: reuse `workflow_id`)
- internal truth remains `Intent → Plan → Actions(Jobs) → Outcomes → Events`

The adapter endpoints below should be the primary integration surface for the platform frontend/backend.

#### 3.0.1 Init thread (idempotent by scope)

`POST /v1/threads/init`

Input:

```json
{
  "student_id": 88,
  "funding_request_id": 556,
  "client_context": { "locale": "en-CA", "ui_route": "/funding/556" }
}
```

Behavior:

- Validates session (dev-mode: allow bypass; prod: validate cookie via platform backend).
- Finds or creates a `runtime.threads` row unique on `(tenant_id, student_id, funding_request_id)`.
- Loads minimal platform context to determine onboarding gates (fast path).
- Returns:
  - `thread_id`
  - `thread_status`
  - `onboarding_gate` and optional `missing_requirements` list

Output:

```json
{
  "thread_id": 123,
  "thread_status": "new|active|archived",
  "onboarding_gate": "ready|needs_onboarding",
  "missing_requirements": ["base_profile.first_name", "background.resume"]
}
```

#### 3.0.2 Submit query (creates workflow; returns query_id immediately)

`POST /v1/threads/{thread_id}/queries`

Input:

```json
{
  "message": "Optimize my email and make it shorter.",
  "attachments": [
    { "source": "s3", "object_uri": "s3://.../sandbox/...", "mime": "application/pdf", "name": "paper.pdf" }
  ]
}
```

Behavior:

- Creates a workflow execution (recommended: `query_id == workflow_id`).
- Writes the canonical intake ledgers (intent/plan/events).
- Returns immediately with `query_id` and the SSE URL.

Output:

```json
{
  "query_id": "uuid",
  "sse_url": "/v1/queries/{query_id}/events"
}
```

#### 3.0.3 Query SSE stream (single multiplexed stream)

`GET /v1/queries/{query_id}/events`

This is an adapter alias for:

- `GET /v1/workflows/{workflow_id}/events` (truth projection), plus
- optional multiplexed `progress`, `assistant_delta`, and `action_required` events.

#### 3.0.4 Resolve an action (approve/decline/provide input)

`POST /v1/actions/{action_id}/resolve`

In v1, implement `action_id` using the existing gate primitive (`ledger.gates.gate_id`) so you can reuse:

- gate storage
- gate decision recording
- pause/resume semantics

Input:

```json
{
  "status": "accepted|declined",
  "payload": { "user_input": { }, "platform_patch_receipt": { } }
}
```

Behavior:

- Records a gate decision event.
- If accepted, transitions the waiting step to `READY` and resumes.

#### 3.0.5 Optional reads (v1 nice-to-have)

- `GET /v1/threads/{thread_id}` — summary + latest query statuses
- `GET /v1/threads/{thread_id}/history` — projection of message-like history from ledgers
- `GET /v1/queries/{query_id}` — final snapshot (outcomes + usage)

### 3.1 Submit workflow (unified entrypoint)

`POST /v1/kernel/submit`

Behavior:

- Normalize input into a typed `Intent`.
- Run **Intake Policy**.
- Build `ContextBundle` references (hashes, IDs; no large blobs).
- Build/validate a `Plan`.
- Run **Plan Policy** (may transform plan or add gates).
- Create `workflow_run` + `workflow_steps` projection rows.
- Append events: `INTENT_RECEIVED`, `PLAN_CREATED`, and `WORKFLOW_ACCEPTED`.
- Enqueue execution (DB-leased; optional broker kick).
- Return identifiers and stream URLs.

Request schema (v1):

```json
{
  "source": "chat|api|webhook|system",
  "thread_id": 4512,
  "scope": { "scope_type": "funding_request", "scope_id": "556" },
  "input": {
    "mode": "message|intent",
    "text": "Generate an outreach email…",
    "intent_hint": {
      "intent_type": "Funding.Outreach.Email.Generate",
      "schema_version": "1.0",
      "inputs": { "goal": "initial_outreach" }
    }
  },
  "constraints": { "tone": "professional-warm", "length": "short", "language": "en" },
  "execution": {
    "mode": "draft_only|human_gated|dry_run|auto_exec",
    "replay_mode": "reproduce|replay|regenerate"
  }
}
```

Response schema:

```json
{
  "workflow_id": "uuid",
  "intent_id": "uuid",
  "plan_id": "uuid",
  "correlation_id": "uuid",
  "status": "accepted|attached",
  "stream": {
    "events": "/v1/workflows/{workflow_id}/events",
    "progress": "/v1/workflows/{workflow_id}/progress",
    "assistant": "/v1/workflows/{workflow_id}/assistant"
  }
}
```

### 3.2 Get workflow state

`GET /v1/workflows/{workflow_id}`

Returns:

- `status` + `execution_mode` + `replay_mode`
- step states (projection) and blocked reason (if waiting approval)
- summary of produced outcomes (IDs + types + previews)

### 3.3 List outcomes (reproduce by default)

`GET /v1/workflows/{workflow_id}/outcomes?mode=reproduce|replay|regenerate`

Rules:

- `mode=reproduce` returns stored outcomes and lineage.
- `mode=replay` triggers rerun of steps as allowed (effects remain idempotent; may create new draft outcomes if policy allows).
- `mode=regenerate` mints new lineage/version keys for draft-producing steps (explicit).

Return shape:

```json
{
  "workflow_id": "uuid",
  "mode": "reproduce",
  "outcomes": [
    {
      "outcome_id": "uuid",
      "lineage_id": "uuid",
      "version": 2,
      "type": "Email.Draft",
      "status": "draft",
      "schema_version": "1.0",
      "created_at": "2026-01-28T19:03:00Z",
      "content": { }
    }
  ]
}
```

### 3.4 Gate decision (human approval)

`POST /v1/workflows/{workflow_id}/gates/{gate_id}`

Behavior:

- Validates actor permissions (`gate:approve`) and tenant/workflow ownership.
- Records `USER_APPROVED` / `USER_REJECTED` event + a `gate_decision` record.
- If approved, transitions the gated step to `READY` and resumes execution.

Request:

```json
{
  "decision": "approve|reject|edit_then_approve",
  "payload": { "edits": { } }
}
```

### 3.5 Cancel workflow

`POST /v1/workflows/{workflow_id}/cancel`

Behavior:

- Marks runnable steps as `CANCELLED` (projection), appends `WORKFLOW_CANCELLED` event.
- Operators must respect cancellation when feasible; idempotency still prevents double effects.

### 3.6 Retry / resume workflow

`POST /v1/workflows/{workflow_id}/resume`

Behavior:

- Recomputes next runnable steps from ledger/projection, respects existing idempotency keys.
- Appends `WORKFLOW_RESUMED`.

### 3.7 Documents (Layer-owned)

`POST /v1/documents/upload`

- stores raw upload to S3/MinIO (temp/sandbox lifecycle based on request)
- writes document ledger records (document + revision)
- returns `document_id` and processing status

`GET /v1/documents/{document_id}`

- returns metadata + revision list + signed download URLs (short-lived)

### 3.8 Health and readiness

- `GET /healthz` (liveness)
- `GET /readyz` (readiness: DB, Redis, object store connectivity; optionally Qdrant)

### 3.9 Admin/debug (guarded; optional)

- `GET /v1/admin/workflows?status=...` (requires admin scope; safe redacted payloads only)
- `GET /v1/admin/ledgers/events?...` (redacted and tenant-scoped)

---

## 4) Streaming (SSE) design

### 4.1 Endpoints

- `GET /v1/workflows/{workflow_id}/events` (canonical event stream; truth projection)
- `GET /v1/workflows/{workflow_id}/progress` (derived progress; not truth)
- `GET /v1/workflows/{workflow_id}/assistant` (derived narrative deltas; not truth)

All SSE endpoints:

- require auth; verify `tenant_id` access to `workflow_id`
- support resuming via cursor

### 4.2 Cursoring and replay

Use one of:

- `Last-Event-ID` header (recommended) with `event_no`, or
- query param `?after_event_no=123`

Contract:

- events are delivered in increasing `event_no` order per workflow.
- reconnection must not duplicate events from the client’s point of view.

### 4.3 SSE event format (wire protocol)

Use `event:` as a stable category and put full payload JSON into `data:`.

Example (events channel):

```
id: 1842
event: ledger_event
data: {"schema_version":"1.0","tenant_id":1,"workflow_id":"...","event_no":1842,"event_type":"STEP_STARTED","created_at":"...","payload":{...}}
```

Example (progress channel):

```
event: progress
data: {"schema_version":"1.0","workflow_id":"...","percent":60,"stage":"running","current_step":{"step_id":"s4","name":"Review.Email"}}
```

Example (assistant channel):

```
event: assistant_delta
data: {"schema_version":"1.0","workflow_id":"...","delta":"Here is a draft…","final":false}
```

### 4.4 Keep-alives, backpressure, and limits

- Send a heartbeat every 15–30s:
  - `event: keepalive` with minimal JSON.
- Enforce:
  - max concurrent SSE connections per tenant/principal
  - max event payload size (IDs + reason codes, no huge blobs)
  - server-side buffering limits to prevent memory pressure
- If client is too slow:
  - drop connection with a clear reason; client reconnects with cursor.

### 4.5 Event types (v1 stable set)

Minimum stable event types (ledger events):

- `INTENT_RECEIVED`
- `INTENT_REJECTED`
- `PLAN_CREATED`
- `WORKFLOW_ACCEPTED`
- `WORKFLOW_STARTED`
- `STEP_READY`
- `STEP_STARTED`
- `POLICY_DECISION`
- `STEP_SUCCEEDED`
- `STEP_FAILED`
- `OUTCOME_CREATED`
- `GATE_REQUESTED`
- `USER_APPROVED`
- `USER_REJECTED`
- `WORKFLOW_PARTIAL`
- `WORKFLOW_COMPLETED`
- `WORKFLOW_CANCELLED`

> Rule: do not explode event types; add payload fields, not new types, unless semantics differ.

---

## 5) Webhooks (v1: platform backend only)

### 5.1 Scope and consumers (decision)

**v1 decision:** the only webhook consumer is the **CanApply platform backend** (server-to-server). The frontend never receives Intelligence Layer webhooks directly.

Rationale (implementation consequences):

- signature verification and secret rotation stay server-side
- retries/dedupe/audit are centralized
- UI still gets realtime via platform backend fanout (SSE/WebSocket) or UI polling the platform

### 5.2 Outbound webhooks (Intelligence Layer → platform backend)

Outbound webhooks are for notifying the platform backend about:

- the CanApply platform backend (status updates, apply-ready prompts, credit low/exhausted)
- v1 explicitly excludes third-party tenant endpoints (enterprise webhooks are v1.1+)

### 5.3 Delivery model

- Webhooks are **at-least-once** delivered.
- Each delivery is idempotent via:
  - `webhook_event_id` (UUID) unique per event
  - receiver dedupes by this ID
- Signature:
  - `X-Webhook-Signature: v1=<hex(hmac_sha256(secret, raw_body))>`
  - include timestamp header to prevent replay
- Retries:
  - exponential backoff with jitter
  - max retry window (e.g. 24h)
- A dead-letter state is recorded if retries exhausted.

### 5.4 Endpoint configuration (decision: no tenant UI in v1)

**v1 decision:** no per-tenant webhook endpoint management UI.

Implementation:

- the platform backend endpoint + secret is **platform-owned**
- config is either:
  - environment-based (fastest), or
  - admin-only DB rows (recommended for rotations without deploys)

If you choose DB config, keep it simple in v1:

- one active endpoint per environment (dev/stage/prod)
- optional override per tenant later

`POST /v1/webhooks/endpoints` may exist as a **platform-admin-only** internal endpoint (not exposed to tenants) to rotate/update the destination and secret reference.

Stores:

- URL
- secret reference
- enabled event types
- retry policy overrides

### 5.5 Webhook event schema

```json
{
  "schema_version": "1.0",
  "webhook_event_id": "uuid",
  "event_type": "workflow.status|workflow.gate_requested|credits.low|credits.exhausted|...",
  "tenant_id": 1,
  "created_at": "2026-01-28T19:03:00Z",
  "trace": {
    "correlation_id": "uuid",
    "workflow_id": "uuid",
    "thread_id": 4512,
    "intent_id": "uuid"
  },
  "data": {
    "status": "waiting_approval",
    "gate": { "gate_id": "uuid", "summary": "...", "expires_at": "..." },
    "outcomes": [{ "type": "Email.Draft", "outcome_id": "uuid" }]
  }
}
```

### 5.6 Mapping ledger events → webhook events

Default mapping:

- `GATE_REQUESTED` → `workflow.gate_requested`
- `WORKFLOW_COMPLETED` → `workflow.completed`
- `WORKFLOW_PARTIAL` → `workflow.partial`
- `STEP_FAILED` (final) → `workflow.failed`
- `credits.low/exhausted` derived from billing events → `credits.low|credits.exhausted`

Policy:

- webhook payloads must be **redacted** according to data classes; never include regulated identifiers or raw document text.

### 5.7 Inbound webhook pattern (external systems → platform backend → Kernel)

Some integrations will be inbound webhooks (e.g., provider callbacks). v1 pattern:

1. External system delivers webhook to **platform backend** only.
2. Platform backend verifies signature, writes an immutable audit record, dedupes, and enqueues internal processing.
3. Platform backend translates the event into a Kernel `submit` call (adapter behavior), or schedules a workflow run.
4. Intelligence Layer executes via workers and emits status back to platform backend via outbound webhooks (or platform pulls via `/workflows/{id}`).

This keeps browsers out of the trust boundary and centralizes retries/auditability.

### 5.8 Platform backend webhook receiver contract (v1)

Since v1 webhooks are **backend-only**, define a single canonical receiver endpoint on the platform backend.

Recommended endpoint (platform backend):

- `POST /internal/intelligence-layer/webhook`

Required behaviors:

1. **Verify signature**:
   - compute HMAC over raw body using shared secret
   - validate timestamp header (e.g., ±5 minutes) to prevent replay
2. **Deduplicate**:
   - unique constraint on `(webhook_event_id)`
   - return `2xx` for duplicates (idempotent ACK), never reprocess
3. **Persist audit record before processing**:
   - store payload hash, headers, signature validity, and processing status
4. **Enqueue internal processing**:
   - translate into platform-internal events (update product state, notify UI)
5. **Return response**:
   - `200 OK` (or `202 Accepted`) for valid events, including duplicates
   - `4xx` only for permanently invalid requests (bad signature, invalid schema)

Suggested platform tables (consumer-side audit trail; v1 “plumbing now”):

```sql
-- Platform DB (or platform-owned Postgres), not the Intelligence Layer DB.
CREATE TABLE webhook_events (
  id                 BIGSERIAL PRIMARY KEY,
  webhook_event_id   UUID NOT NULL UNIQUE,
  received_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  signature_valid    BOOLEAN NOT NULL,
  headers_json       JSONB NOT NULL,
  payload_hash       BYTEA NOT NULL,
  event_type         TEXT NOT NULL,
  tenant_id          BIGINT NOT NULL,
  workflow_id        UUID NULL,
  processing_status  TEXT NOT NULL, -- received|processing|processed|failed
  last_error         JSONB NULL
);
```

---

## 6) Jobs, queuing, workers, and message brokers

### 6.1 Canonical execution model (v1): DB-leased steps

Primary queue = `runtime.workflow_steps` (projection).

Workers:

- poll/lease runnable steps using:
  - `FOR UPDATE SKIP LOCKED`
  - lease TTL + renewals
- execute operator with idempotency key
- write:
  - `ledger.jobs` (+ `ledger.job_attempts`)
  - `ledger.outcomes` when produced
  - `ledger.events` for every transition
- update projection row (`workflow_steps`) to reflect status

Why this is the right v1 default:

- resumable by construction
- easy to debug (DB state + ledgers)
- no hidden queue state
- failure recovery is deterministic and idempotent

### 6.2 Worker pools and routing

Define worker pools by operator class:

- `worker-default`: LLM calls, formatting, lightweight retrieval
- `worker-doc`: parsing/OCR/chunking/export
- `worker-embed`: embedding and Qdrant upserts
- `worker-io`: external integrations (gmail send, webhooks emit) — highest-risk, strictest policy

Routing rule:

- each plan step includes `queue_lane` or `worker_pool` hint (validated by plan policy)
- workers only lease steps for pools they serve

### 6.3 Priorities and fairness

Add fields (projection) to support fairness:

- `priority` (0–100)
- `scheduled_at` (for deferred step readiness, if using step-level scheduling)

Lease query uses `(priority DESC, scheduled_at ASC, created_at ASC)` ordering.

### 6.4 Scheduling in v1 (decision: basic scheduled jobs, not a workflow engine)

**v1 decision:** support **basic scheduling** for delayed retries/backoffs and internal housekeeping, but do not build a second workflow engine.

Minimum that works well:

- A `scheduled_jobs` table with:
  - `run_after` / `scheduled_at`
  - `status`, `attempt`, `max_attempts`
  - `dedupe_key`
  - `payload` (typed)
- A scheduler loop/service that:
  1. atomically claims due jobs (`FOR UPDATE SKIP LOCKED`)
  2. writes the “job claimed/started” event
  3. publishes a wake-up message to the broker (or triggers a DB state change that workers will observe)
  4. marks the scheduled job succeeded/failed and emits follow-on jobs as needed

This covers:

- delayed retries/backoffs
- periodic maintenance workflows

Defer to later:

- tenant-configurable cron expressions
- complex DAG scheduling
- visual workflow builders

### 6.5 Optional broker “kick” (reduce polling latency)

Even with DB-leasing, you can “kick” workers to react faster:

- Redis Streams: publish `workflow_id` when steps are created or transition to READY
- Workers still lease from DB as the source of truth

### 6.6 Broker direction (v1.1+ decision: Redis Streams → NATS JetStream; managed alt: SQS)

**v1.1+ preference:** Redis Streams now, **NATS JetStream** next if low-latency internal eventing and fan-out patterns grow.

**Managed alternative:** Redis Streams → **AWS SQS** if ops simplicity beats latency/event-bus needs.

If/when you need a broker beyond Redis Streams, pick based on constraints:

- **Redis Streams**: simplest, good for v1; not ideal for long retention.
- **RabbitMQ**: good for complex routing/ack semantics; more ops overhead.
- **NATS JetStream**: modern, fast, durable streams; strong for event fanout.
- **SQS**: great managed queue if you’re on AWS; pairs well with Postgres as the truth store.
- **Kafka**: powerful, but overkill for v1 given the ledger is already your event store.

Non-negotiable rule regardless of broker:

- broker messages are **signals**, not truth
- leasing/idempotency in Postgres remains the correctness mechanism

### 6.7 BrokerAdapter + outbox (critical design move)

Implement two primitives early so broker swaps are tractable:

1) `BrokerAdapter` interface (publish/consume/ack/nack/delay), backing implementations:

- `RedisStreamsBrokerAdapter`
- later `NatsJetStreamBrokerAdapter` or `SqsBrokerAdapter`

2) An **outbox pattern** for “DB write + enqueue” consistency:

- Kernel/worker writes workflow/step state in Postgres and writes an `outbox` row in the same transaction.
- A dispatcher reads `outbox` and publishes to the broker; on success, marks the outbox row delivered.

This avoids the classic failure: “DB committed but message publish failed” (or vice versa).

### 6.8 Failure modes and safeguards

- **Worker crash mid-step**:
  - lease expires → another worker re-leases
  - operator idempotency prevents duplicate side effects
- **Poison pill step**:
  - cap retries by error taxonomy
  - mark `FAILED_FINAL`, emit event, surface blocked state
- **Long-running steps**:
  - heartbeat renewals
  - maximum wall-time; on timeout, mark retryable or final based on category

---

## 7) Schema registry and validation (how we keep it typed)

### 7.1 What must be schema-validated

- Intents (`intent_type` + `schema_version`)
- Plans (step schema: effects/tags/cache/idempotency/gates are mandatory)
- Operator payloads and results
- Outcomes (`outcome_type` + `schema_version`)
- Webhook events

### 7.2 Where schemas live

v1 options:

1. filesystem `schemas/` directory in repo (fastest)
2. DB-backed registry (`registry.*` tables)

Requirement:

- Kernel and operators validate at runtime and record:
  - `schema_version`
  - validation failure codes in job/error payloads

---

## 8) Observability (production-grade)

### 8.1 Correlation model

Every request/step/job/outcome/event carries:

- `tenant_id`
- `correlation_id`
- `workflow_id`
- `step_id` (when applicable)
- `job_id` (operator execution identity)

These must appear in logs, metrics labels (carefully), and traces.

### 8.2 Logs (structured, safe)

- Format: JSON logs.
- Required fields:
  - `timestamp`, `level`, `service`, `env`
  - trace fields above
  - `operator_name`, `operator_version` where relevant
- PII policy:
  - do not log raw document text or regulated identifiers
  - prefer hashes and IDs
  - include “redacted_view” fields if needed for debugging

### 8.3 Metrics (minimum set)

Kernel:

- `http_requests_total{route,method,status}`
- `http_request_duration_seconds_bucket{route,method}`
- `sse_connections{channel}` + disconnect reasons
- `workflows_created_total{intent_type}`
- `workflows_inflight{status}`

Executor/workers:

- `step_leases_total{pool,status}`
- `step_lease_latency_seconds`
- `step_runtime_seconds_bucket{operator_name}`
- `operator_calls_total{operator_name,status,trace_type}`
- `operator_retries_total{operator_name,error_code}`
- `idempotency_hits_total{operator_name}`

Policy:

- `policy_decisions_total{stage,decision,reason_code}`
- `policy_latency_seconds_bucket{stage}`

Caching:

- `cache_hits_total{namespace,layer=redis|postgres}`
- `cache_misses_total{namespace,layer=redis|postgres}`
- `cache_write_errors_total`

Billing:

- `credits_reservations_total{status}`
- `credits_debits_total`
- `credits_insufficient_total`
- `cost_usd_total{provider,model}`

### 8.4 Tracing (OpenTelemetry)

Spans:

- `kernel.submit`
  - `policy.intake`
  - `context.build` (include only hashes/ids)
  - `plan.build`
  - `policy.plan`
- `executor.lease_step`
- `operator.invoke:{operator_name}`
  - `provider.call` (LLM)
  - `s3.put/get`, `qdrant.query/upsert`, `platform_db.write` (apply)

Attach:

- provider trace IDs (where available)
- cache hits as span attributes

### 8.5 SLOs and alerting (suggested v1 targets)

SLOs (initial):

- Kernel API availability: 99.9%
- SSE availability: 99.5% (connections are long-lived; measure successful reconnect)
- Worker execution success rate (non-policy-denied): >= 99%
- Lease reclaim rate: < 0.5% of steps (indicates worker instability)

Alerts:

- DB connection pool saturation
- event write errors / lagged SSE reader
- stuck workflows (no new events > N minutes while RUNNING)
- elevated `FAILED_FINAL` rates per operator
- credits reservation failures spikes

### 8.6 “Operational truth” dashboards

Dashboards centered on:

- workflow funnel (accepted → completed/partial/failed)
- per-operator latency + error taxonomy
- policy denials/approvals heatmap (reason codes)
- cache hit rate and saved cost
- credits burn by tenant/model

---

## 9) Implementation notes and pitfalls (what to enforce in code review)

1. Any new endpoint must:
   - be tenant-scoped and authorize via scopes
   - write relevant events (if it changes workflow state)
2. Any new operator must:
   - declare effects + policy tags
   - implement idempotency key behavior
   - record job metrics (tokens/cost/latency) and nondeterminism flags
3. SSE channels must:
   - stream from ledger events, not logs
   - support cursoring and reconnection
4. Webhooks must:
   - be signed, retried, and idempotent
   - be redacted by data classification
5. No “chat transcript as truth”:
   - assistant channel is a projection; outcomes are the artifacts.

---

## 10) Open questions (need decisions)

### 10.1 Decisions captured

1. Webhook consumer in v1: **platform backend only**.
2. No per-tenant webhook management UI in v1; **do** implement audit plumbing now.
3. v1 supports **basic scheduled jobs** (`runtime.scheduled_jobs`) for deferred work/backoffs; reply checks/reminders are **out of scope** (will be merged from an existing project later).
4. Broker direction for v1.1+: **Redis Streams → NATS JetStream** (managed alternative: Redis Streams → SQS).
5. Dev-mode tenancy defaults: `tenant_id=1`, principal from platform identity, **no Postgres RLS in v1**.
6. `runtime.scheduled_jobs` is system-of-record in the Intelligence Layer Postgres (platform may keep optional UI projections later).

### 10.2 Remaining decisions (still needed)

1. Do we need any other scheduled job categories in v1 besides backoffs and internal housekeeping (since email reminders/reply checks are deferred)?
