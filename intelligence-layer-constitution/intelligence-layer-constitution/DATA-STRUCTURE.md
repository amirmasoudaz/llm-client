# CanApply Intelligence Layer — Data Layer (v1)

This document proposes an implementation-ready **data layer** for the CanApply Intelligence Layer, consistent with the guarantees in `CONSTITUTION.md` and the implementation guidance in `TECHNICAL-SPECS.md`, `LEDGERS-RFC-V1.md`, `EXECUTION-RFC-V1.md`, and `8-ai-credits-and-budget-enforcement.md`.

---

## 0) Goals and non‑negotiable invariants (what the data layer must enforce)

From `CONSTITUTION.md` (summarized as enforceable storage requirements):

- **Workflow-first**: everything is reconstructible as `Intent → Plan → Actions → Outcomes → Events`.
- **Ledgers are sources of truth**: ledgers + policy-gated downstream “apply” writes must answer “what happened?” without rerunning an LLM.
- **Stateless agents**: any durable “memory” is written as typed ledger records.
- **Operators are the only effectful code paths** and are **idempotent** under an `idempotency_key`.
- **Policy decisions are first-class records** and must be deterministic for identical inputs (store decision + inputs hash + engine version).
- **Tenancy is mandatory** (`tenant_id` everywhere; no cross-tenant leakage).
- **Replay semantics are explicit**:
  - reproduce = read stored outcomes/events
  - replay = rerun with same refs/keys (effects remain idempotent)
  - regenerate = new version lineage (never overwrite prior outcomes)
- **Data classification and egress rules are enforceable**: store declared `data_classes` and `effects` so policy can gate `external_send`.

---

## 1) Storage topology (multi-store by design)

### 1.1 PostgreSQL (primary data plane for the Layer)

**System of record** for:

- **Ledgers**: events, jobs/actions, outcomes, documents, memory, entities, policy decisions, gates.
- **Workflow runtime control tables**: workflow runs, step state/leasing, request idempotency map.
- **Cold cache**: durable cached operator/LLM/tool results with provenance and TTL.
- **Credits/billing**: balances, reservations, usage events, credit ledger, pricing snapshots.
- Optional: registry (capability/plugin versions) if you want DB-managed manifests.

Why Postgres:

- strong concurrency primitives for DB-leased workflow execution (`FOR UPDATE SKIP LOCKED`)
- JSONB for typed-but-evolving payloads while still being queryable/indexable
- partitioning for hot ledgers (events/jobs) and retention-friendly archiving

### 1.2 Redis (hot cache + counters)

Used for:

- **hot response cache** (fast reuse)
- **request dedupe** (fast “already running / already done” checks)
- **rate limit / quota counters**
- optional “kick worker now” queue/stream (polling remains viable without it)

### 1.3 Object storage (S3 / MinIO)

Used for large blobs:

- uploaded documents (raw)
- derived artifacts (OCR outputs, chunk manifests, PDFs)
- large outcomes (if too big for Postgres)

Paths should encode `tenant_id` + lifecycle + owner id + hashes (see §6).

### 1.4 Vector DB (Qdrant)

Used for:

- embedding vectors for document chunks and other retrievable artifacts
- tenant-scoped filters (`tenant_id`) and collection versioning for reproducibility

### 1.5 “Platform” domain DB (existing CanApply MySQL/MariaDB)

This remains the **downstream domain state** (funding requests, professors, emails, credentials, etc.).

- The Intelligence Layer **reads** from it via the Context Builder.
- The Intelligence Layer **writes** to it only via **policy-gated apply operators**.
- The platform DB is not the Layer’s ledger; it is a **materialized downstream state** that must be explainable via ledger provenance.

---

## 2) Identity model (keys you must get right)

### 2.1 Required IDs (always present in records)

- `tenant_id` (required everywhere)
- `workflow_id` (canonical execution identity; UUID)
- `thread_id` (UX grouping; may be an integer from platform DB; optional on some records)
- `intent_id` (UUID; one normalized request)
- `plan_id` (UUID; compiled steps program)

### 2.2 Trace + correlation

Every ledger record carries:

- `correlation_id` (UUID; spans API → worker → provider; may equal `workflow_id` in v1 but keep both)
- optional provider traces: `trace_id`, `trace_type`

### 2.3 Version lineages

- Outcomes and other versioned ledgers (documents/memory/entities) use **lineage IDs** and **monotonic versions**:
  - `lineage_id` groups versions
  - `version` is an integer starting at 1
  - `parent_*_id` links to the prior version for audit

This directly supports `replay` vs `regenerate` semantics.

---

## 3) PostgreSQL logical schemas

Recommended DB: `intelligence_layer` (name not important).

Schemas (namespaces):

- `runtime` — workflow engine state and leasing
- `ledger` — append-first ledgers (events/jobs/outcomes/documents/memory/entities/policy/gates)
- `cache` — durable cache
- `billing` — credits and usage accounting
- `registry` — optional plugin/capability registry (if not filesystem-managed)

> Note: DDL below is Postgres-oriented and uses `jsonb` for typed payloads. If you prefer strict relational columns for specific outcomes, add materialized projections (views/tables) **derived from outcomes**, never replacing the outcome ledger as truth.

---

## 4) PostgreSQL DDL (proposed)

### 4.1 Common setup

```sql
-- Optional but useful for server-side UUID generation.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS runtime;
CREATE SCHEMA IF NOT EXISTS ledger;
CREATE SCHEMA IF NOT EXISTS cache;
CREATE SCHEMA IF NOT EXISTS billing;
CREATE SCHEMA IF NOT EXISTS registry;
```

### 4.2 `runtime` schema — workflow engine control plane

#### `runtime.threads`

One row per conversation/workspace in the Intelligence Layer, keyed by the platform scope `(student_id, funding_request_id)` in v1.

```sql
CREATE TABLE IF NOT EXISTS runtime.threads (
  tenant_id              BIGINT       NOT NULL,
  thread_id              BIGINT GENERATED ALWAYS AS IDENTITY,

  student_id             BIGINT       NOT NULL, -- platform students.id
  funding_request_id     BIGINT       NOT NULL, -- platform funding_requests.id

  status                 TEXT         NOT NULL, -- new|active|running|completed|archived|failed

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, thread_id),
  CONSTRAINT threads_scope_uq UNIQUE (tenant_id, student_id, funding_request_id)
);

CREATE INDEX IF NOT EXISTS threads_student_created
  ON runtime.threads (tenant_id, student_id, created_at DESC);
```

Notes:

- The platform `funding_request_id` already implies a single student in your current schema, but we keep `(student_id, funding_request_id)` unique to match the API contract and allow future scope types.
- If you later introduce other scope types (program applications, support tickets), extend this table with `(scope_type, scope_id)` and make uniqueness `(tenant_id, scope_type, scope_id)`.

#### `runtime.workflow_runs`

Single row per workflow execution.

```sql
CREATE TABLE IF NOT EXISTS runtime.workflow_runs (
  tenant_id              BIGINT       NOT NULL,
  workflow_id            UUID         NOT NULL,
  correlation_id         UUID         NOT NULL,

  thread_id              BIGINT       NULL,
  scope_type             TEXT         NULL,
  scope_id               TEXT         NULL,

  intent_id              UUID         NOT NULL,
  plan_id                UUID         NULL,

  capability_name        TEXT         NULL,
  capability_version     TEXT         NULL,

  status                 TEXT         NOT NULL, -- accepted|running|waiting_approval|completed|failed|cancelled
  execution_mode         TEXT         NOT NULL, -- dry_run|draft_only|human_gated|auto_exec
  replay_mode            TEXT         NOT NULL DEFAULT 'reproduce', -- reproduce|replay|regenerate

  request_key            BYTEA        NULL, -- stable hash for request-level idempotency
  parent_workflow_id     UUID         NULL,  -- for regenerate lineage

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  started_at             TIMESTAMPTZ  NULL,
  completed_at           TIMESTAMPTZ  NULL,
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, workflow_id),
  CONSTRAINT workflow_runs_thread_fk
    FOREIGN KEY (tenant_id, thread_id) REFERENCES runtime.threads(tenant_id, thread_id)
);

CREATE INDEX IF NOT EXISTS workflow_runs_tenant_status_created
  ON runtime.workflow_runs (tenant_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS workflow_runs_thread_created
  ON runtime.workflow_runs (tenant_id, thread_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS workflow_runs_request_key_uq
  ON runtime.workflow_runs (tenant_id, request_key)
  WHERE request_key IS NOT NULL;
```

#### `runtime.workflow_steps`

Derived control state per step (used for leasing and resumption). This is a **projection**; the ledger remains the source of truth.

```sql
CREATE TABLE IF NOT EXISTS runtime.workflow_steps (
  tenant_id              BIGINT       NOT NULL,
  workflow_id            UUID         NOT NULL,
  step_id                TEXT         NOT NULL, -- stable within plan (e.g. "s3")

  kind                   TEXT         NOT NULL, -- operator|agent|policy_check|human_gate
  name                   TEXT         NOT NULL,
  operator_name          TEXT         NULL,
  operator_version       TEXT         NULL,

  effects                TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
  policy_tags            TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
  risk_level             TEXT         NOT NULL, -- low|medium|high
  cache_policy           TEXT         NOT NULL, -- never|use_if_safe|ttl_only|force_refresh

  idempotency_key        TEXT         NULL, -- derived concrete key
  input_payload          JSONB        NOT NULL DEFAULT '{}'::jsonb,
  input_hash             BYTEA        NULL,

  status                 TEXT         NOT NULL, -- PENDING|READY|RUNNING|SUCCEEDED|FAILED_RETRYABLE|FAILED_FINAL|WAITING_APPROVAL|SKIPPED|CANCELLED
  attempt_count          INT          NOT NULL DEFAULT 0,
  next_retry_at          TIMESTAMPTZ  NULL,

  lease_owner            TEXT         NULL,
  lease_expires_at       TIMESTAMPTZ  NULL,

  last_job_id            UUID         NULL,
  gate_id                UUID         NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  started_at             TIMESTAMPTZ  NULL,
  finished_at            TIMESTAMPTZ  NULL,
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, workflow_id, step_id),
  CONSTRAINT workflow_steps_run_fk
    FOREIGN KEY (tenant_id, workflow_id) REFERENCES runtime.workflow_runs(tenant_id, workflow_id)
);

-- Leasing query pattern: find runnable steps for a tenant.
CREATE INDEX IF NOT EXISTS workflow_steps_runnable
  ON runtime.workflow_steps (tenant_id, status, next_retry_at, lease_expires_at)
  WHERE status IN ('READY', 'FAILED_RETRYABLE');

CREATE INDEX IF NOT EXISTS workflow_steps_workflow
  ON runtime.workflow_steps (tenant_id, workflow_id);
```

> Leasing pattern (worker side): `SELECT ... FOR UPDATE SKIP LOCKED` on `workflow_steps` where `status='READY'` and `(lease_expires_at IS NULL OR lease_expires_at < now())`, then set `lease_owner`, `lease_expires_at`, `status='RUNNING'` in the same transaction.

---

#### `runtime.scheduled_jobs` (basic scheduling)

Supports “basic scheduling” in v1 (delayed retries/backoffs, internal housekeeping) without introducing a second workflow engine.

**Decision (v1/dev mode):** `runtime.scheduled_jobs` is **system-of-record** in the Intelligence Layer Postgres only. The platform may later maintain a UI projection, but it should be derived (webhook/poll), not treated as a second source of truth.

```sql
CREATE TABLE IF NOT EXISTS runtime.scheduled_jobs (
  tenant_id              BIGINT       NOT NULL,
  scheduled_job_id       UUID         NOT NULL,

  kind                   TEXT         NOT NULL, -- workflow_submit|step_wake|operator_invoke|maintenance
  dedupe_key             TEXT         NOT NULL,

  run_after              TIMESTAMPTZ  NOT NULL,
  status                 TEXT         NOT NULL, -- queued|running|succeeded|failed|cancelled|dead
  attempt_count          INT          NOT NULL DEFAULT 0,
  max_attempts           INT          NOT NULL DEFAULT 5,

  workflow_id            UUID         NULL,
  step_id                TEXT         NULL,
  payload                JSONB        NOT NULL DEFAULT '{}'::jsonb,

  lease_owner            TEXT         NULL,
  lease_expires_at       TIMESTAMPTZ  NULL,

  last_error             JSONB        NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, scheduled_job_id),
  CONSTRAINT scheduled_jobs_dedupe_uq UNIQUE (tenant_id, dedupe_key)
);

CREATE INDEX IF NOT EXISTS scheduled_jobs_due
  ON runtime.scheduled_jobs (tenant_id, status, run_after, lease_expires_at)
  WHERE status IN ('queued', 'failed');
```

Notes:

- `dedupe_key` should encode semantic uniqueness (e.g., `backoff_retry:{tenant}:{workflow_id}:{step_id}:{attempt}`).
- Scheduler loop claims due jobs with `FOR UPDATE SKIP LOCKED`, sets a lease, and emits an outbox message to wake workers.

---

#### `runtime.outbox` (DB outbox for broker/webhook dispatch)

Implements the outbox pattern for “DB write + publish” consistency (broker messages and outbound webhooks are **signals**, not truth).

```sql
CREATE TABLE IF NOT EXISTS runtime.outbox (
  tenant_id              BIGINT       NOT NULL,
  outbox_id              UUID         NOT NULL,

  topic                  TEXT         NOT NULL, -- worker_kick|scheduled_job_due|webhook_deliver|...
  message_key            TEXT         NULL,     -- for ordering/dedupe on the broker side
  payload                JSONB        NOT NULL,

  available_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
  status                 TEXT         NOT NULL, -- pending|delivered|failed|dead
  attempt_count          INT          NOT NULL DEFAULT 0,
  max_attempts           INT          NOT NULL DEFAULT 20,

  lock_owner             TEXT         NULL,
  lock_expires_at        TIMESTAMPTZ  NULL,
  last_error             JSONB        NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, outbox_id)
);

CREATE INDEX IF NOT EXISTS outbox_pending
  ON runtime.outbox (tenant_id, status, available_at, lock_expires_at)
  WHERE status IN ('pending', 'failed');
```

---

### 4.3 `ledger` schema — append-first sources of truth

#### `ledger.intents`

Stores normalized intents (typed, versioned) with redaction support.

```sql
CREATE TABLE IF NOT EXISTS ledger.intents (
  tenant_id              BIGINT       NOT NULL,
  intent_id              UUID         NOT NULL,
  intent_type            TEXT         NOT NULL,
  schema_version         TEXT         NOT NULL,
  source                TEXT          NOT NULL, -- chat|api|webhook|system

  thread_id              BIGINT       NULL,
  scope_type             TEXT         NULL,
  scope_id               TEXT         NULL,

  actor                 JSONB         NOT NULL, -- principal + role + trust_level (per your auth model)
  inputs                JSONB         NOT NULL,
  constraints           JSONB         NOT NULL DEFAULT '{}'::jsonb,
  context_refs          JSONB         NOT NULL DEFAULT '{}'::jsonb,

  data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[], -- Public|Internal|Confidential|Regulated

  redacted_inputs       JSONB         NULL,

  correlation_id         UUID         NOT NULL,
  producer_kind          TEXT         NOT NULL, -- adapter|kernel
  producer_name          TEXT         NOT NULL,
  producer_version       TEXT         NOT NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, intent_id)
);

CREATE INDEX IF NOT EXISTS intents_thread_created
  ON ledger.intents (tenant_id, thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS intents_type_created
  ON ledger.intents (tenant_id, intent_type, created_at DESC);
```

#### `ledger.plans`

Stores validated plans (typed steps, metadata, stop conditions).

```sql
CREATE TABLE IF NOT EXISTS ledger.plans (
  tenant_id              BIGINT       NOT NULL,
  plan_id                UUID         NOT NULL,
  intent_id              UUID         NOT NULL,

  schema_version         TEXT         NOT NULL,
  planner_name           TEXT         NOT NULL,
  planner_version        TEXT         NOT NULL,

  plan                  JSONB         NOT NULL, -- full plan, including steps[]
  plan_hash             BYTEA         NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, plan_id),
  CONSTRAINT plans_intent_fk
    FOREIGN KEY (tenant_id, intent_id) REFERENCES ledger.intents(tenant_id, intent_id)
);

CREATE INDEX IF NOT EXISTS plans_intent_created
  ON ledger.plans (tenant_id, intent_id, created_at DESC);
```

#### `ledger.events` (Event Ledger)

Append-only timeline of “what happened”, ordered by `event_no`.

```sql
CREATE TABLE IF NOT EXISTS ledger.events (
  event_no               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

  tenant_id              BIGINT       NOT NULL,
  event_id               UUID         NOT NULL,
  schema_version         TEXT         NOT NULL DEFAULT '1.0',

  workflow_id            UUID         NOT NULL,
  thread_id              BIGINT       NULL,
  intent_id              UUID         NULL,
  plan_id                UUID         NULL,
  step_id                TEXT         NULL,
  job_id                 UUID         NULL,
  outcome_id             UUID         NULL,
  gate_id                UUID         NULL,
  policy_decision_id     UUID         NULL,

  event_type             TEXT         NOT NULL,
  severity               TEXT         NOT NULL DEFAULT 'info',
  actor                  JSONB        NOT NULL, -- {type,id,role}

  payload                JSONB        NOT NULL DEFAULT '{}'::jsonb,
  payload_hash           BYTEA        NULL,

  correlation_id         UUID         NOT NULL,
  producer_kind          TEXT         NOT NULL, -- kernel|operator|agent|adapter
  producer_name          TEXT         NOT NULL,
  producer_version       TEXT         NOT NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS events_event_id_uq
  ON ledger.events (tenant_id, event_id);

CREATE INDEX IF NOT EXISTS events_workflow_ordered
  ON ledger.events (tenant_id, workflow_id, event_no);

CREATE INDEX IF NOT EXISTS events_thread_created
  ON ledger.events (tenant_id, thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS events_correlation
  ON ledger.events (tenant_id, correlation_id);
```

> Streaming: SSE readers poll by `(tenant_id, workflow_id, event_no > last_seen)` and stream results in order.

#### `ledger.webhook_events` + `ledger.webhook_delivery_attempts` (platform-only outbound webhooks)

Outbound webhooks (v1) are **platform-owned** and delivered **only** to the CanApply platform backend (no per-tenant endpoints in v1). Keep an immutable record of what was intended to be sent and append delivery attempts for audit and incident response.

```sql
CREATE TABLE IF NOT EXISTS ledger.webhook_events (
  tenant_id              BIGINT       NOT NULL,
  webhook_event_id       UUID         NOT NULL,
  schema_version         TEXT         NOT NULL DEFAULT '1.0',

  endpoint_ref           TEXT         NOT NULL, -- e.g., "platform_backend"
  event_type             TEXT         NOT NULL, -- workflow.completed|workflow.gate_requested|credits.low|...
  payload                JSONB        NOT NULL,
  payload_hash           BYTEA        NULL,

  data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],

  correlation_id         UUID         NOT NULL,
  workflow_id            UUID         NULL,
  thread_id              BIGINT       NULL,
  intent_id              UUID         NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, webhook_event_id)
);

CREATE INDEX IF NOT EXISTS webhook_events_type_created
  ON ledger.webhook_events (tenant_id, event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS webhook_events_workflow_created
  ON ledger.webhook_events (tenant_id, workflow_id, created_at DESC);
```

```sql
CREATE TABLE IF NOT EXISTS ledger.webhook_delivery_attempts (
  tenant_id              BIGINT       NOT NULL,
  webhook_event_id       UUID         NOT NULL,
  attempt_no             INT          NOT NULL,

  status                 TEXT         NOT NULL, -- delivered|failed|dead
  http_status            INT          NULL,
  latency_ms             INT          NULL,

  error                  JSONB        NULL,
  response_hash          BYTEA        NULL, -- hash of response body (do not store raw)

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, webhook_event_id, attempt_no),
  CONSTRAINT webhook_delivery_attempts_event_fk
    FOREIGN KEY (tenant_id, webhook_event_id) REFERENCES ledger.webhook_events(tenant_id, webhook_event_id)
);

CREATE INDEX IF NOT EXISTS webhook_delivery_attempts_status_created
  ON ledger.webhook_delivery_attempts (tenant_id, status, created_at DESC);
```

#### `ledger.jobs` + `ledger.job_attempts` (Job Ledger)

`ledger.jobs` is the idempotency record (unique per `(tenant_id, operator_name, idempotency_key)`).
Attempts are appended in `ledger.job_attempts`.

```sql
CREATE TABLE IF NOT EXISTS ledger.jobs (
  tenant_id              BIGINT       NOT NULL,
  job_id                 UUID         NOT NULL,
  schema_version         TEXT         NOT NULL DEFAULT '1.0',

  workflow_id            UUID         NULL,
  thread_id              BIGINT       NULL,
  intent_id              UUID         NULL,
  plan_id                UUID         NULL,
  step_id                TEXT         NULL,

  operator_name          TEXT         NOT NULL,
  operator_version       TEXT         NOT NULL,
  idempotency_key        TEXT         NOT NULL,

  effects                TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
  policy_tags            TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
  data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],

  status                 TEXT         NOT NULL, -- queued|running|succeeded|failed|cancelled
  attempt_count          INT          NOT NULL DEFAULT 0,

  input_payload          JSONB        NOT NULL,
  input_hash             BYTEA        NULL,

  result_payload         JSONB        NULL,
  result_hash            BYTEA        NULL,

  error                 JSONB         NULL, -- {code,category,retryable,...}
  nondeterminism        JSONB         NULL, -- {is_nondeterministic,reasons,stability}

  trace_id               TEXT         NULL,
  trace_type             TEXT         NULL, -- openai|cache|manual|...

  metrics                JSONB        NULL, -- latency_ms,tokens,cost,provider...

  correlation_id         UUID         NOT NULL,
  producer_kind          TEXT         NOT NULL, -- kernel|operator
  producer_name          TEXT         NOT NULL,
  producer_version       TEXT         NOT NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  started_at             TIMESTAMPTZ  NULL,
  finished_at            TIMESTAMPTZ  NULL,

  PRIMARY KEY (tenant_id, job_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency_uq
  ON ledger.jobs (tenant_id, operator_name, idempotency_key);

CREATE INDEX IF NOT EXISTS jobs_workflow_created
  ON ledger.jobs (tenant_id, workflow_id, created_at DESC);

CREATE INDEX IF NOT EXISTS jobs_status_created
  ON ledger.jobs (tenant_id, status, created_at DESC);
```

```sql
CREATE TABLE IF NOT EXISTS ledger.job_attempts (
  tenant_id              BIGINT       NOT NULL,
  job_id                 UUID         NOT NULL,
  attempt_no             INT          NOT NULL,

  status                 TEXT         NOT NULL, -- running|succeeded|failed
  started_at             TIMESTAMPTZ  NULL,
  finished_at            TIMESTAMPTZ  NULL,

  error                 JSONB         NULL,
  metrics                JSONB        NULL,

  trace_id               TEXT         NULL,
  trace_type             TEXT         NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, job_id, attempt_no),
  CONSTRAINT job_attempts_job_fk
    FOREIGN KEY (tenant_id, job_id) REFERENCES ledger.jobs(tenant_id, job_id)
);

CREATE INDEX IF NOT EXISTS job_attempts_status_created
  ON ledger.job_attempts (tenant_id, status, created_at DESC);
```

#### `ledger.outcomes` (Outcome Ledger)

Immutable versioned artifacts with explicit lineage.

```sql
CREATE TABLE IF NOT EXISTS ledger.outcomes (
  tenant_id              BIGINT       NOT NULL,
  outcome_id             UUID         NOT NULL,
  lineage_id             UUID         NOT NULL,
  version                INT          NOT NULL,
  parent_outcome_id      UUID         NULL,

  outcome_type           TEXT         NOT NULL,
  schema_version         TEXT         NOT NULL,
  status                 TEXT         NOT NULL, -- draft|final|partial|failed|retracted
  visibility             TEXT         NOT NULL DEFAULT 'private', -- private|shareable|public

  workflow_id            UUID         NULL,
  thread_id              BIGINT       NULL,
  intent_id              UUID         NULL,
  plan_id                UUID         NULL,
  step_id                TEXT         NULL,
  job_id                 UUID         NULL,

  content                JSONB        NULL,
  content_object_uri     TEXT         NULL, -- for large outcomes stored in S3
  content_hash           BYTEA        NULL,

  confidence             DOUBLE PRECISION NULL,
  data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],

  producer_kind          TEXT         NOT NULL, -- operator|agent
  producer_name          TEXT         NOT NULL,
  producer_version       TEXT         NOT NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, outcome_id),
  CONSTRAINT outcomes_lineage_version_uq UNIQUE (tenant_id, lineage_id, version)
);

CREATE INDEX IF NOT EXISTS outcomes_workflow_created
  ON ledger.outcomes (tenant_id, workflow_id, created_at DESC);

CREATE INDEX IF NOT EXISTS outcomes_thread_type_created
  ON ledger.outcomes (tenant_id, thread_id, outcome_type, created_at DESC);
```

> Superseding: do **not** update old outcomes. Create a new version with the same `lineage_id` and `parent_outcome_id=prior_outcome_id`.

#### `ledger.documents` + `ledger.document_revisions` (Document Ledger)

Uploads are immutable; transformations append new revisions.

```sql
CREATE TABLE IF NOT EXISTS ledger.documents (
  tenant_id              BIGINT       NOT NULL,
  document_id            UUID         NOT NULL,

  owner_principal_id     BIGINT       NOT NULL, -- student/user id from platform auth model
  document_type          TEXT         NOT NULL, -- resume|sop|transcript|portfolio|other
  title                  TEXT         NOT NULL,
  lifecycle              TEXT         NOT NULL, -- temp|sandbox|final

  source_object_uri      TEXT         NOT NULL,
  source_hash            BYTEA        NULL,
  mime_type              TEXT         NULL,
  size_bytes             BIGINT       NULL,

  status                 TEXT         NOT NULL, -- uploaded|queued_processing|processing|processed|queued_export|exporting|exported|failed
  processor_version      TEXT         NULL,

  current_revision_id    UUID         NULL, -- pointer (projection); event changes are required

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, document_id)
);

CREATE INDEX IF NOT EXISTS documents_owner_type_created
  ON ledger.documents (tenant_id, owner_principal_id, document_type, created_at DESC);
```

```sql
CREATE TABLE IF NOT EXISTS ledger.document_revisions (
  tenant_id              BIGINT       NOT NULL,
  revision_id            UUID         NOT NULL,
  document_id            UUID         NOT NULL,
  parent_revision_id     UUID         NULL,

  revision_kind          TEXT         NOT NULL, -- extracted_text|processed_content|exported_pdf|chunks_manifest|...
  object_uri             TEXT         NULL,      -- S3 pointer for large content
  content                JSONB        NULL,      -- small structured content (e.g., missing_fields)
  content_hash           BYTEA        NULL,

  processor_name         TEXT         NOT NULL,
  processor_version      TEXT         NOT NULL,
  job_id                 UUID         NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, revision_id),
  CONSTRAINT doc_revisions_doc_fk
    FOREIGN KEY (tenant_id, document_id) REFERENCES ledger.documents(tenant_id, document_id)
);

CREATE INDEX IF NOT EXISTS doc_revisions_doc_created
  ON ledger.document_revisions (tenant_id, document_id, created_at DESC);
```

#### `ledger.memory_facts` (Memory Ledger)

Typed, versioned, TTL-capable memory facts.

```sql
CREATE TABLE IF NOT EXISTS ledger.memory_facts (
  tenant_id              BIGINT       NOT NULL,
  memory_id              UUID         NOT NULL,
  lineage_id             UUID         NOT NULL,
  version                INT          NOT NULL,
  parent_memory_id       UUID         NULL,

  owner_principal_id     BIGINT       NOT NULL,
  scope_type             TEXT         NULL, -- optional: capability|thread|global
  scope_id               TEXT         NULL,

  memory_type            TEXT         NOT NULL, -- tone|do_dont|preference|goal|bio|instruction|guardrail|other
  content                JSONB        NOT NULL,

  source                 TEXT         NOT NULL DEFAULT 'inferred', -- user|system|inferred
  confidence             DOUBLE PRECISION NOT NULL DEFAULT 0.7,

  is_active              BOOLEAN      NOT NULL DEFAULT true,
  expires_at             TIMESTAMPTZ  NULL,

  workflow_id            UUID         NULL,
  job_id                 UUID         NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, memory_id),
  CONSTRAINT memory_lineage_version_uq UNIQUE (tenant_id, lineage_id, version)
);

CREATE INDEX IF NOT EXISTS memory_owner_type_active
  ON ledger.memory_facts (tenant_id, owner_principal_id, memory_type, is_active);

CREATE INDEX IF NOT EXISTS memory_expires_at
  ON ledger.memory_facts (tenant_id, expires_at)
  WHERE expires_at IS NOT NULL;
```

#### `ledger.entities` + `ledger.entity_versions` (Entity Ledger)

Stable entity IDs with append-only versions.

```sql
CREATE TABLE IF NOT EXISTS ledger.entities (
  tenant_id              BIGINT       NOT NULL,
  entity_id              UUID         NOT NULL,
  entity_type            TEXT         NOT NULL, -- professor|program|institution

  canonical_key_hash     BYTEA        NOT NULL, -- dedupe key hash (e.g., lower(email)+institution_id for professor)
  source                 TEXT         NOT NULL, -- canspider|manual|platform

  is_active              BOOLEAN      NOT NULL DEFAULT true,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, entity_id),
  CONSTRAINT entities_canonical_key_uq UNIQUE (tenant_id, entity_type, canonical_key_hash)
);
```

```sql
CREATE TABLE IF NOT EXISTS ledger.entity_versions (
  tenant_id              BIGINT       NOT NULL,
  entity_version_id      UUID         NOT NULL,
  entity_id              UUID         NOT NULL,
  version                INT          NOT NULL,
  parent_entity_version_id UUID       NULL,

  canonical              JSONB        NOT NULL,
  aliases                JSONB        NOT NULL DEFAULT '[]'::jsonb,
  source_ref             JSONB        NULL, -- e.g., {platform_professor_id: 123, canspider_digest_id: 456}

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, entity_version_id),
  CONSTRAINT entity_versions_entity_fk
    FOREIGN KEY (tenant_id, entity_id) REFERENCES ledger.entities(tenant_id, entity_id),
  CONSTRAINT entity_versions_version_uq UNIQUE (tenant_id, entity_id, version)
);

CREATE INDEX IF NOT EXISTS entity_versions_entity_created
  ON ledger.entity_versions (tenant_id, entity_id, created_at DESC);
```

#### `ledger.policy_decisions`

First-class policy outputs with deterministic reconstruction via inputs hash + engine version.

```sql
CREATE TABLE IF NOT EXISTS ledger.policy_decisions (
  tenant_id              BIGINT       NOT NULL,
  policy_decision_id     UUID         NOT NULL,

  stage                  TEXT         NOT NULL, -- intake|plan|action|outcome|apply
  decision               TEXT         NOT NULL, -- ALLOW|DENY|REQUIRE_APPROVAL|ALLOW_WITH_REDACTION|TRANSFORM
  reason_code            TEXT         NOT NULL,
  reason                 TEXT         NULL,

  requirements           JSONB        NOT NULL DEFAULT '{}'::jsonb,
  limits                 JSONB        NOT NULL DEFAULT '{}'::jsonb,
  redactions             JSONB        NOT NULL DEFAULT '[]'::jsonb,
  transform              JSONB        NULL,

  inputs_hash            BYTEA        NOT NULL, -- hash of normalized PolicyContext (redacted)
  policy_engine_name     TEXT         NOT NULL,
  policy_engine_version  TEXT         NOT NULL,

  workflow_id            UUID         NULL,
  intent_id              UUID         NULL,
  plan_id                UUID         NULL,
  step_id                TEXT         NULL,
  job_id                 UUID         NULL,

  correlation_id         UUID         NOT NULL,
  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, policy_decision_id)
);

CREATE INDEX IF NOT EXISTS policy_decisions_workflow_created
  ON ledger.policy_decisions (tenant_id, workflow_id, created_at DESC);
```

#### `ledger.gates` + `ledger.gate_decisions`

Explicit human gates (approval required to proceed).

```sql
CREATE TABLE IF NOT EXISTS ledger.gates (
  tenant_id              BIGINT       NOT NULL,
  gate_id                UUID         NOT NULL,

  workflow_id            UUID         NOT NULL,
  step_id                TEXT         NOT NULL,

  gate_type              TEXT         NOT NULL, -- human_confirm|human_edit_then_approve|...
  reason_code            TEXT         NOT NULL,
  summary                TEXT         NOT NULL,
  preview                JSONB        NOT NULL DEFAULT '{}'::jsonb,

  target_outcome_id      UUID         NULL, -- the exact version being approved
  status                 TEXT         NOT NULL, -- requested|approved|rejected|expired
  expires_at             TIMESTAMPTZ  NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, gate_id)
);

CREATE INDEX IF NOT EXISTS gates_workflow_step
  ON ledger.gates (tenant_id, workflow_id, step_id);
```

```sql
CREATE TABLE IF NOT EXISTS ledger.gate_decisions (
  tenant_id              BIGINT       NOT NULL,
  gate_decision_id       UUID         NOT NULL,
  gate_id                UUID         NOT NULL,

  actor                  JSONB        NOT NULL, -- {type,id,role}
  decision               TEXT         NOT NULL, -- approve|reject|edit_then_approve
  payload                JSONB        NULL,     -- edits, if any

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, gate_decision_id),
  CONSTRAINT gate_decisions_gate_fk
    FOREIGN KEY (tenant_id, gate_id) REFERENCES ledger.gates(tenant_id, gate_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS gate_decisions_one_per_actor
  ON ledger.gate_decisions (tenant_id, gate_id, (actor->>'id'));
```

---

### 4.4 `cache` schema — durable cache (cold cache)

A single generic cache table works well if you include `namespace` and rich metadata.

```sql
CREATE TABLE IF NOT EXISTS cache.entries (
  tenant_id              BIGINT       NOT NULL,
  namespace              TEXT         NOT NULL, -- llm|tool|context|retrieval|...
  cache_key              BYTEA        NOT NULL, -- stable hash of normalized inputs + versions

  value                  JSONB        NULL,
  value_object_uri       TEXT         NULL,
  value_hash             BYTEA        NULL,

  ttl_seconds            INT          NULL,
  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  expires_at             TIMESTAMPTZ  NULL,
  last_accessed_at       TIMESTAMPTZ  NULL,
  hit_count              BIGINT       NOT NULL DEFAULT 0,

  metadata               JSONB        NOT NULL DEFAULT '{}'::jsonb, -- model, prompt_version, tool_schema_versions, policy_hash, etc.

  PRIMARY KEY (tenant_id, namespace, cache_key)
);

CREATE INDEX IF NOT EXISTS cache_entries_expires
  ON cache.entries (tenant_id, expires_at)
  WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS cache_entries_namespace_created
  ON cache.entries (tenant_id, namespace, created_at DESC);
```

> Hot cache mirrors this table in Redis with shorter TTL and smaller payloads.

---

### 4.5 `billing` schema — credits, reservations, usage, pricing

Implements the normative requirements in `8-ai-credits-and-budget-enforcement.md`.

#### Balances (tenant pool + optional per-user sub-budgets)

```sql
CREATE TABLE IF NOT EXISTS billing.credit_balances (
  tenant_id              BIGINT       NOT NULL,
  principal_id           BIGINT       NOT NULL DEFAULT 0, -- 0 = tenant pool; else per-user

  balance_credits        BIGINT       NOT NULL,
  overdraft_limit        BIGINT       NOT NULL DEFAULT 0,
  expires_at             TIMESTAMPTZ  NULL,

  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, principal_id)
);
```

#### Reservations (pre-auth) and settlement (capture/release)

```sql
CREATE TABLE IF NOT EXISTS billing.credit_reservations (
  tenant_id              BIGINT       NOT NULL,
  reservation_id         UUID         NOT NULL,

  principal_id           BIGINT       NULL,
  workflow_id            UUID         NULL,
  request_key            BYTEA        NOT NULL,

  reserved_credits       BIGINT       NOT NULL,
  status                 TEXT         NOT NULL, -- reserved|captured|released|expired
  expires_at             TIMESTAMPTZ  NOT NULL,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, reservation_id),
  CONSTRAINT credit_reservations_request_uq UNIQUE (tenant_id, request_key)
);

CREATE INDEX IF NOT EXISTS credit_reservations_status_expires
  ON billing.credit_reservations (tenant_id, status, expires_at);
```

#### Pricing snapshots (versioned)

```sql
CREATE TABLE IF NOT EXISTS billing.pricing_versions (
  pricing_version_id     UUID         NOT NULL,
  provider               TEXT         NOT NULL, -- openai|anthropic|...
  version                TEXT         NOT NULL,
  effective_from         TIMESTAMPTZ  NOT NULL,
  effective_to           TIMESTAMPTZ  NULL,
  pricing_json           JSONB        NOT NULL,
  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (pricing_version_id),
  CONSTRAINT pricing_versions_provider_version_uq UNIQUE (provider, version)
);
```

#### Credit rate versions (credits/USD) (versioned)

Normative requirement: the credit conversion rate **R** must be versioned for deterministic reconciliation.

```sql
CREATE TABLE IF NOT EXISTS billing.credit_rate_versions (
  credit_rate_version    TEXT         NOT NULL, -- e.g. "2026-01-01"
  credits_per_usd        NUMERIC(18, 8) NOT NULL, -- R

  effective_from         TIMESTAMPTZ  NOT NULL,
  effective_to           TIMESTAMPTZ  NULL,
  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (credit_rate_version)
);
```

#### Usage events (1 row per billable operation)

```sql
CREATE TABLE IF NOT EXISTS billing.usage_events (
  tenant_id              BIGINT       NOT NULL,
  usage_event_id         UUID         NOT NULL,

  principal_id           BIGINT       NULL,
  workflow_id            UUID         NULL,
  job_id                 UUID         NULL,

  operation_type         TEXT         NOT NULL, -- llm_generate|embedding|rerank|...
  provider               TEXT         NOT NULL,
  model                  TEXT         NOT NULL,

  usage                  JSONB        NOT NULL, -- provider-reported usage
  cost_usd               NUMERIC(18, 8) NOT NULL,
  effective_cost_usd     NUMERIC(18, 8) NOT NULL,

  credits_charged        BIGINT       NOT NULL,
  pricing_version_id     UUID         NOT NULL,
  credit_rate_version    TEXT         NOT NULL,
  estimated              BOOLEAN      NOT NULL DEFAULT false,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, usage_event_id),
  CONSTRAINT usage_events_pricing_fk
    FOREIGN KEY (pricing_version_id) REFERENCES billing.pricing_versions(pricing_version_id),
  CONSTRAINT usage_events_credit_rate_fk
    FOREIGN KEY (credit_rate_version) REFERENCES billing.credit_rate_versions(credit_rate_version)
);

CREATE INDEX IF NOT EXISTS usage_events_workflow_created
  ON billing.usage_events (tenant_id, workflow_id, created_at DESC);
```

#### Credit ledger (immutable accounting trail)

```sql
CREATE TABLE IF NOT EXISTS billing.credit_ledger (
  tenant_id              BIGINT       NOT NULL,
  credit_ledger_id       UUID         NOT NULL,

  principal_id           BIGINT       NULL,
  workflow_id            UUID         NULL,
  request_key            BYTEA        NULL,

  delta_credits          BIGINT       NOT NULL, -- negative = spend, positive = grant/refund
  balance_after          BIGINT       NOT NULL,

  reason_code            TEXT         NOT NULL,
  pricing_version_id     UUID         NULL,
  credit_rate_version    TEXT         NULL,
  metadata               JSONB        NOT NULL DEFAULT '{}'::jsonb,

  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, credit_ledger_id),
  CONSTRAINT credit_ledger_pricing_fk
    FOREIGN KEY (pricing_version_id) REFERENCES billing.pricing_versions(pricing_version_id),
  CONSTRAINT credit_ledger_credit_rate_fk
    FOREIGN KEY (credit_rate_version) REFERENCES billing.credit_rate_versions(credit_rate_version)
);

CREATE UNIQUE INDEX IF NOT EXISTS credit_ledger_request_uq
  ON billing.credit_ledger (tenant_id, request_key)
  WHERE request_key IS NOT NULL;
```

---

### 4.6 `registry` schema (optional but recommended for enterprise)

If manifests are filesystem-managed in v1, you may skip DB registry. If you want DB-managed rollout/pinning, store:

- capability versions and tenant pins
- plugin versions and schema refs

Minimal shape:

```sql
CREATE TABLE IF NOT EXISTS registry.capability_versions (
  capability_name        TEXT         NOT NULL,
  capability_version     TEXT         NOT NULL,
  manifest               JSONB        NOT NULL,
  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  PRIMARY KEY (capability_name, capability_version)
);

CREATE TABLE IF NOT EXISTS registry.tenant_capability_pins (
  tenant_id              BIGINT       NOT NULL,
  capability_name        TEXT         NOT NULL,
  pinned_version         TEXT         NOT NULL,
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, capability_name)
);
```

---

## 5) Redis keyspace design (hot cache + counters)

### 5.1 Request-level idempotency (fast path)

- Key: `req:{tenant_id}:{request_key_hex} -> {workflow_id}`
- TTL: 7–30 days (tunable), with Postgres as durable source.

### 5.2 Hot cache

- Key: `cache:{tenant_id}:{namespace}:{cache_key_hex} -> {compressed_value}`
- TTL: namespace-specific (minutes → days)
- Value: compressed JSON (zstd/gzip) or small msgpack

### 5.3 Rate limits / quotas

- Key: `rl:{tenant_id}:{principal_id}:{dimension}:{window_start}`
- Example dimensions:
  - `external_send_per_minute`
  - `tokens_per_day`
  - `workflows_per_hour`

Use atomic counters with expiry at window end.

### 5.4 Optional: worker “kick” channel

- `streams:workflow_kick` (Redis Streams) or pub/sub to reduce DB polling latency.

---

## 6) Object storage (S3/MinIO) conventions

### 6.1 Lifecycle buckets/prefixes

Aligns with `PREREQUISITE.txt` intent (temp/sandbox/final) while keeping tenancy explicit:

- `s3://{bucket}/layer/{tenant_id}/{principal_id}/temp/{workflow_id}/...`
- `s3://{bucket}/layer/{tenant_id}/{principal_id}/sandbox/{thread_id}/...`
- `s3://{bucket}/layer/{tenant_id}/{principal_id}/final/{sha256}.{ext}`

### 6.2 Dedupe strategy

- Raw uploads dedupe by `(tenant_id, source_hash)` (in DB), not by filename.
- Derived artifacts dedupe by `(tenant_id, content_hash, processor_version)`.

### 6.3 Security

- Server-side encryption (SSE-S3 or SSE-KMS) in prod.
- Never log object contents; store only hashes and URIs in logs/events.

---

## 7) Qdrant collections (embeddings)

### 7.1 Collection naming + versioning

- `doc_chunks__v1`, `doc_chunks__v2`, ...

Payload fields (minimum):

- `tenant_id`
- `document_id`
- `revision_id`
- `chunk_id`
- `chunk_index`
- `text_hash`
- `created_at`

### 7.2 Reproducibility

- Store `collection_name` + `collection_version` in retrieval snapshots (Postgres) and in job/outcome provenance when retrieval is used.

---

## 8) Retention, partitioning, and operational notes

### 8.1 Partition candidates (recommended once volume exists)

- `ledger.events` — partition by month on `created_at` (or by tenant if very large tenants)
- `ledger.jobs` / `ledger.job_attempts` — partition by month on `created_at`
- `billing.usage_events` — partition by month

### 8.2 Retention guidelines (defaults; make tenant-configurable)

- Event ledger: 1–7 years (archive to cold storage before delete, if ever)
- Jobs: hot 30–180 days, archive older
- Outcomes: drafts 90–365 days; finals long-term
- Temp objects: hours-days; sandbox: weeks; finals: indefinite

### 8.3 Auditing / export

Support “audit export” by:

- exporting ledger records with redaction applied (DataClass-aware)
- including hashes and producer versions

---

## 9) Integration with platform DB (downstream “apply” state)

### 9.1 Reads (context building)

Context Builder reads platform tables such as:

- `students` profile fields
- `funding_requests`, `funding_professors`, `funding_institutes`
- user templates and credentials (where applicable)

### 9.2 Writes (apply operators only)

All writes to platform domain tables happen via operators with effects:

- `db_write` to platform DB
- `s3_final_write` for final artifacts
- `external_send` for Gmail send

Every apply/send produces:

- a `ledger.jobs` record (operator execution + idempotency key)
- a `ledger.policy_decisions` record at apply stage
- one or more `ledger.events` including receipt IDs
- an outcome that captures the apply/send receipt (`Email.SendReceipt`, `Workflow.ApplyReceipt`, etc.)

---

## 10) Open design questions (need product/infra decisions)

1. Should ledgers live in the Intelligence Layer Postgres DB only, or do you need/expect some ledgers (documents/threads) to be co-located in the platform MySQL DB for UI convenience?
2. What are the target volumes (workflows/day, events/workflow) and retention requirements (1y vs 7y) to decide when to partition from day 1?
3. Do you need DB-level tenant isolation enforcement (Row Level Security) or is app-layer isolation sufficient initially?
4. What is the authoritative `tenant_id` and principal identity mapping today (single-tenant CanApply vs multi-tenant enterprises)?
5. For Gmail credentials: do tokens remain in the platform DB (`funding_credentials`) or should the Intelligence Layer own an encrypted credentials store (KMS + envelope encryption)?
