# 4. Ledgers (Sources of Truth) Deep Spec

## 4.0 What “ledger” means here

A **ledger** is a persistent collection of records that:

* has a **single responsibility** (events, jobs, documents, outcomes, memory, entities)
* has a **stable schema** with versioning
* is **append-first** (or append + “active pointer” patterns)
* is **provable**: each record is attributable to a producer and traceable to execution

The Kernel **reads ledgers**, writes to ledgers, and **derives runtime state** from them. Capabilities and operators produce artifacts; ledgers store truth.

---

## 4.1 Universal Ledger Record Header (required on every ledger item)

Every ledger item, regardless of type, must include:

```json
{
  "ledger_type": "event|job|document|outcome|memory|entity",
  "schema_version": "1.0",
  "tenant_id": 1,
  "record_id": "uuid",
  "created_at": "2026-01-28T19:03:00Z",

  "producer": {
    "kind": "kernel|operator|agent|adapter",
    "name": "Email.GenerateDraft",
    "version": "1.0"
  },

  "trace": {
    "correlation_id": "corr-uuid",
    "workflow_id": "wf-uuid",
    "thread_id": 4512,
    "intent_id": "intent-uuid",
    "plan_id": "plan-uuid",
    "step_id": "s3",
    "job_id": "act-uuid",
    "trace_id": "openai-...",
    "trace_type": "openai|cache|manual"
  },

  "hash": {
    "payload_hash": "sha256(...)", 
    "prev_hash": "sha256(...optional chain...)"
  }
}
```

### Invariants

* **Tenant isolation is mandatory**: `tenant_id` required everywhere.
* **Traceability is mandatory**: at minimum `correlation_id` and `workflow_id`.
* **Schema versioning is mandatory**: never “just change fields” in production.
* **Producer identity is mandatory**: you need to know what created the record.

---

## 4.2 Mutation rule: “append-first + evented mutation”

You already wrote it, here’s the precise enforcement:

### Rule A: If something changes, you must record why it changed

* Any mutation to a record that is not naturally append-only must produce:

  1. an **event** (Event Ledger)
  2. a **new record version** (if using versioned ledger) or a **pointer update** (if using active pointers)

### Rule B: Derived state is not source of truth

* “Current email draft” is a *projection*:

  * from Outcome Ledger (latest draft outcome) plus apply events
* “Professor summary” is an outcome; professor entity stores normalized identity; the summary is not “the professor”.

---

# 4.3 Required Ledgers: deep definitions

## 1) Event Ledger (immutable, append-only)

**Purpose:** the authoritative history of “what happened”.

### What it stores

* intake accepted/rejected
* plan created
* step transitions
* policy decisions
* operator started/succeeded/failed
* outcomes created
* gates requested/approved/rejected
* apply actions completed

### Minimal schema

```json
{
  "event_type": "INTENT_RECEIVED|PLAN_CREATED|ACTION_STARTED|ACTION_SUCCEEDED|ACTION_FAILED|OUTCOME_CREATED|GATE_REQUESTED|USER_APPROVED|WORKFLOW_COMPLETED|...",
  "severity": "info|warn|error",
  "payload": { "..." : "typed per event_type" }
}
```

### Invariants

* append-only
* totally ordered per workflow (use `sequence_no` or monotonic timestamp + tie-breaker)
* every non-trivial workflow transition must have a corresponding event

### Retention

* enterprise friendly: 1–7 years typical (configurable)
* can archive cold storage, but never silently delete

### Indexes you will need

* `(tenant_id, workflow_id, sequence_no)`
* `(tenant_id, thread_id, created_at)`
* `(tenant_id, correlation_id)`

---

## 2) Job Ledger (execution state, retries, costs)

**Purpose:** the authoritative record of each action execution attempt (LLM calls, web fetches, exports, DB writes via operators).

### What it stores

* operator payload hash + result
* timing
* retry count + retry policy
* cost/tokens
* error taxonomy

### Minimal schema

```json
{
  "job_type": "chat_thread|email_generate|email_review|doc_to_json|...",
  "operator_name": "Email.GenerateDraft",
  "status": "queued|running|succeeded|failed|cancelled",
  "attempt": 1,
  "idempotency_key": "string",
  "input_payload": {},
  "result_payload": {},
  "error": { "code": "...", "retryable": true, "category": "transient" },
  "usage": { "token_in": 0, "token_out": 0, "cost_total": 0.0 }
}
```

### Invariants

* job records are appendable by attempt:

  * either create one row per attempt, or keep one row and append attempts in a JSON array
* must support “exactly-once-ish” via idempotency key
* result payload must be immutable after success

### Retention

* keep hot for 30–180 days; archive older
* costs aggregated into a separate billing table is fine, but job ledger remains audit trail

### Indexes

* `(tenant_id, workflow_id, created_at)`
* `(tenant_id, status, created_at)`
* `(tenant_id, idempotency_key)` unique (or unique per operator)

---

## 3) Document Ledger (uploads, processed content, exports)

**Purpose:** authoritative record of documents and transformations (raw upload → extracted text → processed JSON → exported PDF).

### What it stores

* pointers to S3 paths for raw/sandbox/final
* content hashes for dedupe
* extracted_text, processed_content refs (or store separately)
* missing fields analysis
* processing status and processor version

### Minimal schema (aligned with your `student_documents`)

Key upgrades in “ledger spirit”:

* treat each transformation as a **DocumentRevision** outcome or sub-record.

**Document record**

```json
{
  "document_id": 2001,
  "document_type": "resume|sop|transcript|other",
  "title": "User CV",
  "lifecycle": "temp|sandbox|final",
  "source": { "path": "s3://.../raw.docx", "hash": "..." },
  "derived": {
    "extracted_text_hash": "...",
    "processed_content_hash": "...",
    "exported_pdf_path": "s3://.../cv.pdf"
  },
  "status": "uploaded|processed|exported|failed",
  "processor_version": "docproc-1.2"
}
```

### Invariants

* raw upload is immutable (new upload = new document record)
* transformations are reproducible (store hashes + processor versions)
* “final” documents are only produced via apply/approval (policy controlled)
* dedupe uses `source_file_hash` and `processed_content_hash`

### Retention/TTL

* temps: TTL hours-days
* sandbox: TTL weeks
* finals: indefinite (or enterprise retention rules)

### Indexes

* `(tenant_id, student_id, document_type, created_at)`
* `(tenant_id, source_file_hash)`
* `(tenant_id, processed_content_hash)`

---

## 4) Outcome Ledger (produced outputs, version history)

**Purpose:** authoritative record of produced artifacts from workflows: drafts, reviews, alignments, summaries, recommendations.

This is the **single most important ledger** after events.

### What it stores

* typed artifact content
* versioning and lineage
* “draft vs final”
* links to the job that produced it
* links to the prior outcome it supersedes

### Minimal schema

```json
{
  "outcome_id": "out-uuid",
  "outcome_type": "Draft.Email|Review.Email|Summary.Professor|Alignment.Professor|...",
  "status": "draft|final|superseded|retracted",
  "version": 1,
  "parent_outcome_id": null,
  "content": {},
  "confidence": 0.72,
  "visibility": "private|shareable|public"
}
```

### Invariants

* outcomes are immutable once written
* updates create new versions:

  * v1 draft → v2 optimized draft (parent references v1)
* “current” is derived by querying latest version not superseded

### Retention

* drafts: keep for a while (90–365 days)
* finals: keep long-term
* enterprise export: outcomes may be pulled into external systems, but the ledger is canonical

### Indexes

* `(tenant_id, workflow_id, created_at)`
* `(tenant_id, outcome_type, created_at)`
* `(tenant_id, thread_id, outcome_type, version)` unique-ish

---

## 5) Memory Ledger (typed, explicit, TTL capable)

**Purpose:** store durable user-specific preferences and constraints with provenance.

### What it stores

* tone preferences
* do/don’t lists
* goals
* guardrails
* “facts learned from user” vs “system inferred”

### Minimal schema (aligned with your `ai_memory`)

```json
{
  "memory_id": 123,
  "memory_type": "tone|do_dont|preference|goal|bio|instruction|guardrail|other",
  "content": "string",
  "source": "user|system|inferred",
  "confidence": 0.7,
  "is_active": true,
  "expires_at": null
}
```

### Invariants (very important)

* memory must be **typed** (no dumping random text blobs)
* memory must be **scoped**:

  * per student
  * optionally per capability (email tone vs immigration answers)
* memory changes must be evented:

  * `MEMORY_CREATED`, `MEMORY_DEACTIVATED`, `MEMORY_EXPIRED`

### TTL behavior

* TTL is not deletion. TTL means `is_active=false` via expiry event.
* Always preserve history for audit.

### Indexes

* `(tenant_id, student_id, memory_type, is_active)`
* `(tenant_id, expires_at)`

---

## 6) Entity Ledger (normalized entities: professors/programs/institutions)

**Purpose:** canonical identity and normalization for external-world objects.

This ledger prevents a thousand duplicate professors and inconsistent program names.

### What it stores

* normalized identity, stable IDs
* provenance (source: canspider/manual)
* contact info, URLs
* normalized name fields
* change history via events

### What it must NOT store

* “summaries” as the entity truth (those are outcomes)
* transient scraped junk without provenance

### Minimal schema

```json
{
  "entity_id": 910,
  "entity_type": "professor|program|institution",
  "canonical": {
    "full_name": "José Denis-Robichaud",
    "email": "...",
    "url": "...",
    "institution_id": 33
  },
  "aliases": ["J. Denis-Robichaud", "..."],
  "source": "canspider|manual",
  "is_active": true
}
```

### Invariants

* entity IDs are stable and referenced everywhere
* entity updates are evented
* dedupe strategy exists:

  * professors: hash(email + institution) usually beats name matching
  * programs: canonical key = (institution, degree level, normalized program name)

### Indexes

* `(tenant_id, entity_type, canonical_key_hash)` unique
* `(tenant_id, email)` for professor

---

# 4.4 Ledger relationships: the “truth graph”

Think of this as your internal causality DAG:

* **Event Ledger** references everything (job_id, outcome_id, entity_id)
* **Job Ledger** produces outcomes and references operator payload hashes
* **Outcome Ledger** references parent outcomes and the job that produced it
* **Document Ledger** references jobs that processed/exported it
* **Memory Ledger** references events that created/deactivated it
* **Entity Ledger** references events for mutations + outcomes for summaries

This is how you can answer:

* “why did the system write this email?”
* “what model produced it?”
* “what data did it use?”
* “what changed between draft v1 and v2?”

---

# 4.5 Versioning strategy (how you evolve safely)

### Schema versioning rules

* `schema_version` is per ledger type.
* additive changes are okay (new optional fields)
* breaking changes require bump and support reading old versions

### Producer versioning rules

* every operator/agent has its own `version`
* outcomes store producer version, so you can later compare performance across versions

---

# 4.6 The “append-first” patterns you’ll actually use

### Pattern 1: Pure append (events, outcomes)

* never update; supersede with new record

### Pattern 2: Append + active pointer (memory, entities)

* record new version and mark old inactive
* write an event describing the change

### Pattern 3: Append attempts (jobs)

* job attempts are appended; the “current state” is derived

---

# 4.7 What gets streamed vs stored

* Stream: events + progress + assistant deltas (projection)
* Store: events, jobs, outcomes, documents, memory, entities (truth)

The assistant message stream can be regenerated; ledgers cannot.

---

# 4.8 Practical v1 action plan for ledgers (no fluff)

If you want this to work fast:

1. Make **Event Ledger** real first (append-only, queryable).
2. Make **Job Ledger** your single truth for execution attempts (idempotency keys mandatory).
3. Add **Outcome Ledger** next (so drafts aren’t shoved into random tables).
4. Keep **Document Ledger** as you already planned (hashes + lifecycle).
5. Add **Memory Ledger** typed and TTL capable.
6. Keep **Entity Ledger** minimal but normalized.

Everything else can be derived.
