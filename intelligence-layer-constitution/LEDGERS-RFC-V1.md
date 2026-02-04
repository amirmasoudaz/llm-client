# Ledgers RFC v1 (Sources of Truth and Record Shapes)

This RFC captures **implementation-facing** ledger expectations and example shapes. It is not constitutional law and may evolve without changing `CONSTITUTION.md`.

## 1) Ledger types (v1 minimum)

- Event Ledger (append-only)
- Job Ledger (action attempts, idempotency, costs)
- Outcome Ledger (versioned artifacts: draft/final, lineage)
- Document Ledger (uploads + transformations)
- Memory Ledger (typed “learned” facts/preferences; append + active pointer)
- Entity Ledger (normalized entities; append + active pointer)

## 2) Universal record header (recommended)

Every ledger record SHOULD carry a common header for traceability and hashing:

```json
{
  "schema_version": "1.0",
  "tenant_id": 1,
  "record_id": "uuid",
  "created_at": "2026-01-28T19:03:00Z",
  "producer": { "kind": "kernel|operator|agent|adapter", "name": "Email.GenerateDraft", "version": "1.0" },
  "trace": {
    "correlation_id": "corr-uuid",
    "workflow_id": "wf-uuid",
    "thread_id": 4512,
    "intent_id": "intent-uuid",
    "plan_id": "plan-uuid",
    "step_id": "s3",
    "job_id": "act-uuid"
  },
  "hash": { "payload_hash": "sha256(...)" }
}
```

## 3) Event ledger (append-only)

Purpose: authoritative timeline of “what happened”.

Minimum event fields:

```json
{
  "event_type": "INTENT_RECEIVED|PLAN_CREATED|ACTION_STARTED|ACTION_SUCCEEDED|ACTION_FAILED|OUTCOME_CREATED|GATE_REQUESTED|USER_APPROVED|WORKFLOW_COMPLETED",
  "severity": "info|warn|error",
  "payload": { "typed": "per event_type" }
}
```

Requirements:

- total ordering per workflow (`sequence_no` or monotonic timestamp + tiebreaker)
- compact payloads: IDs + reason codes, not large blobs

## 4) Job ledger (actions, retries, costs)

Purpose: record each operator attempt and enforce idempotency.

Minimum job fields:

```json
{
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

Idempotency requirements:

- `(tenant_id, operator_name, idempotency_key)` MUST uniquely identify the effect.
- repeated calls with the same key MUST NOT repeat side effects; return cached result/receipt.

## 5) Outcome ledger (versioned artifacts)

Purpose: stable, retrievable artifacts without recomputation.

Minimum outcome fields:

```json
{
  "outcome_type": "Email.Draft|Email.Review|Professor.Summary|Alignment.Score|...",
  "schema_version": "1.0",
  "status": "draft|final|superseded|retracted|failed",
  "version": 1,
  "parent_outcome_id": null,
  "content": {},
  "confidence": 0.72
}
```

Rules:

- drafts MUST NOT overwrite finals
- regenerate creates a new version lineage (new `version`/new `outcome_id`)

## 6) Document ledger (uploads + transformations)

Purpose: document lifecycle and reproducible transformations.

Rules:

- raw uploads are immutable (new upload = new record)
- transformations record processor version + hashes
- “final” writes require policy-gated apply

## 7) Memory and entity ledgers (append + active pointer)

Rules:

- updates are new versions plus pointer change (or “active” flag)
- every change MUST have an explaining event and provenance

## 8) References

See `4-ledgers-sources-of-truth.md` for the deep spec and patterns.

