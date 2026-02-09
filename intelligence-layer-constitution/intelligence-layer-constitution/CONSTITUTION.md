# CanApply Intelligence Layer Constitution (v0.2)

## 0) Purpose

The CanApply Intelligence Layer (“the Layer”) is a **workflow-centric decision and execution runtime** that converts user and system requests into **auditable outcomes** under **explicit policies**.

Chat is only one interface. The Layer is an integration-ready engine intended to scale across products and integrations without changing its core guarantees.

---

## 1) Non‑Negotiable Laws

### Law 1 — Workflow-first, interface-second

1. Every request MUST normalize into a typed **Intent**.
2. Every Intent MUST execute as a workflow: **Intent → Plan → Actions → Outcomes → Events**.
3. Interfaces (chat, API, webhook, button) are adapters; they MUST NOT be sources of truth.

### Law 2 — The sources of truth are ledgers and approved downstream state

1. The system’s truth MUST be reconstructible from:
   - **Ledgers** (events/jobs/outcomes/documents/memory/entities), and
   - **Approved downstream domain writes** (materialized state written by a gated apply/finalization step).
2. The assistant message stream and chat history are **projections** and MUST NOT be required to audit what happened.
3. “What happened?” MUST be answerable **without rerunning the LLM** by reading ledger records.

### Law 3 — Agents are stateless; state lives in ledgers

1. Agents MUST be pure decision functions: read-only inputs, structured outputs.
2. Any persistent change (including “memory”) MUST be produced via an **Action** that invokes an **Operator**, and MUST pass policy.
3. Agents MUST NOT depend on raw chat history beyond the **ContextBundle** provided by the Kernel.

### Law 4 — Effects are operator-only; credentials are operator-only

1. Any side effect MUST happen only via Operators.
2. Operators MUST declare normalized **Effect** values (e.g., `db_write`, `external_send`, `s3_final_write`), and policy MUST gate on effects.
3. Credential-bearing SDKs/tokens MUST NOT exist in agent runtime. Credentials MAY exist only in operators running in an isolated execution environment.
4. Every operator invocation MUST be idempotent under an `idempotency_key`.

### Law 5 — Policy-first execution with recorded decisions

1. Policy MUST evaluate:
   - intake (allowed intent types / quotas),
   - plan (step legality and required gates),
   - each action (before operator invocation),
   - each outcome (before returning/streaming/applying),
   - apply/finalization (writes to final state and external sends).
2. Every policy evaluation MUST produce a first-class **PolicyDecision** record and a corresponding event.
3. A policy engine decision MUST be deterministic for identical inputs.

### Law 6 — Plans are inspectable, interruptible, resumable

1. A Plan MUST be explicit and machine-checkable (step list), even if it is a single step.
2. Every step MUST declare:
   - effects,
   - policy tags,
   - risk level,
   - cache policy,
   - idempotency strategy,
   - whether a human gate is required.
3. The executor MUST support pause/resume, partial completion, and “dry-run” (no effects).

### Law 7 — Tenancy and scoping are mandatory

1. All records and queries MUST be tenant-scoped (`tenant_id` required everywhere).
2. Every operator MUST receive an **AuthContext** with scoped permissions; operators MUST enforce it (defense-in-depth).
3. Identity semantics MUST be explicit:
   - `workflow_id` is the canonical execution identity,
   - `thread_id` is a UX/conversation grouping,
   - many workflows MAY exist under one thread.

### Law 8 — Data classification and egress rules are enforceable

The Layer MUST classify data used in actions/outcomes into one or more data classes:

- `DataClass.Public`: public web pages, public publications.
- `DataClass.Internal`: non-sensitive operational data; safe for internal processing.
- `DataClass.Confidential`: resumes, transcripts, addresses, private correspondence.
- `DataClass.Regulated`: passport/immigration IDs and similarly sensitive regulated identifiers.

Egress rules MUST be enforced by policy using declared effects (especially `external_send`):

1. If `DataClass.Regulated` is present, `external_send` MUST be denied.
2. If `DataClass.Confidential` is present, `external_send` MUST require approval and MUST apply redaction rules as configured.
3. If only `DataClass.Internal`/`Public` is present, `external_send` MAY be allowed subject to policy, trust, and rate limits.

Jurisdiction- and tenant-specific policy details (e.g., PIPEDA/GDPR) are policy configuration, not constitutional law.

### Law 9 — Replay semantics are explicit

The system MUST distinguish:

1. **Reproduce**: return already-stored outcomes (no recomputation, no operator effects).
2. **Replay**: re-run a workflow using the same stored inputs/context references; nondeterministic steps MAY produce different draft outcomes, but effects MUST remain idempotent.
3. **Regenerate**: explicitly request a new draft/version; MUST mint new version keys and MUST NOT overwrite prior outcomes.

UI defaults:

- Default “view” and “refresh” behavior SHOULD be **reproduce**.
- Default “retry” behavior SHOULD be “resume failed steps” using the same idempotency keys.
- “Regenerate” MUST be an explicit user/system intent and MUST create a new version lineage.

### Law 10 — Capability admission is a hard contract

A new capability MUST NOT ship unless it provides, at minimum:

1. Intent types and JSON schemas (versioned).
2. Outcome types and JSON schemas (versioned).
3. A plan template and/or planner hook that produces machine-checkable steps with effects/tags/gates.
4. Policy tags and effect declarations for every step/operator.
5. Tests or harness coverage for:
   - idempotency (operator keys + no duplicate effects),
   - permissions (AuthContext scopes),
   - redaction/egress enforcement (DataClass + `external_send`).

---

## 2) Required Primitives (Constitutional Definitions)

These primitives MUST exist as versioned, typed records.

### Intent

Structured request: `intent_id`, `intent_type`, `schema_version`, `actor`, `source`, optional `thread_id`, optional `scope`, `inputs`, `constraints`, `context_refs`.

### Plan

Executable step program: `plan_id`, `plan_version`, `intent_id`, `steps[]`, `explanation`, `stop_conditions`.

### Action

Operator invocation: `action_id`, `operator_name`, `operator_version`, `payload`, `idempotency_key`, `auth_context`, `status`, `result|error`, trace fields.

### Outcome

Versioned artifact: `outcome_id`, `outcome_type`, `schema_version`, `status (draft|final|partial|failed)`, `content`, provenance (intent/plan/actions), version lineage.

### Event

Append-only record of transitions/decisions: `event_id`, `event_type`, `timestamp`, trace fields, compact typed payload.

### AuthContext

Mandatory authorization scope passed to executors/operators: `tenant_id`, principal identity, roles, and explicit scopes.

### PolicyDecision

First-class policy output: allow/deny/require-approval/transform/redact with reason codes, requirements, and redaction/transform directives.

### ContextBundle

Deterministic, policy-filtered context package with:

- stable hash(es) for caching/audit,
- a redacted view for logging/exports,
- references (IDs/hashes) instead of raw huge blobs.

### Effect

Normalized effect vocabulary used for policy gating (e.g., `db_write`, `external_send`, `s3_final_write`, `webhook_emit`).

---

## 3) Ledger Doctrine (What must always be true)

1. Ledgers are persistent, versioned stores of record for events, jobs, outcomes, documents, memory, and entities.
2. Ledgers MUST be tenant-scoped, traceable (correlation/workflow/intent/plan/step), and support redacted views.
3. Append-first mutation is mandatory:
   - events: pure append,
   - outcomes: append new versions; never overwrite,
   - jobs: append attempts or immutable final result per idempotency key,
   - memory/entities: append + active pointer patterns.
4. Reproducibility requirement: outcomes MUST be retrievable without rerunning models; recomputation is an explicit replay/regenerate choice.

---

## 4) Forbidden Patterns (Hard “No”)

1. Business logic in the Kernel or adapters (belongs in capabilities + policy + operators).
2. Side effects outside operators (including “helpful” SDK calls inside agents).
3. Credential-bearing code in agent runtime.
4. Storing truth only in chat messages or assistant text.
5. Untyped “blob” outcomes without schema/version/provenance.
6. Operators without idempotency keys or without declared effects.
7. Silent “regenerate” behavior that changes artifacts without minting new versions.

---

## 5) References and Normative Supplements

### Primary references (deep specs preserved as-is)

- `0-introduction.md`
- `1-non-negotiable-principles.md`
- `2-canonical-primitives.md`
- `3-the-kernels.md`
- `4-ledgers-sources-of-truth.md`
- `5-plugin-contracts.md`
- `6-policy-engine.md`
- `7-execution-semantics.md`
- `9-capabilities.md`
- `PREREQUISITE.txt` (historical notes / implementation sketches)

### RFCs and playbook (implementation-specific; can evolve without changing this constitution)

- `KERNEL-RFC-V1.md`
- `LEDGERS-RFC-V1.md`
- `EXECUTION-RFC-V1.md`
- `IMPLEMENTATION-PLAYBOOK.md`

### Normative supplements (normative within their topic, but not part of the constitutional core)

- `8-ai-credits-and-budget-enforcement.md`

