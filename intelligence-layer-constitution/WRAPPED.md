# WRAPPED: CanApply Intelligence Layer (what this repo is becoming)

This repo is a **constitution + RFCs + JSON schemas + manifests + plan templates + runnable demos** for the CanApply “Intelligence Layer” (IL).

The IL is not “a chat app”. It’s a **workflow runtime** that turns requests into **auditable outcomes** under **explicit policy**, with side effects only through **idempotent operators**.

If you build what’s described here, you end up with a production service shaped roughly like:

```
Client (frontend/backend) -> Kernel API (FastAPI)
  -> Ledgers (Postgres) + Cache (Redis) + Blobs (S3/MinIO) + Vectors (Qdrant)
  -> Workers lease steps from DB and run Operators
  -> SSE stream + webhooks are projections of ledger events
  -> Platform DB writes happen only via “apply patch” gates
```

---

## 1) The one-sentence promise

“Reduce user cognitive load while increasing correctness and accountability” by enforcing:

- every request becomes a typed **Intent**
- every intent runs as **Intent → Plan → Actions(Operators) → Outcomes → Events**
- every risky effect is **policy-gated**, **recorded**, and **resumable**
- “what happened?” is answerable from ledgers **without rerunning an LLM**

Canonical law: `CONSTITUTION.md`.

---

## 2) Non‑negotiables (the laws you keep tripping over if you ignore them)

These are repeated across the docs for a reason; they are the “shape” of the system:

1) **Workflow-first, interface-second**: UI/chat are adapters; truth is in ledgers.  
2) **Ledgers + approved downstream writes are truth**: chat transcripts are projections.  
3) **Agents are stateless**: they don’t “remember”; durable state is written as typed records.  
4) **Side effects only through Operators** (with declared `effects[]`) and **idempotency keys**.  
5) **Policy-first** at intake, plan, action, outcome, apply; every decision is recorded.  
6) **Plans are inspectable, interruptible, resumable**: human gates are first-class.  
7) **Tenancy everywhere**: `tenant_id` is required in every record and query.  
8) **Data classification + egress rules** are enforceable (e.g. `Regulated` blocks `external_send`).  
9) **Replay semantics are explicit**: reproduce vs replay vs regenerate (no silent “regen”).  
10) **Capability admission is a contract**: schemas + plan templates + policy tags + coverage.

Start here if you only read one file: `CONSTITUTION.md`.

---

## 3) Core primitives (what the IL actually manipulates)

The system is designed around a small set of typed records (v1 schema versions live under `schemas/`):

- **Intent**: canonical structured request (`schemas/common/intent_base.v1.json` + per-intent schemas).  
- **Plan**: executable step program; steps declare effects/tags/cache/idempotency/gates (`schemas/plans/*`).  
- **Action**: an operator invocation (envelope: `schemas/operators/operator_call_base.v1.json`).  
- **Outcome**: a versioned artifact returned/stored/replayed (`schemas/outcomes/*`).  
- **Event**: append-only timeline used to reconstruct and to stream (`schemas/sse/*` are demo stream shapes).  
- **AuthContext**: scoped permissions passed into operators (see common defs).  
- **PolicyDecision**: deterministic governance record (specified in docs; storage designed in `DATA-STRUCTURE.md`).  
- **ContextBundle**: deterministic, policy-filtered context package built by the Kernel (design doc concept).

The IL treats “LLM calls” as **operators** that produce typed outcomes + usage/cost metadata; the runtime stays deterministic-ish even if models aren’t.

---

## 4) The “Kernel” (what it is, and what it is *not*)

The Kernel is the minimal, stable core (“OS kernel” vibe). It owns control-plane duties:

- intake adapter → normalize and validate intent
- build `AuthContext` + `ContextBundle`
- choose a capability/plan template
- execute a step state machine (pause/resume/retry)
- enforce idempotency (request-level + operator-level)
- run policy at every stage and record decisions
- write ledgers (events/jobs/outcomes/gates/etc.)
- stream projections (SSE / assistant deltas / progress)

It explicitly does **not** contain domain prompts or business heuristics.

Implementation-facing contract: `KERNEL-RFC-V1.md`. Deep narrative: `3-the-kernels.md`.

---

## 5) Execution model (why “it can resume” and “it won’t double-send”)

Execution is defined as:

- **Plan templates** are deterministic step programs (`plan-templates/*.plan.v1.json`).
- The executor resolves **bindings** in payloads and idempotency templates:
  - `{ "from": "context.platform.funding_request.id" }`
  - `{ "const": 123 }`
  - `{ "template": "email_draft:{tenant_id}:{thread_id}:{…}" }`
  - plus `computed.*` values (hashes) for compact, non-PII idempotency keys.
- Steps have a state machine (pending/ready/running/succeeded/failed/waiting_approval/…).
- The runtime supports:
  - **reproduce**: read stored outcomes (default UI behavior)
  - **replay**: rerun steps with the same refs (effects remain idempotent)
  - **regenerate**: create a new version lineage (explicit)

Specs: `EXECUTION-RFC-V1.md` and deep version `7-execution-semantics.md`.

---

## 6) Policy model (governance, not “moderation”)

Policy is a deterministic decision function evaluated at multiple stages:

- intake policy (is this allowed?)
- plan policy (are these steps legal? add gates?)
- action policy (can this operator run with these effects/tags/data classes?)
- outcome policy (redact/transform/require approval?)
- apply policy (writes to platform DB, external sends)

Policy gates on **declared effects** (e.g. `external_send`, `db_write_platform`) and data classes (`Public/Internal/Confidential/Regulated`).

Deep spec: `6-policy-engine.md`. Constitutional requirements: `CONSTITUTION.md`.

---

## 7) Ledgers (how the system stays accountable)

The “sources of truth” are ledgers + approved downstream writes (platform DB changes after an apply gate).

Ledgers (v1 minimum, per docs/RFCs):

- events (append-only timeline)
- jobs/actions (attempts, idempotency, usage/cost)
- outcomes (versioned artifacts; drafts vs finals)
- documents (upload + transforms; hashes and provenance)
- memory and entities (append + active pointer patterns)
- policy decisions
- gates and gate decisions (action_required / approvals / required inputs)

Implementation-ready storage design: `DATA-STRUCTURE.md`. Ledger shapes: `LEDGERS-RFC-V1.md` and `4-ledgers-sources-of-truth.md`.

---

## 8) What “exists” in this repo as contracts (the concrete capability surface)

This repo isn’t abstract: it already pins a v1 capability set as versioned files.

### 8.1 Supported intents (14)

Declared in `manifests/intent-registry.v1.json` and versioned intent schemas under `schemas/intents/`:

- Core runtime: `Thread.Init`, `Workflow.Gate.Resolve`
- Funding outreach: `Funding.Outreach.Professor.Summarize`, `Funding.Outreach.Alignment.Score`, `Funding.Outreach.Email.Generate`, `Funding.Outreach.Email.Review`, `Funding.Outreach.Email.Optimize`, `Funding.Paper.Metadata.Extract`, `Funding.Request.Fields.Update`
- Documents: `Documents.Upload`, `Documents.Process`, `Documents.Review`, `Documents.Optimize`, `Documents.Compose.SOP`

### 8.2 Capabilities (3)

Declared in `manifests/capabilities/*.json`:

- `CoreRuntime@1.0.0` (thread init + gate resolve)
- `FundingOutreach@1.0.0` (professor summarization, alignment, email draft/review/optimize, paper metadata, request field patch proposals)
- `Documents@1.0.0` (upload/process/review/optimize/compose SOP; export; platform patch proposals)

### 8.3 Operator plugins (19)

Declared in `manifests/plugins/operators/*.json` with input/output schemas in `schemas/operators/`:

- Thread and gating: `Thread.CreateOrLoad`, `Workflow.Gate.Resolve`
- Platform read: `Platform.Context.Load`
- Funding outreach core: `Professor.Profile.Retrieve`, `Professor.Summarize`, `Professor.Alignment.Score`
- Email: `Email.GenerateDraft`, `Email.ReviewDraft`, `Email.OptimizeDraft`, `Email.ApplyToPlatform.Propose`
- Funding data: `Paper.Metadata.Extract`, `FundingRequest.Fields.Update.Propose`
- Documents: `Documents.Upload`, `Documents.Process`, `Documents.Review`, `Documents.Optimize`, `Documents.Compose.SOP`, `Documents.Export`, `Documents.ApplyToPlatform.Propose`

### 8.4 Plan templates (14)

Per-intent deterministic programs under `plan-templates/` (validated by `schemas/plans/plan_template.v1.json`).

Notable patterns you see repeatedly:

- `Platform.Context.Load` first (builds a stable context snapshot and hashes)
- a policy check that can emit `action_required`:
  - onboarding prerequisites (`Onboarding.Ensure…`) or
  - missing required source data (`EnsureEmailPresent`, `EnsurePaperSourcePresent`)
- draft/review/optimize operators producing typed outcomes
- `*.ApplyToPlatform.Propose` produces a `platform_patch_proposal` outcome
- a final `human_gate` step `apply_platform_patch` pauses until approval

---

## 9) Data layer + infra (what v1 is designed for)

The storage and runtime design consistently points to:

- Postgres as the primary system of record for ledgers + workflow runtime control tables
- Redis for hot cache + counters + optional “kick worker now”
- S3/MinIO for large artifacts (docs, exports, large outcomes)
- Qdrant for embeddings/retrieval (optional in v1)
- FastAPI for the Kernel API surface + SSE
- DB-leased workers (`FOR UPDATE SKIP LOCKED`) for step execution

Authoritative: `TECHNICAL-SPECS.md`, `DATA-STRUCTURE.md`, `API-RUNTIME-DESIGN.md`.

---

## 10) Credits/budget enforcement (the “economic policy” layer)

AI Credits are treated as a **normative** part of the system:

- record provider usage + cost per AI operation
- use reservation + settlement to avoid overspend mid-stream
- idempotent debits tied to request/workflow IDs
- enforce hard-stop or explicit overage policy

Spec: `8-ai-credits-and-budget-enforcement.md`.

---

## 11) How you extend this system (the “contracts-first” workflow)

The intended feature workflow is:

1) Add/extend an intent schema in `schemas/intents/` (versioned, strict).  
2) Register it in `manifests/intent-registry.v1.json`.  
3) Add a plan template in `plan-templates/` (explicit effects/tags/cache/idempotency/gates).  
4) Ensure every referenced operator has:
   - a plugin manifest under `manifests/plugins/operators/`
   - input/output schemas in `schemas/operators/`
   - an outcome schema in `schemas/outcomes/` (if it produces an outcome)
5) Implement operator runtime code (outside this repo’s core docs), then wire it into the production Kernel.
6) Add policy rules for effects/data classes and gate behaviors.

Practical walkthrough: `FEATURE-DEVELOPMENT-WALKTHROUGH.md`.

---

## 12) Reading order (so you don’t drown)

If you want the “wide picture” fast:

1) `CONSTITUTION.md` (the law)  
2) `FEATURE-DEVELOPMENT-WALKTHROUGH.md` (how the repo artifacts connect)  
3) `API-RUNTIME-DESIGN.md` + `DATA-STRUCTURE.md` (what the production service will look like)  

Deep references (preserved narratives): `0-introduction.md` through `9-capabilities.md`.

---

## Notes about the working tree

- The “source of truth” content in this repo is the tracked docs + schemas + manifests + plan templates listed above.
- There is also a `llm-client/` directory present in this workspace that appears to be its own separate repo (with its own `.git/` and `.venv/`). The IL demos can optionally use it for a real provider, but it is not part of this repo’s tracked contracts (`.gitignore` ignores `llm-client`).
- Some docs still reference historical filenames like `PREREQUISITE.txt` / `PLAN.md`; in the current tree, those don’t exist as normal tracked files (their content appears to have moved into `HISTORICAL-NOTES.txt` and other design docs).
