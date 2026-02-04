
# 2. Canonical Primitives

## 2.1 Intent

### Definition

An **Intent** is the canonical, structured representation of “what should happen” in the Layer.

It is **interface-agnostic**. Chat, API, webhook, button click all become the same thing: an Intent object.

### Invariants (must always be true)

* **I1: Intent is the source of truth for user/system desire.**
* **I2: Intent must be storable and replayable.**
* **I3: Intent must be typed**: `intent_type` + schema version.
* **I4: Intent must be attributable**: who requested it and under which tenant/role.
* **I5: Intent must be safe to log** (or have a redacted view). You don’t “lose” intent because it contains PII; you store it safely.

### Required fields (minimum viable)

* `intent_id` (uuid)
* `intent_type` (enum string, namespaced like `Funding.Outreach.Email.Generate`)
* `schema_version` (e.g., `1.0`)
* `actor`:

  * `tenant_id` (even if single tenant today)
  * `student_id`
  * `role` (`user|admin|system`)
* `source` (`chat|api|webhook|system`)
* `thread_id` (optional but recommended)
* `scope` (optional, strongly recommended):

  * `scope_type` (`funding_request|program_application|general|support_ticket`)
  * `scope_id` (nullable)
* `inputs` (typed; must validate for the intent_type)
* `constraints` (tone, length, language, policy constraints)
* `context_refs` (IDs/URLs, never raw huge blobs)
* `created_at`

### Rules

* **R1: Never put raw documents in intent.** Put document IDs/hashes.
* **R2: Intent inputs are “what the user knows/asks”, not computed things.**
* **R3: If the user request is ambiguous, the intent can be “Clarify.*”** (an intent that produces a question/outcome), rather than forcing bad guesses.
* **R4: Intents must be idempotent-able** (i.e., you can generate an idempotency key from intent_id + stable inputs).

### Exceptions

* Tiny one-shot Q&A intents may not need scope_id.
* “Freeform brainstorm” intents can have looser schema, but they must be explicitly typed as such (e.g., `General.Brainstorm`) and marked **no-side-effects**.

### Failure modes

* Using “chat history” as the intent.
* Making intent schemas too big and embedding derived info.
* Mixing intent with plan (intent should not specify operators).

### v1 example (Funding email generation)

```json
{
  "intent_id": "uuid",
  "intent_type": "Funding.Outreach.Email.Generate",
  "schema_version": "1.0",
  "actor": {"tenant_id": 1, "student_id": 88, "role": "user"},
  "source": "chat",
  "thread_id": 1234,
  "scope": {"scope_type": "funding_request", "scope_id": 556},
  "inputs": {"goal": "initial_outreach", "custom_note": "mention my ViT localization work"},
  "constraints": {"tone": "professional-warm", "length": "short"},
  "context_refs": {"professor_id": 910, "document_ids": [2001]}
}
```

---

## 2.2 Plan

### Definition

A **Plan** is a structured, ordered program that the Layer intends to run to satisfy the Intent.

Plans are produced by the planner and **must be executable** by the executor.

### Invariants

* **P1: Plan must be explicit and step-based.**
* **P2: Plan must declare side-effect boundaries.**
* **P3: Plan must be resumable.**
* **P4: Plan must have provenance** (what created it and why).
* **P5: Plan must be inspectable** (human-readable explanation) and **machine-checkable** (typed steps).

### Required fields

* `plan_id` (uuid)
* `plan_version` (string, planner version)
* `intent_id`
* `created_at`
* `steps[]` each step:

  * `step_id`
  * `step_type` (`agent|operator|policy_check|human_gate`)
  * `name`
  * `requires` (keys from context or previous step outputs)
  * `produces` (output keys)
  * `policy_tags` (PII, external communication, etc.)
  * `risk_level` (`low|medium|high`)
  * `idempotency_key` derivation rule or placeholder
* `explanation` (short)
* `stop_conditions` (when to ask user, when to abort)

### Rules

* **R1: A plan must not call tools directly**. Only operators or agent computations wrapped as steps.
* **R2: Every plan step must be name-stable** (so analytics and debugging work).
* **R3: A plan must declare “human approval” steps** explicitly for high-risk actions.
* **R4: Plans should be minimal in v1.** Prefer 3–6 steps over 17.

### Exceptions

* Single-step plans are fine for trivial intents.
* For “offline compute” tasks (pure processing), you can omit human gate.

### Failure modes

* Plans as prose.
* Steps that don’t declare their inputs/outputs.
* Planner doing execution (“just do it”) instead of returning a plan.

### v1 example (Email generate plan)

Steps:

1. Load context (operator)
2. Draft email (operator calling LLM)
3. Review email (operator calling LLM)
4. Store outcome (operator)
5. Present draft + “Apply” gate (human_gate)

---

## 2.3 Action

### Definition

An **Action** is a concrete invocation of an **Operator** with a typed payload, executed by the executor.

Actions are the only way the Layer changes anything or touches external systems.

### Invariants

* **A1: Every action must be idempotent.**
* **A2: Every action must be authorized.**
* **A3: Every action must be observable** (metrics + status + errors).
* **A4: Every action must be policy-gated** (even if auto-allowed).
* **A5: Every action must produce a typed result** (success or structured failure).

### Required fields

* `action_id` (uuid)
* `operator_name` (e.g., `Email.GenerateDraft`)
* `operator_version`
* `intent_id`, `plan_id`, `step_id`
* `idempotency_key`
* `payload` (typed JSON)
* `auth_context` (scoped permissions)
* `policy_tags`
* `status` (`queued|running|succeeded|failed|cancelled`)
* `started_at`, `finished_at`
* `result` (typed)
* `error` (typed + retryability)

### Rules

* **R1: Operator outputs are not freeform strings.** They are structured objects.
* **R2: Errors must be classified**: retryable vs non-retryable.
* **R3: Actions must be replayable** (given same payload and key, same effect).

### Exceptions

* Some operators call nondeterministic services (LLMs, web). That’s fine, but they must disclose it via `trace_type`, and store inputs/outputs for audit.

### Failure modes

* No idempotency key (duplicate sends, duplicate docs).
* Operator does too much (becomes mini-orchestrator).
* “Soft failures” returned as plain text.

### v1 examples

* `Documents.Process`
* `Professor.ResolveAndSummarize`
* `Email.ReviewDraft`
* `Workflow.ApplyDraftToRequest`

---

## 2.4 Outcome

### Definition

An **Outcome** is the stable, structured result of completing (or partially completing) an Intent.

Outcomes are what consumers care about: UI, enterprise adapters, analytics.

### Invariants

* **O1: Outcomes are versioned.**
* **O2: Outcomes are attributable** (intent/plan/actions).
* **O3: Outcomes must be retrievable without rerunning the LLM.**
* **O4: Outcomes must be typed and validated.**
* **O5: Outcomes must separate draft vs final.**

### Required fields

* `outcome_id` (uuid or DB id)
* `outcome_type` (e.g., `Draft.Email`, `Review.Email`, `Summary.Professor`)
* `schema_version`
* `intent_id`, `plan_id`, `job_ids[]`
* `status` (`draft|final|partial|failed`)
* `content` (typed)
* `rationale` (short, optional)
* `confidence` (0–1)
* `citations` (optional, for web-derived info)
* `created_at`

### Rules

* **R1: Draft outcomes must never overwrite finals.** They create new versions.
* **R2: Applying a change creates a new “final” outcome event** and updates domain tables.
* **R3: Outcomes should be composed.** Email outcome references professor summary outcome, etc.

### Exceptions

* For internal system operations, outcomes can be “internal” but still typed.

### Failure modes

* Storing outcomes only as “assistant message”.
* Overwriting email drafts in place with no version history.
* No separation between “generated” and “approved”.

### v1 examples

* Email draft outcome: `{subject, body, suggested_attachments, personalization_hooks}`
* Review outcome: `{score, issues[], improvements[]}`
* Professor summary outcome: `{bio, research_themes[], recent_papers[]}`

---

## 2.5 Event

### Definition

An **Event** is an immutable record of “something happened” inside the Layer.

Events are the audit trail and the debug timeline.

### Invariants

* **E1: Events are append-only.**
* **E2: Events are time-ordered per thread/workflow.**
* **E3: Every action produces events.**
* **E4: Events have stable types.**
* **E5: Events are safe to export** (with redaction if needed).

### Required fields

* `event_id` (uuid)
* `event_type` (enum)
* `intent_id` (optional for some system events)
* `plan_id` (optional)
* `thread_id`
* `actor` (system/user/operator)
* `timestamp`
* `payload` (typed, small)
* `correlation_id` (trace id)

**Core event types**

* `INTENT_RECEIVED`
* `PLAN_CREATED`
* `POLICY_EVALUATED`
* `ACTION_STARTED`
* `ACTION_SUCCEEDED`
* `ACTION_FAILED`
* `OUTCOME_CREATED`
* `USER_APPROVED`
* `USER_REJECTED`
* `WORKFLOW_COMPLETED`

### Rules

* **R1: Event payloads must be compact.** Reference IDs, not huge data.
* **R2: Every event type must have a schema.**
* **R3: “Progress” is derived from events**, not a separate truth source.

### Exceptions

* None that matter. If you skip events, you lose enterprise readiness.

### Failure modes

* Using logs as events (logs are not structured or queryable).
* Event type explosion (500 event types). Keep them stable and coarse.
* No correlation ID, making debugging impossible.

---

# Cross-Cutting Rules for All Primitives

## Versioning

* Every primitive has `schema_version`.
* Every producer has `producer_version` (agent/operator/planner versions).
* Backward compatibility is enforced by:

  * schema evolution (add fields, don’t rename)
  * migration adapters when needed

## Idempotency

* Intents and Actions must support deterministic idempotency keys.
* “Retry” reuses intent_id and the same idempotency key, unless the user explicitly requests a new run/version.

## Redaction and PII

* Each primitive should support a redacted view:

  * event payload redaction
  * outcome content redaction for exports
  * policy tags for sensitivity

---

# The practical v1 minimum

If you implement Canonical Primitives correctly in v1, you can bolt on anything later.

**v1 must include:**

* Intent objects for the few Funding.Outreach flows
* Plans with named steps (even if simple)
* Actions as operator calls with idempotency
* Outcomes stored and versioned (draft vs final)
* Events append-only with stable types

Everything else is gravy.

---

# Example
* Below is a full **v1 reference workflow** for one capability:

## Capability

**Funding.Outreach.Email.Generate**
“Generate a first outreach email to a professor, based on user profile + professor context + (optional) user resume and a cited hook.”

This is the “golden path” blueprint engineers can clone for other workflows.

---

# 0) The Contract Summary

### Inputs (from the user/UI)

* professor (id or url)
* funding_request_id (optional but common)
* optional resume doc id(s)
* optional “custom note” or constraints

### Outputs (what the layer produces)

* professor summary outcome (if missing)
* alignment outcome (optional but recommended)
* email draft outcome (draft v1)
* email review outcome (score + issues)
* “apply gate” outcome (UI action prompt)

### Side effects

* **No external send in v1** (unless later gated)
* Optional: write draft to `funding_emails` or `funding_requests` as draft state, but only after user clicks “Apply” (recommended)

---

# 1) Intent Schema (canonical)

### Intent Type

`Funding.Outreach.Email.Generate`

### Minimal Intent JSON

```json
{
  "intent_id": "uuid-123",
  "intent_type": "Funding.Outreach.Email.Generate",
  "schema_version": "1.0",
  "actor": { "tenant_id": 1, "student_id": 88, "role": "user" },
  "source": "chat",
  "thread_id": 4512,
  "scope": { "scope_type": "funding_request", "scope_id": 556 },
  "inputs": {
    "goal": "initial_outreach",
    "professor_ref": { "professor_id": 910, "url": null },
    "hook": {
      "type": "paper",
      "ref": { "title": "Some paper title", "url": "https://..." },
      "user_connection_note": "I’m working on a similar approach in my ViT localization project."
    },
    "call_to_action": "ask_for_meeting"
  },
  "constraints": {
    "tone": "professional-warm",
    "length": "short",
    "language": "en",
    "format": "email",
    "include_signature": true
  },
  "context_refs": {
    "document_ids": [2001],
    "funding_request_id": 556
  }
}
```

**Notes**

* `professor_ref` supports either `professor_id` (preferred) or `url` (fallback).
* `hook` is optional; if missing, the system will try to create hooks from professor summary.

---

# 2) Context Package (built by Context Builder)

Context is **read-only** for agents and operators (operators get payloads derived from context).

### Context Package Shape (v1)

```json
{
  "thread": { "thread_id": 4512, "summary": "...", "recent_events": [...] },
  "student": {
    "id": 88,
    "name": "First Last",
    "email": "user@email",
    "phone": "...",
    "citizenship": "...",
    "onboarding_data": { "...": "..." }
  },
  "memory": {
    "tone": ["..."],
    "preferences": ["..."],
    "do_dont": ["..."]
  },
  "funding_request": {
    "id": 556,
    "research_interest": "...",
    "paper_title": "...",
    "research_connection": "...",
    "status": "draft"
  },
  "professor": {
    "id": 910,
    "full_name": "...",
    "email": "...",
    "url": "...",
    "institute": "...",
    "digest_id": 777,
    "cached_summary_outcome_id": "out-abc" 
  },
  "documents": [
    { "document_id": 2001, "type": "resume", "status": "processed", "processed_content_ref": "..." }
  ],
  "policies": {
    "can_generate_email": true,
    "can_send_email": false,
    "requires_apply_confirmation": true
  }
}
```

---

# 3) Plan Template (structured)

### Plan ID and Steps

```json
{
  "plan_id": "plan-001",
  "plan_version": "planner-1.0",
  "intent_id": "uuid-123",
  "explanation": "Generate an outreach email draft using professor context + student profile, review it, and present an apply gate.",
  "steps": [
    {
      "step_id": "s1",
      "step_type": "operator",
      "name": "Context.ResolveProfessorSummary",
      "operator_name": "Professor.ResolveAndSummarize",
      "requires": ["intent.inputs.professor_ref", "context.professor"],
      "produces": ["professor_summary_outcome_id"],
      "risk_level": "low",
      "policy_tags": ["read_only", "public_web"]
    },
    {
      "step_id": "s2",
      "step_type": "operator",
      "name": "Compute.Alignment",
      "operator_name": "Alignment.Compute",
      "requires": ["context.student", "context.documents", "professor_summary_outcome_id", "intent.inputs.hook"],
      "produces": ["alignment_outcome_id"],
      "risk_level": "low",
      "policy_tags": ["pii_internal"]
    },
    {
      "step_id": "s3",
      "step_type": "operator",
      "name": "Draft.Email",
      "operator_name": "Email.GenerateDraft",
      "requires": ["context.student", "context.memory", "context.funding_request", "alignment_outcome_id", "professor_summary_outcome_id", "intent.constraints"],
      "produces": ["email_draft_outcome_id"],
      "risk_level": "medium",
      "policy_tags": ["pii_internal", "external_communication_draft"]
    },
    {
      "step_id": "s4",
      "step_type": "operator",
      "name": "Review.Email",
      "operator_name": "Email.ReviewDraft",
      "requires": ["email_draft_outcome_id", "intent.constraints"],
      "produces": ["email_review_outcome_id"],
      "risk_level": "low",
      "policy_tags": ["quality_check"]
    },
    {
      "step_id": "s5",
      "step_type": "operator",
      "name": "Outcome.Bundle",
      "operator_name": "Outcome.BundleForUI",
      "requires": ["email_draft_outcome_id", "email_review_outcome_id"],
      "produces": ["ui_bundle_outcome_id"],
      "risk_level": "low",
      "policy_tags": ["ui_action"]
    },
    {
      "step_id": "s6",
      "step_type": "human_gate",
      "name": "Apply.DraftToRequest",
      "operator_name": "Workflow.ApplyDraftToRequest",
      "requires": ["user_confirmation", "email_draft_outcome_id"],
      "produces": ["applied_outcome_id"],
      "risk_level": "high",
      "policy_tags": ["write_db", "external_communication_ready"]
    }
  ]
}
```

**Key v1 design choice**

* Step s6 does not run automatically. The system must emit an action prompt: “Apply changes”.

---

# 4) Operator Calls (Actions) and payloads

Each step becomes an **Action** with idempotency key.

## s1) Professor.ResolveAndSummarize

**Idempotency key**
`prof_summary:{professor_id}:{digest_version}:{planner_version}`
(or for url-based: hash(url))

**Payload**

```json
{
  "professor_id": 910,
  "professor_url": "https://...",
  "preferred_sources": ["canspider_digest", "url_fallback"],
  "max_tokens": 1200
}
```

**Result**

```json
{
  "summary": {
    "research_themes": ["..."],
    "recent_work": ["..."],
    "hooks": ["..."],
    "citations": [{"type":"url","ref":"https://..."}]
  },
  "professor_entity_updates": { "digest_id": 777 }
}
```

**Outcome created**
`OutcomeType: Summary.Professor`

---

## s2) Alignment.Compute

**Idempotency key**
`align:{student_id}:{professor_id}:{resume_hash}:{hook_hash}`

**Payload**

```json
{
  "student_profile_ref": { "student_id": 88 },
  "student_resume_ref": { "document_id": 2001, "processed_content_hash": "..." },
  "professor_summary_outcome_id": "out-prof-sum-01",
  "hook": { "type": "paper", "title": "...", "url": "..." }
}
```

**Result**

```json
{
  "alignment_score": 0.78,
  "overlaps": ["..."],
  "gaps": ["..."],
  "recommended_angle": "..."
}
```

**Outcome created**
`OutcomeType: Alignment.Professor`

---

## s3) Email.GenerateDraft

**Idempotency key**
`email_draft:{intent_id}:{draft_version=1}`

**Payload**

```json
{
  "student": { "name": "...", "email": "...", "signature_fields": {...} },
  "professor": { "name": "...", "email": "...", "department": "...", "institute": "..." },
  "research_context": {
    "student_interest": "...",
    "recommended_angle": "...",
    "hook": { "type": "paper", "title": "...", "url": "...", "note": "..." }
  },
  "constraints": { "tone": "professional-warm", "length": "short", "language": "en" },
  "format": "email"
}
```

**Result**

```json
{
  "subject": "Prospective MSc student: ...",
  "body": "Dear Professor ...",
  "attachment_suggestions": ["cv"],
  "rationale": "Used hook from paper X and emphasized overlap Y."
}
```

**Outcome created**
`OutcomeType: Draft.Email`

---

## s4) Email.ReviewDraft

**Idempotency key**
`email_review:{email_draft_outcome_id}:{review_version=1}`

**Payload**

```json
{
  "draft_outcome_id": "out-email-draft-01",
  "rubric_version": "email-quality-v1",
  "constraints": { "tone": "professional-warm", "length": "short" }
}
```

**Result**

```json
{
  "score": 86,
  "issues": [
    {"severity":"medium","msg":"CTA could be clearer"},
    {"severity":"low","msg":"Reduce one sentence length"}
  ],
  "suggested_edits": [
    {"type":"rewrite","target":"sentence_3","proposal":"..."}
  ]
}
```

**Outcome created**
`OutcomeType: Review.Email`

---

## s5) Outcome.BundleForUI

This is a convenience operator that formats what the UI needs:

* draft email
* review summary
* suggested next actions (apply, optimize, upload missing doc, etc.)

**Outcome created**
`OutcomeType: UI.Bundle`

---

## s6) Workflow.ApplyDraftToRequest (human gate)

Triggered only after user clicks “Apply”.

**Idempotency key**
`apply:{funding_request_id}:{email_draft_outcome_id}`

**Payload**

```json
{
  "funding_request_id": 556,
  "email_draft_outcome_id": "out-email-draft-01",
  "write_targets": ["funding_emails", "funding_requests"],
  "attachments": [{"document_id":2001,"purpose":"cv"}]
}
```

**Result**

```json
{
  "funding_emails_id": 9901,
  "funding_request_updated": true,
  "linked_attachments": [301, 302]
}
```

**Outcome created**
`OutcomeType: Applied.EmailDraft`

---

# 5) Outcomes (canonical objects)

## Outcome: Summary.Professor

```json
{
  "outcome_id": "out-prof-sum-01",
  "outcome_type": "Summary.Professor",
  "schema_version": "1.0",
  "status": "final",
  "intent_id": "uuid-123",
  "content": {
    "research_themes": ["..."],
    "hooks": ["..."],
    "citations": [{"url":"https://..."}]
  },
  "confidence": 0.74
}
```

## Outcome: Draft.Email

```json
{
  "outcome_id": "out-email-draft-01",
  "outcome_type": "Draft.Email",
  "schema_version": "1.0",
  "status": "draft",
  "intent_id": "uuid-123",
  "content": {
    "subject": "...",
    "body": "...",
    "attachment_suggestions": ["cv"],
    "placeholders": []
  },
  "confidence": 0.72
}
```

## Outcome: Review.Email

```json
{
  "outcome_id": "out-email-review-01",
  "outcome_type": "Review.Email",
  "schema_version": "1.0",
  "status": "final",
  "intent_id": "uuid-123",
  "content": {
    "score": 86,
    "issues": [...],
    "suggested_edits": [...]
  },
  "confidence": 0.80
}
```

## Outcome: UI.Bundle

Includes “Apply” action prompt.

```json
{
  "outcome_id": "out-ui-bundle-01",
  "outcome_type": "UI.Bundle",
  "schema_version": "1.0",
  "status": "final",
  "content": {
    "primary_artifact": {"type":"Draft.Email","id":"out-email-draft-01"},
    "review": {"id":"out-email-review-01"},
    "actions": [
      {"type":"OPTIMIZE_DRAFT","label":"Improve email"},
      {"type":"APPLY_DRAFT","label":"Apply changes to request"},
      {"type":"UPLOAD_DOC","label":"Upload CV", "required": false}
    ]
  }
}
```

## Outcome: Applied.EmailDraft

```json
{
  "outcome_id": "out-applied-01",
  "outcome_type": "Applied.EmailDraft",
  "schema_version": "1.0",
  "status": "final",
  "content": {
    "funding_request_id": 556,
    "funding_emails_id": 9901,
    "applied_at": "2026-01-28T..."
  }
}
```

---

# 6) Event Timeline (what you stream and store)

This is critical: progress is derived from events.

### Event stream (typical)

1. `INTENT_RECEIVED`
2. `PLAN_CREATED`
3. `POLICY_EVALUATED` (for plan)
4. `ACTION_STARTED` (s1)
5. `ACTION_SUCCEEDED` (s1)
6. `OUTCOME_CREATED` (Summary.Professor)
7. `ACTION_STARTED` (s2)
8. `ACTION_SUCCEEDED` (s2)
9. `OUTCOME_CREATED` (Alignment.Professor)
10. `ACTION_STARTED` (s3)
11. `ACTION_SUCCEEDED` (s3)
12. `OUTCOME_CREATED` (Draft.Email)
13. `ACTION_STARTED` (s4)
14. `ACTION_SUCCEEDED` (s4)
15. `OUTCOME_CREATED` (Review.Email)
16. `ACTION_STARTED` (s5)
17. `ACTION_SUCCEEDED` (s5)
18. `OUTCOME_CREATED` (UI.Bundle)
19. `WORKFLOW_COMPLETED` (execution paused at human gate)

Then user clicks Apply:

20. `USER_APPROVED` (Apply)
21. `POLICY_EVALUATED` (apply action)
22. `ACTION_STARTED` (s6)
23. `ACTION_SUCCEEDED` (s6)
24. `OUTCOME_CREATED` (Applied.EmailDraft)
25. `WORKFLOW_COMPLETED` (final)

---

# 7) Retry semantics (how you avoid duplicates)

### Retry rules

* Retrying the workflow reuses the same `intent_id` unless user asks for “new version”.
* Actions reuse idempotency keys, so:

  * professor summary isn’t re-fetched unnecessarily
  * apply doesn’t duplicate DB writes
* If user requests “regenerate fresh”, you generate `draft_version=2` and a new idempotency key:

  * `email_draft:{intent_id}:{draft_version=2}`

### User-visible versions

* `Draft.Email` outcomes are versioned (v1, v2, v3).
* Apply always targets a specific draft outcome id.

---

# 8) Where this maps to your existing DB (pragmatic)

Without redesigning everything today:

* Use `ai_jobs` to store:

  * input_payload: intent + plan
  * result_payload: step results + created outcome IDs
* Use `chat_thread_messages` (fixed schema) as your event ledger for v1.
* Store email draft content either:

  * in an outcomes table (ideal), or
  * in `ai_jobs.result_payload` + return to UI (acceptable for v1)
    but once user clicks Apply, persist to:
  * `funding_emails.main_email_subject/body` (draft state)
  * or `funding_requests.email_subject/content` depending on your current flow

My strong opinion: **persist only on Apply**. Drafts should live as outcomes, not as domain writes.

---

# 9) This workflow is the template for every new capability:

* define intent type + schema
* define context build keys
* define plan steps with named operators
* define action payload/result schemas
* define outcome objects
* define event stream
* define apply/human gate points
* define idempotency rules
