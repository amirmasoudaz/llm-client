
## 3. The Kernel

The Kernel is the minimal, stable core. It should change slowly.

Yes. The Kernel is the thing that makes this an **“intelligence layer”** instead of “a chat app with some prompts”. Treat it like an OS kernel: boring, minimal, stable, paranoid, and extremely good at bookkeeping.

I’m going to go deep in a very specific way: **contracts, invariants, boundaries, failure modes, and the extension points** that keep v1 shippable and v4 inevitable.

---

## 0) What the Kernel actually is

**Kernel = the runtime that turns external requests into accountable workflows.**
Not the brain. Not the features. Not the prompts.

The Kernel owns **control-plane**: identity, policy, orchestration, ledgers, progress, retries.
Everything “business” lives outside as **capabilities/plugins** that the Kernel runs.

Strong opinion: if something can’t be expressed as **Intent → Plan → Actions → Outcomes → Events**, it does not belong in the Kernel.

---

## 1) Kernel invariants (the laws it must enforce)

These are non-negotiable. They’re how you stop agent soup.

1. **All execution is workflow execution.**
   Even a “simple answer” is a 1-step plan.

2. **No hidden state.**
   The Kernel must be able to explain “why” using only ledgers.

3. **Side effects are operator-only.**
   Kernel never directly sends email, writes docs, or touches third-party APIs except through operators.

4. **Policy gates every step that matters.**
   Especially anything external or anything that writes to “final” state.

5. **Idempotency is mandatory.**
   Retry cannot duplicate: emails, DB writes, exports, attachments.

6. **Streaming is derived from events.**
   Progress is not a vibe. It is a view over event log + step states.

7. **Capability isolation.**
   Kernel does not “know” about funding outreach. It knows about running a capability named `Funding.Outreach.Email.Generate`.

---

## 2) Kernel = eight responsibilities, deep version

### 2.1 Intake

**Contract:** accept request from chat/API/webhook and emit a *canonical Intent*.

**Kernel must do:**

* normalize input into: `Intent { type, actor, scope, inputs, constraints, context_refs }`
* assign `intent_id`, `correlation_id`, `thread_id/workflow_id`
* record `INTENT_RECEIVED`

**Kernel must not do:**

* interpret business meaning beyond intent typing
* fetch professor pages
* ask model to “just handle everything”

**Key design choice:** Intake adapters are thin. They don’t execute. They only:

* authenticate caller
* create intent + enqueue workflow execution
* open SSE

---

### 2.2 Identity + Authorization

This is where enterprise-readiness starts.

**Contract:** given `(tenant_id, principal, intent)`, return an `AuthContext` with scopes.

**Kernel must enforce:**

* tenant isolation (every DB query is tenant-scoped)
* role-based permissions (student vs admin vs support agent)
* capability-level permissions (can generate vs can apply vs can send)

**Outputs:**

* `AuthContext { tenant_id, principal_id, role, scopes[], data_access_policy }`

**Hard rule:** Operators never get raw “user tokens” without explicit permission and scoping.

**v1 pragmatic move:** even if you’re single-tenant today, **include `tenant_id` everywhere**. Retrofitting multi-tenancy later is pain.

---

### 2.3 Context Builder

This is the Kernel’s most underestimated piece. It’s what makes agents stateless.

**Contract:** build a deterministic context bundle for planner/executor from ledgers and refs.

**Inputs:**

* intent
* thread_id
* scope_id (funding_request_id)
* context_refs (professor_id, document_ids)

**Output:**

* `ContextBundle` that contains:

  * core profile fields
  * relevant memory (typed)
  * scope entity (funding_request)
  * professor entity (and summary outcome id if cached)
  * doc ledger refs (processed_content hashes, missing fields)
  * thread summary (optional) + last N events (not raw chat spam)

**Rules (very important):**

* Context Builder returns **references + compact summaries**, not giant blobs.
* It must be **cacheable** keyed by `(intent_id, latest_ledger_versions)` or at least `(thread_id, last_event_id)`.
* It must be able to **redact** for policy and logging.

**Failure mode to avoid:** agents doing DB reads on their own. That breaks determinism and security.

---

### 2.4 Planner

**Contract:** convert `(intent + context)` into a typed `Plan`.

**Planner output must be:**

* step list with names, operator/capability call, required inputs, outputs, policy tags, risk level
* declared human gates
* step-level idempotency strategy (or derivation inputs)

**Kernel rule:** Planner can be replaced/upgraded without breaking the Kernel. So:

* planner must output a stable plan schema
* plan must reference stable operator names

**v1 recommendation:** planner can be a deterministic router + small LLM assist for textual explanation. Don’t start with “LLM makes the plan” as your only plan.

---

### 2.5 Executor

This is the runtime. It is not “a loop that calls tools”. It’s a state machine.

**Contract:** execute plan steps with retries and produce outcomes and events.

**Executor MUST support:**

* **state machine per workflow**:

  * `accepted → planned → running → waiting_for_user → running → completed|failed|cancelled`
* **step execution**:

  * `queued → running → succeeded|failed|skipped`
* **retries**:

  * retry policy based on error type: transient vs permanent
  * exponential backoff where needed
* **idempotency**:

  * enforce idempotency keys at operator call boundary
* **resume**:

  * resume from step N using ledger state
* **human gate**:

  * pause and emit UI action request, then resume when user approves

**Executor MUST NOT:**

* contain special cases like “if funding_request then do X”
* perform side effects directly
* silently skip steps without emitting events

**Big design win:** treat “LLM call” as just another operator action (with trace).

---

### 2.6 Policy Engine

Policy is the Kernel’s immune system.

**Contract:** given `(auth_context, intent, plan_step, action_payload, current_context)`, return:

* `ALLOW | DENY | REQUIRE_APPROVAL | TRANSFORM | REDACT`

**Policy must gate:**

* external communication (email content, recipients)
* writes to final ledgers (apply)
* access to sensitive memory or documents
* web retrieval scope if you care about compliance later

**Kernel rule:** policy decisions are events. Always.

**v1 policy stance (strong opinion):**

* **No auto-send in v1.**
* “Apply” is required for any write to request/email tables.
* Any action that touches Gmail tokens requires explicit scope + reminders_on toggle.

---

### 2.7 Ledger writes

This is what makes the system accountable and debuggable.

At minimum, the Kernel writes:

* events
* jobs (actions)
* outcomes
* documents ledger updates
* memory ledger updates (typed, with source/confidence)

**Ledger design rules:**

* append-only event log
* outcomes are versioned (draft vs final)
* action records include payload hash + result + error classification
* everything has correlation_id / trace_id

**If you cut corners here, you’ll pay forever.** This is the part enterprises love, and startups skip.

---

### 2.8 Streaming + Progress

You already have the right instinct: multiple channels.

**Kernel contract:** stream deterministic views:

* `events` channel: raw canonical event stream
* `progress` channel: derived from step transitions and weights
* `assistant` channel: user-facing narrative and artifacts (a projection, not truth)

**Rule:** progress numbers come from plan structure, not ad-hoc prints.

**Example:**

* each step has weight (default equal)
* when step transitions to succeeded, progress updates
* while running, you can emit “subprogress” but it must be clearly labeled as non-final

---

## 3) Kernel MUST NOT, deep version

You already wrote two bullets. Here’s the full “nope list” that keeps the Kernel clean:

### Must not contain business logic

* no funding-specific fields
* no “SOP rules”
* no prompt text
* no “email tone heuristics”
  Those belong to capability packages.

### Must not contain integration conditionals

* no “if Salesforce then…”
* no “if university uses system X then…”
  Adapters handle that.

### Must not directly perform side effects

* Kernel does not send emails, upload to external systems, mutate final docs
  Only operators do.

### Must not implement domain-specific data models

Kernel stores generic primitives.
Domain tables are updated via apply operators.

---

## 4) The Kernel’s internal interface map (what calls what)

This is the stable “spine” that should change slowly:

1. **IntakeAdapter** → `Kernel.submit(request)`
2. **AuthZ** → `auth_context`
3. **IntentNormalizer** → canonical intent
4. **Ledger.append(INTENT_RECEIVED)**
5. **ContextBuilder.build(intent, auth_context)**
6. **Planner.plan(intent, context)**
7. **Ledger.append(PLAN_CREATED)**
8. **Executor.run(plan, context, auth_context)**

   * for each step:

     * Policy.evaluate(step/action)
     * Operator.invoke(action)
     * Ledger.append(ACTION_*, OUTCOME_*)
     * Stream updates
9. Pause at human gate → emit UI action
10. Resume on approval → apply operator → done

That’s the Kernel. Everything else is plugins/operators/adapters.

---

## 5) Capability packaging (how Kernel stays stable while features explode)

You need a **Capability Registry**. Not optional.

Each capability provides:

* intent schema + version
* plan template (or planner hook)
* operator set it uses
* outcome schemas it produces
* policy tags per step
* UI action descriptors (apply/optimize/upload missing docs)

Kernel loads capabilities like packages:

* `Funding.Outreach.Email.Generate@1.0`
* `Admission.Program.Apply@1.0`
* `Support.Ticket.Resolve@1.0`

Kernel does not know what those mean. It just runs them.

---

## 6) Idempotency and “exactly-once-ish” semantics

You will live or die here.

### Rules

* Every operator call must include `idempotency_key`.
* Operator implementations must enforce:

  * if key seen and succeeded → return cached result
  * if key seen and running → return “in-progress” handle
  * if key failed with non-retryable → return same failure

### Where keys come from

* derived from intent_id + step_id + stable inputs hash
* “regenerate” increments a draft_version and thus changes key
* “apply” key ties to (funding_request_id + draft_outcome_id)

This prevents:

* duplicate emails
* duplicate attachment links
* double-exports

---

## 7) Kernel data model: what it truly needs (minimal)

Even if you store in your existing tables for v1, conceptually the Kernel needs:

* `workflow` (thread/job grouping)
* `event_log`
* `action_log` (ai_jobs)
* `outcomes`
* `documents_ledger`
* `memory_ledger`

You can implement outcomes inside `ai_jobs.result_payload` in early v1, but **don’t pretend that’s final**. You will want an outcomes table soon for versioning and retrieval.

---

## 8) Practical v1 Kernel build order (the shortest path that won’t collapse)

1. **Event log + correlation IDs** (non-negotiable)
2. **Intent normalization + schema validation**
3. **AuthZ + tenant scoping** (even if single tenant)
4. **Context Builder (minimal)** for funding outreach
5. **Plan schema + deterministic planner** for 3–6 intents
6. **Executor state machine** with retries + idempotency enforcement
7. **Operator interface + 6–10 operators**
8. **SSE streaming** from events/progress
9. **Human gate apply** as the only write path to domain tables

---

## 9) Two sharp observations from your current DB sketch (because this WILL bite you)

I’ll keep it respectful to your “check with me” constraint, but you should know:

1. Your `chat_thread_messages` DDL references `student_id` in the FK constraint, but the column isn’t defined in the table in the snippet. Also there’s a trailing comma before `)`. That table won’t run as written.
   You can keep it simple: include `student_id`, `tenant_id`, and remove that trailing comma.

2. You currently mix “workflow” and “funding_request” 1:1 in `chat_threads` (`UNIQUE funding_request_id`). That’s fine for v1, but v2/v3 will need threads that aren’t tied to a funding request.
   Keep the unique constraint only if you also allow `scope_type/scope_id` later.

No need to redesign now, just don’t lock yourself in.

---

## 10) What “Kernel changes slowly” means in practice

Kernel evolves only in these cases:

* new primitive schema versions (with backward compatibility)
* new policy rule sets
* improved planner/executor internals without changing interfaces
* new operator types added to the registry
* performance scaling (queueing, concurrency), without semantic changes

Kernel should not change because you added:

* a new email style
* a new university source
* a new admission workflow
  That’s all capability land.

---

Below is a **v1 Kernel Interface Spec**. It’s written like an internal RFC: crisp contracts, stable schemas, and the smallest set of interfaces you need to run workflows deterministically. No business logic, no prompts.

---

# Kernel Interface Spec v1.0

## 1) Scope and goals

This spec defines the Kernel’s public and internal interfaces to:

* accept requests (chat/api/webhook)
* normalize to intents
* authorize and build context
* plan and execute workflows
* write ledgers (events/actions/outcomes)
* stream progress and artifacts deterministically

**Non-goals**

* defining business prompts, templates, or domain-specific heuristics
* defining adapters for specific enterprise systems
* defining operator internals

---

## 2) Core types (canonical primitives)

### 2.1 Identifiers

* `TenantID`: int (required everywhere)
* `StudentID`: int
* `ThreadID`: int (workflow thread)
* `WorkflowID`: uuid (or bigint, but must be globally unique)
* `IntentID`: uuid
* `PlanID`: uuid
* `StepID`: string (stable within plan, e.g., "s3")
* `ActionID`: uuid
* `OutcomeID`: uuid
* `EventID`: uuid
* `CorrelationID`: uuid (ties all logs/streams)
* `IdempotencyKey`: string

---

## 3) Public Kernel API (external callers)

### 3.1 Submit intent (unified entrypoint)

**POST** `/v1/kernel/submit`

**Request**

```json
{
  "source": "chat",
  "tenant_id": 1,
  "principal": { "type": "student", "id": 88, "role": "user" },
  "thread_id": 4512,
  "scope": { "scope_type": "funding_request", "scope_id": 556 },
  "input": {
    "mode": "message",
    "text": "Generate an email to Prof X mentioning my ViT work",
    "attachments": { "document_ids": [2001] }
  },
  "intent_hint": {
    "intent_type": "Funding.Outreach.Email.Generate",
    "inputs": { "goal": "initial_outreach" }
  },
  "constraints": { "tone": "professional-warm", "length": "short", "language": "en" }
}
```

**Response**

```json
{
  "workflow_id": "wf-uuid",
  "intent_id": "intent-uuid",
  "plan_id": "plan-uuid",
  "correlation_id": "corr-uuid",
  "stream": {
    "events": "/v1/kernel/stream/wf-uuid/events",
    "progress": "/v1/kernel/stream/wf-uuid/progress",
    "assistant": "/v1/kernel/stream/wf-uuid/assistant"
  },
  "status": "accepted"
}
```

**Rules**

* Submission returns immediately after enqueueing workflow execution.
* Kernel must write `INTENT_RECEIVED` and `PLAN_CREATED` events (or `INTENT_REJECTED` on failure).

---

### 3.2 Approve or reject a human gate (apply)

**POST** `/v1/kernel/workflows/{workflow_id}/gate`

**Request**

```json
{
  "tenant_id": 1,
  "principal": { "type": "student", "id": 88, "role": "user" },
  "gate_id": "gate-s6",
  "decision": "approve",
  "payload": {
    "target": { "type": "Draft.Email", "outcome_id": "out-email-draft-01" },
    "apply_targets": ["funding_emails", "funding_requests"]
  }
}
```

**Response**

```json
{
  "workflow_id": "wf-uuid",
  "status": "resumed",
  "correlation_id": "corr-uuid"
}
```

**Rules**

* Must emit `USER_APPROVED` or `USER_REJECTED`.
* Must be idempotent: approving twice returns the same result.

---

### 3.3 Fetch outcomes

**GET** `/v1/kernel/workflows/{workflow_id}/outcomes`

**Response**

```json
{
  "workflow_id": "wf-uuid",
  "outcomes": [
    { "outcome_id": "out-1", "outcome_type": "Draft.Email", "status": "draft", "content": {...} }
  ]
}
```

---

### 3.4 Retry workflow

**POST** `/v1/kernel/workflows/{workflow_id}/retry`

**Request**

```json
{
  "tenant_id": 1,
  "principal": { "type": "student", "id": 88, "role": "user" },
  "mode": "resume_failed_steps"
}
```

**Response**

```json
{
  "workflow_id": "wf-uuid",
  "status": "restarted",
  "correlation_id": "corr-uuid"
}
```

**Rules**

* Retries reuse existing intent by default.
* Regenerate requires a new draft version and new idempotency keys for affected steps.

---

## 4) Streaming spec (SSE)

All streams are SSE. Each event line contains a JSON payload with stable schema.

### 4.1 Events stream

**GET** `/v1/kernel/stream/{workflow_id}/events`

SSE message format:

* event: `kernel.event`
* data: `KernelEvent`

### 4.2 Progress stream

**GET** `/v1/kernel/stream/{workflow_id}/progress`

* event: `kernel.progress`
* data: `KernelProgress`

### 4.3 Assistant stream

**GET** `/v1/kernel/stream/{workflow_id}/assistant`

* event: `kernel.assistant.delta`
* data: deltas or complete artifacts (projection, not truth)

---

## 5) Internal Kernel services (stable interfaces)

### 5.1 AuthZ Service

`authorize(principal, tenant_id, intent_hint, scope) -> AuthContext`

**AuthContext**

```json
{
  "tenant_id": 1,
  "principal_id": 88,
  "role": "user",
  "scopes": [
    "intent:submit",
    "workflow:execute",
    "outcome:read",
    "gate:approve"
  ],
  "data_policy": { "allow_pii": true, "allow_external_send": false }
}
```

---

### 5.2 Intent Normalizer

`normalize(submit_request, auth_context) -> Intent`

Must validate:

* intent type exists in registry
* schema validation for typed `inputs`

---

### 5.3 Context Builder

`build(intent, auth_context) -> ContextBundle`

**Rules**

* Must be deterministic (same ledger state → same context)
* Must return references and compact summaries, not raw huge docs

---

### 5.4 Planner

`plan(intent, context, auth_context) -> Plan`

Plan must:

* have stable step names
* declare policy tags and risk levels
* declare gates

---

### 5.5 Policy Engine

`evaluate(stage, subject, auth_context, intent, context) -> PolicyDecision`

* `stage`: `plan|action|outcome|gate`
* `subject`: plan step or action payload or outcome

**PolicyDecision**

```json
{
  "decision": "ALLOW",
  "reason": "draft only, no external send",
  "transform": null,
  "redactions": []
}
```

Allowed decisions:

* `ALLOW`
* `DENY`
* `REQUIRE_HUMAN_APPROVAL`
* `TRANSFORM` (returns transformed payload)
* `ALLOW_WITH_REDACTION`

---

### 5.6 Executor

`execute(workflow_id) -> void`

Executor runs the state machine:

* loads intent/plan/context from ledgers
* executes next runnable step
* enforces idempotency
* records events and outcomes
* pauses at gates

Internal step runner:
`run_step(step, context) -> StepResult`

---

### 5.7 Operator Invoker (Kernel side)

`invoke(operator_name, payload, auth_context, idempotency_key, correlation_id) -> OperatorResult`

**Operator contract**

* must be pure function w.r.t idempotency key (exactly-once-ish)
* must return structured errors with retry hints

---

### 5.8 Ledger Service

Append-only and versioned writes.

* `append_event(event: KernelEvent) -> EventID`
* `record_action(action: ActionRecord) -> ActionID`
* `record_outcome(outcome: Outcome) -> OutcomeID`
* `read_workflow_state(workflow_id) -> WorkflowState`

---

## 6) Kernel schemas

### 6.1 KernelEvent

```json
{
  "event_id": "evt-uuid",
  "event_type": "ACTION_SUCCEEDED",
  "timestamp": "2026-01-28T19:03:00Z",
  "tenant_id": 1,
  "workflow_id": "wf-uuid",
  "thread_id": 4512,
  "intent_id": "intent-uuid",
  "plan_id": "plan-uuid",
  "step_id": "s3",
  "actor": { "type": "system", "id": "kernel" },
  "correlation_id": "corr-uuid",
  "payload": {
    "operator_name": "Email.GenerateDraft",
    "action_id": "act-uuid",
    "outcome_id": "out-email-draft-01"
  }
}
```

### 6.2 KernelProgress

```json
{
  "tenant_id": 1,
  "workflow_id": "wf-uuid",
  "correlation_id": "corr-uuid",
  "percent": 60,
  "stage": "running",
  "current_step": { "step_id": "s4", "name": "Review.Email" }
}
```

### 6.3 ActionRecord

```json
{
  "action_id": "act-uuid",
  "tenant_id": 1,
  "workflow_id": "wf-uuid",
  "intent_id": "intent-uuid",
  "plan_id": "plan-uuid",
  "step_id": "s3",
  "operator_name": "Email.GenerateDraft",
  "operator_version": "1.0",
  "idempotency_key": "email_draft:intent-uuid:v1",
  "status": "succeeded",
  "payload_hash": "sha256...",
  "input_payload": { "...": "typed payload" },
  "result_payload": { "...": "typed result" },
  "error": null,
  "trace": { "trace_id": "openai-...", "trace_type": "openai" },
  "timing": { "started_at": "...", "finished_at": "..." },
  "tokens": { "in": 1200, "out": 380, "total": 1580 },
  "cost": { "total": 0.0123 }
}
```

### 6.4 Outcome

```json
{
  "outcome_id": "out-uuid",
  "tenant_id": 1,
  "workflow_id": "wf-uuid",
  "intent_id": "intent-uuid",
  "outcome_type": "Draft.Email",
  "schema_version": "1.0",
  "status": "draft",
  "content": { "subject": "...", "body": "...", "attachments": ["cv"] },
  "confidence": 0.72,
  "created_at": "..."
}
```

---

## 7) Idempotency spec (mandatory)

### 7.1 Rules

* Every operator invocation MUST include `idempotency_key`.
* Operator MUST implement:

  * `key not seen` → run, store result, return
  * `key seen succeeded` → return stored result
  * `key seen running` → return “in_progress” with handle
  * `key seen failed` → return same failure unless retry_policy allows rerun

### 7.2 Key derivation guideline

Default:
`{operator}:{tenant_id}:{stable_hash(payload_normalized)}:{version}`

Special cases:

* Regeneration increments version (`draft_version`)
* Apply keys tie to target outcome:

  * `apply:{tenant_id}:{funding_request_id}:{draft_outcome_id}`

---

## 8) Error taxonomy (Kernel and Operators)

### 8.1 Error object

```json
{
  "code": "RATE_LIMITED",
  "message": "Provider rate limit",
  "retryable": true,
  "retry_after_ms": 2000,
  "category": "transient",
  "details": { "provider": "openai" }
}
```

### 8.2 Categories

* `validation` (non-retryable)
* `authorization` (non-retryable)
* `policy_denied` (non-retryable)
* `transient` (retryable)
* `dependency` (retryable or not depending)
* `operator_bug` (non-retryable, alerts)
* `timeout` (retryable)

Kernel executor behavior:

* retry `transient|timeout` up to `N`
* stop workflow on non-retryable with `WORKFLOW_FAILED`
* emit events for every failure transition

---

## 9) Capability Registry interface (how Kernel stays generic)

### 9.1 Registry

`get_capability(intent_type) -> CapabilityDefinition`

CapabilityDefinition includes:

* intent schema + version
* planner hook or plan templates
* allowed operators list
* outcome schema registry
* default step weights (progress)

Kernel uses this registry to validate intent types and plan steps without hardcoding business logic.

---

## 10) Minimal v1 compliance checklist

Kernel is v1-compliant only if:

* all execution starts with `/kernel/submit`
* every workflow has:

  * intent recorded
  * plan recorded
  * actions recorded with idempotency
  * outcomes recorded
  * append-only event stream
* human gate exists for apply/write actions
* policy engine runs on plan and on each action

---

## 11) One concrete mapping to the current tables (practical)

Without forcing redesign now:

* `ai_jobs` can store ActionRecord (one row per action)
* `chat_thread_messages` can store KernelEvent records (as JSON `content`)
* outcomes can be stored either:

  * in a new `outcomes` table (recommended)
  * or in `ai_jobs.result_payload` for v1 (acceptable but you’ll outgrow it fast)
