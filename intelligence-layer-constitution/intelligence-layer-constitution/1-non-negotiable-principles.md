
# 1. Non-Negotiable Principles

## 1.1) Workflow-first, interface-second

### What it means

The system’s core unit is **a job that produces an outcome**, not a conversation turn.

Chat is just one way to submit intents. Tomorrow it’s an API call from Salesforce, a webhook from Intercom, or a button inside an SIS. Same core runtime.

### Why it exists

Conversation-first systems become:

* hard to integrate (everything depends on “chat history”)
* hard to audit (why did it do that?)
* hard to test (prompts drift, context drift)

Workflow-first systems become:

* composable
* replayable
* measurable
* enterprise-friendly

### Rules

* Every user request is normalized into an **Intent** (structured).
* Every Intent produces one or more **Outcomes** (structured).
* Chat messages must never be the source of truth. They are input.
* “What happened?” must be answerable without re-running the LLM by reading the ledgers.

### Exceptions

* Pure Q&A with no side effects can stay conversational. But still: log intent/outcome lightly.
* Exploratory brainstorming can skip formal planning, but it must not mutate state.

### Common failure modes

* Building features that only work if the chat history is perfect.
* Embedding business logic in “assistant responses” instead of operators/outcomes.
* Letting UI define system reality (UI says it sent email, but nothing was actually sent).

---

## 1.2) Everything is Intent → Plan → Actions → Outcomes

### What it means

The layer is basically: **compiler + runtime**.

* Intent is the “source code”
* Plan is the “compiled program”
* Actions are “syscalls”
* Outcomes are “artifacts and state changes”

### Why it exists

This is how you get:

* deterministic-ish behavior
* idempotency (retries don’t duplicate)
* clean debug paths
* policy gating
* real metrics

### Rules

* An Intent must be serializable and stored.
* A Plan must be explicit (step list), named, typed.
* Every side effect must be an Action (operator call).
* Outcomes must be versioned and refer back to the plan/actions that produced them.

### Exceptions

* Single-step “pure compute” intents can have implicit plans (still representable as one step).
* If the system is in “assistant mode” for trivial tasks, you can skip exposing the plan to the user, but you still create it internally.

### Common failure modes

* “The agent just did stuff” with no plan.
* Plans that are prose paragraphs instead of machine-checked steps.
* Outcomes stored as blobs without schema and provenance.

---

## 1.3) Agents are stateless. State lives in ledgers.

### What it means

Agents are **functions**. They read context and return outputs.

State lives in explicit stores:

* event log
* memory ledger (typed)
* document ledger
* outcome ledger
* entity ledger

### Why it exists

Stateful agents cause:

* nondeterminism (“why did it do that today?”)
* hidden coupling (“agent A assumes agent B already did X”)
* un-debuggable drift

Ledger state gives you:

* reproducibility
* controlled memory
* safe multi-agent parallelism
* enterprise audit trails

### Rules

* Agents cannot “remember” outside ledgers.
* Any “learned preference” must be written as typed memory (with source + confidence).
* No agent may depend on ephemeral chat history beyond what the Context Builder provides.

### Exceptions

* In-process ephemeral scratch (within a single job execution) is allowed, but must not become persistent truth.
* Cached computations are allowed if derived deterministically and tied to input hashes.

### Common failure modes

* Prompting agents to “remember this next time” without writing memory.
* Storing important facts only inside thread summaries with no schema.
* Multiple “versions of the truth” across DB tables and prompt text.

---

## 1.4) Side effects only through Operators

### What it means

Agents decide. Operators execute.

If it changes the world, it goes through an Operator:

* sending email
* updating CRM
* saving a final document
* modifying a request status
* calling third-party APIs with consequences

### Why it exists

Because enterprise needs:

* permission boundaries
* idempotency
* auditability
* rate limiting
* safe retries

And you need:

* clean interfaces
* testing with mocks
* separation of responsibility

### Rules

* Agents can only propose actions, never execute side effects directly.
* Operators must be:

  * typed inputs/outputs
  * idempotent via idempotency_key
  * observable (metrics, error types)
* Operators are the only place where credentials live.

### Exceptions

* Read-only calls can be tools (retrieval, parsing).
* “Write” that is internal but reversible (like storing a draft in sandbox) can be an operator still. Basically: just make it an operator anyway.

### Common failure modes

* “Just call Gmail API from the agent” (nightmare).
* Side effects scattered across services without a shared audit trail.
* Operators that are not idempotent (retries duplicate emails).

---

## 1.5) Policy-first execution

### What it means

Policy is not moderation. Policy is **governance**.

Policy decides:

* what can be accessed
* what can be generated
* what can be sent
* what requires human approval
* what must be redacted
* what must be logged

### Why it exists

You’re heading toward immigration + admissions + enterprise systems. That’s compliance terrain.

If policy is not a core primitive, you will end up with:

* “unsafe by default”
* last-minute bolt-ons
* inconsistent behavior across agents

### Rules

* Every Action is evaluated by policy *before* execution.
* Policy engine returns: allow/deny/require-approval/transform/redact.
* Policies are tenant-aware and role-aware.
* Policy decisions are logged as events.

### Exceptions

* Pure read-only safe operations may be auto-allowed under strict scopes.
* Internal dev mode can relax some policies, but must be explicit and never ship unnoticed.

### Common failure modes

* Relying on “LLM safety” instead of explicit rules.
* Mixing “policy logic” into prompts.
* No separation between “draft” vs “final” actions.

---

## 1.6) Plans are inspectable and interruptible

### What it means

Users and enterprises need control over high-stakes tasks.

Your runtime must support:

* showing a plan
* pausing execution
* asking for missing inputs
* resuming from step N
* human approval gates

### Why it exists

Because “seamless automation” without brakes becomes:

* wrong emails sent
* wrong docs submitted
* reputation damage
* compliance failures

### Rules

* Plans must declare:

  * required inputs
  * dependencies
  * side effect steps
  * risk level
* Plan execution must support:

  * pause/resume
  * partial completion
  * “dry run” (no side effects)

### Exceptions

* For low-risk tasks, you can run without showing the plan. But it still exists internally.
* If a user has high trust and autopilot on, you can skip asking for approval for certain action classes.

### Common failure modes

* “Everything is automatic” and users feel unsafe.
* Plans that can’t resume; any failure forces restart.
* No dry-run mode, so testing costs real side effects.

---

## 1.7) Adapters isolate integrations

### What it means

Your core stays canonical. Enterprise systems are just external shapes.

Adapters translate:

* external events → intents
* outcomes → external updates

### Why it exists

Otherwise you’ll get:

* “Salesforce version” logic fork
* “University A SIS” fork
* dozens of conditional branches in your orchestrator

That kills scalability and correctness.

### Rules

* Core capabilities are integration-agnostic.
* Adapters never implement business logic.
* Adapters operate only through operators and policies.
* Same intent schema regardless of integration source.

### Exceptions

* Some integrations require special auth flows or object mappings. That belongs in adapter configuration, not core behavior.
* Some enterprise-specific constraints can be expressed as policies, not code forks.

### Common failure modes

* Hardcoding per-client quirks in the core.
* Treating “integration” as a different product instead of a connector.
* Mapping external fields directly into agent prompts without normalization.

---

# Principle #1 Mapping

## Workflow-first, interface-second

### Goal (what “done” means)

In v1, **every user interaction** (chat message, button click, retry, apply-changes) must become:

1. a **canonical Intent**
2. a stored **Plan** (even if 1-step)
3. executed **Actions** (operators only)
4. stored **Outcomes**
5. recorded **Events**

Chat is just an intake adapter. You should be able to run the same workflow through REST or webhook later without touching the core.

---

## A) Canonical entities you must implement in v1

### 1) Intent (stored)

**Where:** `ai_jobs.input_payload` (or a dedicated `intents` ledger table later)

**Minimum schema (v1):**

* `intent_id` (uuid)
* `intent_type` (enum)
* `actor`: `{ student_id, role }`
* `thread_id`
* `request_id` optional (funding_request_id)
* `inputs` (typed per intent)
* `constraints` (tone/length/etc.)
* `context_refs`: `{ document_ids, professor_id, urls }`
* `created_at`

**v1 intent types (keep it tight):**

* `Funding.Outreach.ProfessorProfile` (prof url/digest → summary/hooks)
* `Funding.Outreach.Alignment` (user background + prof summary → alignment)
* `Funding.Outreach.Email.Generate`
* `Funding.Outreach.Email.Review`
* `Funding.Outreach.Email.Optimize`
* `Documents.Ingest` (upload resume/transcript → extracted JSON + missing fields)
* `Workflow.Apply` (apply draft to request/documents)

You can support more later; don’t.

---

### 2) Plan (stored)

**Where:** `ai_jobs.input_payload.plan` + events

**Minimum schema (v1):**

* `plan_id` (uuid)
* `plan_version` (string)
* `steps[]` each step:

  * `step_id`
  * `name`
  * `operator_name` or `agent_name`
  * `requires` (inputs/context)
  * `produces` (outputs/outcomes)
  * `risk_level`: `low|medium|high`
  * `policy_tags`: `{ pii, external_side_effect, ... }`

**Rule:** Even “just answer my question” should internally be a 1-step plan. That’s how you stay workflow-first.

---

### 3) Outcomes (stored)

**Where:** store outcomes in existing domain tables + `ai_jobs.result_payload` for provenance

For v1, outcomes are primarily:

* email draft versions (subject/body + rationale + confidence)
* professor summary + key hooks
* alignment report
* document processed_content + missing_fields
* apply-confirmed change event

**Rule:** outcomes must reference:

* `intent_id`, `plan_id`, `job_id`, `trace_id`
  So you can replay or audit.

---

### 4) Event log (immutable-ish)

**Where:** if you don’t want a new table immediately, store as append-only records in `chat_thread_messages` with role=`system|tool`, type=`tool_call|tool_result`, and a strict schema.

**Event types (v1 minimum):**

* `INTENT_RECEIVED`
* `PLAN_CREATED`
* `STEP_STARTED`
* `POLICY_DECISION`
* `STEP_SUCCEEDED`
* `STEP_FAILED`
* `OUTCOME_STORED`
* `USER_APPROVED` / `USER_REJECTED`
* `WORKFLOW_COMPLETED`

This is what makes chat non-central. The workflow is what happened.

---

## B) Kernel components to enforce Principle #1 (v1)

### 1) Intake Adapter(s)

**v1:** Chat Intake only (SSE endpoint)

**Contract:** convert free text into an Intent:

* parse: message + thread context
* infer: intent_type + minimal inputs
* never execute here

**Enforcement check:** intake layer cannot call operators except “create job” and “write event”.

---

### 2) Context Builder

**Purpose:** build a deterministic context package for planner/agents:

* thread state summary
* user profile + memory (typed)
* funding_request context if present
* professor entity if present
* document ledger items referenced

**Rule:** agents do not read DB directly. Context Builder does.

---

### 3) Planner

**Purpose:** Intent → Plan

* choose minimal steps
* declare required info
* declare policy tags

**v1**: planner can be a simple deterministic router + one LLM call for plan explanation if you want. But the output must be structured.

---

### 4) Executor (workflow runtime)

**Purpose:** run plan steps:

* create `ai_jobs` per step (or reuse one job with step subrecords; your call)
* call operators
* stream progress_toggle
* write events

**Hard rule:** executor is the only thing allowed to run operators.

---

## C) Operators you must implement for v1 to satisfy Principle #1

Minimal set (don’t exceed ~10):

1. `Thread.CreateOrLoad`
2. `Event.Append`
3. `Documents.Upload` (bytes → student_documents row + s3 path)
4. `Documents.Process` (doc → extracted text + processed_content + missing_fields)
5. `Professor.ResolveAndSummarize` (digest_id or url → summary/hooks)
6. `Alignment.Compute` (user + prof summary → alignment + rationale)
7. `Email.GenerateDraft` (context → draft v1)
8. `Email.ReviewDraft` (draft → score + issues + suggestions)
9. `Email.OptimizeDraft` (draft + edits → new draft)
10. `Workflow.ApplyDraftToRequest` (write to funding_requests/funding_emails + link attachments)

Even if some of these internally call agents/LLMs, expose them as operators to keep side effects and audit centralized.

---

## D) API surface (v1) that is “workflow-first”

### 1) Unified execution endpoint

`POST /v1/workflows/execute`
Payload:

* `thread_id`
* `intent` (or raw `message` for chat intake, but normalized before execution)
  Returns:
* `workflow_id` (job/plan id)
* SSE stream channel key

### 2) SSE channels

You said 3 channels; keep them but make them workflow-native:

* `/events` (immutable event stream)
* `/progress` (percent + step name)
* `/assistant` (human-readable streaming response, optional)

This is key: enterprise integrations will consume `/events` and `/outcomes`, not chat text.

### 3) Outcome fetch

`GET /v1/workflows/{workflow_id}/outcomes`
Returns canonical outcome objects (email draft, doc checklist, etc.)

### 4) Apply action

`POST /v1/workflows/{workflow_id}/apply`
This is your human gate.

---

## E) UI behavior rules (so chat doesn’t hijack the architecture)

### Rule 1: Chat is an intent authoring UI

Chat UI never directly “sends email” or “updates DB”.
It requests an intent: “Generate email draft”, then “Apply”.

### Rule 2: All user-visible artifacts must be outcomes

Email drafts, professor summaries, alignment reports, optimized versions must be retrievable via outcome endpoints even if chat UI disappears.

### Rule 3: Thread history is not business truth

Thread is a UX. The truth is:

* ledger events
* stored outcomes
* applied changes in domain tables

---

## F) Testing and guardrails (how you enforce this in code reviews)

### PR checklist (v1)

* [ ] Does this feature define an intent_type and outcome schema?
* [ ] Does it record events from start → finish?
* [ ] Are side effects only via operators?
* [ ] Can the workflow be replayed from stored intent/context?
* [ ] Can an API client run the same workflow without chat UI?

If any “no”, it violates Principle #1.

---

## G) Concrete v1 deliverable list for Principle #1

This is what you can implement in a week or two and call it compliant:

1. Intent schema + enum + validation
2. Plan schema + step executor
3. Event append + event stream
4. Outcomes returned in structured form
5. Chat intake as thin adapter
6. Apply gate endpoint

That’s the foundation. Everything else becomes capabilities built on top.