# CanApply Intelligence Layer (IL) - Deep Technical Overview and Frontend Integration Guide

This document explains the CanApply Intelligence Layer ("IL", "Layer 2") as a complete system: its goals, runtime model, persistence/ledgers, HTTP + SSE contracts, and how to integrate it into CanApply's Funding Outreach frontend (sidebar copilot).

Audience: CanApply Platform Frontend Engineers (and adjacent platform engineers coordinating integration).

This document is self-contained: you should not need to open any source files to understand how to use the system.

---

## 0) TL;DR (integration checklist)

To "feel/see the agent work" in the frontend you need:

1) `POST /v1/threads/init` (create/load a thread for a funding request).
2) `POST /v1/threads/{thread_id}/queries` (submit a user message; receive `query_id` + `sse_url`).
3) `GET {sse_url}` (stream events; render progress + outcomes; handle gates).
4) If you receive `action_required`, call `POST /v1/actions/{action_id}/resolve` to resume.
5) Stop when you see a terminal SSE event: `final_result` or `final_error` (or cancellation).

Critical rule: treat the **SSE stream as the source of truth** for success/failure. The submit endpoint returns quickly and preflight failures may appear only on SSE.

---

## 1) What the Intelligence Layer Is (and is not)

### 1.1 It is

The Intelligence Layer is a workflow-centric AI runtime that converts user/system requests into auditable results under explicit policy and budget constraints:

- Workflow-first: every request becomes a typed **Intent**.
- Plans: every intent executes as a **Plan** (steps) composed of **Operators**.
- Ledgers: everything is recorded (intents, plans, events, outcomes, policy decisions, gates).
- Streaming: events are projected to clients via **Server-Sent Events (SSE)**.
- Side effects: only **operators** can perform effects (DB writes, sending email, external calls), and those are idempotent and policy-gated.

### 1.2 It is not

- Not "just chat". Chat is one UI adapter over a workflow engine.
- Not a replacement for the CanApply Platform DB. Domain truth remains in the platform DB; IL stores execution truth and produces proposals for domain writes.
- Not "best-effort LLM responses". It produces typed, schema-validated outcomes and an event timeline.

---

## 2) Key Concepts and IDs (frontend-relevant)

These terms appear in API payloads and SSE events.

### 2.1 Thread

A **thread** is a UX container that groups workflows for a single user and scope. In Funding Outreach, the scope is typically:

- `student_id`
- `funding_request_id`

A thread is stable across page refreshes; you call `/v1/threads/init` to get it.

### 2.2 Workflow (aka Query)

A **workflow** is one execution instance, identified by a UUID (`workflow_id`). In the adapter API the returned `query_id` is the workflow id.

The workflow is the unit of:

- streaming (`/v1/workflows/{workflow_id}/events`)
- policy decisions
- operator jobs
- outcomes
- credits reservation/settlement
- approvals (gates)

### 2.3 Intent

An **intent** is the normalized, typed meaning of the user's request, e.g.:

- `Funding.Outreach.Email.Review`
- `Funding.Outreach.Email.Generate`
- `Funding.Request.Fields.Update`

Intents are schema-validated. If required fields are missing/invalid, the workflow fails deterministically (and you'll see `final_error`).

### 2.4 Plan and Step

A **plan** is an explicit list of steps that implement an intent. Each step has a `step_id` and a kind, commonly:

- policy check step (ensure prerequisites)
- operator step (run operator implementation)
- gate step (pause for human approval / missing input)

### 2.5 Operator job

An **operator** is an idempotent unit of work with versioning, contract-validated input/output, and declared effects (for policy).

Examples:

- load platform context for the funding request
- review a stored email draft
- generate conversation suggestions
- propose a patch to platform fields (human-gated)

Operators emit job lifecycle events (`job_started`, `job_completed`, `job_failed`) and often produce outcomes.

### 2.6 Outcome

An **outcome** is a typed artifact recorded as part of the workflow result. For the Funding Outreach copilot you'll commonly see:

- `Email.Review`: verdict/score/issues/suggested subject
- `Conversation.Suggestions`: suggested follow-up prompts for the UI chips
- `PlatformPatch.Proposal`: a structured proposal to update platform domain state

### 2.7 Event

An **event** is an append-only record of what happened (progress stages, policy decisions, job lifecycles, gates, terminals).

The SSE stream is a projection of `ledger.events`.

### 2.8 Gate / Action Required

A **gate** is a pause point that requires the frontend (human) to approve or provide missing info. When the kernel hits a gate it emits `action_required` with:

- `action_id` (UUID) to resolve
- `action_type` (e.g., `collect_fields`, `apply_platform_patch`)
- metadata/payload describing what is needed

Frontend must call `POST /v1/actions/{action_id}/resolve` with `accepted` or `declined` plus optional payload.

---

## 3) Architecture Overview (how the system fits together)

### 3.1 Runtime components

At runtime, the system is:

- **IL API service (FastAPI)**: the HTTP adapter endpoints + SSE endpoints.
- **Workflow kernel**: plans and executes steps, records ledgers, produces events/outcomes.
- **Operators**: plugin-like implementations executed by the kernel.
- **Postgres**: system of record for runtime + ledger + billing tables.
- **Platform DB (MySQL/MariaDB)**: CanApply's domain state; IL reads it for context and writes only through gated apply steps.

Redis is present in the docker-compose as an optional dependency (commonly used for queues/caches), but the core system of record is Postgres.

### 3.2 Three layers in this repo

This repo contains three conceptual layers:

- Layer 0 (`llm-client`): provider abstraction/tool execution and structured outputs.
- Layer 1 (`agent-runtime`): orchestration primitives for jobs/actions/events (legacy path).
- Layer 2 (this IL): constitution-driven workflow kernel + CanApply Funding Outreach capabilities.

Frontend should treat Layer 2 as the product integration surface.

---

## 4) Persistence: what is stored (and what this means for UX)

### 4.1 Postgres is the execution source of truth

Postgres stores the "execution truth" in two groups:

1) Runtime control plane tables:
   - threads
   - workflow runs + step states
2) Ledger tables:
   - intents, plans, policy decisions
   - events (streamed to SSE)
   - outcomes (typed artifacts)
   - gates (waiting approvals)

This enables:

- Refresh/reconnect without losing "what happened".
- Audits: render a timeline from recorded events.
- Reproduce-by-default: show stored outcomes without rerunning models.

### 4.2 Platform DB remains domain truth

The platform DB contains funding requests, professors, drafted emails, reminders, replies, and attachments. The IL loads this context and can propose updates via a patch proposal + human gate, but it should not silently mutate platform truth without a gated apply.

Frontend implication: after accepting a patch/apply gate, you should refresh the platform view (or refetch platform context) because the domain state may change.

---

## 5) Execution semantics you need to handle in the frontend

### 5.1 Success is a terminal event, not an HTTP status

The submit endpoint returns quickly with a `query_id` + `sse_url`. Preflight failures (auth, credits, validation) may still yield HTTP 200 and only surface as SSE terminal events:

- `final_result` = success
- `final_error` = failure

Frontend must always stream the SSE.

### 5.2 Workflow states

Your UI should model a workflow as:

- `running`: streaming progress/job events
- `waiting`: `action_required` gate received; workflow blocked
- `completed`: `final_result`
- `failed`: `final_error`
- `cancelled`: cancellation terminal event

### 5.3 Idempotency and "retry vs regenerate"

You have three user intents in UI terms:

- Reconnect (network blip): re-open SSE for the same workflow id; do not submit a new workflow.
- Retry/resume (after gate): resolve the gate action; do not submit a new workflow.
- Regenerate (new output version): submit a new workflow; optionally pass a new `query_id`.

If you want request-level dedupe, you may supply a client-generated UUID as `query_id`.

### 5.4 At-least-once streaming

The SSE workflow stream can replay earlier events on reconnect. Your reducer should be idempotent:

- track `event_id` or `event_no` to dedupe
- do not assume "exactly once"

---

## 6) HTTP API Contract (Layer 2 adapter)

All endpoints are under `/v1`.

### 6.1 `POST /v1/threads/init`

Purpose: create or load a thread for a funding request.

Request:

```json
{
  "funding_request_id": 88,
  "student_id": 1583,
  "client_context": { "ui_route": "/funding/88" }
}
```

Response:

```json
{
  "thread_id": "1",
  "thread_status": "active",
  "is_new": false,
  "message": "existing_thread",
  "onboarding_gate": "ready",
  "missing_requirements": []
}
```

Notes:

- In the current prototype, identity can come from `x-student-id`/`x-principal-id` headers or from a dev bypass. For production, prefer same-origin cookie or bearer auth so SSE can be authenticated reliably.

### 6.2 `POST /v1/threads/{thread_id}/queries`

Purpose: submit a message and start a workflow.

Request:

```json
{
  "message": "review my email draft",
  "attachments": [],
  "query_id": "optional-uuid-for-idempotency"
}
```

Response:

```json
{
  "query_id": "workflow-uuid",
  "sse_url": "/v1/workflows/workflow-uuid/events"
}
```

Notes:

- The API may accept and return even if the workflow will fail in preflight; failure will be streamed via SSE.
- Use `query_id` if you want a deterministic client id for retries/dedupe.
- Treat `sse_url` as an opaque URL provided by the server. Depending on server configuration it may point to `/v1/workflows/{id}/events` (workflow kernel path) or `/v1/queries/{id}/events` (legacy runtime path).

### 6.3 `GET /v1/workflows/{workflow_id}/events` (SSE)

Purpose: stream the event timeline for a workflow.

Each SSE message:

- `event: <event_name>` (e.g. `progress`, `job_started`, `action_required`, `final_result`)
- `data: <json payload>`

The payload includes:

```json
{
  "event_no": 6287,
  "event_id": "uuid",
  "type": "progress",
  "timestamp": 1770300725.650985,
  "workflow_id": "uuid",
  "thread_id": 1,
  "intent_id": "uuid-or-null",
  "plan_id": "uuid-or-null",
  "step_id": "s3-or-null",
  "job_id": "uuid-or-null",
  "correlation_id": "uuid",
  "severity": "info",
  "actor": { "role": "student", "principal": { "id": 1583, "type": "student" } },
  "data": { "stage": "step_completed", "name": "Email.ReviewDraft" }
}
```

Terminal events:

- `final_result`
- `final_error`
- `job_cancelled`

Event naming note:

- The SSE `event:` name is derived from the canonical event type by replacing `.` with `_` (example: `job.started` becomes `event: job_started`).
- The JSON payload field `type` may still contain the canonical dotted name (`job.started`, `policy.decision`, etc.). Prefer `event_no` for ordering and the `event:` name for routing if you use the browser EventSource API.

### 6.3.1 `/v1/workflows/...` vs `/v1/queries/...` (why there are two)

You may encounter two SSE endpoints in this repo because there are two execution backends:

- `GET /v1/workflows/{workflow_id}/events`
  - "Workflow SSE" (newer): streams from the IL ledger (`ledger.events`), produced by the workflow kernel.
  - Has `event_no` for deterministic ordering.
  - Emits IL-native event types like `progress`, `policy.decision`, `job.started`, `action_required`, `final_result`, `final_error`.

- `GET /v1/queries/{query_id}/events`
  - "Query SSE" (legacy): streams events produced by the older agent runtime (Layer 1).
  - Does not have `event_no`; ordering is based on timestamps.
  - Emits agent-runtime event types like `job.started`, `job.status_changed`, `model.token`, `tool.start`, `final.result`, `final.error`.

Frontend rule:

- Treat the server-provided `sse_url` as authoritative and open it as-is.
- If you want to branch behavior, detect the path prefix (`/v1/workflows/` vs `/v1/queries/`) and use the relevant event catalog below.

### 6.4 `POST /v1/actions/{action_id}/resolve`

Purpose: approve/decline gates and provide missing data.

Request:

```json
{ "status": "accepted", "payload": {} }
```

or:

```json
{ "status": "declined", "payload": {} }
```

When the action requires user input, `payload` must include the requested values.

### 6.5 `POST /v1/workflows/{workflow_id}/cancel`

Purpose: cancel a running workflow.

Request:

```json
{ "reason": "user_cancelled" }
```

### 6.6 `GET /v1/queries/{query_id}/events` (legacy SSE)

Purpose: stream events for the legacy execution backend.

This stream uses a slightly different payload shape than the workflow SSE stream. The payload includes:

- `event_id`
- `type` (canonical dotted name, e.g. `job.started`, `model.token`, `final.result`)
- `timestamp`
- `job_id`, `run_id`
- `trace_id`, `span_id`
- `scope_id`, `principal_id`, `session_id`
- `data` (event-specific payload)
- `schema_version`

Important difference:

- The SSE `event:` name is `type` with `.` replaced by `_` (example: `final.result` becomes `event: final_result`).

---

## 7) SSE event types (what to implement in the frontend)

### 7.1 `progress`

Used to drive status UI. Common stages:

- `query_received`
- `checking_auth`
- `reserving_credits`
- `classifying_intent`
- `workflow_started`
- `loading_context`
- `running_operator:<operator_name>`
- `step_completed`
- `completed`

Treat progress as UI hints; the authoritative results are outcomes in `final_result`.

### 7.2 `policy_decision`

Policy decisions are recorded events that describe allow/deny outcomes for:

- plan stage (is this intent/plan allowed?)
- action stage (is this operator invocation allowed?)
- outcome stage (is this outcome allowed to be emitted/egressed?)

Frontend uses:

- Optional: show an "Explain why" panel for blocked workflows.
- Optional: an internal "audit timeline" view for support/debug.

### 7.3 `job_started` / `job_completed` / `job_failed`

Operator execution boundaries. These are useful for:

- showing which step/operator is currently running
- diagnosing failures (you will typically see a `job_failed` prior to a terminal `final_error`)

### 7.4 `action_required`

The workflow is blocked until the client resolves `action_id`.

Typical shapes:

1) Missing input: `action_type = "collect_fields"`
   - `data.missing_fields`: a list of missing flags/fields
   - UI: collect the missing values or route the user to complete onboarding
   - resolve: `POST /v1/actions/{action_id}/resolve` with `accepted` and `payload` containing the values

2) Approval gate: `action_type = "apply_platform_patch"`
   - `proposed_changes`: a structured patch proposal
   - UI: show a clear diff/summary, risk level, and an explicit approve/decline choice
   - resolve: accept/decline (optionally include approval metadata in payload)

Other action types may exist (depending on capability), and the frontend should be forward-compatible:

- `upload_required`: prompt the user to upload a document (or route to an upload UI)
- `select_option`: present a fixed set of choices
- `confirm`: confirmation dialog (yes/no)
- `redirect`: route the user to a page/flow
- `refresh_ui`: prompt the client to refresh/refetch platform state

### 7.5 `final_result`

Terminal success. The payload contains `data.outputs`, which is a map of named outputs (outcomes). Example:

```json
{
  "status": "success",
  "outputs": {
    "email_review": { "payload": { "verdict": "pass", "overall_score": 0.75, "issues": [] } },
    "suggestions": { "payload": { "suggestions": [ { "text": "Generate a shorter version.", "type": "followup" } ] } }
  }
}
```

Frontend rule:

- Render these typed outcomes (not a raw assistant paragraph) wherever possible.

### 7.6 `final_error`

Terminal failure. Example:

```json
{ "error": "insufficient_credits" }
```

Frontend rule:

- Show the error as a blocked state and provide a next action (buy credits, complete onboarding, re-authenticate, etc.).

### 7.7 Legacy Query SSE: event catalog (agent-runtime backend)

If `sse_url` points to `/v1/queries/{id}/events`, you may see these event types.

Progress:

- `progress` (SSE `event: progress`)
  - `data` may include fields like `progress` (0..1), `stage`, `message`, or other progress metadata depending on the execution.

Job lifecycle:

- `job.started` (SSE `event: job_started`)
  - `data.status` is typically `queued`.
- `job.status_changed` (SSE `event: job_status_changed`)
  - `data.status`: `queued` | `running` | `succeeded` | `failed` | `cancelled`
  - `data.previous_status`
  - `data.progress` (optional)
  - `data.error` (optional)
- `job.completed` (SSE `event: job_completed`)
  - indicates success
- `job.failed` (SSE `event: job_failed`)
  - indicates failure
- `job.cancelled` (SSE `event: job_cancelled`)
  - indicates cancellation (terminal)

Model streaming:

- `model.token` (SSE `event: model_token`)
  - `data.token`: a partial token/chunk of assistant output
- `model.reasoning` (SSE `event: model_reasoning`)
  - optional reasoning content (if the model/provider exposes it)
- `model.done` (SSE `event: model_done`)
  - indicates the model response has completed (often followed by `final.result`)

Tool execution:

- `tool.start` (SSE `event: tool_start`)
  - includes tool name/arguments in `data` (shape depends on tool)
- `tool.end` (SSE `event: tool_end`)
  - includes tool result in `data`
- `tool.error` (SSE `event: tool_error`)
  - includes error details in `data`

Gates:

- `action.required` (SSE `event: action_required`)
  - indicates the runtime needs user input or approval
  - resolve via `POST /v1/actions/{action_id}/resolve`
- `action.resolved` (SSE `event: action_resolved`)
  - indicates a previously-required action was resolved
- `action.expired` (SSE `event: action_expired`)
  - indicates an action timed out/expired
- `action.cancelled` (SSE `event: action_cancelled`)
  - indicates an action was cancelled (often followed by job cancellation)

Artifacts:

- `artifact.created` (SSE `event: artifact_created`)
  - indicates a new artifact was produced (file/report/diff)
- `artifact.updated` (SSE `event: artifact_updated`)
  - indicates an existing artifact was modified/updated

Terminal results:

- `final.result` (SSE `event: final_result`)
  - `data.content`: final assistant text content (legacy)
  - `data.usage`: usage/cost info (if enabled)
- `final.error` (SSE `event: final_error`)
  - `data.error`: error string

Frontend rule:

- Prefer the workflow SSE path for typed outcomes. The legacy query SSE path is primarily a "token stream + final text" interface.

---

## 8) Auth model (what the frontend should expect)

### 8.1 Current prototype semantics

The system supports a development-style identity adapter:

- `x-student-id` or `x-principal-id` header: explicit identity in dev tooling/curl.
- Optional `IL_AUTH_BYPASS=true`: in dev mode, the server can derive identity from the funding request context.

In the current design, submit-query preflight auth failures can be returned via SSE as `final_error` while the HTTP submit returns a `query_id` + `sse_url`. This makes the workflow stream the single place to observe success/failure.

### 8.2 Production integration recommendation

For a browser frontend:

- `EventSource` cannot attach arbitrary headers.
- Therefore, header-based auth for SSE is not robust.

Recommended:

1) Use same-origin cookie auth (session cookie) so both HTTP and SSE carry auth implicitly, or
2) Stream via `fetch()` and parse SSE manually, attaching `Authorization` headers.

Security note: in production, SSE endpoints should be authenticated and scoped (tenant + principal + thread ownership). Treat `workflow_id` as sensitive.

---

## 9) Credits (AI budget enforcement)

Credits are enforced per workflow:

1) On submit-query, IL estimates credits and reserves them (a "hold").
2) During execution, usage events are recorded.
3) On completion/cancellation, IL settles the reservation and writes an immutable ledger entry.

Frontend implications:

- A workflow can fail with `final_error` due to insufficient credits even if the submit HTTP request succeeded.
- If you want to display "credits remaining" in the product, the canonical source should be the platform backend (or a dedicated IL read endpoint), not the SSE stream.

---

## 10) Funding Outreach copilot: what the UI should show

### 10.1 The MVP "email review" slice

User story:

- The user is on the Funding Outreach page for a specific `funding_request_id`.
- The sidebar copilot can review the already-drafted outreach email stored in platform context.

Frontend sequence:

1) `POST /v1/threads/init` for the funding request.
2) Submit: `message = "review my email draft"`.
3) Stream SSE and show:
   - progress stages
   - operator jobs (context load, email review, suggestions)
4) On `final_result`, render:
   - `Email.Review` outcome (score, verdict, issues, suggested subject)
   - `Conversation.Suggestions` outcome as quick-prompt chips

### 10.2 Actionable suggestions UX

Suggestions are intended to be rendered as:

- clickable chips that submit a new query
- or buttons that open a refined prompt composer ("shorten email", "draft follow-up", etc.)

These chips help users understand what the copilot can do next without needing prompt engineering.

---

## 11) Frontend implementation guidance (practical patterns)

### 11.1 Suggested state model

Maintain state derived from events:

- `threadId`
- `activeWorkflowId`
- `status`: running | waiting | completed | failed | cancelled
- `progressStage`
- `actionRequired` (if any): latest action gate descriptor
- `outcomes`: map keyed by output name (`email_review`, `suggestions`, etc.)
- `events` (optional): for an audit timeline UI

### 11.2 Handling multiple concurrent workflows

Users may submit multiple queries. Recommended:

- Keep a per-workflow reducer/state keyed by `workflowId`.
- UI can show one "active" workflow while keeping history for prior workflows.

### 11.3 SSE consumption options

Option A: `EventSource` (simple, cookie auth)

- Pros: easy, browser-native reconnection
- Cons: cannot attach custom headers; needs same-origin cookies or query-param auth (not recommended)

Option B: `fetch()` + manual parsing (recommended for bearer auth)

- Pros: can attach `Authorization` and custom headers; fully controlled reconnection and de-dupe
- Cons: requires SSE parsing and reconnect logic

### 11.4 De-dupe strategy on reconnect

Because streams can replay earlier events:

- track `event_id` in a Set, or
- track `maxEventNo` and ignore any event with `event_no <= maxEventNo`

Using `event_no` is typically sufficient for a single workflow stream.

### 11.5 Gate UX checklist

For `action_required`:

- Always show `title`, `description`, and `reason_code` (if provided).
- For patch proposals:
  - Show each target: table, where clause, and fields being set.
  - Show `risk_level` and require explicit confirmation for medium/high.
- Always provide a "Decline" path (decline resolves gate and may cancel the workflow).

---

## 12) Observability and debugging (frontend + support)

### 12.1 Minimal "debug timeline" view

If you add an internal debug mode in the UI, a very effective support tool is a timeline that shows:

- progress stages
- operator job boundaries
- action_required gates
- terminal results/errors

Because events are normalized, this is more useful than raw chat transcripts for diagnosing issues.

### 12.2 Typical failure causes and UI messaging

- `intent validation failed`: show "I need more info" and ask for the missing field(s).
- `policy denied`: show "This action is not allowed" and the reason code (if safe).
- `insufficient_credits`: show upgrade/top-up CTA.
- operator failure (generic): show "Something went wrong; try again" and provide a "Report issue" with workflow id.

Include `workflow_id` in user-visible error details for support.

---

## 13) Extending the system (what changes mean for frontend)

When backend adds a new capability, frontend impact typically falls into:

- new `intent_type` that becomes reachable from natural language classification
- new outcomes in `final_result.outputs`
- new gate types (`action_type`) requiring a new UI panel

Frontend should be written to be forward-compatible:

- ignore unknown event types
- ignore unknown fields
- treat `final_result` outputs as an open map (feature-detect keys)

---

## 14) Local dev flags (useful for integration testing)

Common environment flags:

- `IL_USE_WORKFLOW_KERNEL=true`: use the constitution-style workflow kernel for `/v1/threads/{id}/queries`.
- `IL_DEBUG=true`: enable debug-only endpoints and overrides.
- `IL_AUTH_BYPASS=true`: dev-only identity bypass.
- `IL_CREDITS_BOOTSTRAP=true`: auto-seed credits for new principals.
- `IL_CREDITS_BOOTSTRAP_AMOUNT=1000`: initial credits amount.
- `IL_CREDITS_MIN_RESERVE=1`: minimum reservation per workflow.
- `IL_CREDITS_RESERVATION_TTL_SEC=900`: reservation TTL.

### 14.1 Local smoke test (curl)

This is a minimal end-to-end check that lets you see the agent "work" via SSE.

1) Init/load a thread (replace IDs):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/threads/init \
  -H "Content-Type: application/json" \
  -H "x-student-id: 1583" \
  -d '{"funding_request_id": 88}'
```

2) Submit a query:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/threads/1/queries \
  -H "Content-Type: application/json" \
  -H "x-student-id: 1583" \
  -d '{"message":"review my email draft"}'
```

3) Stream events (`curl -N` is important to disable buffering):

```bash
curl -N "http://127.0.0.1:8080/v1/workflows/<QUERY_ID>/events"
```

Expected: a sequence of `progress` -> operator job events -> optional `action_required` -> terminal `final_result` (or `final_error`).

---

## 15) Summary (what the frontend can rely on)

- The Intelligence Layer is a workflow engine, not a chat transcript generator.
- The submit-query HTTP response is only a handle; the SSE stream is authoritative.
- Results arrive as typed outcomes in `final_result.outputs`.
- Human approvals and missing inputs appear as `action_required` and are resolved via `/v1/actions/{id}/resolve`.
- Credits and policy are first-class and can fail the workflow with a terminal `final_error`.
