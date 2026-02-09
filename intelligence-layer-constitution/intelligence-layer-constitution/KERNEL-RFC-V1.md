# Kernel RFC v1 (Interface and Responsibilities)

This RFC captures **implementation-facing** Kernel interfaces and responsibilities. It is not constitutional law and may evolve without changing `CONSTITUTION.md`.

## 1) Scope

Define the Kernel’s stable runtime contracts for:

- intake → intent normalization
- auth/tenancy propagation
- context building
- planning
- execution (operators only)
- ledger writes
- streaming projections (events/progress/assistant)

Non-goals:

- business prompts or domain heuristics
- operator internals
- adapter-specific logic

## 2) Kernel responsibilities (v1)

The Kernel owns control-plane:

1. Intake and intent normalization
2. AuthContext creation and propagation (tenant + scopes)
3. ContextBundle construction (policy-filtered, hashed)
4. Planning (Plan schema validation; planner invocation)
5. Execution state machine (step transitions, retries, idempotency enforcement)
6. Policy evaluation (intake/plan/action/outcome/apply)
7. Ledger writes (events/jobs/outcomes + pointers to documents/memory/entities)
8. Streaming (derived views)

## 3) Proposed external API surface (v1)

These are suggested shapes. Endpoint paths are not constitutional; only the behaviors are.

### 3.1 Submit (unified entrypoint)

`POST /v1/kernel/submit`

Request (example):

```json
{
  "source": "chat",
  "tenant_id": 1,
  "principal": { "type": "student", "id": 88, "role": "user" },
  "thread_id": 4512,
  "scope": { "scope_type": "funding_request", "scope_id": 556 },
  "input": { "mode": "message", "text": "Generate an outreach email..." },
  "intent_hint": { "intent_type": "Funding.Outreach.Email.Generate", "inputs": { "goal": "initial_outreach" } },
  "constraints": { "tone": "professional-warm", "length": "short", "language": "en" }
}
```

Response (example):

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

Behavior requirements:

- returns after enqueueing workflow execution (not after completion)
- writes `INTENT_RECEIVED` and `PLAN_CREATED` (or `INTENT_REJECTED`)

### 3.2 Human gate decision (apply/approve/reject)

`POST /v1/kernel/workflows/{workflow_id}/gate`

Request (example):

```json
{
  "tenant_id": 1,
  "principal": { "type": "student", "id": 88, "role": "user" },
  "gate_id": "gate-s6",
  "decision": "approve",
  "payload": { "target": { "type": "Email.Draft", "outcome_id": "out-..." } }
}
```

Behavior requirements:

- records a gate decision event (`USER_APPROVED` / `USER_REJECTED`)
- resumes execution from the gated step using the original idempotency keys

### 3.3 Outcome fetch

`GET /v1/workflows/{workflow_id}/outcomes`

Returns typed outcome objects and their version lineage. Defaults to **reproduce** (read stored outcomes).

## 4) Streaming model (v1)

Three streams, all derived from ledgers:

1. `/events`: canonical event stream (truth projection)
2. `/progress`: derived progress view (step transitions + weights)
3. `/assistant`: user-facing narrative deltas (projection only)

## 5) Internal Kernel interfaces (stable seams)

These are conceptual service boundaries that should remain stable:

- `IntakeAdapter.submit(request) -> IntentDraft`
- `AuthZ.authorize(intent, principal) -> AuthContext`
- `ContextBuilder.build(intent, auth) -> ContextBundle`
- `Planner.plan(intent, context) -> Plan`
- `Policy.evaluate(stage, policy_context) -> PolicyDecision`
- `Executor.run(plan, context, auth) -> WorkflowResult`
- `Ledger.append_event(...)`, `Ledger.write_job(...)`, `Ledger.write_outcome(...)`

## 6) References

See:

- `3-the-kernels.md` for deep narrative + examples.
- `6-policy-engine.md` for policy stages and decision schema.
- `7-execution-semantics.md` for the step state machine and idempotency patterns.

