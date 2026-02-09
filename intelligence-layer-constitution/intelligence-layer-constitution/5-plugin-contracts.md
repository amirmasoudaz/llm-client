# 5. Plugin Contracts Deep Spec

## 5.0 Plugin taxonomy and why it matters

A plugin is any swappable module that the Kernel can invoke by name + version.

You have four plugin types:

1. **Agents**: decision-making, planning, reasoning, text production (no side effects)
2. **Operators**: execution units that may perform side effects (DB writes, Gmail send, exports)
3. **Tools**: pure, read-only helpers (parse, score, retrieve, classify)
4. **Adapters**: glue between external systems and Kernel intents/outcomes (no business rules)

The enforcement goal:

* Agents can “think”, Operators can “do”, Tools can “compute”, Adapters can “translate”.

If any plugin crosses its boundary, your system becomes un-debuggable.

---

# 5.1 Plugin packaging and registration (the boring part that saves you)

Every plugin MUST ship with a **manifest** that the Kernel reads at startup (or from registry).

### 5.1.1 Universal plugin manifest

```json
{
  "name": "Email.GenerateDraft",
  "type": "operator",
  "version": "1.2.0",
  "owner": "canapply",
  "description": "Generate a funding outreach email draft",
  "schemas": {
    "input": "schemas/email_generate_input.v1.json",
    "output": "schemas/email_generate_output.v1.json"
  },
  "requires": {
    "tools": ["Doc.Summarize@1.x", "Text.Dedupe@1.x"],
    "capabilities": [],
    "context_fields": ["student.profile", "professor.entity", "request.scope"]
  },
  "policy_tags": ["draft_only", "pii_allowed"],
  "observability": {
    "log_level": "info",
    "emit_samples": false
  }
}
```

### 5.1.2 Registry contract

Kernel must expose:

* `resolve_plugin(name, version_constraint) -> plugin_handle`
* `list_plugins(type?) -> []`
* `validate_schema(input, schema_ref) -> ok|error`

**Rule:** Kernel never calls “random code”. It calls only registered plugins with known schemas.

---

# 5.2 Agent Contract (deep)

## 5.2.1 What “stateless” actually means

Stateless does NOT mean “doesn’t use memory”. It means:

* Agent cannot write to DB
* Agent cannot store hidden state across calls
* Anything it “remembers” must come from **context package** (read-only) or memory ledger already fetched by Kernel

If an agent wants to “update memory”, it must propose a **MemoryWrite action** for an operator, and policy can gate it.

## 5.2.2 Agent I/O contract

### Inputs (required)

* `intent` (typed, validated)
* `context` (read-only, includes only what policy allows)
* `policy_snapshot` (read-only)
* `tool_registry` (available tools/operators it’s allowed to propose)
* `constraints` (tone/length/etc.)
* `runtime` (limits: tokens/time/credits)

### Output (required)

An agent returns one of:

* `plan_proposal` (for planners/switchboard agents)
* `step_output` (for content-producing agents)
* `next_step_suggestions` (structured)

#### Agent output schema

```json
{
  "agent_name": "Switchboard",
  "agent_version": "1.0",
  "kind": "plan_proposal",
  "confidence": 0.78,
  "rationale": {
    "short": "User wants an outreach draft. Need professor summary and user's CV highlights.",
    "signals": ["intent_type:Funding.Outreach.Email.Generate", "has_professor_id", "has_resume_doc"]
  },
  "required_actions": [
    {
      "operator": "Professor.Summarize",
      "payload": { "professor_id": 910, "source": "canspider" },
      "policy_tags": ["read_only"],
      "idempotency_hint": "prof_sum:{professor_id}:{digest_hash}"
    }
  ],
  "proposed_plan": {
    "steps": [
      { "step_id": "s1", "action": "Professor.Summarize", "gate": "none" },
      { "step_id": "s2", "action": "Email.GenerateDraft", "gate": "human_apply" }
    ]
  },
  "suggestions": [
    { "type": "followup", "text": "Do you want a short or medium-length email?" }
  ]
}
```

## 5.2.3 Agent declarations (enforced at runtime)

Agents MUST declare:

1. **required context fields**

* Example: `["student.profile", "professor.entity", "documents.cv.processed_content"]`

2. **produced output schema**

* Example: `"schemas/draft_email.v1.json"`

3. **allowed actions**

* Agents are only allowed to propose operators/tools in a whitelist.

Kernel behavior:

* If required context fields are missing: agent returns `NEEDS_CONTEXT` not “hallucinate”.
* If agent proposes a non-allowed operator: Kernel rejects plan with `POLICY_DENIED`.

## 5.2.4 Agent determinism: your realistic policy

Agents are not fully deterministic (LLMs), so we do this:

Agents MUST:

* emit a `confidence`
* emit `rationale.signals` grounded in context keys (not raw chain-of-thought)
* optionally emit `randomness_disclosure` if temperature > 0

Kernel MUST:

* record agent output in Job Ledger with model + parameters
* treat agent output as a proposal, not truth, unless policy allows auto-exec

---

# 5.3 Operator Contract (deep)

## 5.3.1 Operators are “effects with rules”

Operators are the only modules that can:

* mutate DB
* call external systems (Gmail, enterprise APIs)
* write to S3 “final”
* trigger webhooks

## 5.3.2 Operator function signature

Operator MUST accept:

* `payload` (typed)
* `idempotency_key`
* `auth_context` (scoped permissions)
* `trace_context` (correlation/workflow/step)
* `policy_snapshot` (optional but recommended so operator can enforce last-mile checks)

```json
{
  "payload": { "..." : "typed" },
  "idempotency_key": "string",
  "auth_context": { "tenant_id": 1, "principal_id": 88, "scopes": ["email:generate"] },
  "trace_context": { "workflow_id": "...", "step_id": "s3", "correlation_id": "..." }
}
```

## 5.3.3 Operator return schema

```json
{
  "status": "succeeded|failed|in_progress",
  "result": { "..." : "typed" },
  "artifacts": [
    { "type": "s3_object", "path": "s3://.../file.pdf", "hash": "..." }
  ],
  "metrics": {
    "latency_ms": 912,
    "tokens_in": 1200,
    "tokens_out": 380,
    "cost_total": 0.0123,
    "provider": "openai"
  },
  "error": null,
  "nondeterminism": {
    "is_nondeterministic": true,
    "reasons": ["llm_sampling"],
    "stability": "high|medium|low"
  }
}
```

## 5.3.4 Mandatory operator rules (hard)

### A) Idempotency

Under the same `idempotency_key`, operator MUST:

* return the exact same result if already succeeded
* never repeat side effects

This implies operator must store an idempotency record keyed by:

* `(tenant_id, operator_name, idempotency_key)`

### B) Side effects must be declared

Operators must declare:

* `effects: ["db_write", "gmail_send", "webhook_emit", "s3_final_write"]`

Kernel policy engine can then gate by effects.

### C) Determinism disclosure

If operator calls nondeterministic third parties (LLM sampling, web content), it MUST set `nondeterminism.is_nondeterministic=true`.

### D) Input validation

Operator MUST validate payload against schema and return `VALIDATION_ERROR` if invalid.

### E) Permission enforcement

Operators must enforce `auth_context.scopes`, even if Kernel already did, because “defense in depth”.

---

# 5.4 Tool Contract (deep)

Tools are:

* read-only retrieval
* pure computation
* parsing
* scoring/classification

**Tools MUST NOT**

* write DB
* send email
* call external side-effecting APIs (if it does, it’s an operator)

## 5.4.1 Tool signature

```json
{
  "payload": { "..." : "typed" },
  "trace_context": { "...": "..." }
}
```

## 5.4.2 Tool return schema

```json
{
  "result": { "...": "typed" },
  "metrics": { "latency_ms": 40 },
  "error": null
}
```

**Tool caching:** encouraged. Tools are the perfect caching layer because they’re pure.

---

# 5.5 Adapter Contract (deep)

Adapters are translators between:

* external events/messages ↔ Kernel intents
* Kernel outcomes ↔ external writes (via operators)

Adapters are not a second brain. They are pipes.

## 5.5.1 Adapter inbound: external → intent

`map_external_event_to_intent(external_event) -> KernelSubmitRequest`

Examples:

* Salesforce case created → `Support.Ticket.Resolve`
* University portal webhook → `Admission.Application.StatusUpdate`
* Slack mention in enterprise workspace → `Support.QA.Answer`

## 5.5.2 Adapter outbound: outcomes → external writes

Adapters must never write directly.
They must request an operator execution:

* `ExternalSystem.WriteOutcome`

## 5.5.3 Adapter auth requirements

Adapters MUST:

* translate external user identity into Kernel principal
* scope permissions using Kernel auth model
* never bypass policy (no “admin by adapter” hacks)

---

# 5.6 Policy tags and effect gating (the glue that makes this safe)

Every plugin must declare:

* `policy_tags` (semantic labels: `draft_only`, `external_send`, `pii_access`, `writes_final`)
* `effects` (actual action kinds)

Kernel policy engine uses both:

* tags for intent-level rules
* effects for risk gating

Example policy:

* Allow `Email.GenerateDraft` always.
* Require human gate for `Email.ApplyDraft` (writes final).
* Deny `Gmail.SendEmail` unless user explicitly enabled and has consent.

---

# 5.7 Plugin versioning and compatibility rules

This is how you avoid breaking enterprise integrations.

### Semantic versioning

* MAJOR: breaking schema
* MINOR: additive fields, new optional outputs
* PATCH: bug fixes, no schema change

### Compatibility guarantees

* Kernel must be able to run older plugin versions for existing workflows if needed
* Outcome schemas must include `schema_version` so older outcomes remain readable

---

# 5.8 Observability requirements (plugins must be debuggable)

Every plugin execution must emit to Job Ledger:

* plugin name + version
* payload hash
* result hash
* metrics
* error taxonomy
* nondeterminism flags

Every plugin must support:

* structured logs (not raw print dumps)
* correlation_id propagation

---

# 5.9 Testing contracts (enforcement, not vibes)

You should require these tests per plugin type:

### Agents

* schema validation tests
* “required context fields” tests (returns NEEDS_CONTEXT)
* “no forbidden operators proposed” tests

### Operators

* idempotency tests (same key twice)
* permission tests (scope denied)
* determinism disclosure tests
* side effect declaration tests

### Tools

* purity tests (no side effects)
* caching correctness tests

### Adapters

* mapping tests (external event → correct intent schema)
* policy enforcement tests (cannot bypass)

---

# 5.10 Practical v1 plugin set (minimum)

For Funding Outreach v1, you likely need:

**Agents**

* Switchboard Agent
* Email Draft Agent (optional if draft is operator-driven)
* FollowUp Suggestion Agent (optional)

**Operators**

* Professor.Summarize (read-only)
* Document.Process (doc → json)
* Email.GenerateDraft
* Email.ReviewDraft
* Email.OptimizeDraft
* Apply.EmailDraft (writes to funding_emails/funding_requests)
* Gmail.Send (optional, gated hard)

**Tools**

* Text.Score (quality rubric)
* Similarity.Match (professor alignment signals)
* Template.Fill (pure formatting)

**Adapters**

* Platform Chat Interface Adapter (chat)
* Platform Backend Adapter (credits, auth)
* Gmail Adapter (OAuth flow callback)
* etc.