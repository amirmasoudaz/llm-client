# Feature Development Walkthrough (Schemas → Manifests → Plan Templates → Operators) — v1

This repo is **contracts + blueprints + runnable demos** for the CanApply Intelligence Layer.

If you want to build the full system, the core idea is:

> Define **schemas** first (what “valid” means), wire them into **manifests** (what exists), write a deterministic **plan template** (what steps run), then implement the **operators** (where real work/side effects happen).

This document is a practical walkthrough so you can add features without getting confused.

---

## 0) The mental model (one picture)

When a user sends a message (via frontend or platform backend):

1) **Switchboard** decides what the user wants (intent detection)
2) The Kernel normalizes the request into a typed **Intent** (JSON object) and validates it
3) The Kernel selects a **Plan Template** for that intent (via manifests)
4) The Executor runs the plan step-by-step:
   - `policy_check` steps can pause the workflow and emit `action_required`
   - `operator` steps do the work (LLM calls, DB reads, S3 ops, platform patch proposals)
   - `human_gate` steps emit `action_required` (apply/confirm/upload/redirect) and pause
5) The system streams typed events (`progress`, `token_delta`, `action_required`, `final`)
6) When the user resolves a gate, the client submits a **Workflow.Gate.Resolve** intent, and the workflow resumes

Key v1 rule: the Intelligence Layer does **not** write to platform DB directly. It proposes patches and waits for approval (`apply_platform_patch` gate).

---

## 1) Repo map (what each folder does)

### 1.1 `schemas/` — “contracts” (versioned JSON Schema)

This folder defines the shapes of everything. If a value is not allowed by schema, the workflow should fail early.

- `schemas/common/`
  - Shared definitions used across all schemas (TenantId, Principal, Binding syntax, RiskLevel, CachePolicy, etc.).
  - Start here: `schemas/common/defs.v1.json`.
- `schemas/intents/`
  - Schemas for **Intent** records (“what the user asked for”, normalized).
  - Example: `schemas/intents/funding_outreach_email_optimize.v1.json`.
- `schemas/plans/`
  - Schemas for **plan templates** and plan steps.
  - Example: `schemas/plans/plan_template.v1.json`, `schemas/plans/step.v1.json`.
- `schemas/operators/`
  - Operator call/return **envelopes** + per-operator input/output schemas.
  - The input envelope always includes: `payload`, `idempotency_key`, `auth_context`, `trace_context`.
  - The output envelope always includes: `status`, `metrics`, and either `result` or `error`.
  - Example: `schemas/operators/email_optimize_draft.input.v1.json`, `schemas/operators/email_optimize_draft.output.v1.json`.
- `schemas/outcomes/`
  - Schemas for **Outcomes** (typed results stored/returned/replayed).
  - Example: `schemas/outcomes/email_draft.v1.json`, `schemas/outcomes/platform_patch_proposal.v1.json`.
- `schemas/sse/`
  - Schemas for streaming events (SSE):
    - `progress`, `token_delta`, `action_required`, `artifact_ready`, `final`, `error`.

Index: `schemas/INDEX.md`  
Conventions + validation boundaries: `schemas/README.md`

### 1.2 `manifests/` — discovery (what exists and how the kernel finds it)

The kernel should not hardcode what intents or operators exist. It discovers them via manifests.

- `manifests/intent-registry.v1.json`
  - Maps: `intent_type → schemas/intents/...`
- `manifests/capabilities/*.json`
  - “Capability” manifests group intents and map:
    - `intent_type → plan-templates/...`
    - dependencies (operators, agents)
    - required entities and outcomes
  - Example: `manifests/capabilities/FundingOutreach.1.0.0.json`
- `manifests/plugins/operators/*.json`
  - Per-operator plugin manifests (name + version + input/output schema refs + tags/effects).
  - Example: `manifests/plugins/operators/Email.OptimizeDraft.1.0.0.json`

Overview: `manifests/README.md`

### 1.3 `plan-templates/` — deterministic workflows per intent

Each plan template is a “program” the executor runs step-by-step.

- Must validate against `schemas/plans/plan_template.v1.json`
- Steps can be: `operator`, `policy_check`, `human_gate` (and `agent` is reserved for later)
- Step payloads can use **bindings**:
  - `{ "from": "context.platform.funding_request.id" }`
  - `{ "const": 123 }`
  - `{ "template": "some:{tenant_id}:{computed.hash}" }`

Read: `plan-templates/README.md`

### 1.4 `demo/` — runnable reference implementations (for learning)

These are intentionally minimal and are *not production-ready*. They exist to make the architecture tangible.

- `demo/kernel_email_review.py`
  - `SchemaRegistry` (loads/validates schemas)
  - `ManifestRegistry` (loads intent registry + capability plan mappings + operator manifests)
  - Binding resolution (`from/const/template`)
  - A small executor and a few operators/policy checks
- `demo/run_email_review_demo.py`
  - CLI run of the deterministic email review plan (no AI)
- `demo/kernel_ai_outreach.py`
  - Adds LLM-backed switchboard + LLM-backed email review/optimize operators
  - Streams `token_delta` events and triggers `apply_platform_patch` gate
- `demo/run_ai_outreach_chat.py`
  - Interactive CLI that mimics: “user query → switchboard → plan execution → stream → gate approval”
- `demo/fastapi_email_review_app.py`
  - Optional FastAPI wrapper showing the rough `/v1/...` + SSE shape

Start here: `demo/README.md`

### 1.5 `llm-client/` — LLM substrate module

This module provides provider abstraction, streaming, tool calling, retries/backoff, caching hooks, and some telemetry primitives.

In the demo, `demo/run_ai_outreach_chat.py` loads it dynamically when `--real` is used.

### 1.6 Root docs (“laws” and “blueprints”)

- `CONSTITUTION.md` — non-negotiable system law
- RFCs: `KERNEL-RFC-V1.md`, `LEDGERS-RFC-V1.md`, `EXECUTION-RFC-V1.md`
- Implementation guidance: `IMPLEMENTATION-PLAYBOOK.md`, `TECHNICAL-SPECS.md`
- Full runtime/API design: `API-RUNTIME-DESIGN.md`
- Data layer design: `DATA-STRUCTURE.md`

---

## 2) How an actual flow works (concrete example: Email Optimize)

This is the best way to “get it”: track one intent across files.

### 2.1 The intent schema

- `schemas/intents/funding_outreach_email_optimize.v1.json`

This defines:
- `intent_type = Funding.Outreach.Email.Optimize`
- allowed `inputs.requested_edits[]` enum values
- optional `email_text_override`, `custom_instructions`, etc.

If the switchboard returns a value not in that enum, validation fails immediately (good).

### 2.2 The plan template

- `plan-templates/funding_outreach_email_optimize.plan.v1.json`

This defines the steps to execute. Typically:
- load platform context
- ensure email exists (policy_check)
- optimize email draft (operator)
- propose platform patch (operator)
- require approval (human_gate apply_platform_patch)

### 2.3 Operator contracts + manifests

For each operator referenced in the plan template, there is:

1) Plugin manifest (discovery + schema refs)  
   - e.g. `manifests/plugins/operators/Email.OptimizeDraft.1.0.0.json`

2) Operator input schema  
   - e.g. `schemas/operators/email_optimize_draft.input.v1.json`

3) Operator output schema  
   - e.g. `schemas/operators/email_optimize_draft.output.v1.json`

4) Outcome schema produced by the operator  
   - e.g. `schemas/outcomes/email_draft.v1.json`

### 2.4 What the demo kernel does

In `demo/kernel_ai_outreach.py`:

- Switchboard decision (AI decision point #1):
  - `AIDemoKernel.switchboard_intent()` returns `intent_type` + `inputs`
  - It normalizes AI output to match the intent schema enums
- Operators (AI decision point #2 and #3):
  - `Email.OptimizeDraft` calls the LLM to produce a new draft outcome
  - `Email.ReviewDraft` calls the LLM to produce a review outcome
- The kernel streams user-facing text via `token_delta` (narration) and then pauses at `apply_platform_patch`

This exactly matches your intended real system pattern:
- AI makes decisions inside bounded, schema-validated boxes
- the executor stays deterministic
- anything “dangerous” is gated

---

## 3) How to add a new feature (new intent) — step-by-step

When you “add a feature” in this architecture, you are usually adding a new `intent_type` that maps to a new plan template.

Do it in this order.

### 3.1 Choose a new `intent_type`

Pick a stable name (don’t rename later). Examples:
- `Funding.Outreach.Email.Translate`
- `Documents.Resume.Review`
- `Onboarding.Gmail.Connect`

Decide:
- What is the scope? (thread-scoped, funding_request scoped, document scoped)
- Is there a human gate? (platform writes ⇒ yes)

### 3.2 Create the intent schema (`schemas/intents/...`)

Create a new file:
- `schemas/intents/<your_intent_snake>.v1.json`

Must:
- extend `schemas/common/intent_base.v1.json`
- set `schema_version` and `intent_type` constants
- define `inputs` shape (use `additionalProperties: false` for user-facing inputs)

Skeleton:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "schemas/intents/your_intent.v1.json",
  "title": "Intent: Your.Intent.Type (v1)",
  "allOf": [
    { "$ref": "../common/intent_base.v1.json" },
    {
      "type": "object",
      "properties": {
        "schema_version": { "const": "1.0" },
        "intent_type": { "const": "Your.Intent.Type" },
        "inputs": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "example_field": { "type": "string" }
          },
          "default": {}
        }
      }
    }
  ]
}
```

### 3.3 Register the intent (`manifests/intent-registry.v1.json`)

Add an entry:

```json
{
  "intent_type": "Your.Intent.Type",
  "schema_version": "1.0",
  "schema_ref": "schemas/intents/your_intent.v1.json"
}
```

### 3.4 Decide outcomes (what “truth” gets stored/returned)

Pick existing outcome types (preferred), or add new ones:
- `schemas/outcomes/...`

Typical patterns:
- draft-like output ⇒ `Email.Draft`, `Document.Composed`, etc.
- review output ⇒ `Email.Review`, `Document.Review`
- platform changes ⇒ `PlatformPatch.Proposal` + later `PlatformPatch.Receipt`

### 3.5 Define/reuse operators (plugin contracts)

For each operator you will call from the plan template, you need:

1) `schemas/operators/<op>.input.v1.json`  
2) `schemas/operators/<op>.output.v1.json`  
3) `manifests/plugins/operators/<Operator.Name>.<version>.json`  
4) a real code implementation (in the real service; in the demo kernels, you can stub)

### 3.6 Create a plan template (`plan-templates/...`)

Create:
- `plan-templates/<your_intent>.plan.v1.json`

Rules:
- validates against `schemas/plans/plan_template.v1.json`
- stable `step_id`s: `s1`, `s2`, … (never reuse a step id for different semantics)
- use bindings for inputs/context
- include `human_gate` for apply/confirm actions

Skeleton:

```json
{
  "schema_version": "1.0",
  "plan_template_id": "your_intent@1.0.0",
  "intent_type": "Your.Intent.Type",
  "plan_version": "1.0.0",
  "description": "Describe what this plan does",
  "steps": [
    {
      "step_id": "s1",
      "kind": "operator",
      "name": "Load platform context",
      "operator_name": "Platform.Context.Load",
      "operator_version": "1.0.0",
      "effects": ["read_platform"],
      "policy_tags": ["pii_allowed"],
      "risk_level": "low",
      "cache_policy": "cacheable",
      "idempotency_template": "ctx:{tenant_id}:{intent.thread_id}:{intent.scope.scope_id}",
      "payload": {
        "thread_id": { "from": "intent.thread_id" },
        "funding_request_id": { "from": "intent.scope.scope_id" }
      },
      "produces": ["context.platform"]
    },
    {
      "step_id": "s2",
      "kind": "policy_check",
      "name": "Ensure required inputs exist",
      "effects": [],
      "policy_tags": [],
      "risk_level": "low",
      "cache_policy": "no_cache",
      "check": {
        "check_name": "EnsureEmailPresent",
        "params": {
          "sources": [
            "intent.inputs.email_text_override",
            "context.platform.funding_request.email_content"
          ],
          "on_missing_action_type": "collect_fields"
        }
      }
    },
    {
      "step_id": "s3",
      "kind": "operator",
      "name": "Do the work",
      "operator_name": "Your.Operator.Name",
      "operator_version": "1.0.0",
      "effects": ["produce_outcome"],
      "policy_tags": ["draft_only"],
      "risk_level": "medium",
      "cache_policy": "cacheable",
      "idempotency_template": "op:{tenant_id}:{intent.thread_id}:{computed.some_hash}",
      "payload": { "some": { "from": "context.platform.some_field" } },
      "produces": ["outcome.your_outcome"]
    },
    {
      "step_id": "s4",
      "kind": "human_gate",
      "name": "Apply changes",
      "effects": ["write_platform"],
      "policy_tags": ["requires_approval"],
      "risk_level": "high",
      "cache_policy": "no_cache",
      "gate": {
        "gate_type": "apply_platform_patch",
        "title": "Apply changes",
        "description": "Apply the proposed patch to the platform DB.",
        "target_outcome_ref": { "from": "outcome.platform_patch_proposal" }
      }
    }
  ]
}
```

### 3.7 Register the plan template under a capability (`manifests/capabilities/...`)

Pick a capability manifest (or create a new one):
- `manifests/capabilities/<Capability>.x.y.z.json`

Update:
- `supported_intents[]` include your intent
- `plan_templates[]` add mapping: `{ intent_type, plan_template_ref }`
- `dependencies.operators[]` include operators you call
- `outcomes[]` include outcomes you produce

### 3.8 Implement operators (real system) + update the switchboard

In production:
- Operators are normal Python functions (async), invoked by workers/executor.
- The switchboard must be updated to produce schema-valid intents.

Rule of thumb:
- **AI outputs must be normalized before validation**, not “trusted”.

---

## 4) How to add an operator (plugin) — contract-first checklist

### 4.1 Add operator schemas (`schemas/operators/*`)

Input schema must extend:
- `schemas/operators/operator_call_base.v1.json`

Output schema must extend:
- `schemas/operators/operator_result_base.v1.json`

### 4.2 Add plugin manifest (`manifests/plugins/operators/*`)

Create:
- `manifests/plugins/operators/Your.Operator.Name.1.0.0.json`

It points to your operator input/output schemas and declares tags/effects.

### 4.3 Implement operator behavior (where your real logic lives)

Operators are where you put:
- DB reads/writes (your intelligence layer DB)
- S3 reads/writes
- LLM calls (via `llm-client`)
- platform patch proposals (never direct platform writes in v1)
- idempotency enforcement and retries

Operators must return:
- `status: succeeded|failed|in_progress`
- `metrics` (latency_ms + token/cost fields when relevant)
- typed result/outcome when succeeded

---

## 5) Policy checks + human gates (how they pause workflows)

### 5.1 `policy_check` steps

Use a `policy_check` step when:
- you must ensure prerequisites exist before calling an operator
- you need deterministic validation that can request more input (`collect_fields`)

Example check used in demos:
- `EnsureEmailPresent` (pauses if no email draft available)

### 5.2 `human_gate` steps

Use a `human_gate` step when:
- user approval is required
- the workflow needs the client to perform an external action (upload, redirect, refresh)

Gate types are enumerated in:
- `schemas/plans/step.v1.json` → `gate.gate_type`

Gate resolution is represented as a real intent:
- `Workflow.Gate.Resolve`

---

## 6) Versioning rules (don’t break yourself later)

### 6.1 Schemas

Prefer additive changes when possible.

If you need a breaking change:
- create a new schema file (new version)
- register it as a new intent schema version
- keep the old one around while clients migrate

### 6.2 Plan templates

Keep step ids stable.
- Do not reuse `s3` for a different meaning in a later revision.
- Add new steps at the end or insert with new ids (e.g. `s3b`), depending on your discipline.

### 6.3 Operator versions

If an operator’s behavior changes in a way that can affect outcomes/idempotency:
- bump operator version and publish a new plugin manifest
- keep old version for replayability if needed

---

## 7) Debugging (what errors mean)

### 7.1 Enum validation failures

Example error:
- `ValidationError: 'subject_line' is not one of [...]`

Meaning:
- AI/switchboard produced a value not allowed by schema.

Fix:
- normalize AI output to allowed enums (recommended), or
- expand schema enum (then bump version and ensure you really want that), or
- drop the field when invalid

### 7.2 “No plan template for intent”

Meaning:
- intent exists in intent registry, but no capability manifest maps it to a plan template.

Fix:
- add mapping to `manifests/capabilities/*.json`

### 7.3 Bindings resolved to `None`

Meaning:
- `{ "from": "path.to.value" }` didn’t exist in binding context.

Fix:
- add a `policy_check` before the operator step
- provide fallback fields

Note: demo kernels prune `None` values before schema validation (`prune_nones`), because optional fields are usually represented by absence rather than `null`.

---

## 8) How to run the demos (sanity loop)

Deterministic (no LLM):

```bash
python demo/run_email_review_demo.py
```

AI switchboard + AI operator demo (offline mock LLM):

```bash
python demo/run_ai_outreach_chat.py
```

AI switchboard + real OpenAI provider via `llm-client`:

```bash
python demo/run_ai_outreach_chat.py --real --model gpt-5-nano
```

Multi-line emails: wrap the email body in either:
- triple backticks ``` … ```
- double quotes " … "

---

## 9) Recommended read order (minimal confusion)

1) `CONSTITUTION.md` (laws)
2) `2-canonical-primitives.md` (Intent/Plan/Operator/Outcome/Event/Gate definitions)
3) `5-plugin-contracts.md` and `7-execution-semantics.md` (operator contracts, idempotency/replay)
4) `schemas/README.md` and `plan-templates/README.md` (how contracts are expressed)
5) `API-RUNTIME-DESIGN.md` (FastAPI + SSE + jobs/workers/webhooks/brokers/observability)
6) `demo/README.md` and run the demos (see it running)

