# 9. Capabilities

## 9.0 What a Capability is

A **Capability** is a versioned product module that answers:

* What problems can we solve (intents)?
* What data must exist (entities + documents)?
* What operators/tools/agents do we rely on?
* What must be verified before actions?
* What outcomes do we produce and in what schemas?
* How do we measure quality + business impact?

A capability is not:

* a microservice
* a single agent
* a single workflow
* a UI feature

It’s the **unit of maturity** you can certify, test, and sell.

---

# 9.1 Capability manifest (this should exist as a real file)

Every capability ships as a manifest plus test suite.

### 9.1.1 Capability Manifest Schema

```json
{
  "capability": {
    "name": "FundingOutreach",
    "version": "1.0.0",
    "stage": "v1",
    "description": "Draft, refine, and optionally send professor outreach emails with tracked outcomes.",
    "supported_intents": [
      "Funding.Outreach.Professor.Summarize",
      "Funding.Outreach.Alignment.Score",
      "Funding.Outreach.Email.Generate",
      "Funding.Outreach.Email.Review",
      "Funding.Outreach.Email.Optimize",
      "Funding.Outreach.Email.ApplyDraft",
      "Funding.Outreach.Email.Send"
    ],
    "required_entities": {
      "professor": { "source": ["canspider", "manual"], "required_fields": ["full_name", "email_address", "department"] },
      "institute": { "required_fields": ["institution_name", "country"] },
      "student": { "required_fields": ["first_name", "last_name", "email"] }
    },
    "required_documents": [
      { "type": "resume", "min_status": "processed", "required": true },
      { "type": "transcript", "min_status": "uploaded", "required": false }
    ],
    "dependencies": {
      "agents": ["Switchboard@1.x", "FollowUp@1.x"],
      "operators": [
        "Professor.Summarize@1.x",
        "Alignment.Score@1.x",
        "Email.GenerateDraft@1.x",
        "Email.ReviewDraft@1.x",
        "Email.OptimizeDraft@1.x",
        "Email.ApplyDraft@1.x",
        "Gmail.SendEmail@1.x"
      ],
      "tools": ["Doc.Parse@1.x", "Text.RubricScore@1.x"]
    },
    "verification_checks": {
      "preconditions": [
        { "check": "Auth.Scope", "params": { "scope": "funding:outreach" } },
        { "check": "Entity.Exists", "params": { "entity": "professor" } },
        { "check": "Doc.Available", "params": { "type": "resume", "status": "processed" } }
      ],
      "pre_send": [
        { "check": "Policy.Allows", "params": { "effect": "external_send" } },
        { "check": "PII.Redaction", "params": { "deny_tags": ["PASSPORT_ID", "GOV_ID"] } },
        { "check": "UserApproval", "params": { "required": true, "trust_level_lt": 3 } }
      ]
    },
    "outcomes": [
      { "type": "Professor.Summary", "schema": "schemas/outcomes/prof_summary.v1.json" },
      { "type": "Alignment.Score", "schema": "schemas/outcomes/alignment_score.v1.json" },
      { "type": "Email.Draft", "schema": "schemas/outcomes/email_draft.v1.json" },
      { "type": "Email.Review", "schema": "schemas/outcomes/email_review.v1.json" },
      { "type": "Email.SendReceipt", "schema": "schemas/outcomes/email_send_receipt.v1.json" }
    ],
    "metrics": {
      "quality": [
        { "name": "EmailRubricScore", "target": ">=0.80" },
        { "name": "HallucinationRisk", "target": "<=0.10" },
        { "name": "PIILeakRate", "target": "0" }
      ],
      "business": [
        { "name": "SendConversion", "target": ">=0.60" },
        { "name": "ReplyRate", "target": ">=0.08" },
        { "name": "TimeToFirstSendMinutes", "target": "<=10" }
      ],
      "ops": [
        { "name": "P95LatencySeconds", "target": "<=12" },
        { "name": "CostPerWorkflowUSD", "target": "<=0.20" }
      ]
    },
    "evaluation": {
      "offline_suite": "eval/funding_outreach_v1.yaml",
      "online_monitoring": ["reply_rate", "bounce_rate", "complaint_rate"]
    }
  }
}
```

**Non-negotiable:** capabilities are versioned like software. No “latest”.

---

# 9.2 Capability lifecycle and maturity model

This is how you keep enterprise trust while moving fast.

### Levels (suggested)

* **P0 Prototype:** works sometimes, no guarantees
* **P1 Repeatable:** stable schemas, basic logging, manual gates
* **P2 Reliable:** resumable execution, idempotency, policy coverage, eval suite
* **P3 Enterprise:** tenant config, audit trails, SLA targets, adapter hardening
* **P4 Certified:** compliance mapping (GDPR/PIPEDA), pentest posture, formal change control

A capability cannot claim P2+ without:

* idempotent operators
* a resumable plan execution model
* defined outcomes + schemas
* evaluation harness

---

# 9.3 Capability boundaries (what’s inside vs outside)

Inside capability:

* intent definitions and routing rules
* verification checks
* required entities/documents
* operator graph (dependency DAG)
* outcome schema list
* eval metrics, tests, and thresholds

Outside capability:

* Kernel runtime
* ledgers themselves
* auth system implementation
* plugin implementations

Capabilities configure the system, they don’t replace it.

---

# 9.4 How the Kernel uses capabilities

On intake:

1. identify capability by intent type (routing table)
2. load capability manifest (version pinned per tenant or per request)
3. run capability preconditions checks
4. allow the planner/switchboard to propose a plan only from declared dependencies
5. enforce verification checks at gates (eg pre_send)
6. validate produced outcomes against declared schemas
7. log capability metrics per run

This is how you avoid “agent called random operator and nuked prod.”

---

# 9.5 Verification checks (make them real primitives)

Define checks as named functions (tools or kernel primitives), eg:

* `Auth.Scope(scope)`
* `Entity.Exists(entity_type)`
* `Entity.FieldsPresent(entity_type, fields)`
* `Doc.Available(type, min_status)`
* `Policy.Allows(effect|tags)`
* `PII.Redaction(deny_tags)`
* `Citations.Required(if_intent_types)`
* `UserApproval(required, gate_type, conditions...)`
* `RateLimit(window, max)`
* `Quota(max_cost|max_tokens)`

Each check produces:

* pass/fail
* reason_code
* remediation hints

---

# 9.6 Outcome schemas: the “contract with the outside world”

Each outcome must have:

* `outcome_id`
* `type`
* `schema_version`
* `producer` (plugin + version)
* `trace_id / job_id`
* `created_at`
* `payload` (typed)

Example: Email Draft outcome should include:

* subject/body
* attachments list (by document IDs, not raw bytes)
* redaction summary
* confidence + rubric score
* dependency hashes (cv hash, professor summary hash, template hash)

This makes caching and dedupe provable.

---

# 9.7 Evaluation harness (testable means measurable)

Capabilities must ship tests in two layers:

### Offline eval (pre-release)

* golden set of inputs (requests + professor profiles + resumes)
* expected rubric outcomes
* regression thresholds (quality must not drop)

### Online monitoring (post-release)

* reply rate
* bounce rate
* complaint rate
* user edits ratio (how often user rewrites the draft)
* time-to-first-send
* cost per workflow
* policy denials count

If you don’t measure, you don’t have a capability, you have vibes.

---

# 9.8 Capability versioning strategy (how to not break tenants)

Three knobs:

1. **Global default capability version** (platform)
2. **Tenant pinned version** (enterprise wants stability)
3. **Per-request override** (experiments, A/B)

Rules:

* Tenant pin wins unless explicit override allowed.
* Outcomes must record capability version.
* Operators referenced should accept version constraints (eg `@1.x`).

---

# 9.9 v1 Funding Outreach capability: the minimum DAG

The workflow DAG for v1 is basically:

1. `Professor.Summarize` (read-only, cacheable)
2. `Alignment.Score` (read-only, cacheable)
3. `Email.GenerateDraft` (draft outcome, cacheable)
4. `Email.ReviewDraft` (review outcome, cacheable)
5. `Email.OptimizeDraft` (new draft revision, cacheable)
6. `Email.ApplyDraft` (write internal final, gated optional)
7. `Gmail.SendEmail` (external side effect, human-gated until trust)

This is simple enough to stabilize, measure, and sell.
