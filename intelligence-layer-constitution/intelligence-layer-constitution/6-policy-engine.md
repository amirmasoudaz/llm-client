# 6. Policy Engine

## 6.0 What Policy is and what it is not

**Policy Engine = decision function**:

> `(actor, tenant_config, intent, action, data, risk_signals) -> decision`

It must be:

* deterministic for identical inputs
* explainable (short reason codes)
* auditable (decision written as event)
* composable (global + tenant + capability policies)

It is NOT:

* business logic (belongs in plugins/capabilities)
* “agent feelings”
* a prompt

---

## 6.1 Policy evaluation stages (where policy runs)

You need policy checks at multiple points, otherwise people bypass it accidentally.

### Stage 1: **Intake Policy**

Checks: actor, tenant, quota, forbidden intents, jurisdiction.
Output: allow/deny/require extra verification.

### Stage 2: **Plan Policy**

Checks proposed plan steps and tags before execution.
Output: allow/deny/transform plan (eg remove forbidden step) / require approval gates.

### Stage 3: **Action Policy**

Checks each operator invocation before it executes.
Output: allow/deny/require approval/redact inputs/enforce constraints.

### Stage 4: **Outcome Policy**

Checks produced output before it is returned/streamed/applied.
Output: allow, redact, transform, require citations, require user approval.

### Stage 5: **Apply Policy**

Checks “write to final” actions (DB writes, external sends).
Output: allow/deny/require approval.

**Hard rule:** any step with side effects must pass Action Policy + Apply Policy.

---

## 6.2 Policy inputs: normalize them into a single PolicyContext

Instead of ad hoc inputs, define a stable object:

```json
{
  "tenant": {
    "tenant_id": 1,
    "plan": "pro",
    "locale": "CA",
    "regulatory_flags": ["PIPEDA"],
    "features": { "gmail_send_enabled": true }
  },
  "actor": {
    "type": "student",
    "id": 88,
    "role": "user",
    "trust_level": 1,
    "auth_scopes": ["intent:submit", "outcome:read", "gate:approve"]
  },
  "intent": {
    "type": "Funding.Outreach.Email.Generate",
    "risk_class": "medium",
    "scope": { "type": "funding_request", "id": 556 }
  },
  "action": {
    "type": "operator_call",
    "operator": "Gmail.SendEmail",
    "effects": ["external_send"],
    "tags": ["email_send", "pii", "third_party"]
  },
  "data": {
    "sensitivity_tags": ["PII", "EDU_RECORD"],
    "destinations": ["gmail"],
    "contains": { "passport": false, "dob": true }
  },
  "signals": {
    "model_confidence": 0.62,
    "risk_score": 0.77,
    "anomaly_flags": ["new_device", "first_send"],
    "rate": { "per_minute": 3, "per_day": 40 },
    "cost_estimate": { "tokens": 2100, "usd": 0.03 }
  }
}
```

**Key idea:** policy runs on normalized tags and effects, not on raw payloads.

---

## 6.3 Policy outputs: a single Decision object

You already have the core, here’s the concrete schema:

```json
{
  "decision": "ALLOW|DENY|REQUIRE_APPROVAL|ALLOW_WITH_REDACTION|TRANSFORM",
  "reason_code": "EMAIL_SEND_REQUIRES_TRUST",
  "reason": "External send is not allowed until trust level >= 3.",
  "requirements": {
    "approval": { "gate_type": "human_confirm", "message": "Send this email now?" },
    "citations": { "required": false },
    "verification": { "required": false }
  },
  "limits": {
    "rate_limit": { "window_s": 60, "max": 2 },
    "quota": { "max_tokens": 5000 }
  },
  "redactions": [
    { "path": "$.email.body", "rule": "mask_phone" }
  ],
  "transform": {
    "action": { "operator": "Gmail.SendEmail", "payload_patch": { "dry_run": true } }
  }
}
```

**Invariant:** every decision must produce a `POLICY_DECISION` event with `reason_code`.

---

# 6.4 Policy rule model (how you express policies without writing spaghetti)

You need three layers:

### 1) Global baseline policies (hard safety)

Non-negotiables that apply everywhere.

### 2) Tenant policies (configurable)

Per enterprise, per geography, per product tier.

### 3) Capability policies (local)

Funding outreach v1, admissions v2, support v3.

### Precedence

`DENY` wins over everything.
Then `REQUIRE_APPROVAL`.
Then transforms/redactions.

---

## 6.5 Policy language: keep it simple but expressive

You don’t need OPA complexity on day 1. You need:

* match conditions on:

  * `intent.type`
  * `action.effects`
  * `action.tags`
  * `data.sensitivity_tags`
  * `actor.trust_level`
  * `tenant.locale/regulatory_flags`
  * `signals.risk_score`

* decide:

  * allow/deny/require approval
  * redact/transform
  * rate limit/quota

Example rule (conceptual):

**Rule: Gmail sending requires trust**

* IF `action.effects` contains `external_send`
* AND `actor.trust_level < 3`
* THEN `REQUIRE_APPROVAL` reason `EMAIL_SEND_REQUIRES_TRUST`

**Rule: passport blocks external send**

* IF data contains `passport = true`
* THEN `DENY` reason `SENSITIVE_ID_BLOCKED`

---

# 6.6 Trust levels (critical for your “earned autonomy”)

This is the cleanest way to scale approvals without annoying users forever.

### Trust Level 0: Unverified

* no external sends
* no application submissions
* drafts only
* heavy redaction

### Trust Level 1: Verified account

* can generate drafts and apply to internal records
* external send requires approval every time

### Trust Level 2: Demonstrated safe usage

* external send requires approval but can enable “approve for this thread”

### Trust Level 3: Trusted sender

* external send allowed with safeguards (rate limits, anomaly checks)
* random spot checks or escalation if risk spikes

### Trust Level 4: Enterprise managed

* tenant admin policies govern, can allow fully automated sequences

**How trust increases (policy-controlled):**

* verified email integration success + time
* successful human-approved sends with no complaints/bounces
* no abnormal spikes
* optional KYC-like enterprise checks (for B2B)

Trust changes must be recorded in:

* Memory Ledger or dedicated Trust Ledger
* and an Event: `TRUST_LEVEL_CHANGED`

---

# 6.7 Approval workflow spec (gates)

Approvals are first-class.

## Gate object

```json
{
  "gate_id": "gate-s7",
  "gate_type": "human_confirm",
  "reason_code": "EMAIL_SEND_REQUIRES_TRUST",
  "summary": "Send email to prof@example.edu",
  "preview": {
    "subject": "...",
    "body": "...",
    "attachments": ["CV.pdf"]
  },
  "expires_at": "2026-01-28T20:03:00Z",
  "allowed_decisions": ["approve", "reject", "edit_then_approve"]
}
```

### Approval invariants

* approval must reference the exact `outcome_id` version being approved
* edits produce a new draft outcome
* approval event ties to `(gate_id, outcome_id, actor_id)`

---

# 6.8 Redaction and transform rules (the “make it safe without killing UX” toolkit)

### Redaction examples

* mask phone numbers unless explicitly required
* strip DOB unless used for admission form filling
* remove passport/ID from email content always

### Transform examples

* change `Gmail.SendEmail` → `Gmail.SendEmail(dry_run=true)` for untrusted users
* downgrade “apply final” → “save draft only”
* enforce “require citation” for immigration advice outcomes

Transforms must be explicit in the decision output, and evented.

---

# 6.9 Risk scoring (simple v1)

Don’t overcomplicate. Use a weighted score from known signals:

Inputs:

* new user? (higher risk)
* first time external send? (higher)
* contains PII/EDU_RECORD? (higher)
* bulk volume? (higher)
* low model confidence? (higher)
* destination is third-party? (higher)

Output:

* `risk_score 0..1`
* `risk_band low|medium|high`

Policy uses risk_band to choose:

* auto allow
* require approval
* deny

---

# 6.10 Rate limits and quotas (policy-owned)

Policy should own:

* max actions per minute/day
* max tokens/cost per workflow/day
* max external sends per day
* cooldown after repeated failures

These belong here because “abuse prevention” is policy, not business logic.

---

# 6.11 Auditing: policy decisions must be reconstructable

Every policy decision emits an immutable event:
`POLICY_DECISION` with:

* stage
* inputs hash (not raw PII)
* decision + reason_code
* any redactions/transforms
* actor + tenant

This is what enterprises ask for in compliance reviews.

---

# 6.12 v1 Funding Outreach Policy baseline (concrete defaults)

Here’s a sane v1 baseline:

### Allowed without approval

* Professor summarization (read-only)
* alignment computation
* draft generation, review, optimization
* storing drafts internally

### Require approval

* apply draft to final records (writing to “finals” or “send queue”)
* sending email via Gmail
* scheduling reminders
* attaching transcripts or sensitive docs

### Deny by default

* sending passport/ID/DOB in external emails
* bulk sends > N per day for low trust
* immigration legal conclusions without citations + disclaimers (v2/v3)
