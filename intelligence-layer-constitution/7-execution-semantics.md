# 7. Execution Semantics Deep Spec

## 7.0 Core promise

For any submitted request, the Layer must guarantee:

1. **Resumability:** if the process crashes mid-run, it can continue without duplicating side effects.
2. **Idempotency:** repeated submissions do not create duplicate external actions or duplicate “final” writes.
3. **Best safe partial value:** if some steps succeed and others fail, return what you have, with clear status, and never pretend it’s complete.

This is achieved by a combination of:

* a **Plan model**
* a **Step state machine**
* **idempotency keys**
* **event + job + outcome ledgers**
* deterministic decision points in the Kernel

---

# 7.1 Plan execution model

## 7.1.1 The Plan is data, not code

A plan is a typed object produced by a planner agent, but validated and normalized by the Kernel.

Minimal plan schema:

```json
{
  "plan_id": "plan-uuid",
  "schema_version": "1.0",
  "intent_id": "intent-uuid",
  "steps": [
    {
      "step_id": "s1",
      "kind": "operator",
      "name": "Professor.Summarize",
      "payload": { "professor_id": 910, "source": "canspider" },
      "effects": ["read_only"],
      "policy_tags": ["read_only"],
      "gate": "none",
      "cache_policy": "use_if_safe",
      "idempotency_template": "prof_summary:{professor_id}:{digest_hash}"
    },
    {
      "step_id": "s2",
      "kind": "operator",
      "name": "Email.GenerateDraft",
      "payload": { "request_id": 556 },
      "effects": ["produce_outcome"],
      "policy_tags": ["draft_only"],
      "gate": "none",
      "cache_policy": "use_if_safe",
      "idempotency_template": "email_draft:{request_id}:{cv_hash}:{prof_sum_hash}:{template_hash}"
    },
    {
      "step_id": "s3",
      "kind": "operator",
      "name": "Gmail.SendEmail",
      "payload": { "draft_outcome_id": "out-..." },
      "effects": ["external_send"],
      "policy_tags": ["external_send"],
      "gate": "human_confirm",
      "cache_policy": "never",
      "idempotency_template": "gmail_send:{draft_outcome_id}"
    }
  ]
}
```

### Plan invariants

* **Step IDs are stable** within a plan.
* **No side effects happen outside operators.**
* Every step declares:

  * effects
  * policy tags
  * cache policy
  * idempotency template

Kernel refuses a plan that omits these.

---

# 7.2 Step state machine (the heart of resumability)

Each step has a state stored in the Job Ledger + referenced via Events.

Recommended states:

* `PENDING`
* `READY`
* `RUNNING`
* `SUCCEEDED`
* `FAILED_RETRYABLE`
* `FAILED_FINAL`
* `WAITING_APPROVAL`
* `SKIPPED`
* `CANCELLED`

### Transition rules

* `PENDING → READY` when dependencies satisfied
* `READY → WAITING_APPROVAL` if gate required and not approved
* `READY → RUNNING` if allowed
* `RUNNING → SUCCEEDED` on success
* `RUNNING → FAILED_RETRYABLE` if retryable error
* `RUNNING → FAILED_FINAL` if non-retryable error
* `FAILED_RETRYABLE → READY` after backoff window passes
* `WAITING_APPROVAL → READY` after approval event

### Dependency model

A step can declare `depends_on: ["s1"]`. Kernel must enforce dependency completion and pass required artifacts (eg professor summary outcome id) into next step payload via safe bindings.

---

# 7.3 Determinism and Idempotency

## 7.3.1 Two keys: Request Idempotency and Action Idempotency

### A) Request-level idempotency

Used to dedupe identical inbound requests.

Compute:
`request_key = hash(tenant_id + actor_id + intent_type + normalized_intent_payload + contexts_hash + policy_snapshot_version)`

Behavior:

* If request_key exists and workflow is complete: return existing final outcome.
* If request_key exists and workflow is in progress: attach to the existing workflow and stream status.
* If request_key exists and workflow failed: return partial outcomes + allow retry.

Store `request_key → workflow_id` mapping.

### B) Action-level idempotency (mandatory)

Every operator call must accept `idempotency_key`.
Store `operator_name + idempotency_key → job_id + result_hash + side_effect_receipt`.

If operator is called again with same key:

* return prior result without repeating side effects.

## 7.3.2 Determinism policy for different step types

You should categorize steps:

### 1) Pure deterministic steps

* parsing, hashing, formatting, schema validation
* must be byte-identical for same input

### 2) “Stable” nondeterministic steps (LLMs)

* allow nondeterminism but record inputs + model params
* prefer caching by input-hash when safe (draft generation usually safe)
* disclose nondeterminism flag in metrics

### 3) External nondeterministic steps (web, external APIs)

* never assume repeatable results
* cache only with explicit TTL and provenance
* policy can require citations/verification

## 7.3.3 “Duplicate requests should return existing outcomes when safe”

Define “safe reuse” explicitly:

Safe to reuse if:

* outcome type is **draft/review/summary/alignment**
* sensitivity constraints unchanged
* policy snapshot equivalent
* input hashes equivalent (doc hashes, professor digest hash, template hash)
* outcome was produced by an allowed plugin version range

Not safe to reuse if:

* action had **external side effects** (send, submit, write-final) unless you’re returning the receipt of that side effect
* policy snapshot changed in a way that affects permission or redaction
* intent includes time-sensitive facts (immigration regulations, deadlines) unless within TTL + verified

---

# 7.4 Observability (make the system self-explaining)

## 7.4.1 Trace context propagation (non-negotiable)

Every step execution must carry:

* `correlation_id` (the whole request)
* `workflow_id`
* `plan_id`
* `step_id`
* `job_id` (created per attempt)
* `trace_id` (LLM provider trace or internal)

These must show up in:

* Event Ledger records
* Job Ledger records
* Outcome Ledger records (as provenance)

## 7.4.2 Event log as a protocol (not random logs)

Events should be typed and minimal, eg:

* `INTENT_ACCEPTED`
* `PLAN_CREATED`
* `STEP_STARTED`
* `STEP_SUCCEEDED`
* `STEP_FAILED`
* `GATE_REQUESTED`
* `GATE_APPROVED`
* `WORKFLOW_PARTIAL`
* `WORKFLOW_COMPLETED`

## 7.4.3 Cost accounting

Every job must record:

* tokens in/out
* model
* cost in USD
* credits consumed
* cache hit/miss

And you also want aggregation:

* per workflow
* per user per day
* per tenant per month

Rule: costs are attached to jobs, not to messages.

---

# 7.5 Failure handling (taxonomy + behavior)

## 7.5.1 Error taxonomy (typed)

You want a small set of categories that every operator/tool uses:

### Retryable

* `NETWORK_TIMEOUT`
* `RATE_LIMIT`
* `TEMPORARY_PROVIDER_ERROR`
* `TRANSIENT_DB_LOCK`
* `DEPENDENCY_UNAVAILABLE`

### Non-retryable

* `POLICY_DENIED`
* `INVALID_INPUT`
* `SCHEMA_VALIDATION_FAILED`
* `MISSING_REQUIRED_CONTEXT`
* `AUTH_FORBIDDEN`

### Partial completion

Not an error category, but a workflow outcome:

* some steps succeeded and produced outcomes
* later steps failed or are waiting for approval

## 7.5.2 Retry policy (simple v1)

* exponential backoff with jitter
* max attempts per step based on error category
* never retry `POLICY_DENIED`, `INVALID_INPUT`, `AUTH_FORBIDDEN`

Example defaults:

* RATE_LIMIT: retry up to 5, backoff 2^n seconds, cap at 2 minutes
* NETWORK_TIMEOUT: retry up to 3
* TEMP_PROVIDER: retry up to 3

## 7.5.3 Partial value guarantee

When workflow cannot complete, the Layer returns:

* status: `partial`
* list of produced outcomes (IDs + previews)
* failed step(s) with reason codes
* next actionable instruction (eg approve, upload doc, fix field)

Example response:

```json
{
  "status": "partial",
  "workflow_id": "wf-...",
  "plan_id": "plan-...",
  "outcomes": [
    { "type": "Summary.Professor", "outcome_id": "out-1", "status": "final" },
    { "type": "Draft.Email", "outcome_id": "out-2", "status": "draft" }
  ],
  "blocked_on": {
    "step_id": "s3",
    "reason_code": "REQUIRES_APPROVAL",
    "gate_id": "gate-s3"
  }
}
```

## 7.5.4 “Best possible partial value safely”

This implies strict safety:

* never fabricate completion
* never auto-send
* if a later step fails, earlier safe outputs should still be returned

---

# 7.6 Resumability: how you actually resume

On restart or retry, Kernel does:

1. Load workflow state from Event + Job ledgers
2. Determine latest step states
3. For each step:

   * if SUCCEEDED: reuse result/outcome
   * if FAILED_RETRYABLE and within retry window: schedule retry
   * if WAITING_APPROVAL: surface gate
   * if PENDING/READY: proceed in order
4. Continue streaming based on current progress

No “in-memory” is required to resume. Only ledgers.

---

# 7.7 Execution modes (v1 pragmatic)

Define modes so policy can constrain them:

* `DRY_RUN`: compute everything, no external side effects
* `DRAFT_ONLY`: allow internal writes, no external sends/submissions
* `AUTO_EXEC`: allowed only at high trust levels and tenant-enabled
* `HUMAN_GATED`: default for actions like send/submit/apply-final

---

# 7.8 v1 Funding Outreach semantics (concrete)

For v1, your golden flow should be:

* Summarize professor (cacheable)
* Build alignment (cacheable)
* Generate email draft (cacheable)
* Review/optimize (cacheable)
* Apply to request (requires approval? optional but recommended)
* Send via Gmail (requires approval until trust earned; never cache)

This gives you:

* fast retries
* no duplicate sends
* drafts survive failures
