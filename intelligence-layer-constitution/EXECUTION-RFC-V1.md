# Execution RFC v1 (Plan, Idempotency, Replay)

This RFC captures **implementation-facing** execution semantics. It is not constitutional law and may evolve without changing `CONSTITUTION.md`.

## 1) Plan as data (not code)

Plans are typed objects validated by the Kernel.

Minimal step requirements:

- `step_id` (stable within the plan)
- `kind` (`operator|agent|policy_check|human_gate`)
- declared `effects[]`
- `policy_tags[]`
- `risk_level`
- `cache_policy`
- idempotency template/derivation inputs
- optional `gate` descriptor

## 2) Step state machine (v1)

Recommended states:

- `PENDING`
- `READY`
- `RUNNING`
- `SUCCEEDED`
- `FAILED_RETRYABLE`
- `FAILED_FINAL`
- `WAITING_APPROVAL`
- `SKIPPED`
- `CANCELLED`

Core transitions:

- `READY → WAITING_APPROVAL` when a gate is required and not yet approved
- `READY → RUNNING → SUCCEEDED|FAILED_*`
- `FAILED_RETRYABLE → READY` after backoff
- `WAITING_APPROVAL → READY` after approval event

## 3) Two idempotency layers

### 3.1 Request-level idempotency (dedupe identical inbound requests)

Maintain a mapping of a stable `request_key -> workflow_id`.

Behavior:

- if complete: return existing outcomes (reproduce)
- if running: attach and stream current state
- if failed: return partial outcomes and allow resume/retry

### 3.2 Action-level idempotency (mandatory)

Every operator call uses an `idempotency_key`.

Required behavior:

- same `(tenant_id, operator_name, idempotency_key)` returns the same receipt/result
- side effects MUST NOT repeat under the same key

## 4) Caching and determinism

Classify steps:

1. deterministic (hash/parse/validate): must be byte-identical on same inputs
2. stable nondeterministic (LLM drafts): allow variance, but store inputs+params and allow cache-by-hash when policy permits
3. external nondeterministic (web/external APIs): cache only with explicit TTL and provenance

## 5) Replay semantics (operational)

Define three modes:

- **reproduce**: fetch stored outcomes and events (no recomputation)
- **replay**: re-run steps using stored inputs/context refs; new draft outcomes MAY be produced, but effectful steps remain idempotent
- **regenerate**: create a new version lineage (new idempotency/version keys for draft-producing steps)

Default UI behavior:

- “Open workflow” and “refresh” should use reproduce
- “Retry” should resume failed steps with the same keys
- “Regenerate” should be explicit and produce a new draft version

## 6) Failure taxonomy (v1)

Retryable:

- timeouts, rate limits, transient provider errors, transient DB locks

Non-retryable:

- policy denied, invalid input, missing required context, auth forbidden

Partial completion:

- return produced outcomes + explicit blocked step and reason codes

## 7) References

See `7-execution-semantics.md` for the deep spec and examples.

