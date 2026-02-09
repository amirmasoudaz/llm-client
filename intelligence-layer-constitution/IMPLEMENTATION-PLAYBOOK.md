# Implementation Playbook (v1)

This playbook is practical guidance and checklists for building v1 safely. It is not constitutional law.

## 1) What to build first (shortest safe path)

1. Event Ledger is real and queryable (append-only, stable event types, correlation IDs).
2. Intent normalization + schema validation for a tight v1 set.
3. AuthZ + tenant scoping everywhere (even if single-tenant today).
4. ContextBundle builder (policy-filtered, hashed).
5. Plan schema + deterministic planner for a few intents.
6. Executor state machine with retries + idempotency enforcement.
7. Operator interface + a small operator set.
8. Streaming (events/progress/assistant) derived from ledgers.
9. Human gate (“Apply”) as the only write path to domain tables in v1.

## 2) Keep the Kernel clean

- No business logic in Kernel.
- No integration conditionals in Kernel.
- All effects in Operators, all governance in Policy.

## 3) Where to put “deep specs”

- Kernel interface and responsibilities: `KERNEL-RFC-V1.md`
- Ledgers record shapes and expectations: `LEDGERS-RFC-V1.md`
- Execution semantics (idempotency, replay, step machine): `EXECUTION-RFC-V1.md`

The preserved deep narrative references remain authoritative context:

- `3-the-kernels.md`
- `4-ledgers-sources-of-truth.md`
- `7-execution-semantics.md`

## 4) Database and schema sketches

Implementation sketches and DDLS exist in:

- `PREREQUISITE.txt`

Treat them as **reference notes**, not as drop-in migrations. Before adopting any DDL:

- ensure tenant scoping (`tenant_id`) is explicit where required
- ensure FK constraints and columns line up (no missing columns / syntax errors)
- ensure `workflow_id` vs `thread_id` semantics match the constitution

## 5) Capability admission checklist (make shipping boring)

For any new capability:

1. Define intent schemas and outcome schemas (versioned).
2. Define a plan template with explicit effects/tags/gates.
3. Implement operators with idempotency keys and declared effects.
4. Add policy rules for egress, redaction, and apply.
5. Add tests/harness for idempotency, permissions, and redaction/egress.

## 6) Credits/budget enforcement

Budgeting is a normative supplement:

- `8-ai-credits-and-budget-enforcement.md`

Integrate it by attaching usage/cost to the Job Ledger and enforcing reservations/settlement at the workflow/request boundary.

