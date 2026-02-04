# JSON Schemas (v1)

This folder contains **versioned JSON Schemas** used to validate:

- API requests/responses (`schemas/api/*`)
- SSE event payloads (`schemas/sse/*`)
- Intent records written to ledgers (`schemas/intents/*`)
- Plan records and per‑intent plan templates (`schemas/plans/*`, `plan-templates/*`)
- Operator payloads and results (`schemas/operators/*`)
- Outcome records returned to clients and stored in ledgers (`schemas/outcomes/*`)

## Conventions

- JSON Schema draft: **2020-12**.
- Every schema has:
  - `$schema` = `https://json-schema.org/draft/2020-12/schema`
  - `$id` = repo-relative path (e.g., `schemas/intents/...`)
- Every typed object includes `schema_version` (string like `"1.0"`).
- Prefer `additionalProperties: false` for externally-facing schemas (API/SSE).
- For internal payloads (operator results, context snapshots), allow extra fields only where necessary and document them.

## Validation boundaries (non-negotiable)

Validate at:

- Intake: incoming thread/query requests → normalize into a typed Intent
- Planning: generated Plan must validate and include effects/tags/cache/idempotency/gates
- Operator invocation: operator payload validates against the operator input schema
- Operator return: operator result validates against the operator output schema
- Outcomes: every produced outcome validates against its outcome schema before being stored/returned
- SSE/Webhooks: emitted event payloads validate against their schemas (at least in staging)

## Plan templates and bindings

Per-intent plan templates live under `plan-templates/`.

Plan templates are **not** raw operator payloads; they include a `payload` that may contain **bindings**:

- `{ "from": "context.platform.funding_request.id" }` — read a value from runtime context by dotted path
- `{ "const": 123 }` — use a literal
- `{ "template": "email_draft:{thread_id}:{request_id}:{resume_hash}" }` — string interpolation template (resolved by the planner/executor)

The executor resolves bindings into a concrete operator payload and then validates it against the operator input schema.

## Registry and discovery

- Capability manifests under `manifests/capabilities/` reference:
  - supported intents + their schemas
  - plan templates per intent
  - operator dependencies + schema refs
  - outcomes produced + schema refs

