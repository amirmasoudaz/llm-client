# Plan templates (v1)

These files define the **deterministic step programs** per intent type.

Each template conforms to `schemas/plans/plan_template.v1.json` and is used by the planner/kernel to produce a concrete `Plan` (with resolved bindings and validated operator payloads).

Guidelines:

- Step IDs must be stable (`s1`, `s2`, …) and never reused for different semantics.
- Every step declares:
  - `effects[]`, `policy_tags[]`, `risk_level`, `cache_policy`
  - `idempotency_template` (operator steps)
  - `gate` (human_gate steps) or `check` (policy_check steps)
- Operator payloads are **templates**: values can be bindings like `{ "from": "context.platform.funding_request.id" }`.

## Binding resolution model (what the executor must implement)

At execution time, the Kernel builds a **binding context** object and resolves every `payload` binding and every `idempotency_template`.

### 1) Dotted-path reads

`{ "from": "x.y.z" }` means: read a value from the binding context using a dotted path.

### 2) Constants

`{ "const": ... }` means: use the literal value as-is.

### 3) String interpolation templates

Any `idempotency_template` (string) is treated as an interpolation template.
The executor should also support `{ "template": "..." }` values inside payloads.

Template placeholders reference the same dotted-path namespace as `from` bindings.

Example:

- `doc_export:{tenant_id}:{intent.thread_id}:{outcome.document_optimized.hash.value}:pdf`

### 4) Computed values

Some idempotency keys use values that must be computed deterministically by the executor and placed under `computed.*`, e.g.:

- `computed.email_body_hash` (blake3 hash of the effective email body used for review/optimize)
- `computed.requested_edits_hash` (blake3 hash of normalized requested edits arrays)
- `computed.fields_hash` (blake3 hash of normalized patch field maps)
- `computed.source_hash`, `computed.target_fields_hash` (for paper metadata extraction)

This keeps idempotency keys short and avoids embedding long/PII strings.
