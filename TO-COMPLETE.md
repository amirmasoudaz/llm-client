# TO-COMPLETE

This document captures remaining gaps from `INTELLIGENCE_LAYER_IMPLEMENTATION_PLAN_V1.md` and a concrete action plan to fully complete Phases A, B, C, D, E, F, G, H, and I.

Last audit: 2026-02-11

## Current status summary
- Phase A: Complete
- Phase B: Complete
- Phase C: Complete
- Phase D: Complete
- Phase E: Complete
- Phase F: Complete
- Phase G: Complete
- Phase H: Complete
- Phase I: Complete

## Definition of done
- Every listed gap below is closed in code.
- Contract validation passes.
- Runtime and operator behavior are covered by automated tests.
- A production-safe path exists (auth, billing, policy, replay semantics).
- Acceptance criteria in Phases A-I are demonstrably met, not only partially implemented.

## Phase A - Foundations

### Remaining gaps
- None.

### Plan of action
- [x] A1. Migrate prompt layout from flat files to versioned operator folders and update loader resolution in `src/intelligence_layer_kernel/prompts/loader.py`.
- [x] A2. Add manifest-level prompt template binding fields and loader checks so template lookup is contract-driven.
- [x] A3. Extend outcome persistence to include prompt template metadata (`template_id`, `template_hash`) in `src/intelligence_layer_kernel/runtime/store.py`.
- [x] A4. Extend usage event persistence to include prompt template metadata in `src/intelligence_layer_api/billing.py` and related usage-writing call paths.
- [x] A5. Add a reproducibility API path for stored outcomes/events (reproduce mode) and verify no recomputation occurs.
- [x] A6. Add tests covering prompt path resolution, strict variable validation, hash persistence, and reproduce semantics.

### Verification
- `PYTHONPATH=src python -m intelligence_layer_kernel.contracts.validate`
- New tests:
- `tests/intelligence_layer_kernel/test_prompt_loader.py`
- `tests/intelligence_layer_kernel/test_outcome_template_hash_persistence.py`
- `tests/intelligence_layer_api/test_workflow_reproduce_mode.py`

## Phase B - Plugin framework and operator core

### Remaining gaps
- None.

### Plan of action
- [x] B1. Replace default-allow skeleton in `src/intelligence_layer_kernel/policy/engine.py` with explicit rule evaluation and deny/requirement support.
- [x] B2. Wire `thread_id` and `intent_id` into operator job claims in `src/intelligence_layer_kernel/operators/executor.py` and `src/intelligence_layer_kernel/operators/store.py`.
- [x] B3. Add capability and policy allow/deny checks in `src/intelligence_layer_kernel/operators/registry.py` and pre-invoke checks in executor.
- [x] B4. Add manifest schema fields for prompt template IDs and enforce render/persist path in executor.
- [x] B5. Ensure policy decisions are emitted at all required stages (plan, action, outcome) with consistent trace IDs and references.
- [x] B6. Add regression tests for deny paths, idempotent replay, and manifest prompt binding failures.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_policy_engine_rules.py`
- `tests/intelligence_layer_kernel/test_operator_executor_trace_propagation.py`
- `tests/intelligence_layer_kernel/test_operator_manifest_prompt_binding.py`

## Phase C - Kernel runtime, planner, and gates

### Remaining gaps
- None.

### Plan of action
- [x] C1. Implement LLM JSON fallback classifier in `src/intelligence_layer_kernel/runtime/switchboard.py` with allowlisted intent validation.
- [x] C2. Add explicit API support for `mode=reproduce|replay|regenerate` for workflow outcome retrieval and execution behavior.
- [x] C3. Implement deterministic replay behavior and explicit regenerate lineage behavior in runtime kernel/store.
- [x] C4. Add full gate lifecycle tests for `collect_profile_fields` and `apply_platform_patch` including resume continuity of idempotency keys.
- [x] C5. Add data-model support for regenerate lineage reuse where needed (parent workflow/outcome lineage tracking).
- [x] C6. Add API and runtime tests proving pause/resume and replay-mode semantics.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_switchboard_llm_fallback.py`
- `tests/intelligence_layer_kernel/test_gate_pause_resume.py`
- `tests/intelligence_layer_kernel/test_replay_regenerate_semantics.py`
- `tests/intelligence_layer_api/test_workflow_outcomes_modes.py`

## Phase D - Funding Outreach MVP slice

### Remaining gaps
- None.

### Plan of action
- [x] D1. Implement production `AuthAdapter` with cookie/session validation and principal propagation in `src/intelligence_layer_api/auth.py`.
- [x] D2. Enforce ownership checks for funding request scope pre-context-load in `src/intelligence_layer_api/app.py`.
- [x] D3. Replace reserve heuristic with provider/model-aware estimate from real pricing versions.
- [x] D4. Ensure usage events are written from actual LLM/tool runtime usage before settlement in `src/intelligence_layer_api/billing.py`.
- [x] D5. Enforce `BudgetSpec.max_cost` using reserved credits and fail-fast when projected spend exceeds hold.
- [x] D6. Emit `model_token` events in workflow-kernel execution path for LLM-backed operators.
- [x] D7. Add staged rollout config to move `IL_USE_WORKFLOW_KERNEL` toward default-on in production-like environments.
- [x] D8. Add tests for auth denied and insufficient credits paths verifying no model token leakage.

### Verification
- New tests:
- `tests/intelligence_layer_api/test_auth_adapter_production.py`
- `tests/intelligence_layer_api/test_credit_reserve_settle_actual_usage.py`
- `tests/intelligence_layer_api/test_budget_enforcement.py`
- `tests/intelligence_layer_api/test_no_tokens_on_auth_or_credit_failure.py`
- `tests/intelligence_layer_api/test_workflow_model_token_streaming.py`

## Phase E - Student profile onboarding and memory

### Remaining gaps
- None.

### Plan of action
- [x] E1. Add end-to-end tests for `Student.Profile.Collect` from missing fields to satisfied requirements.
- [x] E2. Ensure invalid profile update errors are normalized into user-facing actionable messages in workflow responses.
- [x] E3. Add tests for prefill from platform fields and onboarding JSON (`funding_template_initial_data`).
- [x] E4. Add tests for memory type constraints and retrieval filtering.
- [x] E5. Add strict schema-validity test after every update path.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_student_profile_collect_flow.py`
- `tests/intelligence_layer_kernel/test_student_profile_validation_messages.py`
- `tests/intelligence_layer_kernel/test_memory_upsert_retrieve_constraints.py`

## Phase F - Funding request field completion and apply gate

### Remaining gaps
- None.

### Plan of action
- [x] F1. Add end-to-end test: propose -> gate -> apply -> verify platform row updates.
- [x] F2. Add idempotency replay test ensuring repeated apply with same idempotency key cannot double-apply.
- [x] F3. Add stale `updated_at` conflict tests for optimistic locking.
- [x] F4. Add contract test for `action_required.apply_action_id` and `ui.refresh_required` payload shape.
- [x] F5. Add negative tests for unsupported fields and invalid data types/lengths.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_funding_request_update_apply_flow.py`
- `tests/intelligence_layer_kernel/test_funding_request_update_idempotency.py`
- `tests/intelligence_layer_kernel/test_funding_request_update_conflicts.py`
- `PYTHONPATH=src:. .venv/bin/pytest -q tests/intelligence_layer_kernel/test_funding_request_update_apply_flow.py tests/intelligence_layer_kernel/test_funding_request_update_idempotency.py tests/intelligence_layer_kernel/test_funding_request_update_conflicts.py`

## Phase G - Email optimize loop

### Remaining gaps
- None.

### Plan of action
- [x] G1. Implement explicit outcome lineage/version increment behavior for optimized drafts in outcome storage.
- [x] G2. Bind source-draft selection to lineage/version fields rather than ad-hoc payload-only references.
- [x] G3. Add tests for iterative optimization across multiple versions in one thread.
- [x] G4. Add tests for blocking apply when `main_sent=1` and pivot guidance response.
- [x] G5. Add tests for missing-draft gate behavior and resume after draft appears.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_email_optimize_lineage.py`
- `tests/intelligence_layer_kernel/test_email_apply_block_when_sent.py`
- `tests/intelligence_layer_kernel/test_email_optimize_missing_draft_gate.py`
- Runtime updates:
- `src/intelligence_layer_kernel/runtime/store.py`
- `src/intelligence_layer_kernel/runtime/kernel.py`
- `PYTHONPATH=src:. .venv/bin/pytest -q tests/intelligence_layer_kernel/test_email_optimize_lineage.py tests/intelligence_layer_kernel/test_email_apply_block_when_sent.py tests/intelligence_layer_kernel/test_email_optimize_missing_draft_gate.py`

## Phase H - Professor alignment

### Remaining gaps
- None.

### Plan of action
- [x] H1. Add deterministic test fixtures for profile -> summarize -> alignment score chain.
- [x] H2. Add schema contract tests for `Professor.Profile`, `Professor.Summary`, and `Alignment.Score` outputs.
- [x] H3. Add evidence consistency checks (matched topics and rationale coherence).
- [x] H4. Add regression tests for edge-case sparse professor data.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_professor_alignment_determinism.py`
- `tests/intelligence_layer_kernel/test_professor_alignment_schema_contracts.py`
- `PYTHONPATH=src:. .venv/bin/pytest -q tests/intelligence_layer_kernel/test_professor_alignment_determinism.py tests/intelligence_layer_kernel/test_professor_alignment_schema_contracts.py`

## Phase I - CV/SOP/cover-letter review from attachments

### Remaining gaps
- None.

### Plan of action
- [x] I1. Convert missing-attachment condition into `action_required` gate flow (upload required) rather than terminal operator failure.
- [x] I2. Refactor download path to true streaming-to-disk and hash-on-stream in `src/intelligence_layer_kernel/operators/implementations/documents_common.py`.
- [x] I3. Add document-type-specialized review prompts (`cv`, `sop`, `letter`) with typed output schemas and manifest bindings.
- [x] I4. Update `Documents.Review` to use prompt loader outputs and persist prompt hash metadata.
- [x] I5. Add tests for MIME allowlist, size limits, and parsing artifacts across PDF/DOCX/text.
- [x] I6. Add end-to-end test for "Review my CV" including progress events and structured report.

### Verification
- New tests:
- `tests/intelligence_layer_kernel/test_documents_missing_attachment_gate.py`
- `tests/intelligence_layer_kernel/test_documents_streaming_fetch.py`
- `tests/intelligence_layer_kernel/test_documents_review_specialized_outputs.py`
- `tests/intelligence_layer_kernel/test_documents_review_e2e_cv.py`
- `PYTHONPATH=src:. .venv/bin/pytest -q tests/intelligence_layer_kernel/test_documents_missing_attachment_gate.py tests/intelligence_layer_kernel/test_documents_streaming_fetch.py tests/intelligence_layer_kernel/test_documents_review_specialized_outputs.py tests/intelligence_layer_kernel/test_documents_review_e2e_cv.py`
- `PYTHONPATH=src:. .venv/bin/python -m intelligence_layer_kernel.contracts.validate`

## Cross-cutting execution order
- [ ] X1. Finish contract and schema changes first (prompt bindings, replay modes, specialized review schemas).
- [ ] X2. Implement foundational runtime changes (policy, trace propagation, replay semantics) before vertical slice hardening.
- [ ] X3. Implement production auth and billing correctness before enabling workflow kernel by default.
- [ ] X4. Add/expand tests phase by phase and keep CI green at each milestone.
- [ ] X5. Run integration contract suite against a live API environment before sign-off.

## Milestones
- M1 (A+B+C): Foundations, policy, replay semantics, prompt binding/hash persistence.
- M2 (D): Production auth and billing correctness, progress/tokens, workflow-kernel rollout readiness.
- M3 (E+F+G+H+I): Vertical slice hardening, specialized document review, end-to-end acceptance tests.

## Final sign-off checklist
- [ ] Contract validation passes.
- [ ] All new and updated tests pass in CI.
- [ ] A-I acceptance criteria are verified with explicit evidence.
- [ ] Feature flags and rollout docs are updated for production enablement.
