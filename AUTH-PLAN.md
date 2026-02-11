# Plan

Implement Laravel Sanctum token validation in the intelligence layer by adding a dedicated auth adapter that resolves the authenticated student from `personal_access_tokens` + `students`, then enforcing that principal consistently across protected endpoints. The approach keeps existing adapter boundaries, minimizes API-surface disruption, and adds test coverage for both token validation and route authorization behavior.

## Scope
- In:
  - Add Sanctum Bearer token auth mode and adapter in `src/intelligence_layer_api/auth.py`.
  - Add auth config for Sanctum mode in `src/intelligence_layer_api/settings.py` and adapter wiring in `src/intelligence_layer_api/app.py`.
  - Enforce auth/ownership on currently unguarded workflow/query/action/cancel endpoints.
  - Add/expand tests under `tests/intelligence_layer_api/` for adapter behavior and endpoint authorization.
  - Document env/setup updates needed to run Sanctum mode.
- Out:
  - Changes to Laravel token issuance or Sanctum internals.
  - Frontend auth UX changes beyond compatibility notes.
  - Non-auth refactors unrelated to token validation and ownership checks.

## Action items
[x] Inventory all Layer 2 endpoints and map required auth + ownership policy, including SSE/outcomes/resolve/cancel paths in `src/intelligence_layer_api/app.py`.
[x] Add Sanctum auth configuration fields (mode selector + query/type/update toggles) in `src/intelligence_layer_api/settings.py` with safe defaults.
[x] Implement `SanctumBearerAuthAdapter` in `src/intelligence_layer_api/auth.py` to parse `Authorization: Bearer {id}|{plain}`, hash plaintext with SHA-256, validate token row + expiry, resolve student principal, and return `AuthResult`.
[x] Wire `IL_AUTH_MODE=sanctum_bearer` startup selection in `src/intelligence_layer_api/app.py` while preserving existing `platform_session` and `dev_bypass` behavior.
[x] Apply route-level auth hardening for `/v1/queries/{query_id}/events`, `/v1/workflows/{workflow_id}/events`, `/v1/workflows/{workflow_id}/outcomes`, `/v1/actions/{action_id}/resolve`, and cancel endpoints by verifying caller principal ownership.
[x] Add unit tests for Sanctum adapter success/failure cases (missing header, malformed token, hash mismatch, expired token, principal mismatch) in `tests/intelligence_layer_api/test_auth_adapter_production.py` or a dedicated Sanctum auth test file.
[x] Add API-level tests for newly protected endpoints to ensure unauthorized/forbidden scenarios are rejected and valid owner access still works.
[x] Run targeted test suite (`pytest tests/intelligence_layer_api/test_auth_adapter_production.py tests/intelligence_layer_api/test_no_tokens_on_auth_or_credit_failure.py`) plus any newly added tests; fix regressions before merge.
[x] Update docs/env guidance (`docs/LAYER2.md` and/or companion auth docs) with required DB permissions, env vars, and rollout sequence from `dev_bypass` to `sanctum_bearer`.

## Open questions
- Should Sanctum mode accept only `Authorization` Bearer tokens, or also keep cookie/session fallback for backward compatibility?
- Is `personal_access_tokens.last_used_at` update required in production, or should we stay strictly read-only from this service account?
- Do we want strict auth on workflow/event streaming endpoints immediately, or behind a temporary feature flag for staged rollout?
