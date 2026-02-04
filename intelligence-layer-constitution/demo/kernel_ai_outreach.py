from __future__ import annotations

import dataclasses
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Dict, Optional, Protocol, Tuple, Literal

from pydantic import BaseModel

from demo.kernel_email_review import (
    ManifestRegistry,
    SchemaRegistry,
    dotted_get,
    dotted_set,
    hash_object,
    interpolate_template,
    pick_first_nonempty_string,
    prune_nones,
    resolve_template_value,
    sha256_hex,
    stable_json_dumps,
    utc_now_iso,
)


class LLMClient(Protocol):
    async def complete(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
        ...

    def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:  # pragma: no cover
        ...


@dataclasses.dataclass
class WorkflowRun:
    query_id: str
    thread_id: int
    intent: Dict[str, Any]
    plan_template: Dict[str, Any]
    binding_ctx: Dict[str, Any]
    status: str = "running"  # running | waiting_action | completed | failed
    pending_action_id: Optional[str] = None
    pending_action_type: Optional[str] = None
    resume_step_index: Optional[int] = None
    outcomes: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)  # name -> outcome
    events: list[Dict[str, Any]] = dataclasses.field(default_factory=list)
    event_sink: Optional[Callable[[Dict[str, Any]], None]] = None
    final_text: str = ""


class MockLLM:
    """
    Offline LLM stub.

    This exists so the demo can run without network/API keys.
    """

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        class _Res:
            ok = True
            error = None

            def __init__(self, content: str):
                self.content = content

        # Very small heuristic "JSON mode" responder.
        text = str(messages)
        if "intent_type" in text and "Funding.Outreach.Email" in text:
            # Try to isolate the actual user message (our prompt includes the allowed intents too).
            user = ""
            marker = "User message:"
            if marker in text:
                user = text.split(marker, 1)[1].strip()
            else:
                user = text
            user = user.lower()
            if any(k in user for k in ("optimize", "improve", "rewrite", "humanize", "shorten")):
                payload = {
                    "intent_type": "Funding.Outreach.Email.Optimize",
                    "requested_edits": ["improve_clarity", "humanize"],
                    "custom_instructions": "Keep it concise and professional.",
                }
            else:
                payload = {"intent_type": "Funding.Outreach.Email.Review", "review_mode": "rubric"}
            return _Res(json.dumps(payload))

        if "optimized_email" in text:
            payload = {
                "subject": "Quick question about your lab’s work",
                "body": (
                    "Dear Prof. Demo,\n\n"
                    "I’m a prospective graduate applicant interested in your group’s work on X. "
                    "I recently worked on Y and would love to ask whether you’re taking students.\n\n"
                    "If helpful, I can share a short summary of my background and a CV.\n\n"
                    "Best regards,\n"
                    "Demo Student"
                ),
                "notes": "Mock optimization.",
            }
            return _Res(json.dumps(payload))

        if "email_review" in text:
            payload = {
                "verdict": "needs_edits",
                "overall_score": 0.78,
                "issues": [
                    {
                        "code": "SPECIFICITY",
                        "severity": "warning",
                        "message": "Be more specific about alignment with the professor’s work.",
                        "suggestion": "Mention 1 specific paper/topic and your related experience.",
                    }
                ],
                "notes": "Mock review.",
            }
            return _Res(json.dumps(payload))

        return _Res(json.dumps({"ok": True}))

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        class _Event:
            def __init__(self, type_: str, data: Any):
                self.type = type_
                self.data = data

        # Very small token stream for the assistant narrative.
        text = str(messages)
        if "NARRATE_RESULT" in text:
            out = "Here’s my review + an improved version. Want me to apply it?"
        else:
            out = "OK."

        for ch in out:
            yield _Event("TOKEN", ch)
        yield _Event("DONE", {"ok": True})


class AIDemoKernel:
    """
    Async demo kernel that can use an LLM client for:
    - switchboard intent detection
    - draft optimization + review
    - streaming the final assistant narrative
    """

    def __init__(self, repo_root: str, *, llm: Any | None = None):
        self._repo_root = repo_root
        self.schemas = SchemaRegistry(repo_root=repo_root)
        self.manifests = ManifestRegistry(repo_root=repo_root)
        self.platform_store_by_thread_id: Dict[int, Dict[str, Any]] = {}
        self.operator_idempotency: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.workflow_by_query_id: Dict[str, WorkflowRun] = {}
        self.workflow_by_action_id: Dict[str, str] = {}
        self.llm = llm or MockLLM()

    # -------------------------
    # Demo helpers
    # -------------------------
    def seed_platform_context(
        self,
        *,
        thread_id: int,
        student_id: int,
        funding_request_id: int,
        platform_email_subject: Optional[str],
        platform_email_body: Optional[str],
    ) -> None:
        self.platform_store_by_thread_id[thread_id] = {
            "student": {"id": student_id, "first_name": "Demo", "last_name": "Student", "email": "demo@example.com"},
            "funding_request": {
                "id": funding_request_id,
                "email_subject": platform_email_subject,
                "email_content": platform_email_body,
            },
            "email": {"id": None, "main_email_subject": platform_email_subject, "main_email_body": platform_email_body},
            "professor": {"id": 910, "full_name": "Prof. Demo", "email_address": "prof@example.edu", "department": "CS"},
            "institute": {"institution_name": "Demo University", "country": "CA"},
        }

    async def switchboard_intent(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Minimal switchboard: choose between Review and Optimize based on message.
        Uses LLM JSON mode when available; falls back to heuristics via MockLLM.
        """
        allowed_opt_edits = [
            "shorten",
            "humanize",
            "paraphrase",
            "add_bullets",
            "improve_clarity",
            "change_subject",
            "add_custom_hook",
        ]

        def normalize_requested_edits(edits: Any) -> list[str]:
            if not isinstance(edits, list):
                return []
            mapping = {
                "subject_line": "change_subject",
                "subject": "change_subject",
                "tone": "humanize",
                "clarity": "improve_clarity",
                "conciseness": "shorten",
                "shorter": "shorten",
                "call_to_action": "add_custom_hook",
                "cta": "add_custom_hook",
                "formatting": "add_bullets",
                "bullets": "add_bullets",
                "grammar": "improve_clarity",
                "personalization": "add_custom_hook",
                "hook": "add_custom_hook",
            }

            out: list[str] = []
            seen: set[str] = set()
            for raw in edits:
                if not isinstance(raw, str):
                    continue
                key = raw.strip().lower().replace(" ", "_")
                key = mapping.get(key, key)
                if key not in allowed_opt_edits:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return out

        prompt = {
            "role": "user",
            "content": (
                "Return JSON. Decide which intent to run for this user message.\n"
                "Allowed intent_type values:\n"
                "- Funding.Outreach.Email.Review\n"
                "- Funding.Outreach.Email.Optimize\n\n"
                "If intent_type is Optimize, also include requested_edits (array) and optional custom_instructions.\n"
                "If intent_type is Review, include review_mode='rubric'.\n\n"
                f"Allowed requested_edits values for Optimize:\n{json.dumps(allowed_opt_edits)}\n\n"
                f"User message:\n{user_message}\n\n"
                "Return JSON with keys: intent_type, requested_edits?, custom_instructions?, review_mode?."
            ),
        }

        class RespSchema(BaseModel):
            intent_type: Literal["Funding.Outreach.Email.Review", "Funding.Outreach.Email.Optimize"]
            requested_edits: Literal[
                "shorten",
                "humanize",
                "paraphrase",
                "add_bullets",
                "improve_clarity",
                "change_subject",
                "add_custom_hook",
            ]
            custom_instructions: Optional[str]

        res = await self.llm.complete([{"role": "system", "content": "Return JSON only."}, prompt], response_format="json_object")
        if not getattr(res, "ok", True):
            # Hard fallback: heuristic
            msg = user_message.lower()
            if any(k in msg for k in ("optimize", "improve", "rewrite", "humanize", "shorten")):
                return "Funding.Outreach.Email.Optimize", {"requested_edits": ["improve_clarity", "humanize"]}
            return "Funding.Outreach.Email.Review", {"review_mode": "rubric"}

        try:
            data = json.loads(getattr(res, "content", "") or "{}")
        except json.JSONDecodeError:
            data = {}

        intent_type = data.get("intent_type") or "Funding.Outreach.Email.Review"
        inputs: Dict[str, Any] = {}
        if intent_type == "Funding.Outreach.Email.Optimize":
            edits = normalize_requested_edits(data.get("requested_edits"))
            if not edits:
                edits = ["improve_clarity"]
            inputs["requested_edits"] = edits
            if isinstance(data.get("custom_instructions"), str) and data["custom_instructions"].strip():
                inputs["custom_instructions"] = data["custom_instructions"].strip()
        else:
            inputs["review_mode"] = "rubric"
        return intent_type, inputs

    # -------------------------
    # Workflow lifecycle
    # -------------------------
    async def submit_intent(
        self,
        *,
        intent_type: str,
        thread_id: int,
        student_id: int,
        funding_request_id: int,
        inputs: Dict[str, Any],
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> WorkflowRun:
        intent = self._build_intent(
            intent_type=intent_type,
            thread_id=thread_id,
            student_id=student_id,
            funding_request_id=funding_request_id,
            inputs=inputs,
        )

        intent_schema = self.manifests.intent_schema_by_type[intent["intent_type"]]
        self.schemas.validate(intent, intent_schema)

        plan_template_path = self.manifests.plan_template_by_intent_type[intent["intent_type"]]
        plan_template = self._load_json(plan_template_path)
        self.schemas.validate(plan_template, "schemas/plans/plan_template.v1.json")

        query_id = str(uuid.uuid4())
        run = WorkflowRun(
            query_id=query_id,
            thread_id=thread_id,
            intent=intent,
            plan_template=plan_template,
            binding_ctx={
                "tenant_id": intent["actor"]["tenant_id"],
                "thread_id": thread_id,
                "intent": intent,
                "context": {},
                "outcome": {},
                "steps": {},
                "computed": {},
            },
            event_sink=event_sink,
        )
        self.workflow_by_query_id[query_id] = run
        return run

    async def run_to_completion(self, run: WorkflowRun) -> WorkflowRun:
        try:
            await self._execute_plan(run, start_step_index=0)
        except Exception as exc:
            run.status = "failed"
            self._emit_error(run, code="WORKFLOW_EXEC_ERROR", message=str(exc))
            self._emit_final(run, summary="Workflow failed.")
        return run

    async def resolve_action_and_resume(self, *, action_id: str, accepted: bool, payload: Dict[str, Any]) -> WorkflowRun:
        query_id = self.workflow_by_action_id.get(action_id)
        if not query_id:
            raise KeyError(f"Unknown action_id: {action_id}")
        run = self.workflow_by_query_id[query_id]

        if not accepted:
            run.status = "failed"
            self._emit_final(run, summary="Action declined; workflow stopped.")
            return run

        # Record resolution (demo operator) and resume.
        gate_intent = self._build_intent(
            intent_type="Workflow.Gate.Resolve",
            thread_id=run.thread_id,
            student_id=int(run.intent["actor"]["principal"]["id"]),
            funding_request_id=int(run.intent["scope"]["scope_id"]),
            inputs={"action_id": action_id, "status": "accepted", "payload": payload},
        )
        gate_schema = self.manifests.intent_schema_by_type[gate_intent["intent_type"]]
        self.schemas.validate(gate_intent, gate_schema)

        gate_tpl_path = self.manifests.plan_template_by_intent_type[gate_intent["intent_type"]]
        gate_tpl = self._load_json(gate_tpl_path)
        self.schemas.validate(gate_tpl, "schemas/plans/plan_template.v1.json")

        gate_run = WorkflowRun(
            query_id=str(uuid.uuid4()),
            thread_id=run.thread_id,
            intent=gate_intent,
            plan_template=gate_tpl,
            binding_ctx={
                "tenant_id": gate_intent["actor"]["tenant_id"],
                "thread_id": run.thread_id,
                "intent": gate_intent,
                "context": {},
                "outcome": {},
                "steps": {},
                "computed": {},
            },
        )
        await self._execute_plan(gate_run, start_step_index=0)

        # Apply collected fields to the original intent inputs (demo behavior).
        if isinstance(run.intent.get("inputs"), dict) and isinstance(payload, dict):
            for k, v in payload.items():
                if v is None:
                    run.intent["inputs"].pop(k, None)
                else:
                    run.intent["inputs"][k] = v

        run.binding_ctx["intent"] = run.intent
        run.binding_ctx["computed"] = {}

        resume_at = run.resume_step_index or 0
        run.status = "running"
        self._emit_progress(run, percent=30, stage="resuming", message="Resuming after action resolution…")
        await self._execute_plan(run, start_step_index=resume_at)
        return run

    # -------------------------
    # Core plan execution
    # -------------------------
    async def _execute_plan(self, run: WorkflowRun, *, start_step_index: int) -> None:
        steps = run.plan_template["steps"]
        for idx in range(start_step_index, len(steps)):
            step = steps[idx]
            step_id = step["step_id"]
            kind = step["kind"]

            self._emit_progress(run, percent=min(95, 10 + idx * 15), stage=step_id, message=f"Running {step_id} ({kind})")

            if kind == "policy_check":
                ok, action = self._run_policy_check(step["check"], run.binding_ctx, thread_id=run.thread_id, query_id=run.query_id)
                if not ok:
                    action_id = action["payload"]["action_id"]
                    run.status = "waiting_action"
                    run.pending_action_id = action_id
                    run.pending_action_type = action["payload"]["action_type"]
                    run.resume_step_index = idx + 1
                    self.workflow_by_action_id[action_id] = run.query_id
                    self._emit_event(run, action)
                    self._emit_progress(run, percent=25, stage="paused", message="Paused: waiting for required user input.")
                    return
                continue

            if kind == "human_gate":
                # Stream a user-facing message before requesting approval.
                await self._stream_narrative(run, stage="pre_gate")

                action_id = str(uuid.uuid4())
                run.status = "waiting_action"
                run.pending_action_id = action_id
                run.pending_action_type = step["gate"]["gate_type"]
                run.resume_step_index = idx + 1
                self.workflow_by_action_id[action_id] = run.query_id

                proposed = None
                target_ref = step["gate"].get("target_outcome_ref")
                if target_ref:
                    target = resolve_template_value(target_ref, run.binding_ctx)
                    if isinstance(target, dict):
                        proposed = target.get("payload") or target

                self._emit_event(
                    run,
                    self._make_action_required(
                        thread_id=run.thread_id,
                        query_id=run.query_id,
                        action_id=action_id,
                        action_type=step["gate"]["gate_type"],
                        title=step["gate"]["title"],
                        description=step["gate"]["description"],
                        reason_code=step["gate"].get("reason_code"),
                        ui_hints=step["gate"].get("ui_hints") or {},
                        proposed_changes=proposed,
                        data={"source_step_id": step_id},
                    ),
                )
                self._emit_progress(run, percent=80, stage="paused", message="Paused: waiting for user approval.")
                return

            if kind != "operator":
                raise ValueError(f"Demo executor only supports operator/policy_check/human_gate steps. Got: {kind}")

            operator_name = step["operator_name"]
            operator_version = step["operator_version"]
            payload_template = step["payload"]

            payload = prune_nones(resolve_template_value(payload_template, run.binding_ctx))

            # Compute `computed.*` values when required for idempotency keys.
            self._populate_computed_for_step(run.binding_ctx, operator_name, payload)

            idempotency_key = interpolate_template(step["idempotency_template"], run.binding_ctx)

            manifest = self.manifests.get_operator_manifest(operator_name, operator_version)

            operator_call = {
                "schema_version": "1.0",
                "payload": payload,
                "idempotency_key": idempotency_key,
                "auth_context": {
                    "tenant_id": run.intent["actor"]["tenant_id"],
                    "principal": run.intent["actor"]["principal"],
                    "scopes": run.intent["actor"].get("scopes", []),
                },
                "trace_context": {
                    "correlation_id": run.intent.get("correlation_id"),
                    "workflow_id": run.intent.get("intent_id"),
                    "step_id": step_id,
                    "job_id": str(uuid.uuid4()),
                },
            }
            self.schemas.validate(operator_call, manifest.input_schema)

            operator_result = await self._run_operator(operator_name, operator_call)
            self.schemas.validate(operator_result, manifest.output_schema)
            if operator_result.get("status") != "succeeded":
                run.status = "failed"
                err = operator_result.get("error") or {}
                self._emit_error(run, code=err.get("code", "OPERATOR_FAILED"), message=err.get("message", "Operator failed."))
                self._emit_final(run, summary="Workflow failed.")
                return

            self._apply_produces(run.binding_ctx, produces=step.get("produces", []), operator_result=operator_result)
            self._capture_outcomes_from_produces(run, produces=step.get("produces", []))

        run.status = "completed"

        # Stream a user-facing narrative at completion.
        if run.intent.get("intent_type") != "Workflow.Gate.Resolve" and not run.final_text:
            await self._stream_narrative(run, stage="final")

        summary = f"Workflow completed ({run.intent.get('intent_type')})."
        self._emit_final(run, summary=summary)

    def _populate_computed_for_step(self, binding_ctx: Dict[str, Any], operator_name: str, payload: Dict[str, Any]) -> None:
        if operator_name == "Email.ReviewDraft":
            effective_body = pick_first_nonempty_string([payload.get("body"), payload.get("fallback_body")]) or ""
            dotted_set(binding_ctx, "computed.email_body_hash", sha256_hex(effective_body))
        if operator_name == "Email.OptimizeDraft":
            effective_body = pick_first_nonempty_string([payload.get("current_body"), payload.get("fallback_body")]) or ""
            dotted_set(binding_ctx, "computed.email_body_hash", sha256_hex(effective_body))
            dotted_set(binding_ctx, "computed.requested_edits_hash", sha256_hex(stable_json_dumps(payload.get("requested_edits") or [])))

    # -------------------------
    # Policy checks
    # -------------------------
    def _run_policy_check(self, check: Dict[str, Any], binding_ctx: Dict[str, Any], *, thread_id: int, query_id: str) -> Tuple[bool, Dict[str, Any]]:
        name = check["check_name"]
        params = check.get("params", {})

        if name == "EnsureEmailPresent":
            sources = params.get("sources", [])
            vals = [dotted_get(binding_ctx, p) for p in sources]
            if pick_first_nonempty_string(vals) is not None:
                return True, {}
            action_id = str(uuid.uuid4())
            return False, self._make_action_required(
                thread_id=thread_id,
                query_id=query_id,
                action_id=action_id,
                action_type=params.get("on_missing_action_type", "collect_fields"),
                title="Email text required",
                description="Provide the email body to review/optimize (either override or ensure the platform has a draft).",
                data={"missing_sources": sources},
            )

        raise ValueError(f"Unknown check_name: {name}")

    # -------------------------
    # Operators
    # -------------------------
    async def _run_operator(self, operator_name: str, call: Dict[str, Any]) -> Dict[str, Any]:
        key = (operator_name, call["idempotency_key"])
        cached = self.operator_idempotency.get(key)
        if cached is not None:
            return cached

        if operator_name == "Platform.Context.Load":
            res = self._op_platform_context_load(call)
        elif operator_name == "Email.OptimizeDraft":
            res = await self._op_email_optimize_draft(call)
        elif operator_name == "Email.ReviewDraft":
            res = await self._op_email_review_draft(call)
        elif operator_name == "Email.ApplyToPlatform.Propose":
            res = self._op_email_apply_to_platform_propose(call)
        elif operator_name == "Workflow.Gate.Resolve":
            res = self._op_workflow_gate_resolve(call)
        else:
            raise ValueError(f"Operator not implemented in demo: {operator_name}")

        self.operator_idempotency[key] = res
        return res

    def _op_platform_context_load(self, call: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = int(call["payload"]["thread_id"])
        platform = self.platform_store_by_thread_id.get(thread_id)
        if not platform:
            return self._op_error_result(code="PLATFORM_CONTEXT_NOT_FOUND", message=f"No platform context for thread_id={thread_id}")

        fr: Dict[str, Any] = {"id": platform["funding_request"]["id"]}
        if platform["funding_request"].get("email_subject") is not None:
            fr["email_subject"] = platform["funding_request"]["email_subject"]
        if platform["funding_request"].get("email_content") is not None:
            fr["email_content"] = platform["funding_request"]["email_content"]

        email: Dict[str, Any] = {}
        if platform.get("email", {}).get("id") is not None:
            email["id"] = platform["email"]["id"]
        if platform.get("email", {}).get("main_email_subject") is not None:
            email["main_email_subject"] = platform["email"]["main_email_subject"]
        if platform.get("email", {}).get("main_email_body") is not None:
            email["main_email_body"] = platform["email"]["main_email_body"]

        result = {
            "platform": {
                "student": platform["student"],
                "funding_request": fr,
                "email": email or None,
                "professor": platform["professor"],
                "institute": platform["institute"],
                "metas": None,
                "reply": None,
            },
            "intelligence": {
                "requirements": {},
                "preferences": {},
                "background": {},
                "composer_prereqs": {},
                "gmail_connected": False,
                "templates_finalized": False,
            },
            "platform_context_hash": hash_object(platform),
            "student_background_hash": hash_object({}),
            "composer_prereqs_hash": hash_object({}),
        }

        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": result,
            "artifacts": [],
            "metrics": {"latency_ms": 1, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    async def _op_email_optimize_draft(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call["payload"]
        current_subject = pick_first_nonempty_string([payload.get("current_subject"), payload.get("fallback_subject")]) or ""
        current_body = pick_first_nonempty_string([payload.get("current_body"), payload.get("fallback_body")]) or ""

        professor = payload.get("professor") or {}
        requested_edits = payload.get("requested_edits") or []
        custom_instructions = payload.get("custom_instructions") or ""

        # Ask the LLM to output JSON.
        messages = [
            {"role": "system", "content": "You are an expert outreach email editor. Return JSON only."},
            {
                "role": "user",
                "content": (
                    "optimized_email\n"
                    "Return JSON with keys: subject (string), body (string), notes (string optional).\n\n"
                    f"Professor context: {json.dumps(professor, ensure_ascii=False)}\n"
                    f"Requested edits: {json.dumps(requested_edits)}\n"
                    f"Custom instructions: {custom_instructions}\n\n"
                    f"Current subject: {current_subject}\n"
                    f"Current body:\n{current_body}\n\n"
                    "Now produce an improved subject and body."
                ),
            },
        ]

        res = await self.llm.complete(messages, response_format="json_object")
        if not getattr(res, "ok", True):
            return self._op_error_result(code="LLM_OPTIMIZE_FAILED", message=str(getattr(res, "error", "unknown error")))

        try:
            out = json.loads(getattr(res, "content", "") or "{}")
        except json.JSONDecodeError:
            return self._op_error_result(code="LLM_OPTIMIZE_INVALID_JSON", message="Model did not return valid JSON.")

        subject = out.get("subject")
        body = out.get("body")
        if not isinstance(subject, str) or not subject.strip():
            return self._op_error_result(code="LLM_OPTIMIZE_BAD_SUBJECT", message="Missing/invalid subject.")
        if not isinstance(body, str) or not body.strip():
            return self._op_error_result(code="LLM_OPTIMIZE_BAD_BODY", message="Missing/invalid body.")

        outcome_payload = {
            "funding_request_id": int(payload.get("funding_request", {}).get("id") or 0) or None,
            "subject": subject.strip(),
            "body": body.strip(),
            "custom_instructions": custom_instructions or None,
            "notes": out.get("notes"),
        }
        # Prune Nones so we don't store nulls where schemas expect absent.
        outcome_payload = prune_nones(outcome_payload)

        outcome = {
            "schema_version": "1.0",
            "outcome_id": str(uuid.uuid4()),
            "outcome_type": "Email.Draft",
            "created_at": utc_now_iso(),
            "hash": hash_object(outcome_payload),
            "producer": {"name": "Email.OptimizeDraft", "version": "1.0.0", "plugin_type": "operator"},
            "payload": outcome_payload,
        }
        self.schemas.validate(outcome, "schemas/outcomes/email_draft.v1.json")

        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": {"outcome": outcome},
            "artifacts": [],
            "metrics": {"latency_ms": 10, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": True, "reasons": ["llm"], "stability": "medium"},
        }

    async def _op_email_review_draft(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call["payload"]

        draft = payload.get("draft")
        if isinstance(draft, dict) and isinstance(draft.get("payload"), dict):
            subject = draft["payload"].get("subject") or ""
            body = draft["payload"].get("body") or ""
        else:
            subject = pick_first_nonempty_string([payload.get("subject"), payload.get("fallback_subject")]) or ""
            body = pick_first_nonempty_string([payload.get("body"), payload.get("fallback_body")]) or ""

        professor = payload.get("professor") or {}
        custom_instructions = payload.get("custom_instructions") or ""

        messages = [
            {"role": "system", "content": "You are a strict outreach email reviewer. Return JSON only."},
            {
                "role": "user",
                "content": (
                    "email_review\n"
                    "Return JSON with keys:\n"
                    "- verdict: one of [pass, needs_edits, unsafe]\n"
                    "- overall_score: number 0..1\n"
                    "- issues: array of {code, severity(info|warning|error), message, suggestion?}\n"
                    "- notes: string optional\n\n"
                    f"Professor context: {json.dumps(professor, ensure_ascii=False)}\n"
                    f"Custom instructions: {custom_instructions}\n\n"
                    f"Email subject: {subject}\n"
                    f"Email body:\n{body}\n\n"
                    "Review this email."
                ),
            },
        ]

        res = await self.llm.complete(messages, response_format="json_object")
        if not getattr(res, "ok", True):
            return self._op_error_result(code="LLM_REVIEW_FAILED", message=str(getattr(res, "error", "unknown error")))

        try:
            out = json.loads(getattr(res, "content", "") or "{}")
        except json.JSONDecodeError:
            return self._op_error_result(code="LLM_REVIEW_INVALID_JSON", message="Model did not return valid JSON.")

        verdict = out.get("verdict")
        overall_score = out.get("overall_score")
        issues = out.get("issues")
        if verdict not in ("pass", "needs_edits", "unsafe"):
            verdict = "needs_edits"
        if not isinstance(overall_score, (int, float)):
            overall_score = 0.6
        if not isinstance(issues, list):
            issues = []

        outcome_payload = prune_nones(
            {
                "verdict": verdict,
                "overall_score": float(max(0.0, min(1.0, overall_score))),
                "issues": issues,
                "notes": out.get("notes"),
            }
        )

        outcome = {
            "schema_version": "1.0",
            "outcome_id": str(uuid.uuid4()),
            "outcome_type": "Email.Review",
            "created_at": utc_now_iso(),
            "hash": hash_object(outcome_payload),
            "producer": {"name": "Email.ReviewDraft", "version": "1.0.0", "plugin_type": "operator"},
            "payload": outcome_payload,
        }
        self.schemas.validate(outcome, "schemas/outcomes/email_review.v1.json")

        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": {"outcome": outcome},
            "artifacts": [],
            "metrics": {"latency_ms": 10, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": True, "reasons": ["llm"], "stability": "medium"},
        }

    def _op_email_apply_to_platform_propose(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call["payload"]
        fr_id = int(payload["funding_request_id"])
        draft = payload["draft"]
        draft_payload = (draft.get("payload") or {}) if isinstance(draft, dict) else {}

        patch = {
            "schema_version": "1.0",
            "patch_id": str(uuid.uuid4()),
            "targets": [
                {
                    "table": "funding_requests",
                    "where": {"id": fr_id},
                    "set": {
                        "email_subject": draft_payload.get("subject"),
                        "email_content": draft_payload.get("body"),
                    },
                }
            ],
            "human_summary": "Update outreach email subject/body from optimized draft.",
            "risk_level": "medium",
            "requires_approval": True,
            "idempotency_key": call.get("idempotency_key"),
        }
        patch = prune_nones(patch)

        outcome_payload = patch
        outcome = {
            "schema_version": "1.0",
            "outcome_id": str(uuid.uuid4()),
            "outcome_type": "PlatformPatch.Proposal",
            "created_at": utc_now_iso(),
            "hash": hash_object(outcome_payload),
            "producer": {"name": "Email.ApplyToPlatform.Propose", "version": "1.0.0", "plugin_type": "operator"},
            "payload": outcome_payload,
        }
        self.schemas.validate(outcome, "schemas/outcomes/platform_patch_proposal.v1.json")

        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": {"outcome": outcome},
            "artifacts": [],
            "metrics": {"latency_ms": 1, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    def _op_workflow_gate_resolve(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call["payload"]
        result: Dict[str, Any] = {
            "action_id": payload["action_id"],
            "status": payload["status"],
            "resolved_at": utc_now_iso(),
            "message": "Recorded gate resolution (demo).",
        }
        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": result,
            "artifacts": [],
            "metrics": {"latency_ms": 1, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    def _op_error_result(self, *, code: str, message: str) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "status": "failed",
            "result": None,
            "artifacts": [],
            "metrics": {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": {"code": code, "message": message, "category": "dependency", "retryable": False},
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    # -------------------------
    # Produces wiring
    # -------------------------
    def _apply_produces(self, binding_ctx: Dict[str, Any], *, produces: list[str], operator_result: Dict[str, Any]) -> None:
        result = operator_result.get("result") or {}
        for key in produces:
            if key.startswith("context."):
                field = key[len("context.") :]
                dotted_set(binding_ctx, key, dotted_get(result, field))
                continue
            if key.startswith("outcome."):
                name = key[len("outcome.") :]
                val = result.get("outcome", result.get(name))
                dotted_set(binding_ctx, key, val)
                continue
            if key.startswith("steps."):
                if ".result." in key:
                    _, _, rest = key.partition(".result.")
                    dotted_set(binding_ctx, key, result.get(rest))
                else:
                    dotted_set(binding_ctx, key, result)
                continue
            if key in result:
                dotted_set(binding_ctx, key, result[key])

    def _capture_outcomes_from_produces(self, run: WorkflowRun, *, produces: list[str]) -> None:
        for key in produces:
            if not key.startswith("outcome."):
                continue
            name = key[len("outcome.") :]
            out = dotted_get(run.binding_ctx, f"outcome.{name}")
            if isinstance(out, dict) and out.get("outcome_id") and out.get("outcome_type"):
                run.outcomes[name] = out

    # -------------------------
    # Narrative streaming (token_delta)
    # -------------------------
    async def _stream_narrative(self, run: WorkflowRun, *, stage: str) -> None:
        # Avoid narrating for internal intents.
        if run.intent.get("intent_type") in ("Workflow.Gate.Resolve",):
            return

        email_draft = run.outcomes.get("email_draft")
        email_review = run.outcomes.get("email_review")
        patch = run.outcomes.get("platform_patch_proposal")

        context = {
            "intent_type": run.intent.get("intent_type"),
            "stage": stage,
            "email_draft": email_draft.get("payload") if isinstance(email_draft, dict) else None,
            "email_review": email_review.get("payload") if isinstance(email_review, dict) else None,
            "patch_proposal": patch.get("payload") if isinstance(patch, dict) else None,
        }

        messages = [
            {"role": "system", "content": "You are Dana. Produce a concise, helpful response."},
            {
                "role": "user",
                "content": (
                    "NARRATE_RESULT\n"
                    "Write a user-facing response that:\n"
                    "1) briefly explains the review verdict and top issues,\n"
                    "2) shows the optimized subject/body if available,\n"
                    "3) ends with a clear question to apply changes if a patch_proposal exists.\n\n"
                    f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
                ),
            },
        ]

        # If llm.stream isn't available (or is sync), skip narration.
        stream = getattr(self.llm, "stream", None)
        if stream is None:
            return

        async for ev in stream(messages):
            t = getattr(ev, "type", None)
            t_name = getattr(t, "name", None)
            if t in ("TOKEN", "token") or t_name == "TOKEN":
                delta = getattr(ev, "data", "")
                if isinstance(delta, str) and delta:
                    run.final_text += delta
                    self._emit_token_delta(run, delta=delta)
            if t in ("DONE", "done") or t_name == "DONE":
                break

    # -------------------------
    # Events (validated)
    # -------------------------
    def _make_event_base(self, *, thread_id: int, query_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "thread_id": thread_id,
            "query_id": query_id,
            "ts": utc_now_iso(),
            "payload": payload,
        }

    def _emit_progress(self, run: WorkflowRun, *, percent: int, stage: str, message: str) -> None:
        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="progress",
            payload={"percent": percent, "stage": stage, "message": message},
        )
        self.schemas.validate(event, "schemas/sse/progress.v1.json")
        self._emit_event(run, event)

    def _emit_token_delta(self, run: WorkflowRun, *, delta: str) -> None:
        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="token_delta",
            payload={"delta": delta, "role": "assistant"},
        )
        self.schemas.validate(event, "schemas/sse/token_delta.v1.json")
        self._emit_event(run, event)

    def _make_action_required(
        self,
        *,
        thread_id: int,
        query_id: str,
        action_id: str,
        action_type: str,
        title: str,
        description: str,
        data: Dict[str, Any],
        reason_code: str | None = None,
        ui_hints: Dict[str, Any] | None = None,
        proposed_changes: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action_id": action_id,
            "action_type": action_type,
            "title": title,
            "description": description,
            "requires_user_input": True,
            "ui_hints": ui_hints or {"primary_button": "Submit", "secondary_button": "Cancel"},
            "data": data,
        }
        if reason_code:
            payload["reason_code"] = reason_code
        if proposed_changes:
            payload["proposed_changes"] = proposed_changes

        event = self._make_event_base(
            thread_id=thread_id,
            query_id=query_id,
            event_type="action_required",
            payload=payload,
        )
        self.schemas.validate(event, "schemas/sse/action_required.v1.json")
        return event

    def _emit_final(self, run: WorkflowRun, *, summary: str) -> None:
        outcomes_emitted = []
        for _, out in run.outcomes.items():
            outcomes_emitted.append({"outcome_id": out["outcome_id"], "outcome_type": out["outcome_type"], "hash": out["hash"]})

        payload: Dict[str, Any] = {"summary": summary, "outcomes_emitted": outcomes_emitted, "actions_emitted": []}
        if run.final_text:
            payload["final_text"] = run.final_text

        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="final",
            payload=payload,
        )
        self.schemas.validate(event, "schemas/sse/final.v1.json")
        self._emit_event(run, event)

    def _emit_error(self, run: WorkflowRun, *, code: str, message: str) -> None:
        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="error",
            payload={"error": {"code": code, "message": message, "category": "operator_bug", "retryable": False}},
        )
        self.schemas.validate(event, "schemas/sse/error.v1.json")
        self._emit_event(run, event)

    def _emit_event(self, run: WorkflowRun, event: Dict[str, Any]) -> None:
        self.schemas.validate(event, "schemas/sse/event.v1.json")
        run.events.append(event)
        if run.event_sink:
            run.event_sink(event)

    # -------------------------
    # Intent builders
    # -------------------------
    def _build_intent(
        self,
        *,
        intent_type: str,
        thread_id: int,
        student_id: int,
        funding_request_id: int,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "intent_id": str(uuid.uuid4()),
            "intent_type": intent_type,
            "actor": {
                "tenant_id": 1,
                "principal": {"type": "student", "id": student_id},
                "scopes": ["funding:outreach"],
                "trust_level": 1,
            },
            "source": "api",
            "thread_id": thread_id,
            "scope": {"scope_type": "funding_request", "scope_id": str(funding_request_id)},
            "inputs": inputs,
            "constraints": {},
            "context_refs": {},
            "data_classes": ["Confidential"],
            "correlation_id": str(uuid.uuid4()),
            "created_at": utc_now_iso(),
        }

    # -------------------------
    # Local file loads
    # -------------------------
    def _load_json(self, repo_rel_path: str) -> Dict[str, Any]:
        path = os.path.join(self._repo_root, repo_rel_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
