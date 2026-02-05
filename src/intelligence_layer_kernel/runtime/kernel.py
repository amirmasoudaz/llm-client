from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3
from jsonschema import Draft202012Validator

from ..contracts import ContractRegistry
from ..events import EventWriter, LedgerEvent
from ..policy import PolicyEngine, PolicyDecisionStore, PolicyContext
from ..operators import OperatorExecutor, AuthContext, TraceContext as OperatorTraceContext, OperatorCall
from .bindings import BindingContext, resolve_template_value, render_template, set_path, prune_nulls
from .store import IntentStore, PlanStore, WorkflowStore, OutcomeStore, GateStore
from .switchboard import IntentSwitchboard
from .types import IntentRecord, PlanRecord, WorkflowResult


@dataclass(frozen=True)
class WorkflowKernel:
    contracts: ContractRegistry
    operator_executor: OperatorExecutor
    policy_engine: PolicyEngine
    policy_store: PolicyDecisionStore
    event_writer: EventWriter
    pool: Any
    tenant_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_intent_store", IntentStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_plan_store", PlanStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_workflow_store", WorkflowStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_outcome_store", OutcomeStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_gate_store", GateStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_switchboard", IntentSwitchboard())

    async def handle_message(
        self,
        *,
        thread_id: int,
        scope_type: str,
        scope_id: str,
        actor: dict[str, Any],
        message: str,
        source: str = "chat",
        workflow_id: uuid.UUID | None = None,
    ) -> WorkflowResult:
        intent_type, inputs = self._switchboard.classify(message)
        return await self.start_intent(
            intent_type=intent_type,
            inputs=inputs,
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=scope_id,
            actor=actor,
            source=source,
            workflow_id=workflow_id,
        )

    async def start_intent(
        self,
        *,
        intent_type: str,
        inputs: dict[str, Any],
        thread_id: int,
        scope_type: str,
        scope_id: str,
        actor: dict[str, Any],
        source: str = "chat",
        execution_mode: str = "live",
        replay_mode: str = "reproduce",
        parent_workflow_id: uuid.UUID | None = None,
        workflow_id: uuid.UUID | None = None,
    ) -> WorkflowResult:
        intent = self._build_intent(
            intent_type=intent_type,
            inputs=inputs,
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=scope_id,
            actor=actor,
            source=source,
        )
        self._validate_intent(intent)

        plan = self._build_plan(intent)
        workflow_id = workflow_id or uuid.uuid4()
        await self._emit_progress(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage="classifying_intent",
            detail={"intent_type": intent.intent_type},
        )
        plan_payload = plan.to_dict()
        plan_payload.pop("created_at", None)
        plan_hash = _hash_json(plan_payload).digest()

        await self._intent_store.insert(intent)
        await self._plan_store.insert(plan, plan_hash)

        await self._workflow_store.create_run(
            workflow_id=workflow_id,
            correlation_id=intent.correlation_id,
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=scope_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            status="running",
            execution_mode=execution_mode,
            replay_mode=replay_mode,
            parent_workflow_id=parent_workflow_id,
        )
        await self._workflow_store.create_steps(workflow_id=workflow_id, steps=plan.steps)

        await self._emit_progress(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage="workflow_started",
            detail={"intent_type": intent.intent_type},
        )

        return await self._execute_workflow(
            workflow_id=workflow_id,
            intent=intent,
            plan=plan,
            actor=actor,
        )

    async def resolve_action(
        self,
        *,
        action_id: str,
        status: str,
        payload: dict[str, Any],
        actor: dict[str, Any],
        source: str = "api",
    ) -> WorkflowResult:
        try:
            gate_id = uuid.UUID(action_id)
        except ValueError:
            raise ValueError("action_id must be a UUID")

        gate = await self._gate_store.get_gate(gate_id=gate_id)
        if gate is None:
            raise ValueError("gate not found")

        run = await self._workflow_store.get_run(workflow_id=gate["workflow_id"])
        if run is None or run.thread_id is None:
            raise ValueError("workflow run not found for gate")

        intent_type = "Workflow.Gate.Resolve"
        inputs = {"action_id": action_id, "status": status, "payload": payload}
        resolution = await self.start_intent(
            intent_type=intent_type,
            inputs=inputs,
            thread_id=run.thread_id,
            scope_type=run.scope_type or "funding_request",
            scope_id=run.scope_id or str(run.thread_id),
            actor=actor,
            source=source,
            parent_workflow_id=gate["workflow_id"],
        )

        if status == "accepted":
            await self._execute_existing_workflow(
                workflow_id=gate["workflow_id"],
                actor=actor,
            )

        return resolution

    def _build_intent(
        self,
        *,
        intent_type: str,
        inputs: dict[str, Any],
        thread_id: int,
        scope_type: str,
        scope_id: str,
        actor: dict[str, Any],
        source: str,
    ) -> IntentRecord:
        return IntentRecord(
            intent_id=uuid.uuid4(),
            intent_type=intent_type,
            schema_version="1.0",
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=scope_id,
            actor=actor,
            source=source,
            inputs=inputs,
            constraints={},
            context_refs={},
            data_classes=[],
            correlation_id=uuid.uuid4(),
            created_at=datetime.now(timezone.utc),
        )

    def _validate_intent(self, intent: IntentRecord) -> None:
        schema = self.contracts.get_intent_schema(intent.intent_type)
        validator = Draft202012Validator(schema, resolver=self.contracts.resolver_for(schema))
        errors = list(validator.iter_errors(intent.to_dict()))
        if errors:
            messages = "; ".join(err.message for err in errors)
            raise ValueError(f"intent validation failed: {messages}")

    def _build_plan(self, intent: IntentRecord) -> PlanRecord:
        template = self.contracts.get_plan_template(intent.intent_type)
        return PlanRecord(
            plan_id=uuid.uuid4(),
            intent_id=intent.intent_id,
            plan_version=str(template.get("plan_version") or "planner-1.0"),
            intent_type=intent.intent_type,
            thread_id=intent.thread_id,
            steps=list(template.get("steps") or []),
            created_at=datetime.now(timezone.utc),
        )

    async def _execute_existing_workflow(
        self,
        *,
        workflow_id: uuid.UUID,
        actor: dict[str, Any],
    ) -> None:
        run = await self._workflow_store.get_run(workflow_id=workflow_id)
        if run is None or run.plan_id is None:
            return
        plan_row = await self._plan_store.fetch(run.plan_id)
        intent_row = await self._intent_store.fetch(run.intent_id)
        if not plan_row or not intent_row:
            return
        plan_dict = plan_row["plan"]
        plan = PlanRecord(
            plan_id=plan_row["plan_id"],
            intent_id=plan_row["intent_id"],
            plan_version=str(plan_dict.get("plan_version") or "planner-1.0"),
            intent_type=str(plan_dict.get("intent_type") or intent_row["intent_type"]),
            thread_id=int(plan_dict.get("thread_id") or intent_row["thread_id"]),
            steps=list(plan_dict.get("steps") or []),
            created_at=datetime.now(timezone.utc),
        )
        intent = IntentRecord(
            intent_id=intent_row["intent_id"],
            intent_type=intent_row["intent_type"],
            schema_version=intent_row["schema_version"],
            thread_id=intent_row["thread_id"],
            scope_type=intent_row["scope_type"],
            scope_id=intent_row["scope_id"],
            actor=intent_row["actor"],
            source=intent_row["source"],
            inputs=intent_row["inputs"],
            constraints=intent_row["constraints"],
            context_refs=intent_row["context_refs"],
            data_classes=intent_row["data_classes"],
            correlation_id=intent_row["correlation_id"],
            created_at=intent_row["created_at"],
        )
        await self._workflow_store.update_run_status(workflow_id=workflow_id, status="running")
        await self._execute_workflow(
            workflow_id=workflow_id,
            intent=intent,
            plan=plan,
            actor=actor,
            resume_only=True,
        )

    async def _execute_workflow(
        self,
        *,
        workflow_id: uuid.UUID,
        intent: IntentRecord,
        plan: PlanRecord,
        actor: dict[str, Any],
        resume_only: bool = False,
    ) -> WorkflowResult:
        context = await self._build_context(intent=intent, plan=plan, workflow_id=workflow_id)
        ctx = BindingContext(context)

        policy_ctx = PolicyContext(
            stage="plan",
            effects=[],
            policy_tags=[],
            data_classes=intent.data_classes,
            auth_context=actor,
            trace_context={
                "workflow_id": str(workflow_id),
                "intent_id": str(intent.intent_id),
                "plan_id": str(plan.plan_id),
            },
            input_payload={"intent_type": intent.intent_type},
        )
        decision = self.policy_engine.evaluate(policy_ctx)
        decision.workflow_id = str(workflow_id)
        decision.intent_id = str(intent.intent_id)
        decision.plan_id = str(plan.plan_id)
        decision.correlation_id = str(intent.correlation_id)
        await self.policy_store.record(decision)
        if decision.decision != "ALLOW":
            await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
            await self._emit_final_error(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                message="policy denied",
            )
            return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

        steps_state = await self._workflow_store.list_steps(workflow_id=workflow_id)
        state_by_id = {row["step_id"]: row for row in steps_state}

        for step in plan.steps:
            step_id = step["step_id"]
            state = state_by_id.get(step_id)
            if state is None:
                continue
            status = str(state["status"])
            if status == "SUCCEEDED":
                continue
            if status in {"FAILED_FINAL", "CANCELLED"}:
                return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

            if status == "WAITING_APPROVAL":
                gate_id = state.get("gate_id")
                if gate_id is None:
                    return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "waiting")
                decision = await self._gate_store.latest_decision(gate_id=gate_id)
                if decision is None:
                    return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "waiting", gate_id=gate_id)
                if decision["decision"] == "declined":
                    await self._workflow_store.mark_step_cancelled(workflow_id=workflow_id, step_id=step_id)
                    await self._workflow_store.finish_run(workflow_id=workflow_id, status="cancelled")
                    await self._emit_final_error(
                        workflow_id=workflow_id,
                        intent_id=intent.intent_id,
                        plan_id=plan.plan_id,
                        thread_id=intent.thread_id,
                        correlation_id=intent.correlation_id,
                        actor=actor,
                        message="gate declined",
                    )
                    return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "cancelled", gate_id=gate_id)
                await self._workflow_store.mark_step_ready(workflow_id=workflow_id, step_id=step_id)
                state["status"] = "READY"
                status = "READY"

            if not self._deps_satisfied(step, state_by_id):
                continue

            kind = step["kind"]
            if kind == "human_gate" and state.get("gate_id"):
                decision = await self._gate_store.latest_decision(gate_id=state["gate_id"])
                if decision and decision["decision"] == "accepted":
                    await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)
                    state["status"] = "SUCCEEDED"
                    continue
            if kind == "operator":
                result = await self._run_operator_step(
                    workflow_id=workflow_id,
                    intent=intent,
                    plan=plan,
                    step=step,
                    ctx=ctx,
                    actor=actor,
                )
                if result.status != "completed":
                    return result
                state["status"] = "SUCCEEDED"
                continue
            if kind == "policy_check":
                gate_id = await self._run_policy_check_step(
                    workflow_id=workflow_id,
                    intent=intent,
                    plan=plan,
                    step=step,
                    ctx=ctx,
                    actor=actor,
                )
                if gate_id is not None:
                    return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "waiting", gate_id=gate_id)
                state["status"] = "SUCCEEDED"
                continue
            if kind == "human_gate":
                gate_id = await self._run_human_gate_step(
                    workflow_id=workflow_id,
                    intent=intent,
                    plan=plan,
                    step=step,
                    ctx=ctx,
                    actor=actor,
                )
                return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "waiting", gate_id=gate_id)

            await self._workflow_store.mark_step_failed(
                workflow_id=workflow_id,
                step_id=step_id,
                status="FAILED_FINAL",
            )
            await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
            await self._emit_final_error(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                message=f"unsupported step kind: {kind}",
            )
            return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

        await self._workflow_store.finish_run(workflow_id=workflow_id, status="completed")
        await self._emit_progress(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage="completed",
        )
        await self._emit_final_result(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            payload={"status": "success", "outputs": context.get("outcome", {})},
        )
        return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "completed")

    async def _run_operator_step(
        self,
        *,
        workflow_id: uuid.UUID,
        intent: IntentRecord,
        plan: PlanRecord,
        step: dict[str, Any],
        ctx: BindingContext,
        actor: dict[str, Any],
    ) -> WorkflowResult:
        step_id = step["step_id"]
        await self._workflow_store.mark_step_running(workflow_id=workflow_id, step_id=step_id)
        op_name = step.get("operator_name")
        stage = "loading_context" if op_name == "Platform.Context.Load" else f"running_operator:{op_name}"
        await self._emit_progress(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage=stage,
            detail={"step_id": step_id, "name": step.get("name"), "operator": op_name},
        )

        payload = resolve_template_value(step.get("payload") or {}, ctx)
        payload = prune_nulls(payload)
        self._apply_operator_defaults(step.get("operator_name"), payload)
        idempotency_key = render_template(step.get("idempotency_template", ""), ctx)
        await self._workflow_store.update_step_payload(
            workflow_id=workflow_id,
            step_id=step_id,
            idempotency_key=idempotency_key,
            input_payload=payload,
        )

        op_name = step.get("operator_name")
        op_version = step.get("operator_version")
        if not op_name or not op_version:
            await self._workflow_store.mark_step_failed(workflow_id=workflow_id, step_id=step_id, status="FAILED_FINAL")
            await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
            await self._emit_final_error(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                message="operator metadata missing",
            )
            return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

        call = OperatorCall(
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=AuthContext(tenant_id=self.tenant_id, principal=actor.get("principal") or {}, scopes=actor.get("scopes", [])),
            trace_context=OperatorTraceContext(
                correlation_id=str(intent.correlation_id),
                workflow_id=str(workflow_id),
                step_id=step_id,
                plan_id=str(plan.plan_id),
            ),
        )

        result = await self.operator_executor.execute(
            operator_name=op_name,
            operator_version=op_version,
            call=call,
        )

        if result.status != "succeeded":
            status = "FAILED_RETRYABLE" if result.error and result.error.retryable else "FAILED_FINAL"
            await self._workflow_store.mark_step_failed(workflow_id=workflow_id, step_id=step_id, status=status)
            await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
            await self._emit_final_error(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                message=result.error.message if result.error else "operator failed",
            )
            return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

        await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)

        await self._outcome_store.record(
            workflow_id=workflow_id,
            thread_id=intent.thread_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            step_id=step_id,
            job_id=None,
            operator_name=op_name,
            operator_version=op_version,
            status="succeeded",
            content=result.result,
        )

        self._apply_produces(ctx.data, step.get("produces") or [], result.result)
        await self._emit_progress(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage="step_completed",
            detail={"step_id": step_id, "name": step.get("name")},
        )
        return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "completed")

    def _apply_operator_defaults(self, operator_name: str | None, payload: dict[str, Any]) -> None:
        if operator_name == "FundingRequest.Fields.Update.Propose":
            summary = payload.get("human_summary")
            if summary is None or (isinstance(summary, str) and summary.strip() == ""):
                fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
                if fields:
                    keys = ", ".join(sorted(str(k) for k in fields.keys()))
                    payload["human_summary"] = f"Update funding request fields: {keys}"
                else:
                    payload["human_summary"] = "Update funding request fields"

    async def _run_policy_check_step(
        self,
        *,
        workflow_id: uuid.UUID,
        intent: IntentRecord,
        plan: PlanRecord,
        step: dict[str, Any],
        ctx: BindingContext,
        actor: dict[str, Any],
    ) -> uuid.UUID | None:
        step_id = step["step_id"]
        check = step.get("check") or {}
        check_name = str(check.get("check_name") or "")
        params = dict(check.get("params") or {})

        missing_fields: list[str] = []
        if check_name == "EnsureEmailPresent":
            sources = params.get("sources") or []
            if not any(_has_value(ctx.get(src)) for src in sources if isinstance(src, str)):
                missing_fields = ["email_text"]
        elif check_name == "Onboarding.Ensure":
            required = params.get("required_gates") or []
            required = [str(item) for item in required]
            if required:
                missing_fields = required
        else:
            # Unknown check: treat as passed for now.
            missing_fields = []

        if missing_fields:
            gate_type = str(params.get("on_missing_action_type") or "collect_fields")
            gate_id = await self._open_gate(
                workflow_id=workflow_id,
                intent=intent,
                plan=plan,
                step=step,
                ctx=ctx,
                actor=actor,
                gate_type=gate_type,
                title="Provide required information",
                description="Please provide the missing fields to continue.",
                reason_code="MISSING_REQUIRED_FIELDS",
                data={"missing_fields": missing_fields},
            )
            return gate_id

        await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)
        return None

    async def _run_human_gate_step(
        self,
        *,
        workflow_id: uuid.UUID,
        intent: IntentRecord,
        plan: PlanRecord,
        step: dict[str, Any],
        ctx: BindingContext,
        actor: dict[str, Any],
    ) -> uuid.UUID:
        gate_cfg = step.get("gate") or {}
        gate_type = str(gate_cfg.get("gate_type") or "confirm")
        title = str(gate_cfg.get("title") or "Approval required")
        description = str(gate_cfg.get("description") or "")
        reason_code = str(gate_cfg.get("reason_code") or "REQUIRES_APPROVAL")
        requires_user_input = bool(gate_cfg.get("requires_user_input", True))
        ui_hints = gate_cfg.get("ui_hints") or {}
        data: dict[str, Any] = {}

        target_outcome = None
        if "target_outcome_ref" in gate_cfg:
            target_outcome = resolve_template_value(gate_cfg["target_outcome_ref"], ctx)
        proposed_changes = None
        target_outcome_id = None
        if isinstance(target_outcome, dict):
            target_outcome_id = _safe_uuid(target_outcome.get("outcome_id"))
            if "payload" in target_outcome:
                proposed_changes = target_outcome.get("payload")
        gate_id = await self._open_gate(
            workflow_id=workflow_id,
            intent=intent,
            plan=plan,
            step=step,
            ctx=ctx,
            actor=actor,
            gate_type=gate_type,
            title=title,
            description=description,
            reason_code=reason_code,
            requires_user_input=requires_user_input,
            ui_hints=ui_hints,
            target_outcome_id=target_outcome_id,
            proposed_changes=proposed_changes,
            data=data,
        )
        return gate_id

    async def _open_gate(
        self,
        *,
        workflow_id: uuid.UUID,
        intent: IntentRecord,
        plan: PlanRecord,
        step: dict[str, Any],
        ctx: BindingContext,
        actor: dict[str, Any],
        gate_type: str,
        title: str,
        description: str,
        reason_code: str,
        requires_user_input: bool = True,
        ui_hints: dict[str, Any] | None = None,
        target_outcome_id: uuid.UUID | None = None,
        proposed_changes: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        preview = {
            "description": description,
            "ui_hints": ui_hints or {},
            "requires_user_input": requires_user_input,
            "data": data or {},
        }
        gate_id = await self._gate_store.create_gate(
            workflow_id=workflow_id,
            step_id=step["step_id"],
            gate_type=gate_type,
            reason_code=reason_code,
            title=title,
            preview=preview,
            target_outcome_id=target_outcome_id,
            expires_at=None,
        )
        await self._workflow_store.mark_step_waiting(workflow_id=workflow_id, step_id=step["step_id"], gate_id=gate_id)
        await self._workflow_store.update_run_status(workflow_id=workflow_id, status="waiting")

        await self._emit_progress(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage="awaiting_approval",
            detail={"step_id": step["step_id"], "action_id": str(gate_id), "action_type": gate_type},
        )

        payload = {
            "action_id": str(gate_id),
            "action_type": gate_type,
            "reason_code": reason_code,
            "title": title,
            "description": description,
            "requires_user_input": requires_user_input,
            "ui_hints": ui_hints or {},
        }
        if proposed_changes is not None:
            payload["proposed_changes"] = proposed_changes
        if data:
            payload["data"] = data
        await self._emit_event(
            workflow_id=workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            event_type="action_required",
            payload=payload,
            step_id=step["step_id"],
        )
        return gate_id

    async def _build_context(
        self,
        *,
        intent: IntentRecord,
        plan: PlanRecord,
        workflow_id: uuid.UUID,
    ) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "thread_id": intent.thread_id,
            "workflow_id": str(workflow_id),
            "intent": {
                "id": str(intent.intent_id),
                "intent_type": intent.intent_type,
                "thread_id": intent.thread_id,
                "inputs": dict(intent.inputs),
            },
            "context": {},
            "outcome": {},
            "steps": {},
        }

        outcomes = await self._outcome_store.list_by_workflow(workflow_id=workflow_id)
        outcomes_by_step: dict[str, dict[str, Any]] = {}
        for outcome in outcomes:
            step_id = outcome.get("step_id")
            if step_id:
                outcomes_by_step[str(step_id)] = outcome.get("content") or {}

        for step in plan.steps:
            step_id = step["step_id"]
            content = outcomes_by_step.get(step_id)
            if content is not None:
                self._apply_produces(ctx, step.get("produces") or [], content)
                set_path(ctx, f"steps.{step_id}.result", content)

        steps_state = await self._workflow_store.list_steps(workflow_id=workflow_id)
        for row in steps_state:
            gate_id = row.get("gate_id")
            if gate_id:
                decision = await self._gate_store.latest_decision(gate_id=gate_id)
                if decision and decision["decision"] == "accepted":
                    payload = decision.get("payload") or {}
                    if isinstance(payload, dict):
                        ctx["intent"]["inputs"].update(payload)

        return ctx

    def _deps_satisfied(self, step: dict[str, Any], state_by_id: dict[str, dict[str, Any]]) -> bool:
        deps = step.get("depends_on") or []
        for dep in deps:
            state = state_by_id.get(dep)
            if state is None or str(state["status"]) != "SUCCEEDED":
                return False
        return True

    def _apply_produces(self, ctx: dict[str, Any], produces: list[str], result: dict[str, Any] | None) -> None:
        if not result:
            return
        for path in produces:
            value = None
            if path.startswith("outcome.") and "outcome" in result:
                value = result.get("outcome")
            elif path in result:
                value = result.get(path)
            else:
                key = path.split(".")[-1]
                value = result.get(key, result)
            set_path(ctx, path, value)
        # Reset computed cache because context inputs changed.
        ctx["computed"] = {}

    async def _emit_progress(
        self,
        *,
        workflow_id: uuid.UUID,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        thread_id: int,
        correlation_id: uuid.UUID,
        actor: dict[str, Any],
        stage: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        payload = {"stage": stage}
        if detail:
            payload.update(detail)
        step_id = detail.get("step_id") if detail else None
        await self._emit_event(
            workflow_id=workflow_id,
            intent_id=intent_id,
            plan_id=plan_id,
            thread_id=thread_id,
            correlation_id=correlation_id,
            actor=actor,
            event_type="progress",
            payload=payload,
            step_id=step_id,
        )

    async def _emit_final_result(
        self,
        *,
        workflow_id: uuid.UUID,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        thread_id: int,
        correlation_id: uuid.UUID,
        actor: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        await self._emit_event(
            workflow_id=workflow_id,
            intent_id=intent_id,
            plan_id=plan_id,
            thread_id=thread_id,
            correlation_id=correlation_id,
            actor=actor,
            event_type="final_result",
            payload=payload,
        )

    async def _emit_final_error(
        self,
        *,
        workflow_id: uuid.UUID,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        thread_id: int,
        correlation_id: uuid.UUID,
        actor: dict[str, Any],
        message: str,
    ) -> None:
        await self._emit_event(
            workflow_id=workflow_id,
            intent_id=intent_id,
            plan_id=plan_id,
            thread_id=thread_id,
            correlation_id=correlation_id,
            actor=actor,
            event_type="final_error",
            payload={"error": message},
        )

    async def _emit_event(
        self,
        *,
        workflow_id: uuid.UUID,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        thread_id: int,
        correlation_id: uuid.UUID,
        actor: dict[str, Any],
        event_type: str,
        payload: dict[str, Any],
        step_id: str | None = None,
    ) -> None:
        await self.event_writer.append(
            LedgerEvent(
                tenant_id=self.tenant_id,
                event_id=uuid.uuid4(),
                workflow_id=workflow_id,
                thread_id=thread_id,
                intent_id=intent_id,
                plan_id=plan_id,
                step_id=step_id,
                event_type=event_type,
                actor=actor,
                payload=payload,
                correlation_id=correlation_id,
                producer_kind="kernel",
                producer_name="workflow_kernel",
                producer_version="1.0",
            )
        )


def _safe_uuid(value: Any) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except Exception:
        return None


def _hash_json(payload: dict[str, Any]) -> Any:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake3(raw)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True
