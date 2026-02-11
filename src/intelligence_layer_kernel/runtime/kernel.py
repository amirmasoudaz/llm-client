from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

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


class UsageRecorder(Protocol):
    async def quote_usage(
        self,
        *,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> dict[str, Any]:
        ...

    async def record_operator_usage(
        self,
        *,
        principal_id: int,
        workflow_id: uuid.UUID,
        request_key: bytes,
        step_id: str,
        operator_name: str,
        operator_version: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        template_id: str | None = None,
        template_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class WorkflowKernel:
    contracts: ContractRegistry
    operator_executor: OperatorExecutor
    policy_engine: PolicyEngine
    policy_store: PolicyDecisionStore
    event_writer: EventWriter
    pool: Any
    tenant_id: int
    switchboard: IntentSwitchboard | None = None
    usage_recorder: UsageRecorder | None = None
    apply_steps_enabled: bool = True
    apply_shadow_mode: bool = False
    replies_enabled: bool = True
    replies_canary_percent: int = 100
    replies_canary_principals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_intent_store", IntentStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_plan_store", PlanStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_workflow_store", WorkflowStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_outcome_store", OutcomeStore(pool=self.pool, tenant_id=self.tenant_id))
        object.__setattr__(self, "_gate_store", GateStore(pool=self.pool, tenant_id=self.tenant_id))
        resolved_switchboard = self.switchboard or IntentSwitchboard()
        object.__setattr__(self, "_switchboard", resolved_switchboard)

    async def handle_message(
        self,
        *,
        thread_id: int,
        scope_type: str,
        scope_id: str,
        actor: dict[str, Any],
        message: str,
        attachments: list[int] | None = None,
        source: str = "chat",
        workflow_id: uuid.UUID | None = None,
    ) -> WorkflowResult:
        allowed_intents = self._allowed_intents_for_actor(actor=actor)
        if not allowed_intents:
            allowed_intents = self.contracts.list_intent_types()
        intent_type, inputs = self._switchboard.classify(
            message,
            attachment_ids=attachments or [],
            allowed_intents=allowed_intents,
        )
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

        normalized_payload = dict(payload or {})
        if gate.get("gate_type") == "collect_profile_fields":
            has_wrapped_updates = any(
                key in normalized_payload for key in ("profile_updates", "memory_updates")
            )
            if not has_wrapped_updates and normalized_payload:
                normalized_payload = {"profile_updates": normalized_payload}

        intent_type = "Workflow.Gate.Resolve"
        inputs = {"action_id": action_id, "status": status, "payload": normalized_payload}
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

    async def rerun_workflow(
        self,
        *,
        workflow_id: uuid.UUID,
        mode: str,
        actor: dict[str, Any],
        source: str = "api",
    ) -> WorkflowResult:
        replay_mode = mode.strip().lower()
        if replay_mode not in {"replay", "regenerate"}:
            raise ValueError("mode must be replay or regenerate")

        loaded = await self._load_workflow_execution_inputs(workflow_id=workflow_id)
        if loaded is None:
            raise ValueError("workflow not found")
        run, intent, plan = loaded

        replay_workflow_id = uuid.uuid4()
        await self._workflow_store.create_run(
            workflow_id=replay_workflow_id,
            correlation_id=run.correlation_id,
            thread_id=run.thread_id,
            scope_type=run.scope_type or intent.scope_type,
            scope_id=run.scope_id or intent.scope_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            status="running",
            execution_mode=run.execution_mode,
            replay_mode=replay_mode,
            parent_workflow_id=workflow_id,
        )
        await self._workflow_store.create_steps(workflow_id=replay_workflow_id, steps=plan.steps)
        await self._emit_progress(
            workflow_id=replay_workflow_id,
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            thread_id=intent.thread_id,
            correlation_id=intent.correlation_id,
            actor=actor,
            stage="workflow_restarted",
            detail={
                "mode": replay_mode,
                "source_workflow_id": str(workflow_id),
                "source": source,
            },
        )
        return await self._execute_workflow(
            workflow_id=replay_workflow_id,
            intent=intent,
            plan=plan,
            actor=actor,
        )

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

    def _allowed_intents_for_actor(self, *, actor: dict[str, Any]) -> list[str]:
        intents = self.contracts.list_intent_types()
        allowed: list[str] = []
        for intent_type in intents:
            if self._intent_enabled_for_actor(intent_type=intent_type, actor=actor):
                allowed.append(intent_type)
        return allowed

    def _intent_enabled_for_actor(self, *, intent_type: str, actor: dict[str, Any]) -> bool:
        if not _is_reply_intent(intent_type):
            return True
        if not self.replies_enabled:
            return False

        principal = actor.get("principal") if isinstance(actor.get("principal"), dict) else {}
        principal_id = _coerce_positive_int(principal.get("id"))
        if principal_id is None:
            return self._normalized_canary_percent() > 0
        if principal_id in set(self.replies_canary_principals):
            return True

        percent = self._normalized_canary_percent()
        if percent >= 100:
            return True
        if percent <= 0:
            return False
        return _stable_percent_bucket(principal_id) < percent

    def _normalized_canary_percent(self) -> int:
        try:
            value = int(self.replies_canary_percent)
        except Exception:
            return 100
        return max(0, min(100, value))

    async def _execute_existing_workflow(
        self,
        *,
        workflow_id: uuid.UUID,
        actor: dict[str, Any],
    ) -> None:
        loaded = await self._load_workflow_execution_inputs(workflow_id=workflow_id)
        if loaded is None:
            return
        _run, intent, plan = loaded
        await self._workflow_store.update_run_status(workflow_id=workflow_id, status="running")
        await self._execute_workflow(
            workflow_id=workflow_id,
            intent=intent,
            plan=plan,
            actor=actor,
            resume_only=True,
        )

    async def _load_workflow_execution_inputs(
        self,
        *,
        workflow_id: uuid.UUID,
    ) -> tuple[Any, IntentRecord, PlanRecord] | None:
        run = await self._workflow_store.get_run(workflow_id=workflow_id)
        if run is None or run.plan_id is None:
            return None
        plan_row = await self._plan_store.fetch(run.plan_id)
        intent_row = await self._intent_store.fetch(run.intent_id)
        if not plan_row or not intent_row:
            return None

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
        return run, intent, plan

    async def _execute_workflow(
        self,
        *,
        workflow_id: uuid.UUID,
        intent: IntentRecord,
        plan: PlanRecord,
        actor: dict[str, Any],
        resume_only: bool = False,
    ) -> WorkflowResult:
        run = await self._workflow_store.get_run(workflow_id=workflow_id)
        replay_mode = run.replay_mode if run is not None else "reproduce"
        parent_workflow_id = run.parent_workflow_id if run is not None else None
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
                if step["kind"] == "policy_check":
                    # Policy-check gates are data collection pauses; once accepted,
                    # proceed to downstream steps and rely on explicit re-check steps for revalidation.
                    await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)
                    state["status"] = "SUCCEEDED"
                    continue
                await self._workflow_store.mark_step_ready(workflow_id=workflow_id, step_id=step_id)
                state["status"] = "READY"
                status = "READY"

            if not self._deps_satisfied(step, state_by_id):
                continue

            apply_mode = self._apply_step_mode(step)
            if apply_mode is not None:
                await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)
                state["status"] = "SUCCEEDED"
                stage = "apply_shadow_mode" if apply_mode == "shadow" else "apply_steps_disabled"
                await self._emit_progress(
                    workflow_id=workflow_id,
                    intent_id=intent.intent_id,
                    plan_id=plan.plan_id,
                    thread_id=intent.thread_id,
                    correlation_id=intent.correlation_id,
                    actor=actor,
                    stage=stage,
                    detail={
                        "step_id": step_id,
                        "name": step.get("name"),
                        "mode": apply_mode,
                    },
                )
                await self._emit_event(
                    workflow_id=workflow_id,
                    intent_id=intent.intent_id,
                    plan_id=plan.plan_id,
                    thread_id=intent.thread_id,
                    correlation_id=intent.correlation_id,
                    actor=actor,
                    event_type="apply_step_skipped",
                    payload={
                        "step_id": step_id,
                        "operator_name": step.get("operator_name"),
                        "mode": apply_mode,
                    },
                    step_id=step_id,
                )
                continue

            kind = step["kind"]
            if kind == "human_gate" and state.get("gate_id"):
                decision = await self._gate_store.latest_decision(gate_id=state["gate_id"])
                if decision and decision["decision"] == "accepted":
                    await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)
                    state["status"] = "SUCCEEDED"
                    continue
            if kind == "policy_check" and state.get("gate_id"):
                decision = await self._gate_store.latest_decision(gate_id=state["gate_id"])
                if decision and decision["decision"] == "accepted":
                    await self._workflow_store.mark_step_succeeded(workflow_id=workflow_id, step_id=step_id)
                    state["status"] = "SUCCEEDED"
                    continue
                if decision and decision["decision"] == "declined":
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
                    return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "cancelled", gate_id=state["gate_id"])
            if kind == "operator":
                result = await self._run_operator_step(
                    workflow_id=workflow_id,
                    intent=intent,
                    plan=plan,
                    step=step,
                    ctx=ctx,
                    actor=actor,
                    replay_mode=replay_mode,
                    parent_workflow_id=parent_workflow_id,
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
        replay_mode: str,
        parent_workflow_id: uuid.UUID | None,
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
        await self._hydrate_operator_payload(
            operator_name=step.get("operator_name"),
            thread_id=intent.thread_id,
            payload=payload,
        )
        self._apply_operator_defaults(step.get("operator_name"), payload)
        idempotency_key = render_template(step.get("idempotency_template", ""), ctx)
        idempotency_key = self._with_replay_mode_idempotency(
            idempotency_key=idempotency_key,
            step=step,
            replay_mode=replay_mode,
            workflow_id=workflow_id,
        )
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
        try:
            manifest = self.contracts.get_operator_manifest(op_name, op_version)
        except Exception:
            manifest = {}
        llm_backed = _manifest_has_prompt_template(manifest)

        if llm_backed:
            budget_reason = await self._ensure_budget_allows_llm_step(
                actor=actor,
                payload=payload,
                operator_name=op_name,
                operator_version=op_version,
            )
            if budget_reason is not None:
                await self._workflow_store.mark_step_failed(workflow_id=workflow_id, step_id=step_id, status="FAILED_FINAL")
                await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
                await self._emit_final_error(
                    workflow_id=workflow_id,
                    intent_id=intent.intent_id,
                    plan_id=plan.plan_id,
                    thread_id=intent.thread_id,
                    correlation_id=intent.correlation_id,
                    actor=actor,
                    message=budget_reason,
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
                thread_id=intent.thread_id,
                intent_id=str(intent.intent_id),
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
            normalized_error = self._normalize_operator_failure_message(
                operator_name=op_name,
                error=result.error,
            )
            await self._workflow_store.mark_step_failed(workflow_id=workflow_id, step_id=step_id, status=status)
            await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
            await self._emit_final_error(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                message=normalized_error,
            )
            return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

        if llm_backed:
            usage_error = await self._record_llm_usage(
                actor=actor,
                workflow_id=workflow_id,
                step_id=step_id,
                operator_name=op_name,
                operator_version=op_version,
                payload=payload,
                result=result.result,
                template_id=result.prompt_template_id,
                template_hash=result.prompt_template_hash,
            )
            if usage_error is not None:
                await self._workflow_store.mark_step_failed(workflow_id=workflow_id, step_id=step_id, status="FAILED_FINAL")
                await self._workflow_store.finish_run(workflow_id=workflow_id, status="failed")
                await self._emit_final_error(
                    workflow_id=workflow_id,
                    intent_id=intent.intent_id,
                    plan_id=plan.plan_id,
                    thread_id=intent.thread_id,
                    correlation_id=intent.correlation_id,
                    actor=actor,
                    message=usage_error,
                )
                return WorkflowResult(workflow_id, intent.intent_id, plan.plan_id, "failed")

            await self._emit_model_tokens(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                step_id=step_id,
                payload=result.result,
            )

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
            template_id=result.prompt_template_id,
            template_hash=result.prompt_template_hash,
            replay_mode=replay_mode,
            parent_workflow_id=parent_workflow_id,
        )

        self._apply_produces(ctx.data, step.get("produces") or [], result.result)
        if op_name in {"FundingRequest.Fields.Update.Apply", "FundingEmail.Draft.Update.Apply"}:
            refresh_payload: dict[str, Any] = {"target": "funding_request"}
            if op_name == "FundingRequest.Fields.Update.Apply":
                refresh_payload["reason"] = "funding_request_updated"
            else:
                refresh_payload["reason"] = "funding_email_draft_updated"
            if isinstance(result.result, dict):
                request_id = result.result.get("request_id")
                if request_id is not None:
                    refresh_payload["request_id"] = request_id
                email_id = result.result.get("email_id")
                if email_id is not None:
                    refresh_payload["email_id"] = email_id
            await self._emit_event(
                workflow_id=workflow_id,
                intent_id=intent.intent_id,
                plan_id=plan.plan_id,
                thread_id=intent.thread_id,
                correlation_id=intent.correlation_id,
                actor=actor,
                event_type="ui.refresh_required",
                payload=refresh_payload,
                step_id=step_id,
            )
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

    async def _ensure_budget_allows_llm_step(
        self,
        *,
        actor: dict[str, Any],
        payload: dict[str, Any],
        operator_name: str,
        operator_version: str,
    ) -> str | None:
        if self.usage_recorder is None:
            return None
        billing = actor.get("billing")
        if not isinstance(billing, dict):
            return None
        reserved = _coerce_positive_int(billing.get("reserved_credits"))
        used = _coerce_non_negative_int(billing.get("used_credits"))
        if reserved is None:
            return None
        provider = str(billing.get("provider") or "openai")
        model = str(billing.get("model") or "gpt-5-nano")
        tokens_in = _estimate_token_count(payload)
        tokens_out = _estimate_output_tokens(operator_name)
        quote = await self.usage_recorder.quote_usage(
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        projected_credits = used + int(quote.get("credits_charged") or 0)
        if projected_credits > reserved:
            return (
                "budget exceeded: projected credits "
                f"{projected_credits} would exceed reserved credits {reserved} for {operator_name}@{operator_version}"
            )
        return None

    async def _record_llm_usage(
        self,
        *,
        actor: dict[str, Any],
        workflow_id: uuid.UUID,
        step_id: str,
        operator_name: str,
        operator_version: str,
        payload: dict[str, Any],
        result: dict[str, Any] | None,
        template_id: str | None,
        template_hash: str | None,
    ) -> str | None:
        if self.usage_recorder is None:
            return None
        billing = actor.get("billing")
        if not isinstance(billing, dict):
            return None
        request_key_hex = billing.get("request_key")
        if not isinstance(request_key_hex, str) or not request_key_hex:
            return "missing billing request key"
        try:
            request_key = bytes.fromhex(request_key_hex)
        except ValueError:
            return "invalid billing request key"
        principal_id = _coerce_positive_int(billing.get("principal_id"))
        if principal_id is None:
            principal_id = _coerce_positive_int((actor.get("principal") or {}).get("id"))
        if principal_id is None:
            return "missing principal for usage recording"

        provider = str(billing.get("provider") or "openai")
        model = str(billing.get("model") or "gpt-5-nano")
        tokens_in = _estimate_token_count(payload)
        tokens_out = _estimate_token_count(result)
        usage = await self.usage_recorder.record_operator_usage(
            principal_id=principal_id,
            workflow_id=workflow_id,
            request_key=request_key,
            step_id=step_id,
            operator_name=operator_name,
            operator_version=operator_version,
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            template_id=template_id,
            template_hash=template_hash,
            metadata={"workflow_kernel": True},
        )
        if hasattr(usage, "ok"):
            if not bool(getattr(usage, "ok")):
                reason = getattr(usage, "reason", None) or "usage_record_failed"
                if reason == "budget_exceeded":
                    return "budget exceeded while recording usage"
                return f"usage recording failed: {reason}"
            total = getattr(usage, "total_credits_charged", None)
            if total is not None:
                billing["used_credits"] = int(total)
            return None
        if isinstance(usage, dict):
            if not bool(usage.get("ok", True)):
                reason = usage.get("reason") or "usage_record_failed"
                if reason == "budget_exceeded":
                    return "budget exceeded while recording usage"
                return f"usage recording failed: {reason}"
            if usage.get("total_credits_charged") is not None:
                billing["used_credits"] = int(usage["total_credits_charged"])
        return None

    async def _emit_model_tokens(
        self,
        *,
        workflow_id: uuid.UUID,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        thread_id: int,
        correlation_id: uuid.UUID,
        actor: dict[str, Any],
        step_id: str,
        payload: dict[str, Any] | None,
    ) -> None:
        chunks = _split_model_tokens(payload, limit=64)
        for index, token in enumerate(chunks):
            await self._emit_event(
                workflow_id=workflow_id,
                intent_id=intent_id,
                plan_id=plan_id,
                thread_id=thread_id,
                correlation_id=correlation_id,
                actor=actor,
                event_type="model_token",
                payload={"token": token, "index": index},
                step_id=step_id,
            )

    def _normalize_operator_failure_message(
        self,
        *,
        operator_name: str,
        error: Any | None,
    ) -> str:
        if error is None:
            return "operator failed"
        message = str(getattr(error, "message", "") or "operator failed")
        code = str(getattr(error, "code", "") or "")
        details = getattr(error, "details", None)
        if operator_name == "StudentProfile.Update" and code == "invalid_profile_update":
            normalized = _format_profile_update_validation_message(details)
            if normalized is not None:
                return normalized
        return message

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
        if operator_name == "Email.OptimizeDraft" and "requested_edits" not in payload:
            payload["requested_edits"] = []
        if operator_name == "StudentProfile.Update" and "profile_updates" not in payload:
            payload["profile_updates"] = {}
        if operator_name == "Memory.Upsert" and "entries" not in payload:
            payload["entries"] = []

    def _with_replay_mode_idempotency(
        self,
        *,
        idempotency_key: str,
        step: dict[str, Any],
        replay_mode: str,
        workflow_id: uuid.UUID,
    ) -> str:
        if replay_mode != "regenerate":
            return idempotency_key
        effects = step.get("effects")
        if _step_has_effectful_writes(effects):
            return idempotency_key
        suffix = f"regen:{workflow_id}"
        if idempotency_key:
            return f"{idempotency_key}:{suffix}"
        return suffix

    async def _hydrate_operator_payload(
        self,
        *,
        operator_name: str | None,
        thread_id: int,
        payload: dict[str, Any],
    ) -> None:
        if operator_name == "Email.OptimizeDraft":
            await self._hydrate_email_optimize_source(thread_id=thread_id, payload=payload)

    async def _hydrate_email_optimize_source(self, *, thread_id: int, payload: dict[str, Any]) -> None:
        source_outcome_id = _safe_uuid(payload.get("source_draft_outcome_id"))
        source_version_raw = payload.get("source_draft_version")
        source_version = source_version_raw if isinstance(source_version_raw, int) and source_version_raw > 0 else None
        if source_outcome_id is None and source_version is None:
            return

        source = await self._resolve_email_draft_source(
            thread_id=thread_id,
            source_outcome_id=source_outcome_id,
            source_version=source_version,
        )
        if source is None:
            return

        subject = source.get("subject")
        body = source.get("body")
        if isinstance(subject, str) and subject.strip():
            payload["current_subject"] = subject
        if isinstance(body, str) and body.strip():
            payload["current_body"] = body

        resolved_outcome_id = source.get("outcome_id")
        if resolved_outcome_id is not None:
            payload["source_draft_outcome_id"] = str(resolved_outcome_id)
        resolved_version = source.get("version")
        if not (isinstance(resolved_version, int) and resolved_version > 0):
            resolved_version = source.get("version_number")
        if isinstance(resolved_version, int) and resolved_version > 0:
            payload["source_draft_version"] = resolved_version

    async def _resolve_email_draft_source(
        self,
        *,
        thread_id: int,
        source_outcome_id: uuid.UUID | None,
        source_version: int | None,
    ) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT outcome_id, lineage_id, version, content, created_at
                FROM ledger.outcomes
                WHERE tenant_id=$1
                  AND thread_id=$2
                  AND status='succeeded'
                  AND content->'outcome'->>'outcome_type'='Email.Draft'
                ORDER BY created_at DESC
                LIMIT 100;
                """,
                self.tenant_id,
                thread_id,
            )

        drafts: list[dict[str, Any]] = []
        for row in rows:
            content = row["content"]
            if not isinstance(content, dict):
                continue
            outcome = content.get("outcome")
            if not isinstance(outcome, dict):
                continue
            payload = outcome.get("payload")
            if not isinstance(payload, dict):
                continue
            subject = payload.get("subject")
            body = payload.get("body")
            if not isinstance(subject, str) or not isinstance(body, str):
                continue
            row_version_raw = row.get("version") if hasattr(row, "get") else row["version"]
            row_version = row_version_raw if isinstance(row_version_raw, int) and row_version_raw > 0 else None
            payload_version = payload.get("version_number")
            version_number = payload_version if isinstance(payload_version, int) and payload_version > 0 else None
            content_outcome_id = _safe_uuid(outcome.get("outcome_id"))
            row_outcome_id = row.get("outcome_id") if hasattr(row, "get") else row["outcome_id"]
            outcome_id = row_outcome_id if isinstance(row_outcome_id, uuid.UUID) else _safe_uuid(row_outcome_id)
            if outcome_id is None:
                outcome_id = content_outcome_id
            if outcome_id is None:
                continue
            row_lineage_raw = row.get("lineage_id") if hasattr(row, "get") else row["lineage_id"]
            lineage_id = row_lineage_raw if isinstance(row_lineage_raw, uuid.UUID) else _safe_uuid(row_lineage_raw)
            if lineage_id is None:
                lineage_id = outcome_id
            drafts.append(
                {
                    "outcome_id": outcome_id,
                    "content_outcome_id": content_outcome_id,
                    "lineage_id": lineage_id,
                    "version": row_version,
                    "version_number": version_number,
                    "subject": subject,
                    "body": body,
                }
            )

        if source_outcome_id is not None:
            for draft in drafts:
                if draft["outcome_id"] == source_outcome_id or draft.get("content_outcome_id") == source_outcome_id:
                    return draft

        if source_version is not None:
            if drafts:
                latest_lineage_id = drafts[0].get("lineage_id")
                for draft in drafts:
                    if draft.get("lineage_id") != latest_lineage_id:
                        continue
                    draft_version = draft.get("version")
                    if isinstance(draft_version, int) and draft_version == source_version:
                        return draft
            for draft in drafts:
                draft_version = draft.get("version")
                if isinstance(draft_version, int) and draft_version == source_version:
                    return draft
                if draft.get("version_number") == source_version:
                    return draft

        return None

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
        resolved_params = resolve_template_value(dict(check.get("params") or {}), ctx)
        params = resolved_params if isinstance(resolved_params, dict) else {}

        missing_requirements: list[str] = []
        missing_fields: list[str] = []
        targeted_questions: list[dict[str, Any]] = []
        missing_attachment_detail: dict[str, Any] = {}
        if check_name == "EnsureEmailPresent":
            sources = params.get("sources") or []
            if not any(_has_value(ctx.get(src)) for src in sources if isinstance(src, str)):
                missing_fields = [str(params.get("missing_field_key") or "email_draft")]
        elif check_name == "EnsureReplyPresent":
            sources = params.get("sources") or []
            if not any(_has_value(ctx.get(src)) for src in sources if isinstance(src, str)):
                missing_fields = [str(params.get("missing_field_key") or "professor_reply")]
        elif check_name == "Onboarding.Ensure":
            requirements_state = ctx.get("context.profile.requirements")
            if not isinstance(requirements_state, dict):
                requirements_state = ctx.get("context.intelligence.requirements")
            if not isinstance(requirements_state, dict):
                requirements_state = {}

            required_raw = params.get("required_gates")
            required: list[str] = []
            if isinstance(required_raw, list):
                required = [str(item) for item in required_raw if str(item).strip()]
            if not required:
                fallback_required = requirements_state.get("required_requirements")
                if isinstance(fallback_required, list):
                    required = [str(item) for item in fallback_required if str(item).strip()]

            status_by_requirement = requirements_state.get("status_by_requirement")
            if not isinstance(status_by_requirement, dict):
                status_by_requirement = {}

            if required:
                for requirement in required:
                    if not bool(status_by_requirement.get(requirement)):
                        missing_requirements.append(requirement)

            missing_by_requirement = requirements_state.get("missing_fields_by_requirement")
            if isinstance(missing_by_requirement, dict):
                for requirement in missing_requirements:
                    candidates = missing_by_requirement.get(requirement)
                    if not isinstance(candidates, list):
                        continue
                    for field_path in candidates:
                        text = str(field_path).strip()
                        if text and text not in missing_fields:
                            missing_fields.append(text)

            raw_questions = requirements_state.get("targeted_questions")
            if isinstance(raw_questions, list):
                for item in raw_questions:
                    if isinstance(item, dict):
                        targeted_questions.append(item)

            if not missing_fields and missing_requirements:
                missing_fields = list(missing_requirements)
        elif check_name == "EnsureAttachmentPresent":
            attachment_ref = params.get("attachment_ref")
            attachment_id = None
            if isinstance(attachment_ref, dict):
                attachment_id = _coerce_positive_int(attachment_ref.get("attachment_id") or attachment_ref.get("id"))

            requested_document_type = str(params.get("requested_document_type") or "document").strip().lower() or "document"
            if requested_document_type == "other":
                requested_document_type = "document"

            requested_attachment_kinds_raw = params.get("requested_attachment_kinds")
            requested_attachment_kinds: list[str] = []
            if isinstance(requested_attachment_kinds_raw, list):
                for item in requested_attachment_kinds_raw:
                    value = str(item).strip()
                    if value and value not in requested_attachment_kinds:
                        requested_attachment_kinds.append(value)

            if attachment_id is None:
                missing_fields = [f"{requested_document_type}_attachment"]
                missing_requirements = [f"{requested_document_type}_upload"]
                missing_attachment_detail = {
                    "required_document_type": requested_document_type,
                    "required_attachment_kinds": requested_attachment_kinds,
                }
        else:
            # Unknown check: treat as passed for now.
            missing_fields = []

        if missing_fields or missing_requirements:
            if check_name == "Onboarding.Ensure":
                default_gate_type = "collect_profile_fields"
            elif check_name == "EnsureAttachmentPresent":
                default_gate_type = "upload_required"
            else:
                default_gate_type = "collect_fields"
            gate_type = str(params.get("on_missing_action_type") or default_gate_type)
            if check_name == "Onboarding.Ensure":
                reason_code = "MISSING_REQUIRED_PROFILE_FIELDS"
            elif check_name == "EnsureAttachmentPresent":
                reason_code = "MISSING_REQUIRED_ATTACHMENT"
            else:
                reason_code = "MISSING_REQUIRED_FIELDS"
            default_title = "Provide required information"
            default_description = (
                "Please provide the missing profile fields to continue."
                if check_name == "Onboarding.Ensure"
                else "Please provide the missing fields to continue."
            )
            if check_name == "EnsureEmailPresent":
                default_title = "Generate email draft first"
                default_description = (
                    "No draft email was found. Generate a draft in the Funding Outreach UI, "
                    "then retry optimization."
                )
            if check_name == "EnsureReplyPresent":
                default_title = "No professor reply yet"
                default_description = (
                    "No professor reply is available for this request yet. "
                    "Please wait for a reply, then try again."
                )
            if check_name == "EnsureAttachmentPresent":
                requested_document_type = str(
                    missing_attachment_detail.get("required_document_type") or "document"
                ).strip()
                readable_type = requested_document_type.upper() if requested_document_type else "document"
                default_title = f"Upload your {readable_type}"
                default_description = (
                    f"No {requested_document_type} attachment was found for this request. "
                    f"Please upload a {requested_document_type} first, then continue."
                )

            title = str(params.get("on_missing_title") or default_title)
            description = str(params.get("on_missing_description") or default_description)
            requires_user_input = bool(params.get("on_missing_requires_user_input", True))
            gate_data: dict[str, Any] = {
                "missing_requirements": missing_requirements,
                "missing_fields": missing_fields,
                "targeted_questions": targeted_questions[:3],
            }
            if check_name == "EnsureEmailPresent":
                funding_request_id = ctx.get("context.platform.funding_request.id")
                endpoint = "/api/v1/funding/{funding_id}/review"
                if isinstance(funding_request_id, int) and funding_request_id > 0:
                    endpoint = f"/api/v1/funding/{funding_request_id}/review"
                gate_data["action_hint"] = {
                    "action": "generate_draft_preview",
                    "method": "GET",
                    "endpoint": endpoint,
                }
            if check_name == "EnsureAttachmentPresent":
                funding_request_id = ctx.get("context.platform.funding_request.id")
                upload_endpoint = "/api/v1/funding/attachments"
                if isinstance(funding_request_id, int) and funding_request_id > 0:
                    upload_endpoint = f"/api/v1/funding/{funding_request_id}/attachments"
                gate_data.update(missing_attachment_detail)
                gate_data["action_hint"] = {
                    "action": "upload_attachment",
                    "method": "POST",
                    "endpoint": upload_endpoint,
                }
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
                data=gate_data,
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
        if gate_type == "apply_platform_patch":
            payload["apply_action_id"] = str(gate_id)
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
                        _merge_gate_payload_into_intent_inputs(ctx["intent"]["inputs"], payload)

        return ctx

    def _deps_satisfied(self, step: dict[str, Any], state_by_id: dict[str, dict[str, Any]]) -> bool:
        deps = step.get("depends_on") or []
        for dep in deps:
            state = state_by_id.get(dep)
            if state is None or str(state["status"]) != "SUCCEEDED":
                return False
        return True

    def _apply_step_mode(self, step: dict[str, Any]) -> str | None:
        if not _is_apply_step(step):
            return None
        if not self.apply_steps_enabled:
            return "disabled"
        if self.apply_shadow_mode:
            return "shadow"
        return None

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


def _step_has_effectful_writes(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    effectful = {"db_write", "external_send", "s3_final_write", "webhook_emit"}
    for item in value:
        if isinstance(item, str) and item in effectful:
            return True
    return False


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _manifest_has_prompt_template(manifest: dict[str, Any]) -> bool:
    if not isinstance(manifest, dict):
        return False
    if isinstance(manifest.get("prompt_template_id"), str) and manifest["prompt_template_id"].strip():
        return True
    if isinstance(manifest.get("prompt_template"), str) and manifest["prompt_template"].strip():
        return True
    prompt_templates = manifest.get("prompt_templates")
    if isinstance(prompt_templates, str):
        return bool(prompt_templates.strip())
    if isinstance(prompt_templates, list):
        for item in prompt_templates:
            if isinstance(item, str) and item.strip():
                return True
            if isinstance(item, dict):
                template_id = item.get("template_id") or item.get("id")
                if isinstance(template_id, str) and template_id.strip():
                    return True
    if isinstance(prompt_templates, dict):
        for value in prompt_templates.values():
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, dict):
                template_id = value.get("template_id") or value.get("id")
                if isinstance(template_id, str) and template_id.strip():
                    return True
    return False


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def _is_reply_intent(intent_type: str) -> bool:
    return intent_type in {
        "Funding.Outreach.Reply.Interpret",
        "Funding.Outreach.FollowUp.Draft",
    }


def _is_apply_step(step: dict[str, Any]) -> bool:
    effects = step.get("effects")
    if isinstance(effects, list) and any(str(item) == "db_write_platform" for item in effects):
        return True
    kind = str(step.get("kind") or "")
    if kind == "human_gate":
        gate_cfg = step.get("gate")
        if isinstance(gate_cfg, dict):
            gate_type = str(gate_cfg.get("gate_type") or "")
            if gate_type == "apply_platform_patch":
                return True
    operator_name = str(step.get("operator_name") or "")
    return operator_name.endswith(".Apply")


def _stable_percent_bucket(value: int) -> int:
    digest = blake3(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") % 100


def _estimate_output_tokens(operator_name: str) -> int:
    if operator_name.startswith("Email."):
        return 512
    if operator_name.startswith("Documents."):
        return 768
    return 384


def _estimate_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        except TypeError:
            text = str(value)
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _split_model_tokens(payload: dict[str, Any] | None, *, limit: int) -> list[str]:
    if payload is None:
        return []
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        text = str(payload)
    text = text.strip()
    if not text:
        return []
    chunks = text.split()
    if len(chunks) > limit:
        chunks = chunks[:limit]
    return chunks


def _format_profile_update_validation_message(details: Any) -> str | None:
    errors = details.get("errors") if isinstance(details, dict) else None
    if not isinstance(errors, list) or not errors:
        return (
            "I couldn't save those profile updates yet. Please check the values and try again."
        )

    hints = {
        "general.first_name": "enter your first name",
        "general.last_name": "enter your last name",
        "general.email": "use a valid email address like name@example.com",
        "context.background.research_interests": "add at least one research interest topic",
        "inference.sop_intelligence.research_direction.content": "describe your research direction",
        "inference.sop_intelligence.career_trajectory.long_term_goal.content": (
            "describe your long-term academic or career goal"
        ),
    }
    normalized_fields: list[str] = []
    for item in errors:
        text = str(item).strip()
        field = text.split(":", 1)[0].strip() if ":" in text else ""
        if not field:
            continue
        hint = hints.get(field)
        if hint:
            rendered = f"{field} ({hint})"
        else:
            rendered = field
        if rendered not in normalized_fields:
            normalized_fields.append(rendered)
        if len(normalized_fields) >= 3:
            break
    if normalized_fields:
        return (
            "I couldn't save those profile updates yet. Please fix: "
            + ", ".join(normalized_fields)
            + "."
        )
    return (
        "I couldn't save those profile updates yet. Please check the values and try again."
    )


def _merge_gate_payload_into_intent_inputs(target: dict[str, Any], payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if key == "profile_updates":
            existing = target.get("profile_updates")
            if isinstance(existing, dict) and isinstance(value, dict):
                target["profile_updates"] = _deep_merge_dict(existing, value)
                continue
        if key == "memory_updates":
            existing = target.get("memory_updates")
            if isinstance(existing, list) and isinstance(value, list):
                target["memory_updates"] = [*existing, *value]
                continue
        target[key] = value


def _deep_merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged
