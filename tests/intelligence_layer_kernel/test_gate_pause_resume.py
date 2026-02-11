from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from intelligence_layer_kernel.events.types import LedgerEvent
from intelligence_layer_kernel.operators.types import OperatorCall, OperatorMetrics, OperatorResult
from intelligence_layer_kernel.policy import PolicyEngine
from intelligence_layer_kernel.runtime.kernel import WorkflowKernel
from intelligence_layer_kernel.runtime.store import WorkflowRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _InMemoryIntentStore:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, dict[str, Any]] = {}

    async def insert(self, intent) -> None:
        self.rows[intent.intent_id] = {
            "intent_id": intent.intent_id,
            "intent_type": intent.intent_type,
            "schema_version": intent.schema_version,
            "source": intent.source,
            "thread_id": intent.thread_id,
            "scope_type": intent.scope_type,
            "scope_id": intent.scope_id,
            "actor": dict(intent.actor),
            "inputs": dict(intent.inputs),
            "constraints": dict(intent.constraints),
            "context_refs": dict(intent.context_refs),
            "data_classes": list(intent.data_classes),
            "correlation_id": intent.correlation_id,
            "created_at": intent.created_at,
        }

    async def fetch(self, intent_id: uuid.UUID) -> dict[str, Any] | None:
        return self.rows.get(intent_id)


class _InMemoryPlanStore:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, dict[str, Any]] = {}

    async def insert(self, plan, _plan_hash: bytes) -> None:
        self.rows[plan.plan_id] = {
            "plan_id": plan.plan_id,
            "intent_id": plan.intent_id,
            "plan": plan.to_dict(),
        }

    async def fetch(self, plan_id: uuid.UUID) -> dict[str, Any] | None:
        return self.rows.get(plan_id)


class _InMemoryWorkflowStore:
    def __init__(self) -> None:
        self.runs: dict[uuid.UUID, WorkflowRun] = {}
        self.steps: dict[uuid.UUID, list[dict[str, Any]]] = {}

    async def create_run(
        self,
        *,
        workflow_id: uuid.UUID,
        correlation_id: uuid.UUID,
        thread_id: int | None,
        scope_type: str | None,
        scope_id: str | None,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        status: str,
        execution_mode: str,
        replay_mode: str,
        request_key: bytes | None = None,
        parent_workflow_id: uuid.UUID | None = None,
    ) -> None:
        _ = request_key
        self.runs[workflow_id] = WorkflowRun(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=scope_id,
            intent_id=intent_id,
            plan_id=plan_id,
            status=status,
            execution_mode=execution_mode,
            replay_mode=replay_mode,
            parent_workflow_id=parent_workflow_id,
        )

    async def get_run(self, *, workflow_id: uuid.UUID) -> WorkflowRun | None:
        return self.runs.get(workflow_id)

    async def update_run_status(self, *, workflow_id: uuid.UUID, status: str) -> None:
        run = self.runs[workflow_id]
        self.runs[workflow_id] = WorkflowRun(
            workflow_id=run.workflow_id,
            correlation_id=run.correlation_id,
            thread_id=run.thread_id,
            scope_type=run.scope_type,
            scope_id=run.scope_id,
            intent_id=run.intent_id,
            plan_id=run.plan_id,
            status=status,
            execution_mode=run.execution_mode,
            replay_mode=run.replay_mode,
            parent_workflow_id=run.parent_workflow_id,
        )

    async def finish_run(self, *, workflow_id: uuid.UUID, status: str) -> None:
        await self.update_run_status(workflow_id=workflow_id, status=status)

    async def create_steps(self, *, workflow_id: uuid.UUID, steps: list[dict[str, Any]]) -> None:
        rows: list[dict[str, Any]] = []
        for step in steps:
            rows.append(
                {
                    "step_id": step["step_id"],
                    "kind": step["kind"],
                    "name": step["name"],
                    "operator_name": step.get("operator_name"),
                    "operator_version": step.get("operator_version"),
                    "effects": list(step.get("effects") or []),
                    "policy_tags": list(step.get("policy_tags") or []),
                    "risk_level": step.get("risk_level", "low"),
                    "cache_policy": step.get("cache_policy", "never"),
                    "idempotency_key": None,
                    "input_payload": {},
                    "input_hash": None,
                    "status": "PENDING",
                    "attempt_count": 0,
                    "next_retry_at": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "last_job_id": None,
                    "gate_id": None,
                    "created_at": _now(),
                    "started_at": None,
                    "finished_at": None,
                    "updated_at": _now(),
                }
            )
        self.steps[workflow_id] = rows

    async def list_steps(self, *, workflow_id: uuid.UUID) -> list[dict[str, Any]]:
        return [dict(row) for row in self.steps.get(workflow_id, [])]

    async def mark_step_ready(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        self._step(workflow_id, step_id)["status"] = "READY"

    async def mark_step_running(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        row = self._step(workflow_id, step_id)
        row["status"] = "RUNNING"
        row["attempt_count"] = int(row.get("attempt_count") or 0) + 1

    async def mark_step_waiting(self, *, workflow_id: uuid.UUID, step_id: str, gate_id: uuid.UUID) -> None:
        row = self._step(workflow_id, step_id)
        row["status"] = "WAITING_APPROVAL"
        row["gate_id"] = gate_id

    async def mark_step_succeeded(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        self._step(workflow_id, step_id)["status"] = "SUCCEEDED"

    async def mark_step_failed(self, *, workflow_id: uuid.UUID, step_id: str, status: str) -> None:
        self._step(workflow_id, step_id)["status"] = status

    async def mark_step_cancelled(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        self._step(workflow_id, step_id)["status"] = "CANCELLED"

    async def update_step_payload(
        self,
        *,
        workflow_id: uuid.UUID,
        step_id: str,
        idempotency_key: str | None,
        input_payload: dict[str, Any],
        last_job_id: uuid.UUID | None = None,
    ) -> None:
        row = self._step(workflow_id, step_id)
        row["idempotency_key"] = idempotency_key
        row["input_payload"] = dict(input_payload)
        row["last_job_id"] = last_job_id

    def step(self, *, workflow_id: uuid.UUID, step_id: str) -> dict[str, Any]:
        return dict(self._step(workflow_id, step_id))

    def _step(self, workflow_id: uuid.UUID, step_id: str) -> dict[str, Any]:
        for row in self.steps.get(workflow_id, []):
            if row["step_id"] == step_id:
                return row
        raise KeyError(step_id)


class _InMemoryOutcomeStore:
    def __init__(self) -> None:
        self.by_workflow: dict[uuid.UUID, list[dict[str, Any]]] = {}
        self.record_calls: list[dict[str, Any]] = []

    async def record(self, **kwargs) -> uuid.UUID | None:
        self.record_calls.append(dict(kwargs))
        outcome_id = uuid.uuid4()
        workflow_id = kwargs["workflow_id"]
        payload = {
            "outcome_id": outcome_id,
            "lineage_id": outcome_id,
            "version": 1,
            "parent_outcome_id": None,
            "outcome_type": kwargs["operator_name"],
            "status": kwargs["status"],
            "workflow_id": workflow_id,
            "step_id": kwargs["step_id"],
            "content": kwargs["content"],
            "template_id": kwargs.get("template_id"),
            "template_hash": kwargs.get("template_hash"),
            "created_at": _now(),
        }
        self.by_workflow.setdefault(workflow_id, []).append(payload)
        return outcome_id

    async def list_by_workflow(self, *, workflow_id: uuid.UUID) -> list[dict[str, Any]]:
        return [dict(item) for item in self.by_workflow.get(workflow_id, [])]


class _InMemoryGateStore:
    def __init__(self) -> None:
        self.gates: dict[uuid.UUID, dict[str, Any]] = {}
        self.decisions: dict[uuid.UUID, list[dict[str, Any]]] = {}

    async def create_gate(
        self,
        *,
        workflow_id: uuid.UUID,
        step_id: str,
        gate_type: str,
        reason_code: str,
        title: str,
        preview: dict[str, Any],
        target_outcome_id: uuid.UUID | None,
        expires_at: Any | None,
    ) -> uuid.UUID:
        gate_id = uuid.uuid4()
        self.gates[gate_id] = {
            "gate_id": gate_id,
            "workflow_id": workflow_id,
            "step_id": step_id,
            "gate_type": gate_type,
            "reason_code": reason_code,
            "summary": title,
            "preview": dict(preview),
            "target_outcome_id": target_outcome_id,
            "status": "waiting",
            "expires_at": expires_at,
        }
        return gate_id

    async def get_gate(self, *, gate_id: uuid.UUID) -> dict[str, Any] | None:
        gate = self.gates.get(gate_id)
        return dict(gate) if gate is not None else None

    async def resolve_gate(
        self,
        *,
        gate_id: uuid.UUID,
        actor: dict[str, Any],
        decision: str,
        payload: dict[str, Any],
    ) -> uuid.UUID:
        decision_id = uuid.uuid4()
        self.decisions.setdefault(gate_id, []).append(
            {
                "gate_decision_id": decision_id,
                "decision": decision,
                "payload": dict(payload),
                "actor": dict(actor),
            }
        )
        self.gates[gate_id]["status"] = decision
        return decision_id

    async def latest_decision(self, *, gate_id: uuid.UUID) -> dict[str, Any] | None:
        rows = self.decisions.get(gate_id) or []
        if not rows:
            return None
        latest = rows[-1]
        return {
            "gate_decision_id": latest["gate_decision_id"],
            "decision": latest["decision"],
            "payload": dict(latest["payload"]),
        }


class _FakeOperatorExecutor:
    def __init__(self, *, workflow_store: _InMemoryWorkflowStore, gate_store: _InMemoryGateStore) -> None:
        self._workflow_store = workflow_store
        self._gate_store = gate_store
        self.calls: list[dict[str, Any]] = []

    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        self.calls.append(
            {
                "operator_name": operator_name,
                "operator_version": operator_version,
                "idempotency_key": call.idempotency_key,
                "payload": dict(call.payload),
                "workflow_id": call.trace_context.workflow_id,
                "step_id": call.trace_context.step_id,
            }
        )

        if operator_name == "Workflow.Gate.Resolve":
            gate_id = uuid.UUID(str(call.payload["action_id"]))
            status = str(call.payload.get("status") or "declined")
            payload = call.payload.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            gate = await self._gate_store.get_gate(gate_id=gate_id)
            if gate is not None:
                await self._gate_store.resolve_gate(
                    gate_id=gate_id,
                    actor={
                        "tenant_id": call.auth_context.tenant_id,
                        "principal": call.auth_context.principal,
                    },
                    decision=status,
                    payload=payload_dict,
                )
                if status == "accepted":
                    await self._workflow_store.mark_step_ready(workflow_id=gate["workflow_id"], step_id=gate["step_id"])
                    await self._workflow_store.update_run_status(workflow_id=gate["workflow_id"], status="running")
                else:
                    await self._workflow_store.mark_step_cancelled(
                        workflow_id=gate["workflow_id"],
                        step_id=gate["step_id"],
                    )
                    await self._workflow_store.finish_run(workflow_id=gate["workflow_id"], status="cancelled")

            return OperatorResult(
                status="succeeded",
                result={"action_id": str(gate_id), "status": status},
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "FundingRequest.Fields.Update.Propose":
            result_payload = {
                "outcome": {
                    "outcome_id": str(uuid.uuid4()),
                    "payload": {"fields": call.payload.get("fields", {})},
                }
            }
        elif operator_name == "FundingRequest.Fields.Update.Apply":
            result_payload = {"applied": True, "fields": call.payload.get("fields", {})}
        elif operator_name == "StudentProfile.Update":
            result_payload = {"updated": True, "profile_updates": call.payload.get("profile_updates", {})}
        else:
            result_payload = {"ok": True}

        return OperatorResult(
            status="succeeded",
            result=result_payload,
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


class _FakePolicyStore:
    def __init__(self) -> None:
        self.decisions: list[Any] = []

    async def record(self, decision) -> uuid.UUID:
        self.decisions.append(decision)
        return uuid.uuid4()


class _FakeEventWriter:
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def append(self, event: LedgerEvent) -> None:
        self.events.append(event)


class _FakeContracts:
    def __init__(self, templates: dict[str, dict[str, Any]]) -> None:
        self._templates = templates

    def get_intent_schema(self, _intent_type: str) -> dict[str, Any]:
        return {}

    def resolver_for(self, _schema: dict[str, Any]):
        return None

    def get_plan_template(self, intent_type: str) -> dict[str, Any]:
        return dict(self._templates[intent_type])

    def list_intent_types(self) -> list[str]:
        return sorted(self._templates.keys())


def _actor() -> dict[str, Any]:
    return {
        "tenant_id": 1,
        "principal": {"type": "student", "id": 7},
        "role": "student",
        "trust_level": 0,
        "scopes": ["chat"],
    }


def _build_kernel() -> tuple[WorkflowKernel, _InMemoryWorkflowStore, _InMemoryGateStore, _FakeOperatorExecutor]:
    templates = {
        "Student.Profile.Collect": {
            "intent_type": "Student.Profile.Collect",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "policy_check",
                    "name": "EnsureProfileEmail",
                    "effects": ["read_only"],
                    "policy_tags": ["onboarding"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "check": {
                        "check_name": "EnsureEmailPresent",
                        "params": {
                            "sources": ["intent.inputs.profile_updates.email"],
                            "missing_field_key": "email",
                            "on_missing_action_type": "collect_profile_fields",
                        },
                    },
                },
                {
                    "step_id": "s2",
                    "kind": "operator",
                    "name": "StudentProfile.Update",
                    "operator_name": "StudentProfile.Update",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "depends_on": ["s1"],
                    "idempotency_template": "student_update:{tenant_id}:{thread_id}:{intent.inputs.profile_updates.email}",
                    "payload": {"profile_updates": {"from": "intent.inputs.profile_updates"}},
                },
            ],
        },
        "Funding.Request.Fields.Update": {
            "intent_type": "Funding.Request.Fields.Update",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "FundingRequest.Fields.Update.Propose",
                    "operator_name": "FundingRequest.Fields.Update.Propose",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["platform_patch"],
                    "risk_level": "medium",
                    "cache_policy": "never",
                    "idempotency_template": "proposal:{tenant_id}:{thread_id}",
                    "payload": {"fields": {"from": "intent.inputs.fields"}},
                    "produces": ["outcome.platform_patch_proposal"],
                },
                {
                    "step_id": "s2",
                    "kind": "human_gate",
                    "name": "Apply platform patch",
                    "effects": ["db_write"],
                    "policy_tags": ["approval_required"],
                    "risk_level": "high",
                    "cache_policy": "never",
                    "depends_on": ["s1"],
                    "gate": {
                        "gate_type": "apply_platform_patch",
                        "title": "Apply to request",
                        "description": "Approve patch",
                        "reason_code": "REQUIRES_APPROVAL",
                        "requires_user_input": True,
                        "target_outcome_ref": {"from": "outcome.platform_patch_proposal"},
                    },
                },
                {
                    "step_id": "s3",
                    "kind": "operator",
                    "name": "FundingRequest.Fields.Update.Apply",
                    "operator_name": "FundingRequest.Fields.Update.Apply",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["platform_patch"],
                    "risk_level": "high",
                    "cache_policy": "never",
                    "depends_on": ["s2"],
                    "idempotency_template": "apply_patch:{tenant_id}:{thread_id}:{intent.inputs.fields.research_interest}",
                    "payload": {"fields": {"from": "intent.inputs.fields"}},
                },
            ],
        },
        "Workflow.Gate.Resolve": {
            "intent_type": "Workflow.Gate.Resolve",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "Workflow.Gate.Resolve",
                    "operator_name": "Workflow.Gate.Resolve",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["workflow", "gate"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "gate_resolve:{tenant_id}:{intent.inputs.action_id}:{intent.inputs.status}",
                    "payload": {
                        "action_id": {"from": "intent.inputs.action_id"},
                        "status": {"from": "intent.inputs.status"},
                        "payload": {"from": "intent.inputs.payload"},
                    },
                }
            ],
        },
    }

    contracts = _FakeContracts(templates)
    workflow_store = _InMemoryWorkflowStore()
    gate_store = _InMemoryGateStore()
    operator_executor = _FakeOperatorExecutor(workflow_store=workflow_store, gate_store=gate_store)
    kernel = WorkflowKernel(
        contracts=contracts,
        operator_executor=operator_executor,
        policy_engine=PolicyEngine(),
        policy_store=_FakePolicyStore(),
        event_writer=_FakeEventWriter(),
        pool=object(),
        tenant_id=1,
    )
    object.__setattr__(kernel, "_intent_store", _InMemoryIntentStore())
    object.__setattr__(kernel, "_plan_store", _InMemoryPlanStore())
    object.__setattr__(kernel, "_workflow_store", workflow_store)
    object.__setattr__(kernel, "_outcome_store", _InMemoryOutcomeStore())
    object.__setattr__(kernel, "_gate_store", gate_store)
    return kernel, workflow_store, gate_store, operator_executor


@pytest.mark.asyncio
async def test_collect_profile_fields_gate_pause_resume_preserves_idempotency_key() -> None:
    kernel, workflow_store, gate_store, executor = _build_kernel()

    started = await kernel.start_intent(
        intent_type="Student.Profile.Collect",
        inputs={"profile_updates": {}},
        thread_id=11,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )
    assert started.status == "waiting"
    assert started.gate_id is not None

    gate = await gate_store.get_gate(gate_id=started.gate_id)
    assert gate is not None
    assert gate["gate_type"] == "collect_profile_fields"

    await kernel.resolve_action(
        action_id=str(started.gate_id),
        status="accepted",
        payload={"email": "student@example.com"},
        actor=_actor(),
        source="test",
    )

    run = await workflow_store.get_run(workflow_id=started.workflow_id)
    assert run is not None
    assert run.status == "completed"

    profile_update_calls = [item for item in executor.calls if item["operator_name"] == "StudentProfile.Update"]
    assert len(profile_update_calls) == 1
    assert profile_update_calls[0]["idempotency_key"] == "student_update:1:11:student@example.com"
    assert workflow_store.step(workflow_id=started.workflow_id, step_id="s2")["idempotency_key"] == (
        "student_update:1:11:student@example.com"
    )

    await kernel._execute_existing_workflow(workflow_id=started.workflow_id, actor=_actor())
    profile_update_calls = [item for item in executor.calls if item["operator_name"] == "StudentProfile.Update"]
    assert len(profile_update_calls) == 1


@pytest.mark.asyncio
async def test_apply_platform_patch_gate_pause_resume_preserves_idempotency_key() -> None:
    kernel, workflow_store, gate_store, executor = _build_kernel()

    started = await kernel.start_intent(
        intent_type="Funding.Request.Fields.Update",
        inputs={"fields": {"research_interest": "machine learning"}},
        thread_id=12,
        scope_type="funding_request",
        scope_id="55",
        actor=_actor(),
        source="test",
    )
    assert started.status == "waiting"
    assert started.gate_id is not None

    gate = await gate_store.get_gate(gate_id=started.gate_id)
    assert gate is not None
    assert gate["gate_type"] == "apply_platform_patch"

    await kernel.resolve_action(
        action_id=str(started.gate_id),
        status="accepted",
        payload={},
        actor=_actor(),
        source="test",
    )

    run = await workflow_store.get_run(workflow_id=started.workflow_id)
    assert run is not None
    assert run.status == "completed"

    apply_calls = [item for item in executor.calls if item["operator_name"] == "FundingRequest.Fields.Update.Apply"]
    assert len(apply_calls) == 1
    assert apply_calls[0]["idempotency_key"] == "apply_patch:1:12:machine learning"
    assert workflow_store.step(workflow_id=started.workflow_id, step_id="s3")["idempotency_key"] == (
        "apply_patch:1:12:machine learning"
    )

    await kernel._execute_existing_workflow(workflow_id=started.workflow_id, actor=_actor())
    apply_calls = [item for item in executor.calls if item["operator_name"] == "FundingRequest.Fields.Update.Apply"]
    assert len(apply_calls) == 1
