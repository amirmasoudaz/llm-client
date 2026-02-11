from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from intelligence_layer_kernel.events.types import LedgerEvent
from intelligence_layer_kernel.operators.types import OperatorCall, OperatorMetrics, OperatorResult
from intelligence_layer_kernel.policy import PolicyEngine
from intelligence_layer_kernel.runtime.kernel import WorkflowKernel
from intelligence_layer_kernel.runtime.store import WorkflowRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


class IntentStore:
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


class PlanStore:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, dict[str, Any]] = {}

    async def insert(self, plan, _plan_hash: bytes) -> None:
        self.rows[plan.plan_id] = {"plan_id": plan.plan_id, "intent_id": plan.intent_id, "plan": plan.to_dict()}

    async def fetch(self, plan_id: uuid.UUID) -> dict[str, Any] | None:
        return self.rows.get(plan_id)


class WorkflowStore:
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

    def _step(self, workflow_id: uuid.UUID, step_id: str) -> dict[str, Any]:
        for row in self.steps.get(workflow_id, []):
            if row["step_id"] == step_id:
                return row
        raise KeyError(step_id)


class OutcomeStore:
    def __init__(self) -> None:
        self.by_workflow: dict[uuid.UUID, list[dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []

    async def record(self, **kwargs) -> uuid.UUID | None:
        self.calls.append(dict(kwargs))
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


class GateStore:
    async def create_gate(self, **kwargs):
        raise AssertionError("gates are not expected in this test")

    async def get_gate(self, **kwargs):
        _ = kwargs
        return None

    async def resolve_gate(self, **kwargs):
        raise AssertionError("gates are not expected in this test")

    async def latest_decision(self, **kwargs):
        _ = kwargs
        return None


class OperatorExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        self.calls.append(
            {
                "operator_name": operator_name,
                "operator_version": operator_version,
                "idempotency_key": call.idempotency_key,
                "workflow_id": call.trace_context.workflow_id,
                "step_id": call.trace_context.step_id,
                "payload": dict(call.payload),
            }
        )
        return OperatorResult(
            status="succeeded",
            result={"outcome": {"payload": {"summary": "token output from llm-backed operator"}}},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


class PolicyStore:
    async def record(self, _decision) -> uuid.UUID:
        return uuid.uuid4()


class EventWriter:
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def append(self, event: LedgerEvent) -> None:
        self.events.append(event)


class Contracts:
    def __init__(self) -> None:
        self._template = {
            "intent_type": "Funding.Outreach.Email.Review",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "Email.ReviewDraft",
                    "operator_name": "Email.ReviewDraft",
                    "operator_version": "1.0.0",
                    "effects": ["produce_outcome"],
                    "policy_tags": ["review"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "review:{tenant_id}:{thread_id}",
                    "payload": {"body": {"const": "Please review this draft"}},
                }
            ],
        }

    def get_intent_schema(self, _intent_type: str) -> dict[str, Any]:
        return {}

    def resolver_for(self, _schema: dict[str, Any]):
        return None

    def get_plan_template(self, _intent_type: str) -> dict[str, Any]:
        return dict(self._template)

    def list_intent_types(self) -> list[str]:
        return ["Funding.Outreach.Email.Review"]

    def get_operator_manifest(self, operator_name: str, version: str) -> dict[str, Any]:
        if operator_name == "Email.ReviewDraft" and version == "1.0.0":
            return {"name": operator_name, "version": version, "prompt_templates": {"primary": "Email.ReviewDraft/1.0.0/review_draft"}}
        return {"name": operator_name, "version": version}


def build_kernel(*, usage_recorder: Any | None = None) -> tuple[WorkflowKernel, OperatorExecutor, EventWriter]:
    contracts = Contracts()
    operator_executor = OperatorExecutor()
    event_writer = EventWriter()
    kernel = WorkflowKernel(
        contracts=contracts,
        operator_executor=operator_executor,
        policy_engine=PolicyEngine(),
        policy_store=PolicyStore(),
        event_writer=event_writer,
        pool=object(),
        tenant_id=1,
        usage_recorder=usage_recorder,
    )

    object.__setattr__(kernel, "_intent_store", IntentStore())
    object.__setattr__(kernel, "_plan_store", PlanStore())
    object.__setattr__(kernel, "_workflow_store", WorkflowStore())
    object.__setattr__(kernel, "_outcome_store", OutcomeStore())
    object.__setattr__(kernel, "_gate_store", GateStore())
    return kernel, operator_executor, event_writer


def actor(*, reserved_credits: int, used_credits: int = 0) -> dict[str, Any]:
    return {
        "tenant_id": 1,
        "principal": {"type": "student", "id": 77},
        "role": "student",
        "trust_level": 0,
        "scopes": ["chat"],
        "billing": {
            "principal_id": 77,
            "request_key": bytes([1] * 32).hex(),
            "reserved_credits": reserved_credits,
            "used_credits": used_credits,
            "provider": "openai",
            "model": "gpt-5-nano",
        },
    }
