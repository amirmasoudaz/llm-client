from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from intelligence_layer_kernel.events.types import LedgerEvent
from intelligence_layer_kernel.operators.types import OperatorCall, OperatorMetrics, OperatorResult
from intelligence_layer_kernel.policy import PolicyEngine
from intelligence_layer_kernel.runtime.kernel import WorkflowKernel
from intelligence_layer_kernel.runtime.store import OutcomeStore, WorkflowRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _IntentStore:
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


class _PlanStore:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, dict[str, Any]] = {}

    async def insert(self, plan, _plan_hash: bytes) -> None:
        self.rows[plan.plan_id] = {"plan_id": plan.plan_id, "intent_id": plan.intent_id, "plan": plan.to_dict()}

    async def fetch(self, plan_id: uuid.UUID) -> dict[str, Any] | None:
        return self.rows.get(plan_id)


class _WorkflowStore:
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


class _OutcomeStore:
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


class _GateStore:
    async def create_gate(self, **kwargs):
        raise AssertionError("gates are not expected in these tests")

    async def get_gate(self, **kwargs):
        _ = kwargs
        return None

    async def resolve_gate(self, **kwargs):
        raise AssertionError("gates are not expected in these tests")

    async def latest_decision(self, **kwargs):
        _ = kwargs
        return None


class _OperatorExecutor:
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
            result={"ok": True, "operator": operator_name},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


class _PolicyStore:
    async def record(self, _decision) -> uuid.UUID:
        return uuid.uuid4()


class _EventWriter:
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def append(self, event: LedgerEvent) -> None:
        self.events.append(event)


class _Contracts:
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
        "principal": {"type": "student", "id": 77},
        "role": "student",
        "trust_level": 0,
        "scopes": ["chat"],
    }


def _build_kernel() -> tuple[WorkflowKernel, _WorkflowStore, _OperatorExecutor, _OutcomeStore]:
    templates = {
        "Funding.Outreach.Email.Review": {
            "intent_type": "Funding.Outreach.Email.Review",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "Email.ReviewDraft",
                    "operator_name": "Email.ReviewDraft",
                    "operator_version": "1.0.0",
                    "effects": ["read_only"],
                    "policy_tags": ["review"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "review:{tenant_id}:{thread_id}",
                    "payload": {},
                }
            ],
        },
        "Funding.Request.Fields.Update": {
            "intent_type": "Funding.Request.Fields.Update",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "Draft.Generate",
                    "operator_name": "Draft.Generate",
                    "operator_version": "1.0.0",
                    "effects": ["read_only"],
                    "policy_tags": ["draft"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "draft:{tenant_id}:{thread_id}",
                    "payload": {},
                },
                {
                    "step_id": "s2",
                    "kind": "operator",
                    "name": "FundingRequest.Fields.Update.Apply",
                    "operator_name": "FundingRequest.Fields.Update.Apply",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["apply"],
                    "risk_level": "high",
                    "cache_policy": "never",
                    "depends_on": ["s1"],
                    "idempotency_template": "apply:{tenant_id}:{thread_id}",
                    "payload": {},
                },
            ],
        },
    }

    contracts = _Contracts(templates)
    workflow_store = _WorkflowStore()
    operator_executor = _OperatorExecutor()
    outcome_store = _OutcomeStore()

    kernel = WorkflowKernel(
        contracts=contracts,
        operator_executor=operator_executor,
        policy_engine=PolicyEngine(),
        policy_store=_PolicyStore(),
        event_writer=_EventWriter(),
        pool=object(),
        tenant_id=1,
    )
    object.__setattr__(kernel, "_intent_store", _IntentStore())
    object.__setattr__(kernel, "_plan_store", _PlanStore())
    object.__setattr__(kernel, "_workflow_store", workflow_store)
    object.__setattr__(kernel, "_outcome_store", outcome_store)
    object.__setattr__(kernel, "_gate_store", _GateStore())
    return kernel, workflow_store, operator_executor, outcome_store


@pytest.mark.asyncio
async def test_rerun_replay_reuses_idempotency_and_sets_parent_workflow() -> None:
    kernel, workflow_store, executor, outcome_store = _build_kernel()
    started = await kernel.start_intent(
        intent_type="Funding.Outreach.Email.Review",
        inputs={},
        thread_id=21,
        scope_type="funding_request",
        scope_id="500",
        actor=_actor(),
    )
    assert started.status == "completed"

    replayed = await kernel.rerun_workflow(
        workflow_id=started.workflow_id,
        mode="replay",
        actor=_actor(),
        source="test",
    )
    assert replayed.status == "completed"
    assert replayed.workflow_id != started.workflow_id

    calls_by_workflow: dict[str, list[dict[str, Any]]] = {}
    for call in executor.calls:
        calls_by_workflow.setdefault(call["workflow_id"], []).append(call)
    original_calls = calls_by_workflow[str(started.workflow_id)]
    replay_calls = calls_by_workflow[str(replayed.workflow_id)]
    assert original_calls[0]["idempotency_key"] == "review:1:21"
    assert replay_calls[0]["idempotency_key"] == "review:1:21"

    replay_run = await workflow_store.get_run(workflow_id=replayed.workflow_id)
    assert replay_run is not None
    assert replay_run.replay_mode == "replay"
    assert replay_run.parent_workflow_id == started.workflow_id

    replay_outcomes = [item for item in outcome_store.calls if item["workflow_id"] == replayed.workflow_id]
    assert replay_outcomes[0]["replay_mode"] == "replay"
    assert replay_outcomes[0]["parent_workflow_id"] == started.workflow_id


@pytest.mark.asyncio
async def test_rerun_regenerate_mints_new_non_effectful_keys_only() -> None:
    kernel, workflow_store, executor, outcome_store = _build_kernel()
    started = await kernel.start_intent(
        intent_type="Funding.Request.Fields.Update",
        inputs={},
        thread_id=22,
        scope_type="funding_request",
        scope_id="501",
        actor=_actor(),
    )
    assert started.status == "completed"

    regenerated = await kernel.rerun_workflow(
        workflow_id=started.workflow_id,
        mode="regenerate",
        actor=_actor(),
        source="test",
    )
    assert regenerated.status == "completed"
    assert regenerated.workflow_id != started.workflow_id

    original_calls = [item for item in executor.calls if item["workflow_id"] == str(started.workflow_id)]
    regenerated_calls = [item for item in executor.calls if item["workflow_id"] == str(regenerated.workflow_id)]

    original_by_step = {item["step_id"]: item for item in original_calls}
    regenerated_by_step = {item["step_id"]: item for item in regenerated_calls}

    assert original_by_step["s1"]["idempotency_key"] == "draft:1:22"
    regenerated_draft_key = regenerated_by_step["s1"]["idempotency_key"]
    assert regenerated_draft_key.startswith("draft:1:22:regen:")
    assert str(regenerated.workflow_id) in regenerated_draft_key

    assert original_by_step["s2"]["idempotency_key"] == "apply:1:22"
    assert regenerated_by_step["s2"]["idempotency_key"] == "apply:1:22"

    regenerated_run = await workflow_store.get_run(workflow_id=regenerated.workflow_id)
    assert regenerated_run is not None
    assert regenerated_run.replay_mode == "regenerate"
    assert regenerated_run.parent_workflow_id == started.workflow_id

    regenerated_outcomes = [item for item in outcome_store.calls if item["workflow_id"] == regenerated.workflow_id]
    assert len(regenerated_outcomes) == 2
    for call in regenerated_outcomes:
        assert call["replay_mode"] == "regenerate"
        assert call["parent_workflow_id"] == started.workflow_id


class _Acquire:
    def __init__(self, conn: "_Conn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_Conn":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _Conn:
    def __init__(self, *, parent_row: dict[str, Any] | None = None) -> None:
        self.parent_row = parent_row
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append((query, args))

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((query, args))
        return self.parent_row

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        _ = (query, args)
        return []


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


@pytest.mark.asyncio
async def test_outcome_store_regenerate_reuses_parent_lineage() -> None:
    parent_outcome_id = uuid.uuid4()
    parent_lineage_id = uuid.uuid4()
    conn = _Conn(
        parent_row={
            "outcome_id": parent_outcome_id,
            "lineage_id": parent_lineage_id,
            "version": 3,
        }
    )
    store = OutcomeStore(pool=_Pool(conn), tenant_id=1)
    parent_workflow_id = uuid.uuid4()

    await store.record(
        workflow_id=uuid.uuid4(),
        thread_id=9,
        intent_id=uuid.uuid4(),
        plan_id=uuid.uuid4(),
        step_id="s1",
        job_id=None,
        operator_name="Email.ReviewDraft",
        operator_version="1.0.0",
        status="succeeded",
        content={"outcome": {"ok": True}},
        replay_mode="regenerate",
        parent_workflow_id=parent_workflow_id,
    )

    assert len(conn.fetchrow_calls) == 1
    assert len(conn.execute_calls) == 1
    _query, args = conn.execute_calls[0]
    assert args[2] == parent_lineage_id
    assert args[3] == 4
    assert args[4] == parent_outcome_id
