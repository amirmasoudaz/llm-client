from __future__ import annotations

import uuid
from typing import Any

import pytest

from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_apply import (
    FundingRequestFieldsUpdateApplyOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_propose import (
    FundingRequestFieldsUpdateProposeOperator,
)
from intelligence_layer_kernel.operators.types import OperatorCall, OperatorMetrics, OperatorResult
from intelligence_layer_kernel.policy import PolicyEngine
from intelligence_layer_kernel.runtime.kernel import WorkflowKernel
from tests.intelligence_layer_kernel._phase_e_testkit import (
    FakeEventWriter,
    FakePolicyStore,
    InMemoryGateStore,
    InMemoryIntentStore,
    InMemoryOutcomeStore,
    InMemoryPlanStore,
    InMemoryWorkflowStore,
    StaticContracts,
)
from tests.intelligence_layer_kernel._phase_fg_testkit import FakePlatformDB


def _actor() -> dict[str, Any]:
    return {
        "tenant_id": 1,
        "principal": {"type": "student", "id": 7},
        "role": "student",
        "trust_level": 0,
        "scopes": ["chat"],
    }


class _FundingUpdateExecutor:
    def __init__(
        self,
        *,
        workflow_store: InMemoryWorkflowStore,
        gate_store: InMemoryGateStore,
        platform_db: FakePlatformDB,
        funding_request_id: int,
    ) -> None:
        self._workflow_store = workflow_store
        self._gate_store = gate_store
        self._funding_request_id = funding_request_id
        self.calls: list[dict[str, Any]] = []

        self._propose = FundingRequestFieldsUpdateProposeOperator()
        self._propose._db = platform_db
        self._apply = FundingRequestFieldsUpdateApplyOperator()
        self._apply._db = platform_db

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
                    await self._workflow_store.mark_step_ready(
                        workflow_id=gate["workflow_id"],
                        step_id=gate["step_id"],
                    )
                    await self._workflow_store.update_run_status(
                        workflow_id=gate["workflow_id"],
                        status="running",
                    )
                else:
                    await self._workflow_store.mark_step_cancelled(
                        workflow_id=gate["workflow_id"],
                        step_id=gate["step_id"],
                    )
                    await self._workflow_store.finish_run(
                        workflow_id=gate["workflow_id"],
                        status="cancelled",
                    )
            return OperatorResult(
                status="succeeded",
                result={"action_id": str(gate_id), "status": status},
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Platform.Context.Load":
            return OperatorResult(
                status="succeeded",
                result={
                    "platform": {"funding_request": {"id": self._funding_request_id}},
                    "intelligence": {},
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "FundingRequest.Fields.Update.Propose":
            return await self._propose.run(call)

        if operator_name == "FundingRequest.Fields.Update.Apply":
            return await self._apply.run(call)

        return OperatorResult(
            status="succeeded",
            result={"ok": True},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


def _build_kernel(
    platform_db: FakePlatformDB,
    *,
    funding_request_id: int = 41,
) -> tuple[
    WorkflowKernel,
    InMemoryWorkflowStore,
    InMemoryGateStore,
    _FundingUpdateExecutor,
    FakeEventWriter,
]:
    templates = {
        "Funding.Request.Fields.Update": {
            "intent_type": "Funding.Request.Fields.Update",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "Platform.Context.Load",
                    "operator_name": "Platform.Context.Load",
                    "operator_version": "1.0.0",
                    "effects": ["read_only"],
                    "policy_tags": ["read_only", "platform_read"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "platform_context:{tenant_id}:{thread_id}",
                    "payload": {"thread_id": {"from": "intent.thread_id"}},
                    "produces": ["context.platform", "context.intelligence"],
                },
                {
                    "step_id": "s2",
                    "kind": "operator",
                    "name": "FundingRequest.Fields.Update.Propose",
                    "operator_name": "FundingRequest.Fields.Update.Propose",
                    "operator_version": "1.0.0",
                    "effects": ["produce_outcome"],
                    "policy_tags": ["apply_proposal", "platform_write"],
                    "risk_level": "medium",
                    "cache_policy": "never",
                    "idempotency_template": "proposal:{tenant_id}:{thread_id}",
                    "payload": {
                        "funding_request_id": {"from": "context.platform.funding_request.id"},
                        "fields": {"from": "intent.inputs.fields"},
                        "human_summary": {"from": "intent.inputs.human_summary"},
                    },
                    "produces": ["outcome.platform_patch_proposal"],
                },
                {
                    "step_id": "s3",
                    "kind": "human_gate",
                    "name": "Apply.PlatformPatch",
                    "effects": ["db_write_platform"],
                    "policy_tags": ["apply", "platform_write"],
                    "risk_level": "medium",
                    "cache_policy": "never",
                    "gate": {
                        "gate_type": "apply_platform_patch",
                        "reason_code": "REQUIRES_APPROVAL",
                        "title": "Apply funding request updates",
                        "description": "Apply the proposed updates to your funding request.",
                        "requires_user_input": False,
                        "ui_hints": {"primary_button": "Apply changes"},
                        "target_outcome_ref": {"from": "outcome.platform_patch_proposal"},
                    },
                },
                {
                    "step_id": "s4",
                    "kind": "operator",
                    "name": "FundingRequest.Fields.Update.Apply",
                    "operator_name": "FundingRequest.Fields.Update.Apply",
                    "operator_version": "1.0.0",
                    "effects": ["db_write_platform"],
                    "policy_tags": ["apply", "platform_write"],
                    "risk_level": "medium",
                    "cache_policy": "never",
                    "depends_on": ["s3"],
                    "idempotency_template": "apply:{tenant_id}:{thread_id}",
                    "payload": {
                        "funding_request_id": {"from": "context.platform.funding_request.id"},
                        "proposal": {"from": "outcome.platform_patch_proposal"},
                        "strict_optimistic_lock": {"const": True},
                    },
                    "produces": ["outcome.platform_patch_receipt"],
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

    contracts = StaticContracts(templates)
    workflow_store = InMemoryWorkflowStore()
    gate_store = InMemoryGateStore()
    event_writer = FakeEventWriter()
    operator_executor = _FundingUpdateExecutor(
        workflow_store=workflow_store,
        gate_store=gate_store,
        platform_db=platform_db,
        funding_request_id=funding_request_id,
    )
    kernel = WorkflowKernel(
        contracts=contracts,
        operator_executor=operator_executor,
        policy_engine=PolicyEngine(),
        policy_store=FakePolicyStore(),
        event_writer=event_writer,
        pool=object(),
        tenant_id=1,
    )
    object.__setattr__(kernel, "_intent_store", InMemoryIntentStore())
    object.__setattr__(kernel, "_plan_store", InMemoryPlanStore())
    object.__setattr__(kernel, "_workflow_store", workflow_store)
    object.__setattr__(kernel, "_outcome_store", InMemoryOutcomeStore())
    object.__setattr__(kernel, "_gate_store", gate_store)
    return kernel, workflow_store, gate_store, operator_executor, event_writer


@pytest.mark.asyncio
async def test_funding_request_update_end_to_end_gate_apply_updates_platform_and_emits_refresh() -> None:
    platform_db = FakePlatformDB(
        funding_requests={
            41: {
                "research_interest": "robotics",
                "paper_title": "Old title",
                "journal": "Old Journal",
                "year": 2022,
                "research_connection": "Old connection",
            }
        }
    )
    kernel, workflow_store, gate_store, _executor, event_writer = _build_kernel(platform_db, funding_request_id=41)

    started = await kernel.start_intent(
        intent_type="Funding.Request.Fields.Update",
        inputs={
            "fields": {"research_interest": "machine learning", "paper_title": "New title"},
            "human_summary": "Update request metadata",
        },
        thread_id=21,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )
    assert started.status == "waiting"
    assert started.gate_id is not None

    waiting_gate = await gate_store.get_gate(gate_id=started.gate_id)
    assert waiting_gate is not None
    assert waiting_gate["gate_type"] == "apply_platform_patch"

    action_required_events = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "action_required"
    ]
    assert action_required_events
    action_payload = action_required_events[-1].payload
    assert action_payload["action_type"] == "apply_platform_patch"
    assert action_payload["apply_action_id"] == str(started.gate_id)
    assert isinstance(action_payload.get("proposed_changes"), dict)

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

    updated_row = platform_db.funding_requests[41]
    assert updated_row["research_interest"] == "machine learning"
    assert updated_row["paper_title"] == "New title"
    assert updated_row["journal"] == "Old Journal"
    assert updated_row["research_connection"] == "Old connection"

    refresh_events = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "ui.refresh_required"
    ]
    assert refresh_events
    refresh_payload = refresh_events[-1].payload
    assert refresh_payload == {
        "target": "funding_request",
        "reason": "funding_request_updated",
        "request_id": 41,
    }
