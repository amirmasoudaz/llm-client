from __future__ import annotations

import uuid
from typing import Any

import pytest

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


def _actor() -> dict[str, Any]:
    return {
        "tenant_id": 1,
        "principal": {"type": "student", "id": 7},
        "role": "student",
        "trust_level": 0,
        "scopes": ["chat"],
    }


class _EmailOptimizeExecutor:
    def __init__(
        self,
        *,
        workflow_store: InMemoryWorkflowStore,
        gate_store: InMemoryGateStore,
    ) -> None:
        self._workflow_store = workflow_store
        self._gate_store = gate_store
        self.calls: list[dict[str, Any]] = []
        self.platform_context: dict[str, Any] = {
            "funding_request": {"id": 41, "email_subject": "Subject", "email_content": None},
            "email": {"id": 101, "main_email_body": None},
            "professor": {"id": 9001},
        }

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
                result={"platform": self.platform_context, "intelligence": {}},
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Email.OptimizeDraft":
            return OperatorResult(
                status="succeeded",
                result={
                    "outcome": {
                        "outcome_id": str(uuid.uuid4()),
                        "outcome_type": "Email.Draft",
                        "payload": {
                            "subject": call.payload.get("fallback_subject") or "Subject",
                            "body": call.payload.get("fallback_body") or "Body",
                        },
                    }
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        return OperatorResult(
            status="succeeded",
            result={"ok": True},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


def _build_kernel() -> tuple[
    WorkflowKernel,
    InMemoryWorkflowStore,
    InMemoryGateStore,
    _EmailOptimizeExecutor,
    FakeEventWriter,
]:
    templates = {
        "Funding.Outreach.Email.Optimize": {
            "intent_type": "Funding.Outreach.Email.Optimize",
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
                    "kind": "policy_check",
                    "name": "Ensure.Email.Present",
                    "effects": ["read_only"],
                    "policy_tags": ["requires_input"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "check": {
                        "check_name": "EnsureEmailPresent",
                        "params": {
                            "sources": [
                                "intent.inputs.email_text_override",
                                "context.platform.funding_request.email_content",
                                "context.platform.email.main_email_body",
                            ],
                            "on_missing_action_type": "collect_fields",
                            "on_missing_requires_user_input": False,
                        },
                    },
                },
                {
                    "step_id": "s3",
                    "kind": "operator",
                    "name": "Email.OptimizeDraft",
                    "operator_name": "Email.OptimizeDraft",
                    "operator_version": "1.0.0",
                    "effects": ["produce_outcome"],
                    "policy_tags": ["draft_only"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "depends_on": ["s2"],
                    "idempotency_template": "email_optimize:{tenant_id}:{thread_id}",
                    "payload": {
                        "requested_edits": {"from": "intent.inputs.requested_edits"},
                        "fallback_subject": {"from": "context.platform.funding_request.email_subject"},
                        "fallback_body": {"from": "context.platform.funding_request.email_content"},
                        "professor": {"from": "context.platform.professor"},
                        "funding_request": {"from": "context.platform.funding_request"},
                    },
                    "produces": ["outcome.email_draft"],
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
    operator_executor = _EmailOptimizeExecutor(
        workflow_store=workflow_store,
        gate_store=gate_store,
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
async def test_missing_draft_gate_then_resume_after_draft_appears() -> None:
    kernel, workflow_store, gate_store, operator_executor, event_writer = _build_kernel()

    started = await kernel.start_intent(
        intent_type="Funding.Outreach.Email.Optimize",
        inputs={"requested_edits": ["improve_clarity"]},
        thread_id=33,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )
    assert started.status == "waiting"
    assert started.gate_id is not None

    waiting_gate = await gate_store.get_gate(gate_id=started.gate_id)
    assert waiting_gate is not None
    assert waiting_gate["gate_type"] == "collect_fields"
    action_hint = waiting_gate["preview"]["data"]["action_hint"]
    assert action_hint == {
        "action": "generate_draft_preview",
        "method": "GET",
        "endpoint": "/api/v1/funding/41/review",
    }

    operator_executor.platform_context["funding_request"]["email_content"] = "Freshly generated draft body"

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

    optimize_calls = [call for call in operator_executor.calls if call["operator_name"] == "Email.OptimizeDraft"]
    assert len(optimize_calls) == 1
    assert optimize_calls[0]["payload"]["fallback_body"] == "Freshly generated draft body"

    final_results = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "final_result"
    ]
    assert final_results
