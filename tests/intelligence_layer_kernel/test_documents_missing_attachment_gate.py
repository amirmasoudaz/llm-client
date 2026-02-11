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


class _DocumentsExecutor:
    def __init__(self, *, workflow_store: InMemoryWorkflowStore, gate_store: InMemoryGateStore) -> None:
        self._workflow_store = workflow_store
        self._gate_store = gate_store
        self.calls: list[dict[str, Any]] = []
        self.attachments_result: dict[str, Any] = {
            "attachments": [],
            "selected_attachment": None,
            "requested_document_type": "cv",
            "requested_attachment_kinds": ["cv"],
        }

    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        self.calls.append(
            {
                "operator_name": operator_name,
                "operator_version": operator_version,
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
                    await self._workflow_store.mark_step_cancelled(workflow_id=gate["workflow_id"], step_id=gate["step_id"])
                    await self._workflow_store.finish_run(workflow_id=gate["workflow_id"], status="cancelled")
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
                    "platform": {"funding_request": {"id": 41}},
                    "intelligence": {},
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Platform.Attachments.List":
            return OperatorResult(
                status="succeeded",
                result=self.attachments_result,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Documents.ImportFromPlatformAttachment":
            selected = call.payload.get("selected_attachment")
            attachment_id = selected.get("attachment_id") if isinstance(selected, dict) else None
            return OperatorResult(
                status="succeeded",
                result={
                    "outcome": {
                        "outcome_id": str(uuid.uuid4()),
                        "outcome_type": "Document.Uploaded",
                        "payload": {
                            "document_id": str(uuid.uuid4()),
                            "document_type": "cv",
                            "artifact": {"object_uri": "s3://bucket/cv.pdf"},
                            "attachment_id": attachment_id,
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


def _build_kernel() -> tuple[WorkflowKernel, InMemoryWorkflowStore, InMemoryGateStore, _DocumentsExecutor, FakeEventWriter]:
    templates = {
        "Documents.Review": {
            "intent_type": "Documents.Review",
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
                    "idempotency_template": "context:{tenant_id}:{thread_id}",
                    "payload": {"thread_id": {"from": "intent.thread_id"}},
                    "produces": ["context.platform", "context.intelligence"],
                },
                {
                    "step_id": "s2",
                    "kind": "operator",
                    "name": "Platform.Attachments.List",
                    "operator_name": "Platform.Attachments.List",
                    "operator_version": "1.0.0",
                    "effects": ["read_only"],
                    "policy_tags": ["documents", "platform_read"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "attachments:{tenant_id}:{thread_id}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "document_type": {"from": "intent.inputs.document_type"},
                    },
                    "produces": [
                        "context.documents.attachments",
                        "context.documents.selected_attachment",
                        "context.documents.requested_document_type",
                        "context.documents.requested_attachment_kinds",
                    ],
                },
                {
                    "step_id": "s3",
                    "kind": "policy_check",
                    "name": "Ensure.Attachment.Present",
                    "effects": ["read_only"],
                    "policy_tags": ["requires_input"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "check": {
                        "check_name": "EnsureAttachmentPresent",
                        "params": {
                            "attachment_ref": {"from": "context.documents.selected_attachment"},
                            "requested_document_type": {"from": "context.documents.requested_document_type"},
                            "requested_attachment_kinds": {"from": "context.documents.requested_attachment_kinds"},
                            "on_missing_action_type": "upload_required",
                            "on_missing_requires_user_input": False,
                        },
                    },
                },
                {
                    "step_id": "s4",
                    "kind": "operator",
                    "name": "Documents.ImportFromPlatformAttachment",
                    "operator_name": "Documents.ImportFromPlatformAttachment",
                    "operator_version": "1.0.0",
                    "depends_on": ["s3"],
                    "effects": ["produce_outcome"],
                    "policy_tags": ["documents"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "import:{tenant_id}:{thread_id}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "document_type": {"from": "context.documents.requested_document_type"},
                        "selected_attachment": {"from": "context.documents.selected_attachment"},
                    },
                    "produces": ["outcome.document_uploaded"],
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
    operator_executor = _DocumentsExecutor(workflow_store=workflow_store, gate_store=gate_store)

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
async def test_documents_review_missing_attachment_opens_upload_required_gate_then_resumes() -> None:
    kernel, workflow_store, gate_store, operator_executor, event_writer = _build_kernel()

    started = await kernel.start_intent(
        intent_type="Documents.Review",
        inputs={"document_type": "cv", "review_goal": "quality"},
        thread_id=22,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )

    assert started.status == "waiting"
    assert started.gate_id is not None

    waiting_gate = await gate_store.get_gate(gate_id=started.gate_id)
    assert waiting_gate is not None
    assert waiting_gate["gate_type"] == "upload_required"
    assert waiting_gate["reason_code"] == "MISSING_REQUIRED_ATTACHMENT"

    gate_data = waiting_gate["preview"]["data"]
    assert gate_data["missing_fields"] == ["cv_attachment"]
    assert gate_data["required_document_type"] == "cv"
    assert gate_data["required_attachment_kinds"] == ["cv"]
    assert gate_data["action_hint"] == {
        "action": "upload_attachment",
        "method": "POST",
        "endpoint": "/api/v1/funding/41/attachments",
    }

    action_required_events = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "action_required"
    ]
    assert action_required_events
    assert action_required_events[-1].payload["action_type"] == "upload_required"

    operator_executor.attachments_result["selected_attachment"] = {
        "attachment_id": 991,
        "name": "cv.pdf",
        "file_path": "uploads/cv.pdf",
    }

    await kernel.resolve_action(
        action_id=str(started.gate_id),
        status="accepted",
        payload={"attachment_ids": [991]},
        actor=_actor(),
        source="test",
    )

    run = await workflow_store.get_run(workflow_id=started.workflow_id)
    assert run is not None
    assert run.status == "completed"

    import_calls = [call for call in operator_executor.calls if call["operator_name"] == "Documents.ImportFromPlatformAttachment"]
    assert import_calls

    final_results = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "final_result"
    ]
    assert final_results
