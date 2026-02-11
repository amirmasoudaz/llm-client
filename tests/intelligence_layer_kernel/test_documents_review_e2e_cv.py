from __future__ import annotations

import uuid
from typing import Any

import pytest
from blake3 import blake3

from intelligence_layer_kernel.operators import OperatorRegistry
from intelligence_layer_kernel.operators.executor import OperatorExecutor
from intelligence_layer_kernel.operators.implementations.documents_review import DocumentsReviewOperator
from intelligence_layer_kernel.operators.store import JobClaim
from intelligence_layer_kernel.operators.types import OperatorCall, OperatorMetrics, OperatorResult
from intelligence_layer_kernel.policy import PolicyEngine
from intelligence_layer_kernel.prompts.loader import PromptRenderResult
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


class _ReviewAcquire:
    def __init__(self, conn: "_ReviewConn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_ReviewConn":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _ReviewConn:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row

    async def fetchrow(self, _query: str, *_args: Any) -> dict[str, Any] | None:
        return self.row


class _ReviewPool:
    def __init__(self, row: dict[str, Any]) -> None:
        self._conn = _ReviewConn(row)

    def acquire(self) -> _ReviewAcquire:
        return _ReviewAcquire(self._conn)


class _OpContracts:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self._manifest = manifest

    def get_operator_manifest(self, name: str, version: str) -> dict[str, Any]:
        if name != "Documents.Review" or version != "1.0.0":
            raise KeyError(f"unknown manifest: {name}@{version}")
        return dict(self._manifest)

    def get_schema_by_ref(self, _ref: str) -> dict[str, Any]:
        return {}

    def resolver_for(self, _schema: dict[str, Any]):
        return None


class _JobStore:
    async def claim_job(self, **_kwargs) -> JobClaim:
        return JobClaim(status="new", job_id=uuid.uuid4(), attempt_no=1, result_payload=None)

    async def complete_job(self, **_kwargs) -> None:
        return None


class _OperatorEventWriter:
    async def append(self, _event) -> None:
        return None


class _FakePromptLoader:
    def render(self, template_id: str, context: dict[str, Any]) -> PromptRenderResult:
        normalized = template_id.strip().lstrip("/")
        if not normalized.endswith(".j2"):
            normalized = normalized + ".j2"
        rendered = str(sorted(context.keys()))
        return PromptRenderResult(
            template_id=normalized,
            template_hash=blake3(normalized.encode("utf-8")).hexdigest(),
            text=rendered,
        )


class _WorkflowExecutor:
    def __init__(self, *, workflow_store: InMemoryWorkflowStore, gate_store: InMemoryGateStore) -> None:
        self._workflow_store = workflow_store
        self._gate_store = gate_store
        self.document_id = str(uuid.uuid4())
        review_row = {
            "document_type": "cv",
            "extracted_text": "Education Experience Skills",
            "extracted_fields": {
                "word_count": 540,
                "has_education_section": True,
                "has_experience_section": True,
                "has_skills_section": True,
                "emails": ["student@example.com"],
            },
        }

        review_manifest = {
            "name": "Documents.Review",
            "version": "1.0.0",
            "type": "operator",
            "schemas": {
                "input": "schemas/operators/documents_review.input.v1.json",
                "output": "schemas/operators/documents_review.output.v1.json",
            },
            "policy_tags": ["documents", "draft_only"],
            "effects": ["produce_outcome"],
            "prompt_templates": {
                "primary": "Documents.Review/1.0.0/review_generic",
                "cv": "Documents.Review/1.0.0/review_cv",
                "sop": "Documents.Review/1.0.0/review_sop",
                "letter": "Documents.Review/1.0.0/review_letter",
            },
        }

        contracts = _OpContracts(review_manifest)
        registry = OperatorRegistry(contracts)
        registry.register(DocumentsReviewOperator(pool=_ReviewPool(review_row), tenant_id=1))

        self._review_executor = OperatorExecutor(
            contracts=contracts,
            registry=registry,
            job_store=_JobStore(),
            policy_engine=PolicyEngine(),
            policy_store=FakePolicyStore(),
            event_writer=_OperatorEventWriter(),
            prompt_loader=_FakePromptLoader(),
        )

    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        if operator_name == "Workflow.Gate.Resolve":
            gate_id = uuid.UUID(str(call.payload["action_id"]))
            status = str(call.payload.get("status") or "declined")
            payload = call.payload.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            gate = await self._gate_store.get_gate(gate_id=gate_id)
            if gate is not None:
                await self._gate_store.resolve_gate(
                    gate_id=gate_id,
                    actor={"tenant_id": call.auth_context.tenant_id, "principal": call.auth_context.principal},
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
                result={"platform": {"funding_request": {"id": 41}}, "intelligence": {}},
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Platform.Attachments.List":
            return OperatorResult(
                status="succeeded",
                result={
                    "attachments": [
                        {
                            "attachment_id": 9001,
                            "name": "cv.pdf",
                            "file_path": "uploads/cv.pdf",
                            "mime": "application/pdf",
                            "kind": "cv",
                        }
                    ],
                    "selected_attachment": {
                        "attachment_id": 9001,
                        "name": "cv.pdf",
                        "file_path": "uploads/cv.pdf",
                        "mime": "application/pdf",
                        "kind": "cv",
                    },
                    "requested_document_type": "cv",
                    "requested_attachment_kinds": ["cv"],
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Documents.ImportFromPlatformAttachment":
            return OperatorResult(
                status="succeeded",
                result={
                    "outcome": {
                        "outcome_id": str(uuid.uuid4()),
                        "outcome_type": "Document.Uploaded",
                        "payload": {
                            "document_id": self.document_id,
                            "document_type": "cv",
                            "artifact": {"object_uri": "s3://bucket/cv.pdf"},
                        },
                    }
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Documents.Process":
            return OperatorResult(
                status="succeeded",
                result={
                    "outcome": {
                        "outcome_id": str(uuid.uuid4()),
                        "outcome_type": "Document.Processed",
                        "payload": {
                            "document_id": self.document_id,
                            "extracted_fields": {
                                "word_count": 540,
                                "has_education_section": True,
                                "has_experience_section": True,
                                "has_skills_section": True,
                                "emails": ["student@example.com"],
                            },
                            "text_hash": {"alg": "blake3", "value": "hash-1"},
                        },
                        "hash": {"alg": "blake3", "value": "processed-hash"},
                    }
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Documents.Review":
            return await self._review_executor.execute(
                operator_name=operator_name,
                operator_version=operator_version,
                call=call,
            )

        return OperatorResult(
            status="succeeded",
            result={"ok": True},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


def _build_kernel() -> tuple[WorkflowKernel, InMemoryWorkflowStore, InMemoryOutcomeStore, FakeEventWriter]:
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
                    "idempotency_template": "platform_context:{tenant_id}:{thread_id}",
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
                    "idempotency_template": "attachments:{tenant_id}:{thread_id}:{intent.inputs.document_type}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "document_type": {"from": "intent.inputs.document_type"},
                        "attachment_ids": {"from": "intent.inputs.attachment_ids"},
                    },
                    "produces": [
                        "context.documents.attachments",
                        "context.documents.selected_attachment",
                        "context.documents.requested_document_type",
                    ],
                },
                {
                    "step_id": "s3",
                    "kind": "operator",
                    "name": "Documents.ImportFromPlatformAttachment",
                    "operator_name": "Documents.ImportFromPlatformAttachment",
                    "operator_version": "1.0.0",
                    "effects": ["produce_outcome"],
                    "policy_tags": ["documents", "platform_read"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "import:{tenant_id}:{thread_id}:{context.documents.selected_attachment.attachment_id}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "document_type": {"from": "context.documents.requested_document_type"},
                        "selected_attachment": {"from": "context.documents.selected_attachment"},
                    },
                    "produces": ["outcome.document_uploaded"],
                },
                {
                    "step_id": "s4",
                    "kind": "operator",
                    "name": "Documents.Process",
                    "operator_name": "Documents.Process",
                    "operator_version": "1.0.0",
                    "effects": ["produce_outcome"],
                    "policy_tags": ["documents"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "process:{tenant_id}:{outcome.document_uploaded.payload.document_id}",
                    "payload": {
                        "document_id": {"from": "outcome.document_uploaded.payload.document_id"},
                        "processing_profile": {"const": "default"},
                    },
                    "produces": ["outcome.document_processed"],
                },
                {
                    "step_id": "s5",
                    "kind": "operator",
                    "name": "Documents.Review",
                    "operator_name": "Documents.Review",
                    "operator_version": "1.0.0",
                    "effects": ["produce_outcome"],
                    "policy_tags": ["documents", "draft_only"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "review:{tenant_id}:{outcome.document_uploaded.payload.document_id}",
                    "payload": {
                        "document_id": {"from": "outcome.document_uploaded.payload.document_id"},
                        "document_type": {"from": "outcome.document_uploaded.payload.document_type"},
                        "document_processed": {"from": "outcome.document_processed"},
                        "review_goal": {"from": "intent.inputs.review_goal"},
                        "custom_instructions": {"from": "intent.inputs.custom_instructions"},
                    },
                    "produces": ["outcome.document_review"],
                },
            ],
        },
    }

    contracts = StaticContracts(templates)
    workflow_store = InMemoryWorkflowStore()
    gate_store = InMemoryGateStore()
    outcome_store = InMemoryOutcomeStore()
    event_writer = FakeEventWriter()
    operator_executor = _WorkflowExecutor(workflow_store=workflow_store, gate_store=gate_store)

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
    object.__setattr__(kernel, "_outcome_store", outcome_store)
    object.__setattr__(kernel, "_gate_store", gate_store)
    return kernel, workflow_store, outcome_store, event_writer


def _actor() -> dict[str, Any]:
    return {
        "tenant_id": 1,
        "principal": {"type": "student", "id": 7},
        "role": "student",
        "trust_level": 0,
        "scopes": ["chat"],
    }


@pytest.mark.asyncio
async def test_documents_review_cv_e2e_emits_progress_structured_report_and_prompt_hash() -> None:
    kernel, workflow_store, outcome_store, event_writer = _build_kernel()

    started = await kernel.start_intent(
        intent_type="Documents.Review",
        inputs={
            "document_type": "cv",
            "review_goal": "quality",
            "custom_instructions": "Review my CV for graduate funding outreach.",
        },
        thread_id=77,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )

    run = await workflow_store.get_run(workflow_id=started.workflow_id)
    assert run is not None
    assert run.status == "completed"

    progress_events = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "progress"
    ]
    stages = [str(event.payload.get("stage")) for event in progress_events]
    assert "loading_context" in stages
    assert "running_operator:Platform.Attachments.List" in stages
    assert "running_operator:Documents.ImportFromPlatformAttachment" in stages
    assert "running_operator:Documents.Process" in stages
    assert "running_operator:Documents.Review" in stages
    assert "completed" in stages

    final_events = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "final_result"
    ]
    assert final_events

    outputs = final_events[-1].payload["outputs"]
    review_outcome = outputs["document_review"]
    review_payload = review_outcome["payload"]
    assert review_payload["document_type"] == "cv"
    assert review_payload["structured_report"]["type"] == "cv"
    assert set(review_payload["structured_report"]["section_scores"].keys()) == {
        "contact",
        "education",
        "experience",
        "skills",
    }

    review_records = [
        row
        for row in outcome_store.record_calls
        if row.get("operator_name") == "Documents.Review"
    ]
    assert review_records
    assert review_records[-1]["template_id"] == "Documents.Review/1.0.0/review_cv.j2"
    assert isinstance(review_records[-1]["template_hash"], str)
    assert len(review_records[-1]["template_hash"]) == 64
