from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any

import pytest

from intelligence_layer_kernel.operators.implementations.profile_memory_utils import (
    evaluate_requirements,
    group_memory_by_type,
    merge_profile_updates,
    normalize_memory_entries,
    normalize_profile,
    prefill_profile_from_platform,
    validate_profile,
)
from intelligence_layer_kernel.operators.types import (
    OperatorCall,
    OperatorError,
    OperatorMetrics,
    OperatorResult,
)
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


class _CollectOperatorExecutor:
    def __init__(
        self,
        *,
        workflow_store: InMemoryWorkflowStore,
        gate_store: InMemoryGateStore,
        student_id: int = 7,
        funding_request_id: int = 41,
    ) -> None:
        self._workflow_store = workflow_store
        self._gate_store = gate_store
        self._student_id = student_id
        self._funding_request_id = funding_request_id
        self.calls: list[dict[str, Any]] = []
        self.profile = normalize_profile({}, student_id=student_id)
        self.memory_entries: list[dict[str, Any]] = []
        self.profile_snapshots: list[dict[str, Any]] = []
        self._created = False

    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        self.calls.append(
            {
                "operator_name": operator_name,
                "operator_version": operator_version,
                "idempotency_key": call.idempotency_key,
                "payload": deepcopy(call.payload),
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

        if operator_name == "StudentProfile.LoadOrCreate":
            created = not self._created
            self._created = True
            self.profile_snapshots.append({"path": "load_or_create", "profile": deepcopy(self.profile)})
            return OperatorResult(
                status="succeeded",
                result={
                    "student_id": self._student_id,
                    "funding_request_id": self._funding_request_id,
                    "created": created,
                    "profile": deepcopy(self.profile),
                    "memory": {
                        "entries": deepcopy(self.memory_entries),
                        "by_type": group_memory_by_type(self.memory_entries),
                    },
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "StudentProfile.Requirements.Evaluate":
            required = call.payload.get("required_requirements")
            required_requirements = (
                [str(item) for item in required if str(item).strip()]
                if isinstance(required, list)
                else None
            )
            requirements = evaluate_requirements(
                self.profile,
                intent_type=str(call.payload.get("intent_type") or "Student.Profile.Collect"),
                required_requirements=required_requirements,
            )
            return OperatorResult(
                status="succeeded",
                result={"requirements": requirements},
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "StudentProfile.Update":
            updates = call.payload.get("profile_updates")
            updates_dict = updates if isinstance(updates, dict) else {}
            merged_profile, updated_fields = merge_profile_updates(self.profile, updates_dict)
            merged_profile = normalize_profile(merged_profile, student_id=self._student_id)
            errors = validate_profile(merged_profile)
            if errors:
                return OperatorResult(
                    status="failed",
                    result=None,
                    artifacts=[],
                    metrics=OperatorMetrics(latency_ms=1),
                    error=OperatorError(
                        code="invalid_profile_update",
                        message="profile update failed schema validation",
                        category="validation",
                        retryable=False,
                        details={"errors": errors},
                    ),
                )
            self.profile = merged_profile
            self.profile_snapshots.append({"path": "update", "profile": deepcopy(self.profile)})
            return OperatorResult(
                status="succeeded",
                result={
                    "student_id": self._student_id,
                    "profile": deepcopy(self.profile),
                    "updated_fields": updated_fields,
                    "validation": {"is_valid": True, "errors": []},
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=1),
                error=None,
            )

        if operator_name == "Memory.Upsert":
            entries = normalize_memory_entries(call.payload.get("entries"))
            upserted = 0
            for entry in entries:
                self.memory_entries = [
                    item for item in self.memory_entries if str(item.get("type")) != entry["type"]
                ]
                self.memory_entries.append(
                    {
                        "memory_id": str(uuid.uuid4()),
                        "type": entry["type"],
                        "content": entry["content"],
                        "source": entry["source"],
                        "updated_at": "2026-02-11T00:00:00+00:00",
                    }
                )
                upserted += 1
            return OperatorResult(
                status="succeeded",
                result={
                    "student_id": self._student_id,
                    "upserted": upserted,
                    "memory": {
                        "entries": deepcopy(self.memory_entries),
                        "by_type": group_memory_by_type(self.memory_entries),
                    },
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
    _CollectOperatorExecutor,
    FakeEventWriter,
]:
    templates = {
        "Student.Profile.Collect": {
            "intent_type": "Student.Profile.Collect",
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
                    "cache_policy": "ttl_only",
                    "idempotency_template": "platform_context:{tenant_id}:{thread_id}",
                    "payload": {"thread_id": {"from": "intent.thread_id"}},
                    "produces": ["context.platform", "context.intelligence"],
                },
                {
                    "step_id": "s2",
                    "kind": "operator",
                    "name": "StudentProfile.LoadOrCreate",
                    "operator_name": "StudentProfile.LoadOrCreate",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["onboarding", "profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "student_profile_load:{tenant_id}:{thread_id}",
                    "payload": {"thread_id": {"from": "intent.thread_id"}},
                    "produces": ["context.profile", "context.memory"],
                },
                {
                    "step_id": "s3",
                    "kind": "operator",
                    "name": "StudentProfile.Requirements.Evaluate",
                    "operator_name": "StudentProfile.Requirements.Evaluate",
                    "operator_version": "1.0.0",
                    "effects": ["read_only"],
                    "policy_tags": ["onboarding", "profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "student_profile_requirements_pre:{tenant_id}:{thread_id}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "intent_type": {"from": "intent.inputs.for_intent_type"},
                        "required_requirements": {"from": "intent.inputs.required_requirements"},
                        "strict": {"const": False},
                    },
                    "produces": ["context.profile.requirements"],
                },
                {
                    "step_id": "s4",
                    "kind": "policy_check",
                    "name": "Onboarding.EnsureProfile",
                    "effects": ["read_only"],
                    "policy_tags": ["onboarding_gate"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "check": {
                        "check_name": "Onboarding.Ensure",
                        "params": {
                            "required_gates": {"from": "intent.inputs.required_requirements"},
                            "on_missing_action_type": "collect_profile_fields",
                        },
                    },
                },
                {
                    "step_id": "s5",
                    "kind": "operator",
                    "name": "StudentProfile.Update",
                    "operator_name": "StudentProfile.Update",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["onboarding", "profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "depends_on": ["s4"],
                    "idempotency_template": "student_profile_update:{tenant_id}:{thread_id}:{computed.profile_updates_hash}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "profile_updates": {"from": "intent.inputs.profile_updates"},
                        "source": {"const": "user"},
                    },
                    "produces": ["context.profile", "context.profile.validation"],
                },
                {
                    "step_id": "s6",
                    "kind": "operator",
                    "name": "Memory.Upsert",
                    "operator_name": "Memory.Upsert",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["memory", "profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "depends_on": ["s4"],
                    "idempotency_template": "memory_upsert:{tenant_id}:{thread_id}:{computed.memory_updates_hash}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "entries": {"from": "intent.inputs.memory_updates"},
                    },
                    "produces": ["context.memory"],
                },
                {
                    "step_id": "s7",
                    "kind": "operator",
                    "name": "StudentProfile.Requirements.Evaluate",
                    "operator_name": "StudentProfile.Requirements.Evaluate",
                    "operator_version": "1.0.0",
                    "effects": ["read_only"],
                    "policy_tags": ["onboarding", "profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "depends_on": ["s5"],
                    "idempotency_template": "student_profile_requirements_post:{tenant_id}:{thread_id}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "intent_type": {"from": "intent.inputs.for_intent_type"},
                        "required_requirements": {"from": "intent.inputs.required_requirements"},
                        "strict": {"const": False},
                    },
                    "produces": ["context.profile.requirements"],
                },
                {
                    "step_id": "s8",
                    "kind": "policy_check",
                    "name": "Onboarding.RecheckProfile",
                    "effects": ["read_only"],
                    "policy_tags": ["onboarding_gate"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "check": {
                        "check_name": "Onboarding.Ensure",
                        "params": {
                            "required_gates": {
                                "from": "context.profile.requirements.required_requirements"
                            },
                            "on_missing_action_type": "collect_profile_fields",
                        },
                    },
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
    operator_executor = _CollectOperatorExecutor(
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
async def test_student_profile_collect_flow_missing_then_satisfied() -> None:
    kernel, workflow_store, gate_store, executor, event_writer = _build_kernel()
    required = ["base_profile_complete", "background_data_complete"]

    started = await kernel.start_intent(
        intent_type="Student.Profile.Collect",
        inputs={
            "for_intent_type": "Funding.Outreach.Email.Generate",
            "required_requirements": required,
            "profile_updates": {},
            "memory_updates": [],
        },
        thread_id=11,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )
    assert started.status == "waiting"
    assert started.gate_id is not None

    first_gate = await gate_store.get_gate(gate_id=started.gate_id)
    assert first_gate is not None
    assert first_gate["gate_type"] == "collect_profile_fields"
    first_preview_data = first_gate["preview"]["data"]
    assert "general.email" in first_preview_data["missing_fields"]
    assert len(first_preview_data["targeted_questions"]) <= 3

    await kernel.resolve_action(
        action_id=str(started.gate_id),
        status="accepted",
        payload={
            "profile_updates": {
                "email": "ada@example.com",
                "first_name": "Ada",
                "last_name": "Lovelace",
                "research_interest": "machine learning",
            }
        },
        actor=_actor(),
        source="test",
    )

    run = await workflow_store.get_run(workflow_id=started.workflow_id)
    assert run is not None
    assert run.status == "completed"

    requirements = evaluate_requirements(
        executor.profile,
        intent_type="Funding.Outreach.Email.Generate",
        required_requirements=required,
    )
    assert requirements["is_satisfied"] is True
    assert validate_profile(executor.profile) == []

    assert executor.profile_snapshots
    snapshot_paths = {str(item["path"]) for item in executor.profile_snapshots}
    assert "load_or_create" in snapshot_paths
    assert "update" in snapshot_paths
    for snapshot in executor.profile_snapshots:
        assert validate_profile(snapshot["profile"]) == []

    final_results = [
        event
        for event in event_writer.events
        if event.workflow_id == started.workflow_id and event.event_type == "final_result"
    ]
    assert final_results


def test_prefill_from_platform_fields_and_onboarding_json() -> None:
    row = {
        "user_first_name": "Ada",
        "user_last_name": "Lovelace",
        "user_email_address": "ada@example.com",
        "user_phone_number": "+44-0000",
        "user_date_of_birth": "2000-12-01",
        "user_gender": "female",
        "user_country_of_citizenship": "UK",
        "request_research_interest": "machine learning",
        "funding_template_initial_data": json.dumps(
            {
                "YourInterests": "robotics",
                "YourLastDegree": "MSc",
                "UniversityName": "Oxford",
                "YourGPA": 3.8,
                "PreferredEducationalLevel": "PhD",
                "YourName": "Ada",
            }
        ),
    }
    profile = prefill_profile_from_platform(row, student_id=7)
    assert profile["general"]["first_name"] == "Ada"
    assert profile["general"]["last_name"] == "Lovelace"
    assert profile["general"]["email"] == "ada@example.com"

    interests = [
        str(item.get("topic"))
        for item in profile["context"]["background"]["research_interests"]
        if isinstance(item, dict)
    ]
    assert "machine learning" in interests
    assert "robotics" in interests

    assert profile["context"]["targets"]["target_degree_levels"] == ["PhD"]
    assert profile["context"]["personalization"]["preferred_name"] == "Ada"
    assert validate_profile(profile) == []
