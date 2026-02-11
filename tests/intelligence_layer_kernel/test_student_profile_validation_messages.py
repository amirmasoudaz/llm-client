from __future__ import annotations

from typing import Any

import pytest

from intelligence_layer_kernel.operators.types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
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


class _FailingProfileUpdateExecutor:
    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        _ = operator_version
        _ = call
        if operator_name == "StudentProfile.Update":
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
                    details={
                        "errors": [
                            "general.email: 'not-an-email' is not a 'email'",
                            "general.first_name: '' should be non-empty",
                        ]
                    },
                ),
            )
        return OperatorResult(
            status="succeeded",
            result={"ok": True},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


def _build_kernel() -> tuple[WorkflowKernel, FakeEventWriter]:
    templates = {
        "Student.Profile.Collect": {
            "intent_type": "Student.Profile.Collect",
            "plan_version": "planner-1.0",
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "operator",
                    "name": "StudentProfile.Update",
                    "operator_name": "StudentProfile.Update",
                    "operator_version": "1.0.0",
                    "effects": ["db_write"],
                    "policy_tags": ["profile"],
                    "risk_level": "low",
                    "cache_policy": "never",
                    "idempotency_template": "student_update:{tenant_id}:{thread_id}",
                    "payload": {
                        "thread_id": {"from": "intent.thread_id"},
                        "profile_updates": {"from": "intent.inputs.profile_updates"},
                    },
                }
            ],
        }
    }

    event_writer = FakeEventWriter()
    kernel = WorkflowKernel(
        contracts=StaticContracts(templates),
        operator_executor=_FailingProfileUpdateExecutor(),
        policy_engine=PolicyEngine(),
        policy_store=FakePolicyStore(),
        event_writer=event_writer,
        pool=object(),
        tenant_id=1,
    )
    object.__setattr__(kernel, "_intent_store", InMemoryIntentStore())
    object.__setattr__(kernel, "_plan_store", InMemoryPlanStore())
    object.__setattr__(kernel, "_workflow_store", InMemoryWorkflowStore())
    object.__setattr__(kernel, "_outcome_store", InMemoryOutcomeStore())
    object.__setattr__(kernel, "_gate_store", InMemoryGateStore())
    return kernel, event_writer


@pytest.mark.asyncio
async def test_invalid_profile_update_error_is_actionable_in_workflow_response() -> None:
    kernel, event_writer = _build_kernel()

    result = await kernel.start_intent(
        intent_type="Student.Profile.Collect",
        inputs={"profile_updates": {"email": "not-an-email"}},
        thread_id=11,
        scope_type="funding_request",
        scope_id="41",
        actor=_actor(),
        source="test",
    )
    assert result.status == "failed"

    final_errors = [event for event in event_writer.events if event.event_type == "final_error"]
    assert final_errors
    message = str(final_errors[-1].payload.get("error"))
    assert "I couldn't save those profile updates yet." in message
    assert "general.email (use a valid email address like name@example.com)" in message
    assert "general.first_name (enter your first name)" in message
    assert "profile update failed schema validation" not in message

