from __future__ import annotations

import uuid
from typing import Any

import pytest

from intelligence_layer_kernel.events.types import LedgerEvent
from intelligence_layer_kernel.operators.executor import OperatorExecutor
from intelligence_layer_kernel.operators.store import JobClaim
from intelligence_layer_kernel.operators.types import (
    AuthContext,
    OperatorCall,
    OperatorMetrics,
    OperatorResult,
    TraceContext,
)
from intelligence_layer_kernel.policy import PolicyEngine


class _FakeContracts:
    def get_schema_by_ref(self, _ref: str) -> dict[str, Any]:
        return {}

    def resolver_for(self, _schema: dict[str, Any]):
        return None


class _FakeOperator:
    name = "Test.Operator"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls: list[OperatorCall] = []

    async def run(self, call: OperatorCall) -> OperatorResult:
        self.calls.append(call)
        return OperatorResult(
            status="succeeded",
            result={"ok": True},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


class _FakeRegistry:
    def __init__(self, operator: _FakeOperator, manifest: dict[str, Any]) -> None:
        self._operator = operator
        self._manifest = manifest

    def get_manifest(self, _name: str, _version: str) -> dict[str, Any]:
        return dict(self._manifest)

    def get(self, _name: str, _version: str) -> _FakeOperator:
        return self._operator

    def enforce_invocation_policy(self, **_kwargs) -> None:
        return None


class _FakeJobStore:
    def __init__(self, claim: JobClaim) -> None:
        self.claim = claim
        self.claim_calls: list[dict[str, Any]] = []
        self.complete_calls: list[dict[str, Any]] = []

    async def claim_job(self, **kwargs) -> JobClaim:
        self.claim_calls.append(kwargs)
        return self.claim

    async def complete_job(self, **kwargs) -> None:
        self.complete_calls.append(kwargs)


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


@pytest.mark.asyncio
async def test_executor_propagates_thread_intent_trace_into_claims_and_policy() -> None:
    workflow_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    intent_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    claim_job_id = uuid.uuid4()

    operator = _FakeOperator()
    manifest = {
        "name": "Test.Operator",
        "version": "1.0.0",
        "schemas": {"input": "schema.input", "output": "schema.output"},
        "effects": ["db_write"],
        "policy_tags": ["tag_a"],
    }
    job_store = _FakeJobStore(
        JobClaim(status="new", job_id=claim_job_id, attempt_no=1, result_payload=None)
    )
    policy_store = _FakePolicyStore()
    event_writer = _FakeEventWriter()
    executor = OperatorExecutor(
        contracts=_FakeContracts(),
        registry=_FakeRegistry(operator, manifest),
        job_store=job_store,
        policy_engine=PolicyEngine(),
        policy_store=policy_store,
        event_writer=event_writer,
        prompt_loader=None,
    )

    call = OperatorCall(
        payload={"x": 1},
        idempotency_key="k-1",
        auth_context=AuthContext(tenant_id=1, principal={"type": "student", "id": 99}, scopes=["scope:run"]),
        trace_context=TraceContext(
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            step_id="s1",
            thread_id=321,
            intent_id=intent_id,
            plan_id=plan_id,
        ),
    )

    result = await executor.execute(operator_name="Test.Operator", operator_version="1.0.0", call=call)

    assert result.status == "succeeded"
    assert len(job_store.claim_calls) == 1
    claim_call = job_store.claim_calls[0]
    assert claim_call["thread_id"] == 321
    assert claim_call["intent_id"] == intent_id
    assert claim_call["plan_id"] == plan_id

    assert len(policy_store.decisions) == 2
    for decision in policy_store.decisions:
        assert decision.workflow_id == workflow_id
        assert decision.intent_id == intent_id
        assert decision.plan_id == plan_id
        assert decision.step_id == "s1"
        assert decision.job_id == str(claim_job_id)
        assert decision.correlation_id == correlation_id

    assert len(event_writer.events) >= 2
    started = event_writer.events[0]
    assert started.thread_id == 321
    assert str(started.intent_id) == intent_id
    assert str(started.plan_id) == plan_id


@pytest.mark.asyncio
async def test_executor_idempotent_replay_returns_prior_result_without_running_operator() -> None:
    operator = _FakeOperator()
    manifest = {
        "name": "Test.Operator",
        "version": "1.0.0",
        "schemas": {"input": "schema.input", "output": "schema.output"},
        "effects": [],
        "policy_tags": [],
    }
    prior_result_payload = {
        "result": {"from_cache": True},
        "artifacts": [{"id": "a1"}],
    }
    job_store = _FakeJobStore(
        JobClaim(status="existing_success", job_id=uuid.uuid4(), attempt_no=1, result_payload=prior_result_payload)
    )
    executor = OperatorExecutor(
        contracts=_FakeContracts(),
        registry=_FakeRegistry(operator, manifest),
        job_store=job_store,
        policy_engine=PolicyEngine(),
        policy_store=_FakePolicyStore(),
        event_writer=_FakeEventWriter(),
        prompt_loader=None,
    )

    call = OperatorCall(
        payload={"x": 1},
        idempotency_key="k-2",
        auth_context=AuthContext(tenant_id=1, principal={"type": "student", "id": 99}, scopes=[]),
        trace_context=TraceContext(
            correlation_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            step_id="s1",
        ),
    )

    result = await executor.execute(operator_name="Test.Operator", operator_version="1.0.0", call=call)

    assert result.status == "succeeded"
    assert result.result == {"from_cache": True}
    assert result.artifacts == [{"id": "a1"}]
    assert operator.calls == []
