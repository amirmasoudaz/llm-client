from __future__ import annotations

import uuid
from typing import Any

import pytest

from intelligence_layer_kernel.operators import OperatorRegistry
from intelligence_layer_kernel.operators.base import Operator
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
    def __init__(self, manifests: dict[tuple[str, str], dict[str, Any]] | None = None) -> None:
        self._manifests = manifests or {}

    def get_operator_manifest(self, name: str, version: str) -> dict[str, Any]:
        key = (name, version)
        if key not in self._manifests:
            raise KeyError(f"unknown manifest: {name}@{version}")
        return dict(self._manifests[key])

    def get_schema_by_ref(self, _ref: str) -> dict[str, Any]:
        return {}

    def resolver_for(self, _schema: dict[str, Any]):
        return None


class _NoopOperator(Operator):
    name = "Test.Operator"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, call: OperatorCall) -> OperatorResult:
        self.calls += 1
        return OperatorResult(
            status="succeeded",
            result={"ok": True},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=1),
            error=None,
        )


class _FakeJobStore:
    def __init__(self, claim: JobClaim) -> None:
        self._claim = claim
        self.complete_calls: list[dict[str, Any]] = []

    async def claim_job(self, **_kwargs) -> JobClaim:
        return self._claim

    async def complete_job(self, **kwargs) -> None:
        self.complete_calls.append(kwargs)


class _FakePolicyStore:
    def __init__(self) -> None:
        self.decisions = []

    async def record(self, decision) -> uuid.UUID:
        self.decisions.append(decision)
        return uuid.uuid4()


class _FakeEventWriter:
    async def append(self, _event) -> None:
        return None


def _build_call(scopes: list[str] | None = None) -> OperatorCall:
    return OperatorCall(
        payload={"x": 1},
        idempotency_key="id-1",
        auth_context=AuthContext(
            tenant_id=1,
            principal={"type": "student", "id": 11},
            scopes=scopes or [],
        ),
        trace_context=TraceContext(
            correlation_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            step_id="s1",
        ),
    )


def test_registry_denies_operator_by_policy_tag() -> None:
    manifest = {
        "name": "Test.Operator",
        "version": "1.0.0",
        "type": "operator",
        "schemas": {"input": "schema.input", "output": "schema.output"},
        "policy_tags": ["platform_write"],
        "effects": ["db_write_platform"],
    }
    contracts = _FakeContracts({("Test.Operator", "1.0.0"): manifest})
    registry = OperatorRegistry(contracts, policy_tag_denylist={"platform_write"})

    with pytest.raises(KeyError):
        registry.register(_NoopOperator())


@pytest.mark.asyncio
async def test_executor_fails_when_manifest_prompt_binding_exists_but_loader_missing() -> None:
    manifest = {
        "name": "Test.Operator",
        "version": "1.0.0",
        "type": "operator",
        "schemas": {"input": "schema.input", "output": "schema.output"},
        "policy_tags": [],
        "effects": [],
        "prompt_templates": {"primary": "Test.Operator/1.0.0/main_prompt"},
    }
    operator = _NoopOperator()
    contracts = _FakeContracts({("Test.Operator", "1.0.0"): manifest})
    registry = OperatorRegistry(contracts)
    registry.register(operator)

    executor = OperatorExecutor(
        contracts=contracts,
        registry=registry,
        job_store=_FakeJobStore(JobClaim(status="new", job_id=uuid.uuid4(), attempt_no=1, result_payload=None)),
        policy_engine=PolicyEngine(),
        policy_store=_FakePolicyStore(),
        event_writer=_FakeEventWriter(),
        prompt_loader=None,
    )

    with pytest.raises(ValueError, match="prompt loader is required"):
        await executor.execute(operator_name="Test.Operator", operator_version="1.0.0", call=_build_call())


@pytest.mark.asyncio
async def test_executor_preinvoke_scope_denial_returns_failed_without_running_operator() -> None:
    manifest = {
        "name": "Test.Operator",
        "version": "1.0.0",
        "type": "operator",
        "schemas": {"input": "schema.input", "output": "schema.output"},
        "policy_tags": ["internal"],
        "effects": ["db_write"],
        "required_scopes": ["scope:required"],
    }
    operator = _NoopOperator()
    contracts = _FakeContracts({("Test.Operator", "1.0.0"): manifest})
    registry = OperatorRegistry(contracts)
    registry.register(operator)

    policy_store = _FakePolicyStore()
    job_store = _FakeJobStore(JobClaim(status="new", job_id=uuid.uuid4(), attempt_no=1, result_payload=None))
    executor = OperatorExecutor(
        contracts=contracts,
        registry=registry,
        job_store=job_store,
        policy_engine=PolicyEngine(),
        policy_store=policy_store,
        event_writer=_FakeEventWriter(),
        prompt_loader=None,
    )

    result = await executor.execute(
        operator_name="Test.Operator",
        operator_version="1.0.0",
        call=_build_call(scopes=[]),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "policy_denied"
    assert result.error.details == {"reason_code": "missing_required_scope", "requirements": {"required_scopes": ["scope:required"]}}
    assert operator.calls == 0
    assert len(policy_store.decisions) == 1
    assert policy_store.decisions[0].reason_code == "missing_required_scope"
    assert len(job_store.complete_calls) == 1
