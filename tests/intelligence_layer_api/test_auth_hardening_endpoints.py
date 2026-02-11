from __future__ import annotations

import uuid
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
HTTPException = fastapi.HTTPException
Request = pytest.importorskip("starlette.requests").Request
StreamingResponse = pytest.importorskip("starlette.responses").StreamingResponse
app_module = pytest.importorskip("intelligence_layer_api.app")

from intelligence_layer_api.auth import AuthResult


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": []})


class _ILDB:
    tenant_id = 1

    def __init__(self) -> None:
        self.query = {"query_id": str(uuid.uuid4()), "thread_id": 101, "job_id": "job-101"}
        self.thread = {"student_id": 77, "funding_request_id": 88, "status": "active"}
        self.run = {
            "workflow_id": str(uuid.uuid4()),
            "thread_id": 101,
            "status": "completed",
            "parent_workflow_id": None,
        }
        self.gate = {"gate_id": str(uuid.uuid4()), "workflow_id": self.run["workflow_id"], "status": "waiting"}

    async def get_query(self, *, query_id: str) -> dict[str, Any] | None:
        _ = query_id
        return dict(self.query)

    async def get_thread(self, *, thread_id: int) -> dict[str, Any] | None:
        _ = thread_id
        return dict(self.thread)

    async def get_workflow_run(self, *, workflow_id: str) -> dict[str, Any] | None:
        if str(workflow_id) == str(self.run["workflow_id"]):
            return dict(self.run)
        return None

    async def get_gate(self, *, gate_id: str) -> dict[str, Any] | None:
        if str(gate_id) == str(self.gate["gate_id"]):
            return dict(self.gate)
        return None

    async def list_runtime_events(self, *, job_id: str, after_ts: float, limit: int = 200) -> list[dict[str, Any]]:
        _ = (job_id, after_ts, limit)
        return []


class _AuthDenied:
    async def authenticate(self, **kwargs):
        _ = kwargs
        return AuthResult(ok=False, reason="unauthorized", status_code=401)

    async def funding_request_owner_id(self, *, funding_request_id: int):
        _ = funding_request_id
        return None


class _AuthAllowed:
    async def authenticate(self, **kwargs):
        _ = kwargs
        return AuthResult(ok=True, principal_id=77, scopes=["chat"], trust_level=2, bypass=False)

    async def funding_request_owner_id(self, *, funding_request_id: int):
        _ = funding_request_id
        return 77


class _Projector:
    async def stream(self, *, workflow_id: str):
        _ = workflow_id
        if False:
            yield ""


class _OutcomeStore:
    async def list_by_workflow(self, *, workflow_id):
        _ = workflow_id
        return []


class _Kernel:
    def __init__(self) -> None:
        self.cancel_calls: list[dict[str, Any]] = []
        self.resolve_calls: list[dict[str, Any]] = []

    async def cancel(self, job_id: str, reason: str | None = None) -> None:
        self.cancel_calls.append({"job_id": job_id, "reason": reason})

    async def resolve_action(self, **kwargs) -> None:
        self.resolve_calls.append(dict(kwargs))


class _KernelContainer:
    def __init__(self, kernel: _Kernel) -> None:
        self.kernel = kernel


@pytest.mark.asyncio
async def test_query_events_rejects_when_auth_fails(monkeypatch) -> None:
    monkeypatch.setattr(app_module.app.state, "ildb", _ILDB(), raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthDenied(), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.query_events(query_id=str(uuid.uuid4()), request=_request())
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_workflow_events_rejects_when_auth_fails(monkeypatch) -> None:
    ildb = _ILDB()
    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthDenied(), raising=False)
    monkeypatch.setattr(app_module.app.state, "sse_projector", _Projector(), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.workflow_events(workflow_id=str(ildb.run["workflow_id"]), request=_request())
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_workflow_outcomes_rejects_when_auth_fails(monkeypatch) -> None:
    ildb = _ILDB()
    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthDenied(), raising=False)
    monkeypatch.setattr(app_module.app.state, "outcome_store", _OutcomeStore(), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.workflow_outcomes(
            workflow_id=str(ildb.run["workflow_id"]),
            request=_request(),
            mode="reproduce",
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_resolve_action_rejects_when_auth_fails(monkeypatch) -> None:
    ildb = _ILDB()
    kernel = _Kernel()
    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthDenied(), raising=False)
    monkeypatch.setattr(app_module.app.state, "workflow_kernel", None, raising=False)
    monkeypatch.setattr(app_module.app.state, "kernel_container", _KernelContainer(kernel), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.resolve_action(
            action_id=str(ildb.gate["gate_id"]),
            req=app_module.ResolveActionRequest(status="accepted", payload={}),
            request=_request(),
        )
    assert exc_info.value.status_code == 401
    assert kernel.resolve_calls == []


@pytest.mark.asyncio
async def test_cancel_query_rejects_when_auth_fails(monkeypatch) -> None:
    kernel = _Kernel()
    monkeypatch.setattr(app_module.app.state, "ildb", _ILDB(), raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthDenied(), raising=False)
    monkeypatch.setattr(app_module.app.state, "kernel_container", _KernelContainer(kernel), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.cancel_query(query_id=str(uuid.uuid4()), req=None, request=_request())
    assert exc_info.value.status_code == 401
    assert kernel.cancel_calls == []


@pytest.mark.asyncio
async def test_cancel_query_allows_owner_and_cancels_job(monkeypatch) -> None:
    kernel = _Kernel()
    monkeypatch.setattr(app_module.app.state, "ildb", _ILDB(), raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthAllowed(), raising=False)
    monkeypatch.setattr(app_module.app.state, "kernel_container", _KernelContainer(kernel), raising=False)

    response = await app_module.cancel_query(
        query_id=str(uuid.uuid4()),
        req=app_module.CancelQueryRequest(reason="user_cancelled"),
        request=_request(),
    )

    assert response["ok"] is True
    assert len(kernel.cancel_calls) == 1
    assert kernel.cancel_calls[0]["job_id"] == "job-101"
    assert kernel.cancel_calls[0]["reason"] == "user_cancelled"


@pytest.mark.asyncio
async def test_query_events_allows_owner_and_returns_stream(monkeypatch) -> None:
    monkeypatch.setattr(app_module.app.state, "ildb", _ILDB(), raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthAllowed(), raising=False)

    response = await app_module.query_events(query_id=str(uuid.uuid4()), request=_request())

    assert isinstance(response, StreamingResponse)
