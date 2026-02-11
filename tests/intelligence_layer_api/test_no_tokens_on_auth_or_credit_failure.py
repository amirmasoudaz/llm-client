from __future__ import annotations

from types import SimpleNamespace
import uuid

import pytest

fastapi = pytest.importorskip("fastapi")
Request = pytest.importorskip("starlette.requests").Request
app_module = pytest.importorskip("intelligence_layer_api.app")

from intelligence_layer_api.auth import AuthResult
from intelligence_layer_api.billing import CreditReservation


class _EventWriter:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)


class _ILDB:
    tenant_id = 1

    async def get_thread(self, *, thread_id: int):
        _ = thread_id
        return {"student_id": 77, "funding_request_id": 88, "status": "active"}

    async def get_workflow_run(self, *, workflow_id: str):
        _ = workflow_id
        return None

    async def get_latest_waiting_profile_gate(self, *, thread_id: int):
        _ = thread_id
        return None


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
        return AuthResult(ok=True, principal_id=77, scopes=["chat"], trust_level=0, bypass=False)

    async def funding_request_owner_id(self, *, funding_request_id: int):
        _ = funding_request_id
        return 77


class _KernelNever:
    def __init__(self) -> None:
        self.called = False

    async def handle_message(self, **kwargs):
        _ = kwargs
        self.called = True
        raise AssertionError("kernel should not run")

    async def resolve_action(self, **kwargs):
        _ = kwargs
        self.called = True
        raise AssertionError("kernel should not run")


class _CreditDeny:
    async def estimate_reserve_credits(self, *args, **kwargs):
        _ = (args, kwargs)
        return 5

    async def remaining_credits(self, *, principal_id: int):
        _ = principal_id
        return 0

    async def reserve(self, **kwargs):
        _ = kwargs
        return CreditReservation(
            ok=False,
            reservation_id=None,
            reserved_credits=0,
            request_key=b"",
            expires_at=None,
            reason="insufficient_credits",
        )


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


@pytest.mark.asyncio
async def test_auth_denied_query_path_emits_no_model_token(monkeypatch) -> None:
    ildb = _ILDB()
    event_writer = _EventWriter()
    kernel = _KernelNever()
    monkeypatch.setattr(app_module, "get_settings", lambda: SimpleNamespace(use_workflow_kernel=True, debug=False))
    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "event_writer", event_writer, raising=False)
    monkeypatch.setattr(app_module.app.state, "workflow_kernel", kernel, raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthDenied(), raising=False)

    response = await app_module.submit_query(
        thread_id="101",
        req=app_module.SubmitQueryRequest(message="hello"),
        request=_request(),
    )

    assert response.query_id
    assert kernel.called is False
    assert all(event.event_type != "model_token" for event in event_writer.events)


@pytest.mark.asyncio
async def test_insufficient_credits_query_path_emits_no_model_token(monkeypatch) -> None:
    ildb = _ILDB()
    event_writer = _EventWriter()
    kernel = _KernelNever()
    monkeypatch.setattr(app_module, "get_settings", lambda: SimpleNamespace(use_workflow_kernel=True, debug=False))
    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "event_writer", event_writer, raising=False)
    monkeypatch.setattr(app_module.app.state, "workflow_kernel", kernel, raising=False)
    monkeypatch.setattr(app_module.app.state, "auth_adapter", _AuthAllowed(), raising=False)
    monkeypatch.setattr(app_module.app.state, "credit_manager", _CreditDeny(), raising=False)

    response = await app_module.submit_query(
        thread_id="101",
        req=app_module.SubmitQueryRequest(message="hello", query_id=str(uuid.uuid4())),
        request=_request(),
    )

    assert response.query_id
    assert kernel.called is False
    assert all(event.event_type != "model_token" for event in event_writer.events)
