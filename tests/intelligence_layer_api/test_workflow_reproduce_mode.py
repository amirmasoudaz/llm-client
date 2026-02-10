from __future__ import annotations

import uuid
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
HTTPException = fastapi.HTTPException
app_module = pytest.importorskip("intelligence_layer_api.app")


class _StubILDB:
    def __init__(self, run: dict[str, Any] | None) -> None:
        self._run = run
        self.calls: list[str] = []

    async def get_workflow_run(self, *, workflow_id: str) -> dict[str, Any] | None:
        self.calls.append(workflow_id)
        return self._run


class _StubOutcomeStore:
    def __init__(self, outcomes: list[dict[str, Any]]) -> None:
        self._outcomes = outcomes
        self.calls: list[uuid.UUID] = []

    async def list_by_workflow(self, *, workflow_id: uuid.UUID) -> list[dict[str, Any]]:
        self.calls.append(workflow_id)
        return list(self._outcomes)


class _NoRecomputeKernel:
    def __init__(self) -> None:
        self.called = False

    async def handle_message(self, *args, **kwargs):
        self.called = True
        raise AssertionError("workflow_outcomes should not execute workflow code")

    async def start_intent(self, *args, **kwargs):
        self.called = True
        raise AssertionError("workflow_outcomes should not execute workflow code")


@pytest.mark.asyncio
async def test_workflow_outcomes_reproduce_reads_stored_rows_without_recompute(monkeypatch) -> None:
    workflow_id = str(uuid.uuid4())
    outcome_id = uuid.uuid4()
    ildb = _StubILDB({"status": "completed"})
    outcome_store = _StubOutcomeStore(
        [
            {
                "outcome_id": outcome_id,
                "outcome_type": "Email.Review",
                "status": "succeeded",
                "workflow_id": uuid.UUID(workflow_id),
                "step_id": "s1",
                "content": {"outcome": {"ok": True}},
                "template_id": "Email.ReviewDraft/1.0.0/review_draft.j2",
                "template_hash": "hash-1",
            }
        ]
    )
    kernel = _NoRecomputeKernel()

    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "outcome_store", outcome_store, raising=False)
    monkeypatch.setattr(app_module.app.state, "workflow_kernel", kernel, raising=False)

    response = await app_module.workflow_outcomes(workflow_id=workflow_id, mode="reproduce")

    assert response.workflow_id == workflow_id
    assert response.mode == "reproduce"
    assert response.recomputed is False
    assert response.run_status == "completed"
    assert len(response.outcomes) == 1
    assert response.outcomes[0]["outcome_id"] == str(outcome_id)
    assert response.outcomes[0]["template_id"] == "Email.ReviewDraft/1.0.0/review_draft.j2"
    assert response.outcomes[0]["template_hash"] == "hash-1"
    assert ildb.calls == [workflow_id]
    assert outcome_store.calls == [uuid.UUID(workflow_id)]
    assert kernel.called is False


@pytest.mark.asyncio
async def test_workflow_outcomes_reproduce_rejects_unsupported_mode() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await app_module.workflow_outcomes(workflow_id=str(uuid.uuid4()), mode="replay")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_workflow_outcomes_reproduce_404_when_workflow_missing(monkeypatch) -> None:
    monkeypatch.setattr(app_module.app.state, "ildb", _StubILDB(None), raising=False)
    monkeypatch.setattr(app_module.app.state, "outcome_store", _StubOutcomeStore([]), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.workflow_outcomes(workflow_id=str(uuid.uuid4()), mode="reproduce")

    assert exc_info.value.status_code == 404
