from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
HTTPException = fastapi.HTTPException
app_module = pytest.importorskip("intelligence_layer_api.app")

from intelligence_layer_kernel.runtime.types import WorkflowResult


class _StubILDB:
    def __init__(self, runs: dict[str, dict[str, Any]]) -> None:
        self.runs = runs
        self.calls: list[str] = []
        self.tenant_id = 1

    async def get_workflow_run(self, *, workflow_id: str) -> dict[str, Any] | None:
        self.calls.append(workflow_id)
        return self.runs.get(workflow_id)


class _StubOutcomeStore:
    def __init__(self, rows: dict[uuid.UUID, list[dict[str, Any]]]) -> None:
        self.rows = rows
        self.calls: list[uuid.UUID] = []

    async def list_by_workflow(self, *, workflow_id: uuid.UUID) -> list[dict[str, Any]]:
        self.calls.append(workflow_id)
        return list(self.rows.get(workflow_id, []))


class _StubKernel:
    def __init__(self, *, replay_workflow_id: uuid.UUID) -> None:
        self.replay_workflow_id = replay_workflow_id
        self.calls: list[dict[str, Any]] = []

    async def rerun_workflow(
        self,
        *,
        workflow_id: uuid.UUID,
        mode: str,
        actor: dict[str, Any],
        source: str = "api",
    ) -> WorkflowResult:
        self.calls.append(
            {
                "workflow_id": workflow_id,
                "mode": mode,
                "actor": dict(actor),
                "source": source,
            }
        )
        return WorkflowResult(
            workflow_id=self.replay_workflow_id,
            intent_id=uuid.uuid4(),
            plan_id=uuid.uuid4(),
            status="completed",
        )


def _row(outcome_type: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "outcome_id": uuid.uuid4(),
        "lineage_id": uuid.uuid4(),
        "version": 1,
        "parent_outcome_id": None,
        "outcome_type": outcome_type,
        "status": "succeeded",
        "workflow_id": None,
        "step_id": "s1",
        "content": {"ok": True},
        "template_id": None,
        "template_hash": None,
        "created_at": now,
    }


@pytest.mark.asyncio
async def test_workflow_outcomes_replay_mode_triggers_rerun_and_returns_new_workflow(monkeypatch) -> None:
    source_workflow_id = uuid.uuid4()
    replay_workflow_id = uuid.uuid4()
    ildb = _StubILDB(
        {
            str(source_workflow_id): {"status": "completed", "parent_workflow_id": None},
            str(replay_workflow_id): {"status": "completed", "parent_workflow_id": str(source_workflow_id)},
        }
    )
    rows = {replay_workflow_id: [_row("Email.Review")]}
    rows[replay_workflow_id][0]["workflow_id"] = replay_workflow_id
    outcome_store = _StubOutcomeStore(rows)
    kernel = _StubKernel(replay_workflow_id=replay_workflow_id)

    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "outcome_store", outcome_store, raising=False)
    monkeypatch.setattr(app_module.app.state, "workflow_kernel", kernel, raising=False)

    response = await app_module.workflow_outcomes(workflow_id=str(source_workflow_id), mode="replay")

    assert response.mode == "replay"
    assert response.recomputed is True
    assert response.workflow_id == str(replay_workflow_id)
    assert response.source_workflow_id == str(source_workflow_id)
    assert response.parent_workflow_id == str(source_workflow_id)
    assert response.run_status == "completed"
    assert len(response.outcomes) == 1
    assert response.outcomes[0]["workflow_id"] == str(replay_workflow_id)
    assert len(kernel.calls) == 1
    assert kernel.calls[0]["mode"] == "replay"
    assert kernel.calls[0]["workflow_id"] == source_workflow_id


@pytest.mark.asyncio
async def test_workflow_outcomes_regenerate_mode_triggers_rerun(monkeypatch) -> None:
    source_workflow_id = uuid.uuid4()
    regenerate_workflow_id = uuid.uuid4()
    ildb = _StubILDB(
        {
            str(source_workflow_id): {"status": "completed", "parent_workflow_id": None},
            str(regenerate_workflow_id): {"status": "completed", "parent_workflow_id": str(source_workflow_id)},
        }
    )
    rows = {regenerate_workflow_id: [_row("Email.Draft")]}
    rows[regenerate_workflow_id][0]["workflow_id"] = regenerate_workflow_id
    outcome_store = _StubOutcomeStore(rows)
    kernel = _StubKernel(replay_workflow_id=regenerate_workflow_id)

    monkeypatch.setattr(app_module.app.state, "ildb", ildb, raising=False)
    monkeypatch.setattr(app_module.app.state, "outcome_store", outcome_store, raising=False)
    monkeypatch.setattr(app_module.app.state, "workflow_kernel", kernel, raising=False)

    response = await app_module.workflow_outcomes(workflow_id=str(source_workflow_id), mode="regenerate")

    assert response.mode == "regenerate"
    assert response.recomputed is True
    assert response.workflow_id == str(regenerate_workflow_id)
    assert response.source_workflow_id == str(source_workflow_id)
    assert response.parent_workflow_id == str(source_workflow_id)
    assert len(kernel.calls) == 1
    assert kernel.calls[0]["mode"] == "regenerate"


@pytest.mark.asyncio
async def test_workflow_outcomes_replay_requires_kernel() -> None:
    source_workflow_id = uuid.uuid4()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            app_module.app.state,
            "ildb",
            _StubILDB({str(source_workflow_id): {"status": "completed", "parent_workflow_id": None}}),
            raising=False,
        )
        monkeypatch.setattr(app_module.app.state, "outcome_store", _StubOutcomeStore({}), raising=False)
        monkeypatch.setattr(app_module.app.state, "workflow_kernel", None, raising=False)

        with pytest.raises(HTTPException) as exc_info:
            await app_module.workflow_outcomes(workflow_id=str(source_workflow_id), mode="replay")
        assert exc_info.value.status_code == 400
    finally:
        monkeypatch.undo()
