from __future__ import annotations

import uuid
from typing import Any
from datetime import datetime, timezone

import pytest

from intelligence_layer_kernel.runtime.store import OutcomeStore


class _Acquire:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_FakeConn":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConn:
    def __init__(self, *, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append((query, args))

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        return self.rows


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


@pytest.mark.asyncio
async def test_outcome_store_persists_template_metadata() -> None:
    conn = _FakeConn()
    store = OutcomeStore(pool=_FakePool(conn), tenant_id=1)
    workflow_id = uuid.uuid4()
    intent_id = uuid.uuid4()
    plan_id = uuid.uuid4()

    template_id = "Email.ReviewDraft/1.0.0/review_draft.j2"
    template_hash = "abc123"

    outcome_id = await store.record(
        workflow_id=workflow_id,
        thread_id=11,
        intent_id=intent_id,
        plan_id=plan_id,
        step_id="s1",
        job_id=None,
        operator_name="Email.ReviewDraft",
        operator_version="1.0.0",
        status="succeeded",
        content={"outcome": {"outcome_type": "Email.Review"}},
        template_id=template_id,
        template_hash=template_hash,
    )

    assert outcome_id is not None
    assert len(conn.execute_calls) == 1
    query, args = conn.execute_calls[0]
    assert "template_id" in query
    assert "template_hash" in query
    assert args[15] == template_id
    assert args[16] == template_hash


@pytest.mark.asyncio
async def test_outcome_store_lists_template_metadata() -> None:
    workflow_id = uuid.uuid4()
    conn = _FakeConn(
        rows=[
            {
                "outcome_id": uuid.uuid4(),
                "lineage_id": uuid.uuid4(),
                "version": 1,
                "parent_outcome_id": None,
                "outcome_type": "Email.ReviewDraft",
                "status": "succeeded",
                "workflow_id": workflow_id,
                "step_id": "s1",
                "content": {"ok": True},
                "template_id": "Email.ReviewDraft/1.0.0/review_draft.j2",
                "template_hash": "hash-value",
                "created_at": datetime.now(timezone.utc),
            }
        ]
    )
    store = OutcomeStore(pool=_FakePool(conn), tenant_id=1)

    outcomes = await store.list_by_workflow(workflow_id=workflow_id)

    assert len(outcomes) == 1
    assert outcomes[0]["template_id"] == "Email.ReviewDraft/1.0.0/review_draft.j2"
    assert outcomes[0]["template_hash"] == "hash-value"
