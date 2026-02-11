from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from intelligence_layer_kernel.policy import PolicyEngine
from intelligence_layer_kernel.runtime.kernel import WorkflowKernel
from intelligence_layer_kernel.runtime.store import OutcomeStore
from tests.intelligence_layer_kernel._phase_e_testkit import FakeEventWriter, FakePolicyStore, StaticContracts


def _email_optimize_content(
    *,
    subject: str,
    body: str,
    source_version_id: str | None = None,
    payload_version_number: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"subject": subject, "body": body}
    if source_version_id is not None:
        payload["source_version_id"] = source_version_id
    if payload_version_number is not None:
        payload["version_number"] = payload_version_number
    return {
        "outcome": {
            "outcome_id": str(uuid.uuid4()),
            "outcome_type": "Email.Draft",
            "payload": payload,
        }
    }


class _StoreAcquire:
    def __init__(self, conn: _StoreConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _StoreConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _StoreConn:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._counter = 0

    async def execute(self, _query: str, *args: Any) -> None:
        self._counter += 1
        row = {
            "tenant_id": int(args[0]),
            "outcome_id": args[1],
            "lineage_id": args[2],
            "version": int(args[3]),
            "parent_outcome_id": args[4],
            "status": str(args[6]),
            "thread_id": args[8],
            "content": json.loads(args[13]),
            "created_index": self._counter,
        }
        self.rows.append(row)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        normalized = " ".join(query.split()).lower()
        if "content->'outcome'->>'outcome_type'='email.draft'" not in normalized:
            return None

        tenant_id = int(args[0])
        thread_id = int(args[1])
        eligible = [
            row
            for row in self.rows
            if row["tenant_id"] == tenant_id
            and int(row["thread_id"] or 0) == thread_id
            and row["status"] == "succeeded"
            and _is_email_draft(row["content"])
        ]
        if "and outcome_id=$3" in normalized:
            outcome_id = args[2]
            for row in eligible:
                if row["outcome_id"] == outcome_id:
                    return {
                        "outcome_id": row["outcome_id"],
                        "lineage_id": row["lineage_id"],
                        "version": row["version"],
                    }
            return None

        if not eligible:
            return None
        latest = max(eligible, key=lambda item: item["created_index"])
        return {
            "outcome_id": latest["outcome_id"],
            "lineage_id": latest["lineage_id"],
            "version": latest["version"],
        }

    async def fetch(self, _query: str, *_args: Any) -> list[dict[str, Any]]:
        return []


class _StorePool:
    def __init__(self, conn: _StoreConn) -> None:
        self._conn = conn

    def acquire(self) -> _StoreAcquire:
        return _StoreAcquire(self._conn)


def _is_email_draft(content: dict[str, Any]) -> bool:
    outcome = content.get("outcome")
    if not isinstance(outcome, dict):
        return False
    return str(outcome.get("outcome_type") or "") == "Email.Draft"


@pytest.mark.asyncio
async def test_email_optimize_outcomes_create_lineage_chain_with_incrementing_versions() -> None:
    conn = _StoreConn()
    store = OutcomeStore(pool=_StorePool(conn), tenant_id=1)
    workflow_id = uuid.uuid4()

    first_outcome_id = await store.record(
        workflow_id=workflow_id,
        thread_id=33,
        intent_id=uuid.uuid4(),
        plan_id=uuid.uuid4(),
        step_id="s4",
        job_id=None,
        operator_name="Email.OptimizeDraft",
        operator_version="1.0.0",
        status="succeeded",
        content=_email_optimize_content(subject="v1", body="draft one"),
    )
    second_outcome_id = await store.record(
        workflow_id=workflow_id,
        thread_id=33,
        intent_id=uuid.uuid4(),
        plan_id=uuid.uuid4(),
        step_id="s4",
        job_id=None,
        operator_name="Email.OptimizeDraft",
        operator_version="1.0.0",
        status="succeeded",
        content=_email_optimize_content(subject="v2", body="draft two"),
    )
    third_outcome_id = await store.record(
        workflow_id=workflow_id,
        thread_id=33,
        intent_id=uuid.uuid4(),
        plan_id=uuid.uuid4(),
        step_id="s4",
        job_id=None,
        operator_name="Email.OptimizeDraft",
        operator_version="1.0.0",
        status="succeeded",
        content=_email_optimize_content(subject="v3", body="draft three"),
    )

    assert first_outcome_id is not None
    assert second_outcome_id is not None
    assert third_outcome_id is not None
    assert len(conn.rows) == 3

    first, second, third = conn.rows
    assert first["lineage_id"] == first["outcome_id"]
    assert first["version"] == 1
    assert first["parent_outcome_id"] is None

    assert second["lineage_id"] == first["lineage_id"]
    assert second["version"] == 2
    assert second["parent_outcome_id"] == first["outcome_id"]

    assert third["lineage_id"] == first["lineage_id"]
    assert third["version"] == 3
    assert third["parent_outcome_id"] == second["outcome_id"]

    first_payload = first["content"]["outcome"]["payload"]
    second_payload = second["content"]["outcome"]["payload"]
    third_payload = third["content"]["outcome"]["payload"]
    assert first_payload["version_number"] == 1
    assert second_payload["version_number"] == 2
    assert third_payload["version_number"] == 3
    assert second_payload["source_version_id"] == str(first["outcome_id"])
    assert third_payload["source_version_id"] == str(second["outcome_id"])


class _KernelAcquire:
    def __init__(self, conn: _KernelConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _KernelConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _KernelConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def fetch(self, _query: str, *_args: Any) -> list[dict[str, Any]]:
        return list(self.rows)


class _KernelPool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def acquire(self) -> _KernelAcquire:
        return _KernelAcquire(_KernelConn(self._rows))


class _NoopOperatorExecutor:
    async def execute(self, **_kwargs):
        raise AssertionError("not used in this test")


@pytest.mark.asyncio
async def test_source_draft_resolution_prefers_ledger_version_from_latest_lineage() -> None:
    lineage_old = uuid.uuid4()
    lineage_latest = uuid.uuid4()
    outcome_latest_v3 = uuid.uuid4()
    outcome_latest_v2 = uuid.uuid4()
    outcome_old_v2 = uuid.uuid4()

    rows = [
        {
            "outcome_id": outcome_latest_v3,
            "lineage_id": lineage_latest,
            "version": 3,
            "created_at": datetime.now(timezone.utc),
            "content": _email_optimize_content(
                subject="v3 latest lineage",
                body="draft v3",
                source_version_id=str(outcome_latest_v2),
                payload_version_number=99,
            ),
        },
        {
            "outcome_id": outcome_latest_v2,
            "lineage_id": lineage_latest,
            "version": 2,
            "created_at": datetime.now(timezone.utc),
            "content": _email_optimize_content(
                subject="v2 latest lineage",
                body="draft v2 latest",
                source_version_id=str(outcome_old_v2),
                payload_version_number=1,
            ),
        },
        {
            "outcome_id": outcome_old_v2,
            "lineage_id": lineage_old,
            "version": 2,
            "created_at": datetime.now(timezone.utc),
            "content": _email_optimize_content(
                subject="v2 old lineage",
                body="draft v2 old",
                payload_version_number=2,
            ),
        },
    ]

    kernel = WorkflowKernel(
        contracts=StaticContracts({}),
        operator_executor=_NoopOperatorExecutor(),
        policy_engine=PolicyEngine(),
        policy_store=FakePolicyStore(),
        event_writer=FakeEventWriter(),
        pool=_KernelPool(rows),
        tenant_id=1,
    )

    payload = {"source_draft_version": 2}
    await kernel._hydrate_email_optimize_source(thread_id=55, payload=payload)

    assert payload["current_subject"] == "v2 latest lineage"
    assert payload["current_body"] == "draft v2 latest"
    assert payload["source_draft_outcome_id"] == str(outcome_latest_v2)
    assert payload["source_draft_version"] == 2
