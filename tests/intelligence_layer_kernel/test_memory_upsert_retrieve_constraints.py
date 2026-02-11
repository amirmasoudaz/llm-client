from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from intelligence_layer_kernel.operators.implementations.memory_retrieve import MemoryRetrieveOperator
from intelligence_layer_kernel.operators.implementations.memory_upsert import MemoryUpsertOperator
from intelligence_layer_kernel.operators.implementations.profile_memory_utils import MEMORY_TYPES
from intelligence_layer_kernel.operators.types import AuthContext, OperatorCall, TraceContext


class _MemoryPoolState:
    def __init__(self) -> None:
        self.thread_to_student: dict[tuple[int, int], int] = {}
        self.memories: list[dict[str, Any]] = []


class _AcquireCtx:
    def __init__(self, state: _MemoryPoolState) -> None:
        self._state = state

    async def __aenter__(self) -> "_FakeMemoryConn":
        return _FakeMemoryConn(self._state)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakeTx:
    async def __aenter__(self) -> "_FakeTx":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakeMemoryConn:
    def __init__(self, state: _MemoryPoolState) -> None:
        self._state = state

    def transaction(self) -> _FakeTx:
        return _FakeTx()

    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        normalized = " ".join(query.split()).lower()
        if "from runtime.threads" in normalized:
            tenant_id = int(args[0])
            thread_id = int(args[1])
            student_id = self._state.thread_to_student.get((tenant_id, thread_id))
            if student_id is None:
                return None
            return {"student_id": student_id}
        return None

    async def execute(self, query: str, *args) -> str:
        normalized = " ".join(query.split()).lower()
        if "update profile.student_memories" in normalized:
            tenant_id = int(args[0])
            student_id = int(args[1])
            memory_type = str(args[2])
            for row in self._state.memories:
                if (
                    int(row["tenant_id"]) == tenant_id
                    and int(row["student_id"]) == student_id
                    and str(row["memory_type"]) == memory_type
                    and bool(row["is_active"])
                ):
                    row["is_active"] = False
                    row["updated_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"

        if "insert into profile.student_memories" in normalized:
            tenant_id = int(args[0])
            memory_id = args[1]
            student_id = int(args[2])
            memory_type = str(args[3])
            memory_content = str(args[4])
            source = str(args[5])
            self._state.memories.append(
                {
                    "tenant_id": tenant_id,
                    "memory_id": memory_id,
                    "student_id": student_id,
                    "memory_type": memory_type,
                    "memory_content": memory_content,
                    "source": source,
                    "is_active": True,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            return "INSERT 1"
        raise AssertionError(f"unexpected execute query: {query}")

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        normalized = " ".join(query.split()).lower()
        if "from profile.student_memories" not in normalized:
            raise AssertionError(f"unexpected fetch query: {query}")

        tenant_id = int(args[0])
        student_id = int(args[1])
        filter_types: list[str] | None = None
        if "memory_type = any" in normalized:
            raw_types = args[2]
            filter_types = [str(item) for item in raw_types] if isinstance(raw_types, list) else []
            limit = int(args[3])
        else:
            limit = int(args[2]) if len(args) >= 3 else 50

        rows = [
            row
            for row in self._state.memories
            if int(row["tenant_id"]) == tenant_id
            and int(row["student_id"]) == student_id
            and bool(row["is_active"])
        ]
        if filter_types is not None:
            rows = [row for row in rows if str(row["memory_type"]) in set(filter_types)]
        rows.sort(key=lambda item: item["updated_at"], reverse=True)
        rows = rows[:limit]
        return [
            {
                "memory_id": row["memory_id"],
                "memory_type": row["memory_type"],
                "memory_content": row["memory_content"],
                "source": row["source"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]


class _FakePool:
    def __init__(self) -> None:
        self.state = _MemoryPoolState()

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self.state)


def _call(payload: dict[str, Any]) -> OperatorCall:
    return OperatorCall(
        payload=payload,
        idempotency_key=str(uuid.uuid4()),
        auth_context=AuthContext(tenant_id=1, principal={"type": "student", "id": 7}, scopes=["test"]),
        trace_context=TraceContext(correlation_id=str(uuid.uuid4()), workflow_id=str(uuid.uuid4()), step_id="s1"),
    )


@pytest.mark.asyncio
async def test_memory_upsert_enforces_type_constraints() -> None:
    pool = _FakePool()
    pool.state.thread_to_student[(1, 11)] = 7
    operator = MemoryUpsertOperator(pool=pool, tenant_id=1)

    result = await operator.run(
        _call(
            {
                "thread_id": 11,
                "entries": [
                    {"type": "tone_style", "content": "formal and concise"},
                    {"type": "unknown", "content": "should be ignored"},
                    {"type": "do_dont", "content": ""},
                    {"type": "do_dont", "content": "avoid emojis"},
                    {"type": "long_term_goal", "content": "aim for a top CS PhD"},
                ],
            }
        )
    )

    assert result.status == "succeeded"
    payload = result.result or {}
    assert int(payload["upserted"]) == 3
    memory = payload["memory"]
    by_type = memory["by_type"]
    assert set(by_type.keys()) == {"tone_style", "do_dont", "long_term_goal"}
    for entry in memory["entries"]:
        assert entry["type"] in MEMORY_TYPES


@pytest.mark.asyncio
async def test_memory_retrieve_filters_by_allowed_types() -> None:
    pool = _FakePool()
    pool.state.thread_to_student[(1, 11)] = 7
    upsert = MemoryUpsertOperator(pool=pool, tenant_id=1)
    retrieve = MemoryRetrieveOperator(pool=pool, tenant_id=1)

    await upsert.run(
        _call(
            {
                "thread_id": 11,
                "entries": [
                    {"type": "tone_style", "content": "formal"},
                    {"type": "do_dont", "content": "do not mention budget"},
                    {"type": "long_term_goal", "content": "publish in top venues"},
                ],
            }
        )
    )

    filtered = await retrieve.run(
        _call({"thread_id": 11, "types": ["do_dont", "unsupported"], "limit": 10})
    )
    assert filtered.status == "succeeded"
    filtered_entries = filtered.result["memory"]["entries"]
    assert filtered_entries
    assert {entry["type"] for entry in filtered_entries} == {"do_dont"}
    assert set(filtered.result["memory"]["by_type"].keys()) == {"do_dont"}

    limited = await retrieve.run(_call({"thread_id": 11, "limit": 2}))
    assert limited.status == "succeeded"
    assert len(limited.result["memory"]["entries"]) == 2
