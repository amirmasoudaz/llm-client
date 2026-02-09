from __future__ import annotations

import time
from typing import Any

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .profile_memory_utils import MEMORY_TYPES, group_memory_by_type


class MemoryRetrieveOperator(Operator):
    name = "Memory.Retrieve"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        thread_id = call.payload.get("thread_id")
        if not isinstance(thread_id, int) or thread_id <= 0:
            return _failed(
                start=start,
                code="missing_thread_id",
                message="thread_id is required",
            )

        raw_types = call.payload.get("types")
        types: list[str] = []
        if isinstance(raw_types, list):
            for item in raw_types:
                text = str(item).strip()
                if text in MEMORY_TYPES and text not in types:
                    types.append(text)
        limit = int(call.payload.get("limit") or 20)
        limit = max(1, min(100, limit))

        async with self._pool.acquire() as conn:
            thread = await conn.fetchrow(
                """
                SELECT student_id
                FROM runtime.threads
                WHERE tenant_id=$1 AND thread_id=$2;
                """,
                self._tenant_id,
                thread_id,
            )
            if not thread:
                return _failed(
                    start=start,
                    code="thread_not_found",
                    message="thread not found",
                )
            student_id = int(thread["student_id"])

            if types:
                rows = await conn.fetch(
                    """
                    SELECT memory_id, memory_type, memory_content, source, updated_at
                    FROM profile.student_memories
                    WHERE tenant_id=$1
                      AND student_id=$2
                      AND is_active=true
                      AND memory_type = ANY($3::text[])
                    ORDER BY updated_at DESC
                    LIMIT $4;
                    """,
                    self._tenant_id,
                    student_id,
                    types,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT memory_id, memory_type, memory_content, source, updated_at
                    FROM profile.student_memories
                    WHERE tenant_id=$1
                      AND student_id=$2
                      AND is_active=true
                    ORDER BY updated_at DESC
                    LIMIT $3;
                    """,
                    self._tenant_id,
                    student_id,
                    limit,
                )

        entries = [_row_to_entry(row) for row in rows]
        memory = {"entries": entries, "by_type": group_memory_by_type(entries)}

        return OperatorResult(
            status="succeeded",
            result={"student_id": student_id, "memory": memory},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _row_to_entry(row: Any) -> dict[str, Any]:
    return {
        "memory_id": str(row["memory_id"]),
        "type": str(row["memory_type"]),
        "content": str(row["memory_content"]),
        "source": str(row["source"] or "user"),
        "updated_at": row["updated_at"].isoformat(),
    }


def _failed(*, start: float, code: str, message: str) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(
            code=code,
            message=message,
            category="validation",
            retryable=False,
        ),
    )
