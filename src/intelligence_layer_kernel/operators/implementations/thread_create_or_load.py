from __future__ import annotations

import time
from typing import Any

from ..base import Operator
from ..types import OperatorCall, OperatorResult, OperatorMetrics


class ThreadCreateOrLoadOperator(Operator):
    name = "Thread.CreateOrLoad"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        student_id = int(payload["student_id"])
        funding_request_id = int(payload["funding_request_id"])

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT thread_id, status
                FROM runtime.threads
                WHERE tenant_id=$1 AND student_id=$2 AND funding_request_id=$3;
                """,
                self._tenant_id,
                student_id,
                funding_request_id,
            )
            if row:
                thread_id = int(row["thread_id"])
                status = str(row["status"])
                thread_status = status if status in {"active", "archived"} else "active"
                is_new = False
            else:
                row = await conn.fetchrow(
                    """
                    INSERT INTO runtime.threads (tenant_id, student_id, funding_request_id)
                    VALUES ($1, $2, $3)
                    RETURNING thread_id, status;
                    """,
                    self._tenant_id,
                    student_id,
                    funding_request_id,
                )
                thread_id = int(row["thread_id"])
                thread_status = "new"
                is_new = True

        result = {
            "thread_id": thread_id,
            "thread_status": thread_status,
            "onboarding_gate": "ready",
            "missing_requirements": [],
        }

        metrics = OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000))

        return OperatorResult(
            status="succeeded",
            result=result,
            artifacts=[],
            metrics=metrics,
            error=None,
        )
