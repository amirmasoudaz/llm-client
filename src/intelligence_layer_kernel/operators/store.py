from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import asyncpg
from blake3 import blake3


@dataclass(frozen=True)
class JobClaim:
    status: str  # new|retry|existing_success|in_progress
    job_id: uuid.UUID | None
    attempt_no: int | None
    result_payload: dict[str, Any] | None


class OperatorJobStore:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def claim_job(
        self,
        *,
        operator_name: str,
        operator_version: str,
        idempotency_key: str,
        workflow_id: str | None,
        thread_id: int | None,
        intent_id: str | None,
        plan_id: str | None,
        step_id: str | None,
        correlation_id: str,
        input_payload: dict[str, Any],
        effects: list[str],
        policy_tags: list[str],
        data_classes: list[str] | None = None,
    ) -> JobClaim:
        payload_json = json.dumps(input_payload, sort_keys=True, separators=(",", ":"))
        input_hash = blake3(payload_json.encode("utf-8")).digest()

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT job_id, status, attempt_count, result_payload
                    FROM ledger.jobs
                    WHERE tenant_id=$1 AND operator_name=$2 AND idempotency_key=$3
                    FOR UPDATE;
                    """,
                    self._tenant_id,
                    operator_name,
                    idempotency_key,
                )
                if row:
                    status = str(row["status"])
                    if status == "succeeded":
                        result_payload = row["result_payload"]
                        if isinstance(result_payload, str):
                            try:
                                result_payload = json.loads(result_payload)
                            except Exception:
                                result_payload = None
                        return JobClaim("existing_success", uuid.UUID(str(row["job_id"])), int(row["attempt_count"]), result_payload)
                    # Treat running jobs as retryable to avoid stuck operators in single-process runs.
                    attempt_no = int(row["attempt_count"]) + 1
                    await conn.execute(
                        """
                        UPDATE ledger.jobs
                        SET status='running', attempt_count=$4, started_at=now(), input_payload=$5::jsonb, input_hash=$6
                        WHERE tenant_id=$1 AND operator_name=$2 AND idempotency_key=$3;
                        """,
                        self._tenant_id,
                        operator_name,
                        idempotency_key,
                        attempt_no,
                        json.dumps(input_payload),
                        input_hash,
                    )
                    await conn.execute(
                        """
                        INSERT INTO ledger.job_attempts (tenant_id, job_id, attempt_no, status, started_at)
                        VALUES ($1,$2,$3,'running',now());
                        """,
                        self._tenant_id,
                        row["job_id"],
                        attempt_no,
                    )
                    return JobClaim("retry", uuid.UUID(str(row["job_id"])), attempt_no, None)

                job_id = uuid.uuid4()
                try:
                    await conn.execute(
                        """
                        INSERT INTO ledger.jobs (
                          tenant_id, job_id, workflow_id, thread_id, intent_id, plan_id, step_id,
                          operator_name, operator_version, idempotency_key, effects, policy_tags, data_classes,
                          status, attempt_count, input_payload, input_hash,
                          correlation_id, producer_kind, producer_name, producer_version, created_at, started_at
                        ) VALUES (
                          $1,$2,$3::uuid,$4,$5::uuid,$6::uuid,$7,
                          $8,$9,$10,$11,$12,$13,
                          'running',1,$14::jsonb,$15,
                          $16::uuid,'kernel','operator_executor','1.0',now(),now()
                        );
                        """,
                        self._tenant_id,
                        job_id,
                        workflow_id,
                        thread_id,
                        intent_id,
                        plan_id,
                        step_id,
                        operator_name,
                        operator_version,
                        idempotency_key,
                        effects,
                        policy_tags,
                        data_classes or [],
                        json.dumps(input_payload),
                        input_hash,
                        correlation_id,
                    )
                except asyncpg.UniqueViolationError:
                    row = await conn.fetchrow(
                        """
                        SELECT job_id, status, attempt_count, result_payload
                        FROM ledger.jobs
                        WHERE tenant_id=$1 AND operator_name=$2 AND idempotency_key=$3
                        FOR UPDATE;
                        """,
                        self._tenant_id,
                        operator_name,
                        idempotency_key,
                    )
                    if row and str(row["status"]) == "succeeded":
                        result_payload = row["result_payload"]
                        if isinstance(result_payload, str):
                            try:
                                result_payload = json.loads(result_payload)
                            except Exception:
                                result_payload = None
                        return JobClaim(
                            "existing_success",
                            uuid.UUID(str(row["job_id"])),
                            int(row["attempt_count"]),
                            result_payload,
                        )
                    # Fall back to retry semantics
                    attempt_no = int(row["attempt_count"]) + 1 if row else 1
                    if row:
                        await conn.execute(
                            """
                            UPDATE ledger.jobs
                            SET status='running', attempt_count=$4, started_at=now(), input_payload=$5::jsonb, input_hash=$6
                            WHERE tenant_id=$1 AND operator_name=$2 AND idempotency_key=$3;
                            """,
                            self._tenant_id,
                            operator_name,
                            idempotency_key,
                            attempt_no,
                            json.dumps(input_payload),
                            input_hash,
                        )
                        await conn.execute(
                            """
                            INSERT INTO ledger.job_attempts (tenant_id, job_id, attempt_no, status, started_at)
                            VALUES ($1,$2,$3,'running',now());
                            """,
                            self._tenant_id,
                            row["job_id"],
                            attempt_no,
                        )
                        return JobClaim("retry", uuid.UUID(str(row["job_id"])), attempt_no, None)
                    raise
                await conn.execute(
                    """
                    INSERT INTO ledger.job_attempts (tenant_id, job_id, attempt_no, status, started_at)
                    VALUES ($1,$2,1,'running',now());
                    """,
                    self._tenant_id,
                    job_id,
                )
                return JobClaim("new", job_id, 1, None)

    async def complete_job(
        self,
        *,
        job_id: uuid.UUID,
        attempt_no: int,
        status: str,
        result_payload: dict[str, Any] | None,
        error: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
    ) -> None:
        result_hash = None
        if result_payload is not None:
            payload_json = json.dumps(result_payload, sort_keys=True, separators=(",", ":"))
            result_hash = blake3(payload_json.encode("utf-8")).digest()

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE ledger.jobs
                    SET status=$3, result_payload=$4::jsonb, result_hash=$5, error=$6::jsonb, metrics=$7::jsonb, finished_at=now()
                    WHERE tenant_id=$1 AND job_id=$2;
                    """,
                    self._tenant_id,
                    job_id,
                    status,
                    json.dumps(result_payload) if result_payload is not None else None,
                    result_hash,
                    json.dumps(error) if error is not None else None,
                    json.dumps(metrics) if metrics is not None else None,
                )
                await conn.execute(
                    """
                    UPDATE ledger.job_attempts
                    SET status=$4, finished_at=now(), error=$5::jsonb, metrics=$6::jsonb
                    WHERE tenant_id=$1 AND job_id=$2 AND attempt_no=$3;
                    """,
                    self._tenant_id,
                    job_id,
                    attempt_no,
                    status if status != "running" else "failed",
                    json.dumps(error) if error is not None else None,
                    json.dumps(metrics) if metrics is not None else None,
                )
