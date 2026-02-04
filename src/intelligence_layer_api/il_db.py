from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg


@dataclass(frozen=True)
class ILDB:
    pool: asyncpg.Pool
    tenant_id: int = 1

    async def ensure_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS runtime;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime.threads (
                  tenant_id BIGINT NOT NULL,
                  thread_id BIGINT GENERATED ALWAYS AS IDENTITY,
                  student_id BIGINT NOT NULL,
                  funding_request_id BIGINT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'active',
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, thread_id),
                  CONSTRAINT threads_scope_uq UNIQUE (tenant_id, student_id, funding_request_id)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS threads_student_created ON runtime.threads (tenant_id, student_id, created_at DESC);"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime.queries (
                  tenant_id BIGINT NOT NULL,
                  query_id UUID NOT NULL,
                  thread_id BIGINT NOT NULL,
                  job_id TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'accepted',
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, query_id)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS queries_thread_created ON runtime.queries (tenant_id, thread_id, created_at DESC);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS queries_job_id ON runtime.queries (tenant_id, job_id);"
            )

    async def get_or_create_thread(self, *, student_id: int, funding_request_id: int) -> tuple[int, str, bool]:
        async with self.pool.acquire() as conn:
            # Fast path: avoid burning identity values on repeated inits by selecting first.
            row = await conn.fetchrow(
                """
                SELECT thread_id, status
                FROM runtime.threads
                WHERE tenant_id=$1 AND student_id=$2 AND funding_request_id=$3;
                """,
                self.tenant_id,
                student_id,
                funding_request_id,
            )
            if row is not None:
                await conn.execute(
                    "UPDATE runtime.threads SET updated_at=now() WHERE tenant_id=$1 AND thread_id=$2;",
                    self.tenant_id,
                    int(row["thread_id"]),
                )
                return int(row["thread_id"]), str(row["status"]), False

            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO runtime.threads (tenant_id, student_id, funding_request_id)
                    VALUES ($1, $2, $3)
                    RETURNING thread_id, status;
                    """,
                    self.tenant_id,
                    student_id,
                    funding_request_id,
                )
                assert row is not None
                return int(row["thread_id"]), str(row["status"]), True
            except asyncpg.UniqueViolationError:
                # Race: someone else created it between our SELECT and INSERT.
                row = await conn.fetchrow(
                    """
                    SELECT thread_id, status
                    FROM runtime.threads
                    WHERE tenant_id=$1 AND student_id=$2 AND funding_request_id=$3;
                    """,
                    self.tenant_id,
                    student_id,
                    funding_request_id,
                )
                assert row is not None
                return int(row["thread_id"]), str(row["status"]), False

    async def insert_query(self, *, query_id: str, thread_id: int, job_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runtime.queries (tenant_id, query_id, thread_id, job_id, status)
                VALUES ($1, $2::uuid, $3, $4, 'accepted');
                """,
                self.tenant_id,
                query_id,
                thread_id,
                job_id,
            )

    async def get_thread(self, *, thread_id: int) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT student_id, funding_request_id, status FROM runtime.threads WHERE tenant_id=$1 AND thread_id=$2;",
                self.tenant_id,
                thread_id,
            )
            if not row:
                return None
            return dict(row)

    async def get_job_id_for_query(self, *, query_id: str) -> str | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT job_id FROM runtime.queries WHERE tenant_id=$1 AND query_id=$2::uuid;",
                self.tenant_id,
                query_id,
            )
            if not row:
                return None
            return str(row["job_id"])

    async def get_query(self, *, query_id: str) -> dict[str, Any] | None:
        """Fetch an existing query record (idempotency helper)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT query_id::text AS query_id, thread_id, job_id, status, created_at, updated_at
                FROM runtime.queries
                WHERE tenant_id=$1 AND query_id=$2::uuid;
                """,
                self.tenant_id,
                query_id,
            )
            if not row:
                return None
            return dict(row)

    async def get_query_id_for_job(self, *, job_id: str) -> str | None:
        """Fetch the query_id for a given job_id (reverse lookup)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT query_id::text AS query_id
                FROM runtime.queries
                WHERE tenant_id=$1 AND job_id=$2
                ORDER BY created_at ASC
                LIMIT 1;
                """,
                self.tenant_id,
                job_id,
            )
            if not row:
                return None
            return str(row["query_id"])

    async def list_runtime_events(
        self,
        *,
        job_id: str,
        after_ts: float,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_id, job_id, run_id, trace_id, span_id, scope_id, principal_id, session_id, type, ts, data, schema_version
                FROM runtime.runtime_events
                WHERE job_id = $1 AND ts > $2
                ORDER BY ts ASC
                LIMIT $3;
                """,
                job_id,
                after_ts,
                limit,
            )
            out: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                data = d.get("data")
                if isinstance(data, str):
                    try:
                        import json

                        d["data"] = json.loads(data)
                    except Exception:
                        # Leave as-is; callers can handle best-effort.
                        pass
                out.append(d)
            return out


_pool_lock = asyncio.Lock()
_pool: asyncpg.Pool | None = None


async def get_pool(dsn: str) -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        return _pool
