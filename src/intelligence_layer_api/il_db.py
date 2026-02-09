from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg

from intelligence_layer_kernel.db import ensure_kernel_schema

@dataclass(frozen=True)
class ILDB:
    pool: asyncpg.Pool
    tenant_id: int = 1

    async def ensure_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS runtime;")
            await conn.execute("CREATE SCHEMA IF NOT EXISTS profile;")
            await conn.execute("CREATE SCHEMA IF NOT EXISTS billing;")
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
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile.student_profiles (
                  tenant_id BIGINT NOT NULL,
                  student_id BIGINT NOT NULL,
                  profile_json JSONB NOT NULL,
                  schema_version TEXT NOT NULL DEFAULT '2.0.0',
                  completeness_state JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, student_id)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS student_profiles_updated ON profile.student_profiles (tenant_id, updated_at DESC);"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile.student_memories (
                  tenant_id BIGINT NOT NULL,
                  memory_id UUID NOT NULL,
                  student_id BIGINT NOT NULL,
                  memory_type TEXT NOT NULL,
                  memory_content TEXT NOT NULL,
                  source TEXT NOT NULL DEFAULT 'user',
                  is_active BOOLEAN NOT NULL DEFAULT true,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, memory_id)
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS student_memories_student_type_active
                  ON profile.student_memories (tenant_id, student_id, memory_type, is_active, updated_at DESC);
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing.credit_balances (
                  tenant_id BIGINT NOT NULL,
                  principal_id BIGINT NOT NULL DEFAULT 0,
                  balance_credits BIGINT NOT NULL,
                  overdraft_limit BIGINT NOT NULL DEFAULT 0,
                  expires_at TIMESTAMPTZ NULL,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, principal_id)
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing.credit_reservations (
                  tenant_id BIGINT NOT NULL,
                  reservation_id UUID NOT NULL,
                  principal_id BIGINT NULL,
                  workflow_id UUID NULL,
                  request_key BYTEA NOT NULL,
                  reserved_credits BIGINT NOT NULL,
                  status TEXT NOT NULL,
                  expires_at TIMESTAMPTZ NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, reservation_id),
                  CONSTRAINT credit_reservations_request_uq UNIQUE (tenant_id, request_key)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS credit_reservations_status_expires ON billing.credit_reservations (tenant_id, status, expires_at);"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing.pricing_versions (
                  pricing_version_id UUID NOT NULL,
                  provider TEXT NOT NULL,
                  version TEXT NOT NULL,
                  effective_from TIMESTAMPTZ NOT NULL,
                  effective_to TIMESTAMPTZ NULL,
                  pricing_json JSONB NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (pricing_version_id),
                  CONSTRAINT pricing_versions_provider_version_uq UNIQUE (provider, version)
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing.credit_rate_versions (
                  credit_rate_version TEXT NOT NULL,
                  credits_per_usd NUMERIC(18, 8) NOT NULL,
                  effective_from TIMESTAMPTZ NOT NULL,
                  effective_to TIMESTAMPTZ NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (credit_rate_version)
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing.usage_events (
                  tenant_id BIGINT NOT NULL,
                  usage_event_id UUID NOT NULL,
                  principal_id BIGINT NULL,
                  workflow_id UUID NULL,
                  job_id UUID NULL,
                  operation_type TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  usage JSONB NOT NULL,
                  cost_usd NUMERIC(18, 8) NOT NULL,
                  effective_cost_usd NUMERIC(18, 8) NOT NULL,
                  credits_charged BIGINT NOT NULL,
                  pricing_version_id UUID NOT NULL,
                  credit_rate_version TEXT NOT NULL,
                  estimated BOOLEAN NOT NULL DEFAULT false,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, usage_event_id),
                  CONSTRAINT usage_events_pricing_fk
                    FOREIGN KEY (pricing_version_id) REFERENCES billing.pricing_versions(pricing_version_id),
                  CONSTRAINT usage_events_credit_rate_fk
                    FOREIGN KEY (credit_rate_version) REFERENCES billing.credit_rate_versions(credit_rate_version)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS usage_events_workflow_created ON billing.usage_events (tenant_id, workflow_id, created_at DESC);"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing.credit_ledger (
                  tenant_id BIGINT NOT NULL,
                  credit_ledger_id UUID NOT NULL,
                  principal_id BIGINT NULL,
                  workflow_id UUID NULL,
                  request_key BYTEA NULL,
                  delta_credits BIGINT NOT NULL,
                  balance_after BIGINT NOT NULL,
                  reason_code TEXT NOT NULL,
                  pricing_version_id UUID NULL,
                  credit_rate_version TEXT NULL,
                  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (tenant_id, credit_ledger_id),
                  CONSTRAINT credit_ledger_pricing_fk
                    FOREIGN KEY (pricing_version_id) REFERENCES billing.pricing_versions(pricing_version_id),
                  CONSTRAINT credit_ledger_credit_rate_fk
                    FOREIGN KEY (credit_rate_version) REFERENCES billing.credit_rate_versions(credit_rate_version)
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS credit_ledger_request_uq
                  ON billing.credit_ledger (tenant_id, request_key)
                  WHERE request_key IS NOT NULL;
                """
            )
            await ensure_kernel_schema(conn)

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

    async def get_workflow_run(self, *, workflow_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT workflow_id::text AS workflow_id, thread_id, intent_id::text AS intent_id, plan_id::text AS plan_id, status
                FROM runtime.workflow_runs
                WHERE tenant_id=$1 AND workflow_id=$2::uuid;
                """,
                self.tenant_id,
                workflow_id,
            )
            if not row:
                return None
            return dict(row)

    async def get_gate(self, *, gate_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT gate_id::text AS gate_id, workflow_id::text AS workflow_id, status, gate_type
                FROM ledger.gates
                WHERE tenant_id=$1 AND gate_id=$2::uuid;
                """,
                self.tenant_id,
                gate_id,
            )
            if not row:
                return None
            return dict(row)

    async def get_latest_waiting_profile_gate(self, *, thread_id: int) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT g.gate_id::text AS gate_id,
                       g.workflow_id::text AS workflow_id,
                       g.preview,
                       g.reason_code,
                       g.status
                FROM ledger.gates g
                JOIN runtime.workflow_runs wr
                  ON wr.tenant_id=g.tenant_id
                 AND wr.workflow_id=g.workflow_id
                WHERE g.tenant_id=$1
                  AND wr.thread_id=$2
                  AND wr.status='waiting'
                  AND g.gate_type='collect_profile_fields'
                  AND g.status='waiting'
                ORDER BY g.created_at DESC
                LIMIT 1;
                """,
                self.tenant_id,
                thread_id,
            )
            if not row:
                return None
            out = dict(row)
            preview = out.get("preview")
            if isinstance(preview, str):
                try:
                    import json

                    out["preview"] = json.loads(preview)
                except Exception:
                    pass
            return out

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
