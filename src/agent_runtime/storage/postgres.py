"""
PostgreSQL storage adapters for agent runtime.

This module provides persistent storage implementations using PostgreSQL:
- PostgresJobStore: Job record persistence
- PostgresActionStore: Action record persistence
- PostgresLedgerWriter: Ledger event persistence

Requires asyncpg to be installed: pip install asyncpg
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import re
import time
from decimal import Decimal
from typing import Any, TYPE_CHECKING

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    ASYNCPG_AVAILABLE = False

from ..jobs.types import JobRecord, JobStatus
from ..jobs.store import JobStore, JobFilter
from ..actions.types import ActionRecord, ActionStatus
from ..actions.store import ActionStore, ActionFilter
from ..ledger.types import LedgerEvent, LedgerEventType, UsageRecord
from ..ledger.writer import LedgerWriter
from ..context import BudgetSpec, PolicyRef, RunVersions


def _require_asyncpg() -> None:
    """Raise ImportError if asyncpg is not available."""
    if not ASYNCPG_AVAILABLE:
        raise ImportError(
            "PostgreSQL storage requires asyncpg. "
            "Install with: pip install asyncpg"
        )


def _sanitize_table_name(name: str) -> str:
    """Ensure the table name is safe for SQL interpolation."""
    if not name:
        raise ValueError("table_name cannot be empty")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name


def _to_timestamptz(value: Any) -> Any:
    """Convert epoch seconds floats into timezone-aware datetimes for TIMESTAMPTZ columns."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return value


# =============================================================================
# PostgresJobStore
# =============================================================================


class PostgresJobStore(JobStore):
    """PostgreSQL implementation of JobStore.
    
    Table schema:
    - job_id (TEXT PRIMARY KEY)
    - scope_id, principal_id, session_id, run_id (TEXT)
    - parent_job_id, idempotency_key (TEXT)
    - status (TEXT)
    - created_at, updated_at, started_at, completed_at (TIMESTAMPTZ)
    - deadline (FLOAT)
    - progress (FLOAT)
    - current_turn, total_turns (INTEGER)
    - error, error_code, result_ref (TEXT)
    - budgets, policy_ref, versions, metadata, tags (JSONB)
    """
    
    TABLE_NAME = "runtime_jobs"
    
    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        table_name: str | None = None,
    ):
        _require_asyncpg()
        self._pool = pool
        self._table = _sanitize_table_name(table_name or self.TABLE_NAME)
        self._ensured = False
        self._lock = asyncio.Lock()
    
    async def _ensure_table(self) -> None:
        """Create the jobs table if it doesn't exist."""
        async with self._lock:
            if self._ensured:
                return
            
            ddl = f'''
            CREATE TABLE IF NOT EXISTS "{self._table}" (
                job_id TEXT PRIMARY KEY,
                scope_id TEXT,
                principal_id TEXT,
                session_id TEXT,
                run_id TEXT NOT NULL,
                parent_job_id TEXT,
                idempotency_key TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                deadline DOUBLE PRECISION,
                progress DOUBLE PRECISION,
                current_turn INTEGER DEFAULT 0,
                total_turns INTEGER,
                error TEXT,
                error_code TEXT,
                result_ref TEXT,
                budgets JSONB,
                policy_ref JSONB,
                versions JSONB,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                tags JSONB DEFAULT '{{}}'::jsonb,
                schema_version INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS "{self._table}_scope_id_idx" ON "{self._table}" (scope_id);
            CREATE INDEX IF NOT EXISTS "{self._table}_status_idx" ON "{self._table}" (status);
            CREATE INDEX IF NOT EXISTS "{self._table}_idempotency_key_idx" ON "{self._table}" (idempotency_key);
            CREATE INDEX IF NOT EXISTS "{self._table}_created_at_idx" ON "{self._table}" (created_at);
            '''
            
            async with self._pool.acquire() as conn:
                for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                    await conn.execute(stmt)
            
            self._ensured = True
    
    def _job_to_row(self, job: JobRecord) -> dict[str, Any]:
        """Convert JobRecord to database row."""
        return {
            "job_id": job.job_id,
            "scope_id": job.scope_id,
            "principal_id": job.principal_id,
            "session_id": job.session_id,
            "run_id": job.run_id,
            "parent_job_id": job.parent_job_id,
            "idempotency_key": job.idempotency_key,
            "status": job.status.value,
            "created_at": _to_timestamptz(job.created_at),
            "updated_at": _to_timestamptz(job.updated_at),
            "started_at": _to_timestamptz(job.started_at),
            "completed_at": _to_timestamptz(job.completed_at),
            "deadline": job.deadline,
            "progress": job.progress,
            "current_turn": job.current_turn,
            "total_turns": job.total_turns,
            "error": job.error,
            "error_code": job.error_code,
            "result_ref": job.result_ref,
            "budgets": json.dumps(job.budgets.to_dict()) if job.budgets else None,
            "policy_ref": json.dumps(job.policy_ref.to_dict()) if job.policy_ref else None,
            "versions": json.dumps(job.versions.to_dict()) if job.versions else None,
            "metadata": json.dumps(job.metadata),
            "tags": json.dumps(job.tags),
            "schema_version": job.schema_version,
        }
    
    def _row_to_job(self, row: Any) -> JobRecord:
        """Convert database row to JobRecord."""
        budgets = None
        if row["budgets"]:
            budgets_data = row["budgets"] if isinstance(row["budgets"], dict) else json.loads(row["budgets"])
            budgets = BudgetSpec.from_dict(budgets_data)
        
        policy_ref = None
        if row["policy_ref"]:
            policy_data = row["policy_ref"] if isinstance(row["policy_ref"], dict) else json.loads(row["policy_ref"])
            policy_ref = PolicyRef.from_dict(policy_data)
        
        versions = None
        if row["versions"]:
            versions_data = row["versions"] if isinstance(row["versions"], dict) else json.loads(row["versions"])
            versions = RunVersions.from_dict(versions_data)
        
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        tags = row["tags"] if isinstance(row["tags"], dict) else json.loads(row["tags"] or "{}")
        
        return JobRecord(
            job_id=row["job_id"],
            scope_id=row["scope_id"],
            principal_id=row["principal_id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            parent_job_id=row["parent_job_id"],
            idempotency_key=row["idempotency_key"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"].timestamp() if hasattr(row["created_at"], "timestamp") else row["created_at"],
            updated_at=row["updated_at"].timestamp() if hasattr(row["updated_at"], "timestamp") else row["updated_at"],
            started_at=row["started_at"].timestamp() if row["started_at"] and hasattr(row["started_at"], "timestamp") else row["started_at"],
            completed_at=row["completed_at"].timestamp() if row["completed_at"] and hasattr(row["completed_at"], "timestamp") else row["completed_at"],
            deadline=row["deadline"],
            progress=row["progress"],
            current_turn=row["current_turn"] or 0,
            total_turns=row["total_turns"],
            error=row["error"],
            error_code=row["error_code"],
            result_ref=row["result_ref"],
            budgets=budgets,
            policy_ref=policy_ref,
            versions=versions,
            metadata=metadata,
            tags=tags,
            schema_version=row.get("schema_version", 1),
        )
    
    async def create(self, job: JobRecord) -> JobRecord:
        await self._ensure_table()
        
        row = self._job_to_row(job)
        columns = list(row.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        
        q = f'''
        INSERT INTO "{self._table}" ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        '''
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(q, *row.values())
            except asyncpg.UniqueViolationError:
                raise ValueError(f"Job {job.job_id} already exists")
        
        return job
    
    async def get(self, job_id: str) -> JobRecord | None:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}" WHERE job_id = $1'
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(q, job_id)
            if row is None:
                return None
            return self._row_to_job(row)
    
    async def update(self, job: JobRecord) -> JobRecord:
        await self._ensure_table()
        
        row = self._job_to_row(job)
        # Exclude job_id from update
        update_cols = [k for k in row.keys() if k != "job_id"]
        set_clause = ", ".join([f"{col} = ${i+2}" for i, col in enumerate(update_cols)])
        
        q = f'''
        UPDATE "{self._table}"
        SET {set_clause}
        WHERE job_id = $1
        '''
        
        values = [job.job_id] + [row[col] for col in update_cols]
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(q, *values)
            if result == "UPDATE 0":
                raise ValueError(f"Job {job.job_id} not found")
        
        return job
    
    async def delete(self, job_id: str) -> bool:
        await self._ensure_table()
        
        q = f'DELETE FROM "{self._table}" WHERE job_id = $1'
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(q, job_id)
            return result != "DELETE 0"
    
    async def list(self, filter: JobFilter | None = None) -> list[JobRecord]:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}"'
        params: list[Any] = []
        conditions: list[str] = []
        param_idx = 1
        
        if filter:
            if filter.scope_id:
                conditions.append(f"scope_id = ${param_idx}")
                params.append(filter.scope_id)
                param_idx += 1
            if filter.principal_id:
                conditions.append(f"principal_id = ${param_idx}")
                params.append(filter.principal_id)
                param_idx += 1
            if filter.session_id:
                conditions.append(f"session_id = ${param_idx}")
                params.append(filter.session_id)
                param_idx += 1
            if filter.parent_job_id:
                conditions.append(f"parent_job_id = ${param_idx}")
                params.append(filter.parent_job_id)
                param_idx += 1
            if filter.idempotency_key:
                conditions.append(f"idempotency_key = ${param_idx}")
                params.append(filter.idempotency_key)
                param_idx += 1
            if filter.status:
                if isinstance(filter.status, set):
                    placeholders = [f"${param_idx + i}" for i in range(len(filter.status))]
                    conditions.append(f"status IN ({', '.join(placeholders)})")
                    params.extend([s.value for s in filter.status])
                    param_idx += len(filter.status)
                else:
                    conditions.append(f"status = ${param_idx}")
                    params.append(filter.status.value)
                    param_idx += 1
        
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        
        if filter:
            order_dir = "DESC" if filter.order_desc else "ASC"
            q += f" ORDER BY {filter.order_by} {order_dir}"
            q += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([filter.limit, filter.offset])
        else:
            q += " ORDER BY created_at DESC LIMIT 100"
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(q, *params)
            return [self._row_to_job(row) for row in rows]
    
    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        scope_id: str | None = None,
    ) -> JobRecord | None:
        await self._ensure_table()
        
        if scope_id:
            q = f'SELECT * FROM "{self._table}" WHERE idempotency_key = $1 AND scope_id = $2'
            params = [idempotency_key, scope_id]
        else:
            q = f'SELECT * FROM "{self._table}" WHERE idempotency_key = $1'
            params = [idempotency_key]
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(q, *params)
            if row is None:
                return None
            return self._row_to_job(row)
    
    async def count(self, filter: JobFilter | None = None) -> int:
        await self._ensure_table()
        
        q = f'SELECT COUNT(*) FROM "{self._table}"'
        params: list[Any] = []
        conditions: list[str] = []
        param_idx = 1
        
        if filter:
            if filter.scope_id:
                conditions.append(f"scope_id = ${param_idx}")
                params.append(filter.scope_id)
                param_idx += 1
            if filter.status:
                if isinstance(filter.status, set):
                    placeholders = [f"${param_idx + i}" for i in range(len(filter.status))]
                    conditions.append(f"status IN ({', '.join(placeholders)})")
                    params.extend([s.value for s in filter.status])
                else:
                    conditions.append(f"status = ${param_idx}")
                    params.append(filter.status.value)
        
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        
        async with self._pool.acquire() as conn:
            return await conn.fetchval(q, *params)


# =============================================================================
# PostgresActionStore
# =============================================================================


class PostgresActionStore(ActionStore):
    """PostgreSQL implementation of ActionStore.
    
    Table schema:
    - action_id (TEXT PRIMARY KEY)
    - job_id (TEXT NOT NULL)
    - type (TEXT NOT NULL)
    - status (TEXT NOT NULL)
    - payload (JSONB)
    - resolution (JSONB)
    - resolution_error (TEXT)
    - created_at, expires_at, resolved_at (TIMESTAMPTZ)
    - resume_token (TEXT UNIQUE)
    - metadata (JSONB)
    """
    
    TABLE_NAME = "runtime_actions"
    
    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        table_name: str | None = None,
    ):
        _require_asyncpg()
        self._pool = pool
        self._table = _sanitize_table_name(table_name or self.TABLE_NAME)
        self._ensured = False
        self._lock = asyncio.Lock()
    
    async def _ensure_table(self) -> None:
        """Create the actions table if it doesn't exist."""
        async with self._lock:
            if self._ensured:
                return
            
            ddl = f'''
            CREATE TABLE IF NOT EXISTS "{self._table}" (
                action_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                payload JSONB DEFAULT '{{}}'::jsonb,
                resolution JSONB,
                resolution_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ,
                resolved_at TIMESTAMPTZ,
                resume_token TEXT UNIQUE NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                schema_version INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS "{self._table}_job_id_idx" ON "{self._table}" (job_id);
            CREATE INDEX IF NOT EXISTS "{self._table}_resume_token_idx" ON "{self._table}" (resume_token);
            CREATE INDEX IF NOT EXISTS "{self._table}_status_idx" ON "{self._table}" (status);
            CREATE INDEX IF NOT EXISTS "{self._table}_expires_at_idx" ON "{self._table}" (expires_at) WHERE status = 'pending';
            '''
            
            async with self._pool.acquire() as conn:
                for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                    await conn.execute(stmt)
            
            self._ensured = True
    
    def _action_to_row(self, action: ActionRecord) -> dict[str, Any]:
        """Convert ActionRecord to database row."""
        return {
            "action_id": action.action_id,
            "job_id": action.job_id,
            "type": action.type,
            "status": action.status.value,
            "payload": json.dumps(action.payload),
            "resolution": json.dumps(action.resolution) if action.resolution else None,
            "resolution_error": action.resolution_error,
            "created_at": _to_timestamptz(action.created_at),
            "expires_at": _to_timestamptz(action.expires_at),
            "resolved_at": _to_timestamptz(action.resolved_at),
            "resume_token": action.resume_token,
            "metadata": json.dumps(action.metadata),
            "schema_version": action.schema_version,
        }
    
    def _row_to_action(self, row: Any) -> ActionRecord:
        """Convert database row to ActionRecord."""
        payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"] or "{}")
        resolution = None
        if row["resolution"]:
            resolution = row["resolution"] if isinstance(row["resolution"], dict) else json.loads(row["resolution"])
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        
        return ActionRecord(
            action_id=row["action_id"],
            job_id=row["job_id"],
            type=row["type"],
            status=ActionStatus(row["status"]),
            payload=payload,
            resolution=resolution,
            resolution_error=row["resolution_error"],
            created_at=row["created_at"].timestamp() if hasattr(row["created_at"], "timestamp") else row["created_at"],
            expires_at=row["expires_at"].timestamp() if row["expires_at"] and hasattr(row["expires_at"], "timestamp") else row["expires_at"],
            resolved_at=row["resolved_at"].timestamp() if row["resolved_at"] and hasattr(row["resolved_at"], "timestamp") else row["resolved_at"],
            resume_token=row["resume_token"],
            metadata=metadata,
            schema_version=row.get("schema_version", 1),
        )
    
    async def create(self, action: ActionRecord) -> ActionRecord:
        await self._ensure_table()
        
        row = self._action_to_row(action)
        columns = list(row.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        
        q = f'''
        INSERT INTO "{self._table}" ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        '''
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(q, *row.values())
            except asyncpg.UniqueViolationError:
                raise ValueError(f"Action {action.action_id} already exists")
        
        return action
    
    async def get(self, action_id: str) -> ActionRecord | None:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}" WHERE action_id = $1'
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(q, action_id)
            if row is None:
                return None
            return self._row_to_action(row)
    
    async def get_by_resume_token(self, resume_token: str) -> ActionRecord | None:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}" WHERE resume_token = $1'
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(q, resume_token)
            if row is None:
                return None
            return self._row_to_action(row)
    
    async def update(self, action: ActionRecord) -> ActionRecord:
        await self._ensure_table()
        
        row = self._action_to_row(action)
        update_cols = [k for k in row.keys() if k != "action_id"]
        set_clause = ", ".join([f"{col} = ${i+2}" for i, col in enumerate(update_cols)])
        
        q = f'''
        UPDATE "{self._table}"
        SET {set_clause}
        WHERE action_id = $1
        '''
        
        values = [action.action_id] + [row[col] for col in update_cols]
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(q, *values)
            if result == "UPDATE 0":
                raise ValueError(f"Action {action.action_id} not found")
        
        return action
    
    async def delete(self, action_id: str) -> bool:
        await self._ensure_table()
        
        q = f'DELETE FROM "{self._table}" WHERE action_id = $1'
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(q, action_id)
            return result != "DELETE 0"
    
    async def list(self, filter: ActionFilter | None = None) -> list[ActionRecord]:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}"'
        params: list[Any] = []
        conditions: list[str] = []
        param_idx = 1
        
        if filter:
            if filter.job_id:
                conditions.append(f"job_id = ${param_idx}")
                params.append(filter.job_id)
                param_idx += 1
            if filter.type:
                conditions.append(f"type = ${param_idx}")
                params.append(filter.type)
                param_idx += 1
            if filter.status:
                if isinstance(filter.status, set):
                    placeholders = [f"${param_idx + i}" for i in range(len(filter.status))]
                    conditions.append(f"status IN ({', '.join(placeholders)})")
                    params.extend([s.value for s in filter.status])
                    param_idx += len(filter.status)
                else:
                    conditions.append(f"status = ${param_idx}")
                    params.append(filter.status.value)
                    param_idx += 1
        
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        
        q += " ORDER BY created_at DESC"
        
        if filter:
            q += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([filter.limit, filter.offset])
        else:
            q += " LIMIT 100"
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(q, *params)
            return [self._row_to_action(row) for row in rows]
    
    async def list_pending_for_job(self, job_id: str) -> list[ActionRecord]:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}" WHERE job_id = $1 AND status = $2 ORDER BY created_at'
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(q, job_id, ActionStatus.PENDING.value)
            return [self._row_to_action(row) for row in rows]
    
    async def list_expired(self) -> list[ActionRecord]:
        await self._ensure_table()
        
        q = f'''
        SELECT * FROM "{self._table}"
        WHERE status = $1 AND expires_at IS NOT NULL AND expires_at < NOW()
        ORDER BY expires_at
        '''
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(q, ActionStatus.PENDING.value)
            return [self._row_to_action(row) for row in rows]


# =============================================================================
# PostgresLedgerWriter
# =============================================================================


class PostgresLedgerWriter(LedgerWriter):
    """PostgreSQL implementation of LedgerWriter.
    
    Table schema:
    - event_id (TEXT PRIMARY KEY)
    - type (TEXT NOT NULL)
    - timestamp (TIMESTAMPTZ NOT NULL)
    - job_id, run_id, scope_id, principal_id, session_id (TEXT)
    - provider, model, tool_name, connector_name (TEXT)
    - input_tokens, output_tokens, total_tokens, cached_tokens (INTEGER)
    - cost (NUMERIC)
    - duration_ms (DOUBLE PRECISION)
    - metadata (JSONB)
    """
    
    TABLE_NAME = "runtime_ledger"
    
    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        table_name: str | None = None,
    ):
        _require_asyncpg()
        self._pool = pool
        self._table = _sanitize_table_name(table_name or self.TABLE_NAME)
        self._ensured = False
        self._lock = asyncio.Lock()
    
    async def _ensure_table(self) -> None:
        """Create the ledger table if it doesn't exist."""
        async with self._lock:
            if self._ensured:
                return
            
            ddl = f'''
            CREATE TABLE IF NOT EXISTS "{self._table}" (
                event_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                job_id TEXT,
                run_id TEXT,
                scope_id TEXT,
                principal_id TEXT,
                session_id TEXT,
                provider TEXT,
                model TEXT,
                tool_name TEXT,
                connector_name TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cached_tokens INTEGER DEFAULT 0,
                cost NUMERIC(20, 10) DEFAULT 0,
                duration_ms DOUBLE PRECISION,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                schema_version INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS "{self._table}_scope_id_idx" ON "{self._table}" (scope_id);
            CREATE INDEX IF NOT EXISTS "{self._table}_principal_id_idx" ON "{self._table}" (principal_id);
            CREATE INDEX IF NOT EXISTS "{self._table}_job_id_idx" ON "{self._table}" (job_id);
            CREATE INDEX IF NOT EXISTS "{self._table}_timestamp_idx" ON "{self._table}" (timestamp);
            CREATE INDEX IF NOT EXISTS "{self._table}_type_idx" ON "{self._table}" (type);
            '''
            
            async with self._pool.acquire() as conn:
                for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                    await conn.execute(stmt)
            
            self._ensured = True
    
    def _event_to_row(self, event: LedgerEvent) -> dict[str, Any]:
        """Convert LedgerEvent to database row."""
        return {
            "event_id": event.event_id,
            "type": event.type.value,
            "timestamp": _to_timestamptz(event.timestamp),
            "job_id": event.job_id,
            "run_id": event.run_id,
            "scope_id": event.scope_id,
            "principal_id": event.principal_id,
            "session_id": event.session_id,
            "provider": event.provider,
            "model": event.model,
            "tool_name": event.tool_name,
            "connector_name": event.connector_name,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "total_tokens": event.total_tokens,
            "cached_tokens": event.cached_tokens,
            "cost": Decimal(event.cost),
            "duration_ms": event.duration_ms,
            "metadata": json.dumps(event.metadata),
            "schema_version": event.schema_version,
        }
    
    def _row_to_event(self, row: Any) -> LedgerEvent:
        """Convert database row to LedgerEvent."""
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        
        return LedgerEvent(
            event_id=row["event_id"],
            type=LedgerEventType(row["type"]),
            timestamp=row["timestamp"].timestamp() if hasattr(row["timestamp"], "timestamp") else row["timestamp"],
            job_id=row["job_id"],
            run_id=row["run_id"],
            scope_id=row["scope_id"],
            principal_id=row["principal_id"],
            session_id=row["session_id"],
            provider=row["provider"],
            model=row["model"],
            tool_name=row["tool_name"],
            connector_name=row["connector_name"],
            input_tokens=row["input_tokens"] or 0,
            output_tokens=row["output_tokens"] or 0,
            total_tokens=row["total_tokens"] or 0,
            cached_tokens=row["cached_tokens"] or 0,
            cost=str(row["cost"]) if row["cost"] else "0",
            duration_ms=row["duration_ms"],
            metadata=metadata,
            schema_version=row.get("schema_version", 1),
        )
    
    async def write(self, event: LedgerEvent) -> None:
        await self._ensure_table()
        
        row = self._event_to_row(event)
        columns = list(row.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        
        q = f'''
        INSERT INTO "{self._table}" ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        ON CONFLICT (event_id) DO NOTHING
        '''
        
        async with self._pool.acquire() as conn:
            await conn.execute(q, *row.values())
    
    async def get_usage(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> UsageRecord:
        await self._ensure_table()
        
        # Build aggregation query
        q = f'''
        SELECT
            COALESCE(SUM(input_tokens), 0) as total_input_tokens,
            COALESCE(SUM(output_tokens), 0) as total_output_tokens,
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            COALESCE(SUM(cached_tokens), 0) as total_cached_tokens,
            COALESCE(SUM(cost), 0) as total_cost,
            COUNT(*) FILTER (WHERE type = 'model_usage') as request_count,
            COUNT(*) FILTER (WHERE type = 'tool_usage') as tool_call_count,
            MIN(timestamp) as first_event_at,
            MAX(timestamp) as last_event_at
        FROM "{self._table}"
        '''
        
        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1
        
        if scope_id:
            conditions.append(f"scope_id = ${param_idx}")
            params.append(scope_id)
            param_idx += 1
        if principal_id:
            conditions.append(f"principal_id = ${param_idx}")
            params.append(principal_id)
            param_idx += 1
        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1
        if job_id:
            conditions.append(f"job_id = ${param_idx}")
            params.append(job_id)
        
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(q, *params)
            
            first_event = None
            last_event = None
            if row["first_event_at"]:
                first_event = row["first_event_at"].timestamp() if hasattr(row["first_event_at"], "timestamp") else row["first_event_at"]
            if row["last_event_at"]:
                last_event = row["last_event_at"].timestamp() if hasattr(row["last_event_at"], "timestamp") else row["last_event_at"]
            
            return UsageRecord(
                scope_id=scope_id,
                principal_id=principal_id,
                session_id=session_id,
                total_input_tokens=int(row["total_input_tokens"]),
                total_output_tokens=int(row["total_output_tokens"]),
                total_tokens=int(row["total_tokens"]),
                total_cached_tokens=int(row["total_cached_tokens"]),
                total_cost=Decimal(str(row["total_cost"])),
                request_count=int(row["request_count"]),
                tool_call_count=int(row["tool_call_count"]),
                first_event_at=first_event,
                last_event_at=last_event,
            )
    
    async def list_events(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        job_id: str | None = None,
        event_type: LedgerEventType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LedgerEvent]:
        await self._ensure_table()
        
        q = f'SELECT * FROM "{self._table}"'
        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1
        
        if scope_id:
            conditions.append(f"scope_id = ${param_idx}")
            params.append(scope_id)
            param_idx += 1
        if principal_id:
            conditions.append(f"principal_id = ${param_idx}")
            params.append(principal_id)
            param_idx += 1
        if job_id:
            conditions.append(f"job_id = ${param_idx}")
            params.append(job_id)
            param_idx += 1
        if event_type:
            conditions.append(f"type = ${param_idx}")
            params.append(event_type.value)
            param_idx += 1
        
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        
        q += f" ORDER BY timestamp DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(q, *params)
            return [self._row_to_event(row) for row in rows]


__all__ = [
    "PostgresJobStore",
    "PostgresActionStore",
    "PostgresLedgerWriter",
]
