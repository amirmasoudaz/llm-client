"""
Postgres-persisted event bus.

This is a v0.1 implementation to make runtime events auditable (constitution-aligned)
while still supporting low-latency in-process subscriptions (SSE).

Design:
- Persist every published RuntimeEvent to Postgres (append-only).
- Delegate fanout/subscriptions to an in-process InMemoryEventBus.

Limits:
- Subscriptions are process-local (no cross-process streaming).
- Persistence is best-effort; publish does not fail the whole run if DB write fails.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Any

from .bus import EventBus, EventSubscription, InMemoryEventBus
from .types import RuntimeEvent, RuntimeEventType


class PostgresPersistedEventBus(EventBus):
    TABLE = "runtime_events"

    def __init__(self, *, pool: Any, inner: InMemoryEventBus | None = None):
        self._pool = pool
        self._inner = inner or InMemoryEventBus()
        self._ensured = False
        self._lock = asyncio.Lock()

    async def _ensure_table(self) -> None:
        if self._ensured:
            return
        async with self._lock:
            if self._ensured:
                return
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE SCHEMA IF NOT EXISTS runtime;")
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS runtime.{self.TABLE} (
                      event_id TEXT PRIMARY KEY,
                      job_id TEXT,
                      run_id TEXT,
                      trace_id TEXT,
                      span_id TEXT,
                      scope_id TEXT,
                      principal_id TEXT,
                      session_id TEXT,
                      type TEXT NOT NULL,
                      ts DOUBLE PRECISION NOT NULL,
                      data JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                      schema_version INTEGER NOT NULL DEFAULT 1
                    );
                    """
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.TABLE}_job_ts ON runtime.{self.TABLE} (job_id, ts);"
                )
            self._ensured = True

    async def publish(self, event: RuntimeEvent) -> None:
        await self._inner.publish(event)
        try:
            await self._ensure_table()
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO runtime.{self.TABLE}
                      (event_id, job_id, run_id, trace_id, span_id, scope_id, principal_id, session_id, type, ts, data, schema_version)
                    VALUES
                      ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12)
                    ON CONFLICT (event_id) DO NOTHING;
                    """,
                    event.event_id,
                    event.job_id,
                    event.run_id,
                    event.trace_id,
                    event.span_id,
                    event.scope_id,
                    event.principal_id,
                    event.session_id,
                    event.type.value,
                    float(event.timestamp),
                    json.dumps(event.data),
                    int(event.schema_version),
                )
        except Exception:
            # Best-effort persistence only (do not fail job execution).
            return

    def subscribe(
        self,
        job_id: str | None = None,
        event_types: set[RuntimeEventType] | None = None,
        scope_id: str | None = None,
    ) -> EventSubscription:
        return self._inner.subscribe(job_id=job_id, event_types=event_types, scope_id=scope_id)

    async def events(self, subscription: EventSubscription) -> AsyncIterator[RuntimeEvent]:
        async for ev in self._inner.events(subscription):
            yield ev

    def unsubscribe(self, subscription: EventSubscription) -> None:
        self._inner.unsubscribe(subscription)

    async def close(self) -> None:
        await self._inner.close()


__all__ = ["PostgresPersistedEventBus"]

