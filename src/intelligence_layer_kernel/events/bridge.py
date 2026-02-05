from __future__ import annotations

import asyncio
import uuid
from typing import Any

from agent_runtime.events import RuntimeEvent, RuntimeEventType

from .types import LedgerEvent
from .writer import EventWriter


class RuntimeEventBridge:
    def __init__(
        self,
        *,
        event_bus: Any,
        writer: EventWriter,
        tenant_id: int,
        producer_name: str = "agent_runtime",
        producer_version: str = "0.1",
    ) -> None:
        self._event_bus = event_bus
        self._writer = writer
        self._tenant_id = tenant_id
        self._producer_name = producer_name
        self._producer_version = producer_version
        self._subscription = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._subscription is not None:
            return
        self._subscription = self._event_bus.subscribe()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._subscription is not None:
            try:
                self._event_bus.unsubscribe(self._subscription)
            except Exception:
                pass
            self._subscription = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        assert self._subscription is not None
        async for ev in self._event_bus.events(self._subscription):
            try:
                ledger_event = _map_runtime_event(ev, tenant_id=self._tenant_id, producer_name=self._producer_name, producer_version=self._producer_version)
                if ledger_event is None:
                    continue
                await self._writer.append(ledger_event)
            except Exception:
                # Best-effort bridge; never break runtime flow.
                continue


def _map_runtime_event(
    ev: RuntimeEvent,
    *,
    tenant_id: int,
    producer_name: str,
    producer_version: str,
) -> LedgerEvent | None:
    workflow_id = _parse_uuid(ev.run_id)
    if workflow_id is None:
        return None

    correlation_id = _parse_uuid(ev.trace_id) or workflow_id
    job_id = _parse_uuid(ev.job_id)
    thread_id = _parse_int(ev.session_id)
    actor = {
        "type": "principal",
        "id": ev.principal_id or "unknown",
        "role": "user",
    }
    severity = "error" if ev.type in {RuntimeEventType.FINAL_ERROR, RuntimeEventType.TOOL_ERROR, RuntimeEventType.JOB_FAILED} else "info"

    event_id = _parse_uuid(ev.event_id) or uuid.uuid4()

    return LedgerEvent(
        tenant_id=tenant_id,
        event_id=event_id,
        workflow_id=workflow_id,
        thread_id=thread_id,
        job_id=job_id,
        event_type=ev.type.value,
        actor=actor,
        payload=dict(ev.data or {}),
        correlation_id=correlation_id,
        producer_kind="runtime",
        producer_name=producer_name,
        producer_version=producer_version,
        severity=severity,
    )


def _parse_uuid(value: Any) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
