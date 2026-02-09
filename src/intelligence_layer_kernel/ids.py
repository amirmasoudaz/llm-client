from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class TraceContext:
    tenant_id: int
    correlation_id: uuid.UUID
    workflow_id: uuid.UUID
    thread_id: int | None = None
    intent_id: uuid.UUID | None = None
    plan_id: uuid.UUID | None = None
    step_id: str | None = None
    job_id: uuid.UUID | None = None


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()
