from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid


@dataclass(frozen=True)
class LedgerEvent:
    tenant_id: int
    event_id: uuid.UUID
    workflow_id: uuid.UUID
    event_type: str
    actor: dict[str, Any]
    payload: dict[str, Any]
    correlation_id: uuid.UUID
    producer_kind: str
    producer_name: str
    producer_version: str
    schema_version: str = "1.0"
    severity: str = "info"
    thread_id: int | None = None
    intent_id: uuid.UUID | None = None
    plan_id: uuid.UUID | None = None
    step_id: str | None = None
    job_id: uuid.UUID | None = None
    outcome_id: uuid.UUID | None = None
    gate_id: uuid.UUID | None = None
    policy_decision_id: uuid.UUID | None = None
