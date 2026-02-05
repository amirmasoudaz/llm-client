from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass(frozen=True)
class IntentRecord:
    intent_id: uuid.UUID
    intent_type: str
    schema_version: str
    thread_id: int
    scope_type: str
    scope_id: str
    actor: dict[str, Any]
    source: str
    inputs: dict[str, Any]
    constraints: dict[str, Any]
    context_refs: dict[str, Any]
    data_classes: list[str]
    correlation_id: uuid.UUID
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_id": str(self.intent_id),
            "intent_type": self.intent_type,
            "thread_id": self.thread_id,
            "actor": self.actor,
            "source": self.source,
            "scope": {"scope_type": self.scope_type, "scope_id": self.scope_id},
            "inputs": self.inputs,
            "constraints": self.constraints,
            "context_refs": self.context_refs,
            "data_classes": list(self.data_classes),
            "correlation_id": str(self.correlation_id),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class PlanRecord:
    plan_id: uuid.UUID
    intent_id: uuid.UUID
    plan_version: str
    intent_type: str
    thread_id: int
    steps: list[dict[str, Any]]
    created_at: datetime
    schema_version: str = "1.0"
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "plan_id": str(self.plan_id),
            "plan_version": self.plan_version,
            "intent_id": str(self.intent_id),
            "intent_type": self.intent_type,
            "thread_id": self.thread_id,
            "steps": self.steps,
            "created_at": self.created_at.isoformat(),
        }
        if self.explanation:
            data["explanation"] = self.explanation
        return data


@dataclass(frozen=True)
class WorkflowResult:
    workflow_id: uuid.UUID
    intent_id: uuid.UUID
    plan_id: uuid.UUID
    status: str
    gate_id: uuid.UUID | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
