from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyContext:
    stage: str
    operator_name: str | None = None
    operator_version: str | None = None
    effects: list[str] = field(default_factory=list)
    policy_tags: list[str] = field(default_factory=list)
    data_classes: list[str] = field(default_factory=list)
    auth_context: dict[str, Any] | None = None
    trace_context: dict[str, Any] | None = None
    input_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "operator_name": self.operator_name,
            "operator_version": self.operator_version,
            "effects": list(self.effects),
            "policy_tags": list(self.policy_tags),
            "data_classes": list(self.data_classes),
            "auth_context": self.auth_context or {},
            "trace_context": self.trace_context or {},
            "input_payload": self.input_payload or {},
        }


@dataclass
class PolicyDecision:
    stage: str
    decision: str
    reason_code: str
    reason: str | None = None
    requirements: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    redactions: list[dict[str, Any]] = field(default_factory=list)
    transform: dict[str, Any] | None = None
    inputs_hash: bytes | None = None
    policy_engine_name: str = "il_policy"
    policy_engine_version: str = "0.1"
    workflow_id: str | None = None
    intent_id: str | None = None
    plan_id: str | None = None
    step_id: str | None = None
    job_id: str | None = None
    correlation_id: str | None = None

    def to_record(self, *, tenant_id: int) -> dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "stage": self.stage,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "requirements": self.requirements,
            "limits": self.limits,
            "redactions": self.redactions,
            "transform": self.transform,
            "inputs_hash": self.inputs_hash,
            "policy_engine_name": self.policy_engine_name,
            "policy_engine_version": self.policy_engine_version,
            "workflow_id": self.workflow_id,
            "intent_id": self.intent_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "job_id": self.job_id,
            "correlation_id": self.correlation_id,
        }
