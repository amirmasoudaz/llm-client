from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuthContext:
    tenant_id: int
    principal: dict[str, Any]
    scopes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "principal": self.principal,
            "scopes": list(self.scopes),
        }


@dataclass(frozen=True)
class TraceContext:
    correlation_id: str
    workflow_id: str
    step_id: str
    thread_id: int | None = None
    intent_id: str | None = None
    plan_id: str | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "correlation_id": self.correlation_id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
        }
        if self.thread_id is not None:
            data["thread_id"] = int(self.thread_id)
        if self.intent_id:
            data["intent_id"] = self.intent_id
        if self.plan_id:
            data["plan_id"] = self.plan_id
        if self.job_id:
            data["job_id"] = self.job_id
        return data


@dataclass
class OperatorCall:
    payload: dict[str, Any]
    idempotency_key: str
    auth_context: AuthContext
    trace_context: TraceContext
    policy_snapshot: dict[str, Any] | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "payload": self.payload,
            "idempotency_key": self.idempotency_key,
            "auth_context": self.auth_context.to_dict(),
            "trace_context": self.trace_context.to_dict(),
        }
        if self.policy_snapshot is not None:
            data["policy_snapshot"] = self.policy_snapshot
        return data


@dataclass
class OperatorMetrics:
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_total: int | None = None
    cost_total_usd: float | None = None
    provider: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"latency_ms": int(self.latency_ms)}
        if self.tokens_in is not None:
            data["tokens_in"] = int(self.tokens_in)
        if self.tokens_out is not None:
            data["tokens_out"] = int(self.tokens_out)
        if self.tokens_total is not None:
            data["tokens_total"] = int(self.tokens_total)
        if self.cost_total_usd is not None:
            data["cost_total_usd"] = float(self.cost_total_usd)
        if self.provider:
            data["provider"] = self.provider
        if self.model:
            data["model"] = self.model
        return data


@dataclass
class OperatorError:
    code: str
    message: str
    category: str
    retryable: bool
    retry_after_ms: int | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "retryable": self.retryable,
        }
        if self.retry_after_ms is not None:
            data["retry_after_ms"] = int(self.retry_after_ms)
        if self.details:
            data["details"] = self.details
        return data


@dataclass
class OperatorResult:
    status: str
    result: dict[str, Any] | None
    metrics: OperatorMetrics
    error: OperatorError | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    nondeterminism: dict[str, Any] | None = None
    schema_version: str = "1.0"
    prompt_template_id: str | None = field(default=None, repr=False, compare=False)
    prompt_template_hash: str | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "result": self.result,
            "artifacts": list(self.artifacts),
            "metrics": self.metrics.to_dict(),
            "error": self.error.to_dict() if self.error else None,
        }
        if self.nondeterminism is not None:
            data["nondeterminism"] = self.nondeterminism
        return data
