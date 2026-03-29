"""
Shared execution context types for llm-client and higher-level runtimes.

These types are intentionally generic and avoid application-specific policy
semantics. They provide a richer execution envelope than RequestContext while
remaining interoperable with RequestContext for provider calls.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cancellation import CancellationToken
    from .spec import RequestContext


def _default_cancel_token() -> Any:
    """Create a per-execution cancellation token."""
    from .cancellation import CancellationToken

    return CancellationToken()


@dataclass(frozen=True)
class BudgetSpec:
    """Generic budget constraints for an execution."""

    max_tokens: int | None = None
    max_cost: float | None = None
    max_tool_calls: int | None = None
    max_turns: int | None = None
    max_runtime_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "max_tool_calls": self.max_tool_calls,
            "max_turns": self.max_turns,
            "max_runtime_seconds": self.max_runtime_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetSpec:
        return cls(
            max_tokens=data.get("max_tokens"),
            max_cost=data.get("max_cost"),
            max_tool_calls=data.get("max_tool_calls"),
            max_turns=data.get("max_turns"),
            max_runtime_seconds=data.get("max_runtime_seconds"),
        )


@dataclass(frozen=True)
class PolicyRef:
    """Reference to an externally managed policy bundle."""

    name: str
    version: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "overrides": dict(self.overrides),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyRef:
        return cls(
            name=data["name"],
            version=data.get("version"),
            overrides=dict(data.get("overrides", {})),
        )


@dataclass(frozen=True)
class RunVersions:
    """Version metadata used for determinism and replay."""

    runtime_version: str | None = None
    llm_client_version: str | None = None
    operator_version: str | None = None
    model_version: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_version": self.runtime_version,
            "llm_client_version": self.llm_client_version,
            "operator_version": self.operator_version,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunVersions:
        return cls(
            runtime_version=data.get("runtime_version"),
            llm_client_version=data.get("llm_client_version"),
            operator_version=data.get("operator_version"),
            model_version=data.get("model_version"),
            schema_version=data.get("schema_version", 1),
        )


@dataclass(frozen=True)
class ExecutionContext:
    """Generic execution envelope for agentic runtimes.

    This extends RequestContext-style correlation with execution-level budgets,
    policy references, replay versions, and lifecycle identifiers.
    """

    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str | None = None
    parent_span_id: str | None = None

    budgets: BudgetSpec | None = None
    policy_ref: PolicyRef | None = None
    versions: RunVersions | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cancel: Any = field(default_factory=_default_cancel_token)
    schema_version: int = 1

    @classmethod
    def ensure(cls, ctx: ExecutionContext | None) -> ExecutionContext:
        return ctx if ctx is not None else cls()

    def child(self, *, new_span: bool = True) -> ExecutionContext:
        return ExecutionContext(
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=self.session_id,
            run_id=self.run_id,
            job_id=self.job_id,
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()) if new_span else self.span_id,
            parent_span_id=self.span_id,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            metadata=dict(self.metadata),
            created_at=self.created_at,
            cancel=self.cancel,
            schema_version=self.schema_version,
        )

    def with_job(self, job_id: str) -> ExecutionContext:
        return ExecutionContext(
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=self.session_id,
            run_id=self.run_id,
            job_id=job_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            metadata=dict(self.metadata),
            created_at=self.created_at,
            cancel=self.cancel,
            schema_version=self.schema_version,
        )

    def with_session(self, session_id: str) -> ExecutionContext:
        return ExecutionContext(
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=session_id,
            run_id=self.run_id,
            job_id=self.job_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            metadata=dict(self.metadata),
            created_at=self.created_at,
            cancel=self.cancel,
            schema_version=self.schema_version,
        )

    def to_request_context(self) -> RequestContext:
        from .spec import RequestContext

        return RequestContext(
            request_id=self.run_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            tenant_id=self.scope_id,
            user_id=self.principal_id,
            session_id=self.session_id,
            job_id=self.job_id,
            tags=dict(self.metadata),
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            created_at=self.created_at,
            cancellation_token=self.cancel,
        )

    @classmethod
    def from_request_context(cls, ctx: RequestContext) -> ExecutionContext:
        budgets = getattr(ctx, "budgets", None)
        if isinstance(budgets, dict):
            budgets = BudgetSpec.from_dict(budgets)
        policy_ref = getattr(ctx, "policy_ref", None)
        if isinstance(policy_ref, dict):
            policy_ref = PolicyRef.from_dict(policy_ref)
        versions = getattr(ctx, "versions", None)
        if isinstance(versions, dict):
            versions = RunVersions.from_dict(versions)
        return cls(
            scope_id=getattr(ctx, "scope_id", None),
            principal_id=getattr(ctx, "principal_id", None),
            session_id=getattr(ctx, "session_id", None),
            run_id=getattr(ctx, "run_id", getattr(ctx, "request_id", str(uuid.uuid4()))),
            job_id=getattr(ctx, "job_id", None) or str(uuid.uuid4()),
            trace_id=getattr(ctx, "trace_id", None) or str(uuid.uuid4()),
            span_id=getattr(ctx, "span_id", None),
            parent_span_id=getattr(ctx, "parent_span_id", None),
            budgets=budgets,
            policy_ref=policy_ref,
            versions=versions,
            metadata=dict(getattr(ctx, "metadata", getattr(ctx, "tags", {})) or {}),
            created_at=getattr(ctx, "created_at", time.time()),
            cancel=getattr(ctx, "cancellation_token", _default_cancel_token()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "principal_id": self.principal_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "budgets": self.budgets.to_dict() if self.budgets else None,
            "policy_ref": self.policy_ref.to_dict() if self.policy_ref else None,
            "versions": self.versions.to_dict() if self.versions else None,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        return cls(
            scope_id=data.get("scope_id") or data.get("tenant_id"),
            principal_id=data.get("principal_id") or data.get("user_id"),
            session_id=data.get("session_id"),
            run_id=data.get("run_id") or data.get("request_id", str(uuid.uuid4())),
            job_id=data.get("job_id", str(uuid.uuid4())),
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
            budgets=BudgetSpec.from_dict(data["budgets"]) if data.get("budgets") else None,
            policy_ref=PolicyRef.from_dict(data["policy_ref"]) if data.get("policy_ref") else None,
            versions=RunVersions.from_dict(data["versions"]) if data.get("versions") else None,
            metadata=dict(data.get("metadata", data.get("tags", {}))),
            created_at=data.get("created_at", time.time()),
            schema_version=data.get("schema_version", 1),
        )

    @property
    def tenant_id(self) -> str | None:
        return self.scope_id

    @property
    def user_id(self) -> str | None:
        return self.principal_id

    @property
    def request_id(self) -> str:
        return self.run_id


__all__ = ["BudgetSpec", "ExecutionContext", "PolicyRef", "RunVersions"]
