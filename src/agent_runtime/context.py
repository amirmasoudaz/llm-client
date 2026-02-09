"""
Execution context for agent runtime.

This module provides the ExecutionContext that flows through all runtime operations,
carrying identity, policy, budgets, and tracing information.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llm_client.cancellation import CancellationToken


def _default_cancel_token() -> Any:
    """Create a per-execution cancellation token (preferred).

    Returns CancellationToken() if llm_client is available, None otherwise.
    """
    try:
        from llm_client.cancellation import CancellationToken
        return CancellationToken()
    except ImportError:
        return None


@dataclass(frozen=True)
class BudgetSpec:
    """Budget constraints for an execution.
    
    Attributes:
        max_tokens: Maximum total tokens (input + output)
        max_cost: Maximum cost in dollars
        max_tool_calls: Maximum number of tool calls
        max_turns: Maximum conversation turns
        max_runtime_seconds: Maximum execution time
    """
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
    """Reference to a policy configuration.
    
    Policies are identified by name and version, allowing for
    centralized policy management.
    """
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
    """Version metadata for determinism and replay.
    
    Captures versions of all components involved in an execution
    to support replay and debugging.
    """
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
    """Universal context that flows through all runtime operations.
    
    This is the "glue" that:
    - Produces clean, correlated logs
    - Scopes caches and memory keys
    - Enforces multi-tenant boundaries
    - Makes debugging possible
    
    The naming follows infrastructure conventions:
    - scope_id: tenant/org/workspace identifier
    - principal_id: user/service actor identifier
    - session_id: conversation/thread bucket
    - run_id: single execution/request identifier
    - job_id: lifecycle record identifier
    
    Note: This is frozen but contains a mutable CancellationToken.
    The token object itself is mutable even though the reference is not.
    """
    # Identity
    scope_id: str | None = None  # tenant/org/workspace
    principal_id: str | None = None  # user/service actor
    session_id: str | None = None  # conversation/thread bucket
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Tracing
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str | None = None
    parent_span_id: str | None = None

    # Policy and budgets
    budgets: BudgetSpec | None = None
    policy_ref: PolicyRef | None = None

    # Versioning for replay
    versions: RunVersions | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # Cancellation (optional - only when llm_client is available)
    cancel: Any = field(default_factory=_default_cancel_token)  # CancellationToken | None

    # Schema version for serialization compatibility
    schema_version: int = 1

    def child(self, *, new_span: bool = True) -> ExecutionContext:
        """Create a child context for nested operations.
        
        Preserves identity and propagates cancellation.
        Creates new span_id if requested.
        """
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
        """Create a new context with a specific job_id."""
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

    def to_request_context(self) -> Any:
        """Convert to llm_client.RequestContext for integration.
        
        Returns None if llm_client is not available.
        """
        try:
            from llm_client.spec import RequestContext
            return RequestContext(
                request_id=self.run_id,
                trace_id=self.trace_id,
                span_id=self.span_id,
                tenant_id=self.scope_id,
                user_id=self.principal_id,
                session_id=self.session_id,
                job_id=self.job_id,
                tags=dict(self.metadata),
                created_at=self.created_at,
                cancellation_token=self.cancel,
            )
        except ImportError:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excluding cancellation token)."""
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
        """Deserialize from dictionary."""
        return cls(
            scope_id=data.get("scope_id"),
            principal_id=data.get("principal_id"),
            session_id=data.get("session_id"),
            run_id=data.get("run_id", str(uuid.uuid4())),
            job_id=data.get("job_id", str(uuid.uuid4())),
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
            budgets=BudgetSpec.from_dict(data["budgets"]) if data.get("budgets") else None,
            policy_ref=PolicyRef.from_dict(data["policy_ref"]) if data.get("policy_ref") else None,
            versions=RunVersions.from_dict(data["versions"]) if data.get("versions") else None,
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", time.time()),
            schema_version=data.get("schema_version", 1),
        )


__all__ = [
    "ExecutionContext",
    "BudgetSpec",
    "PolicyRef",
    "RunVersions",
]
