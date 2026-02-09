"""
Request specification and context for deterministic execution.

Identity Taxonomy Note
----------------------
This module uses llm-client naming conventions for identity fields:
    - tenant_id -> maps to scope_id in agent-runtime
    - user_id -> maps to principal_id in agent-runtime
    
The canonical taxonomy (per agent-runtime) is:
    - scope_id: tenant/org/workspace
    - principal_id: user/service actor
    - session_id: thread/conversation bucket
    - job_id: lifecycle record
    - run_id: specific execution (maps to request_id here)
    - trace_id/span_id: observability

For interop with agent-runtime, use the scope_id/principal_id aliases which
emit deprecation warnings to encourage migration to the canonical taxonomy.
"""

from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from blake3 import blake3

from .providers.types import Message
from .serialization import stable_json_dumps
from .tools.base import Tool

if TYPE_CHECKING:
    from .cancellation import CancellationToken


def _default_cancel_token() -> CancellationToken:
    """Create a default no-op cancellation token."""
    from .cancellation import CancellationToken
    return CancellationToken.none()


@dataclass(frozen=True)
class RequestContext:
    """Context for request correlation, tracing, cancellation, and multi-tenancy.
    
    Frozen dataclass ensures stable identity for cache keys and tracing.
    Mutable objects (like CancellationToken) remain mutable inside.
    
    Identity fields (for agent-runtime integration):
    - request_id: Unique identifier for this request/run (maps to run_id)
    - session_id: Conversation/thread bucket for grouping related requests
    - job_id: Lifecycle record ID for job state management
    - tenant_id: Scope/org/workspace identifier for multi-tenancy (maps to scope_id)
    - user_id: User/service actor identifier (maps to principal_id)
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None
    span_id: str | None = None  # For distributed tracing
    tenant_id: str | None = None  # scope_id in runtime terminology
    user_id: str | None = None  # principal_id in runtime terminology
    session_id: str | None = None  # thread_id / conversation bucket
    job_id: str | None = None  # lifecycle record id for runtime jobs
    tags: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cancellation_token: CancellationToken = field(default_factory=_default_cancel_token)
    schema_version: int = 2  # Bumped for session_id/job_id addition

    @classmethod
    def ensure(cls, ctx: RequestContext | None) -> RequestContext:
        """Ensure a context exists - create default if None."""
        return ctx if ctx is not None else cls()

    def child(self, *, new_span: bool = True) -> RequestContext:
        """Create a child context for nested operations.
        
        Preserves request_id, trace_id, tenant_id, user_id, session_id, job_id, cancellation_token.
        Generates new span_id if new_span=True.
        """
        return RequestContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()) if new_span else self.span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id,
            job_id=self.job_id,
            tags=dict(self.tags),
            cancellation_token=self.cancellation_token,  # Propagate cancellation
            schema_version=self.schema_version,
        )

    def with_job(self, job_id: str) -> RequestContext:
        """Create a new context with job_id set. Useful for runtime integration."""
        return RequestContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id,
            job_id=job_id,
            tags=dict(self.tags),
            created_at=self.created_at,
            cancellation_token=self.cancellation_token,
            schema_version=self.schema_version,
        )

    def with_session(self, session_id: str) -> RequestContext:
        """Create a new context with session_id set. Useful for conversation tracking."""
        return RequestContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=session_id,
            job_id=self.job_id,
            tags=dict(self.tags),
            created_at=self.created_at,
            cancellation_token=self.cancellation_token,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        # Note: cancellation_token is not serialized
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "tags": dict(self.tags),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestContext:
        """Create a RequestContext from a dictionary.
        
        Accepts both llm-client names (tenant_id, user_id) and 
        agent-runtime names (scope_id, principal_id) for compatibility.
        """
        # Support both naming conventions
        tenant = data.get("tenant_id") or data.get("scope_id")
        user = data.get("user_id") or data.get("principal_id")
        
        return cls(
            request_id=data.get("request_id") or data.get("run_id", str(uuid.uuid4())),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            tenant_id=tenant,
            user_id=user,
            session_id=data.get("session_id"),
            job_id=data.get("job_id"),
            tags=dict(data.get("tags", {})),
            created_at=data.get("created_at", time.time()),
            schema_version=data.get("schema_version", 2),
        )

    # === Aliases for agent-runtime compatibility ===
    # The canonical taxonomy uses scope_id and principal_id.
    # These aliases provide interop while encouraging migration.

    @property
    def scope_id(self) -> str | None:
        """Alias for tenant_id (agent-runtime canonical name).
        
        Prefer using tenant_id directly. This alias exists for
        compatibility with agent-runtime's canonical taxonomy.
        """
        return self.tenant_id

    @property
    def principal_id(self) -> str | None:
        """Alias for user_id (agent-runtime canonical name).
        
        Prefer using user_id directly. This alias exists for
        compatibility with agent-runtime's canonical taxonomy.
        """
        return self.user_id

    @property
    def run_id(self) -> str:
        """Alias for request_id (agent-runtime canonical name).
        
        Prefer using request_id directly. This alias exists for
        compatibility with agent-runtime's canonical taxonomy.
        """
        return self.request_id


@dataclass(frozen=True)
class RequestSpec:
    provider: str
    model: str
    messages: list[Message]
    tools: list[Tool] | None = None
    tool_choice: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | dict[str, Any] | type | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        tools_payload = None
        if self.tools:
            tools_payload_list: list[dict[str, Any]] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in self.tools
            ]
            tools_payload = sorted(tools_payload_list, key=lambda t: t["name"])

        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "tools": tools_payload,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "reasoning_effort": self.reasoning_effort,
            "reasoning": self.reasoning,
            "extra": dict(self.extra),
            "stream": self.stream,
        }

    def cache_key(self) -> str:
        payload = self.to_dict()
        return blake3(stable_json_dumps(payload).encode("utf-8")).hexdigest()


__all__ = ["RequestContext", "RequestSpec"]
