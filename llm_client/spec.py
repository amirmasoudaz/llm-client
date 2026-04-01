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
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .cache_keys import request_cache_key
from .providers.types import Message
from .tools.base import ResponsesBuiltinTool, ResponsesCustomTool, ResponsesMCPTool, Tool, ToolDefinition

if TYPE_CHECKING:
    from .cancellation import CancellationToken
    from .context import BudgetSpec, ExecutionContext, PolicyRef, RunVersions


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
    parent_span_id: str | None = None
    tenant_id: str | None = None  # scope_id in runtime terminology
    user_id: str | None = None  # principal_id in runtime terminology
    session_id: str | None = None  # thread_id / conversation bucket
    job_id: str | None = None  # lifecycle record id for runtime jobs
    tags: dict[str, Any] = field(default_factory=dict)
    budgets: BudgetSpec | None = None
    policy_ref: PolicyRef | None = None
    versions: RunVersions | None = None
    created_at: float = field(default_factory=time.time)
    cancellation_token: CancellationToken = field(default_factory=_default_cancel_token)
    schema_version: int = 3  # Bumped for richer execution-context interop

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
            parent_span_id=self.span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id,
            job_id=self.job_id,
            tags=dict(self.tags),
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            created_at=self.created_at,
            cancellation_token=self.cancellation_token,  # Propagate cancellation
            schema_version=self.schema_version,
        )

    def with_job(self, job_id: str) -> RequestContext:
        """Create a new context with job_id set. Useful for runtime integration."""
        return RequestContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id,
            job_id=job_id,
            tags=dict(self.tags),
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
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
            parent_span_id=self.parent_span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=session_id,
            job_id=self.job_id,
            tags=dict(self.tags),
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            created_at=self.created_at,
            cancellation_token=self.cancellation_token,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_execution_context(cls, ctx: ExecutionContext) -> RequestContext:
        return ctx.to_request_context()

    def to_execution_context(self) -> ExecutionContext:
        from .context import ExecutionContext

        return ExecutionContext.from_request_context(self)

    def to_dict(self) -> dict[str, Any]:
        # Note: cancellation_token is not serialized
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "tags": dict(self.tags),
            "budgets": self.budgets.to_dict() if self.budgets else None,
            "policy_ref": self.policy_ref.to_dict() if self.policy_ref else None,
            "versions": self.versions.to_dict() if self.versions else None,
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
        budgets = data.get("budgets")
        if isinstance(budgets, dict):
            from .context import BudgetSpec

            budgets = BudgetSpec.from_dict(budgets)
        policy_ref = data.get("policy_ref")
        if isinstance(policy_ref, dict):
            from .context import PolicyRef

            policy_ref = PolicyRef.from_dict(policy_ref)
        versions = data.get("versions")
        if isinstance(versions, dict):
            from .context import RunVersions

            versions = RunVersions.from_dict(versions)
        
        return cls(
            request_id=data.get("request_id") or data.get("run_id", str(uuid.uuid4())),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
            tenant_id=tenant,
            user_id=user,
            session_id=data.get("session_id"),
            job_id=data.get("job_id"),
            tags=dict(data.get("tags", data.get("metadata", {}))),
            budgets=budgets,
            policy_ref=policy_ref,
            versions=versions,
            created_at=data.get("created_at", time.time()),
            schema_version=data.get("schema_version", 3),
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

    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for tags when interoperating with execution contexts."""
        return self.tags


def _tool_name_for_sorting(tool: Any) -> str:
    if isinstance(tool, Tool):
        return str(tool.name or "")
    if isinstance(tool, ResponsesCustomTool):
        return str(tool.name or "")
    if isinstance(tool, ResponsesMCPTool):
        return str(tool.server_label or tool.connector_id or tool.server_url or "mcp")
    if isinstance(tool, ResponsesBuiltinTool):
        return str(tool.config.get("name") or tool.type or "")
    if isinstance(tool, dict):
        direct_name = str(tool.get("name") or "").strip()
        if direct_name:
            return direct_name
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "").strip()
    return ""


def _serialize_tool_for_request_spec(tool: Any) -> dict[str, Any]:
    if isinstance(tool, Tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
    if isinstance(tool, ResponsesCustomTool):
        return {
            "name": tool.name,
            "description": tool.description,
            "provider_definition": tool.to_dict(),
        }
    if isinstance(tool, ResponsesMCPTool):
        return {
            "name": str(tool.server_label or tool.connector_id or tool.server_url or "mcp"),
            "description": tool.server_description,
            "provider_definition": tool.to_dict(),
        }
    if isinstance(tool, ResponsesBuiltinTool):
        return {
            "name": str(tool.config.get("name") or tool.type or "tool"),
            "description": tool.config.get("description"),
            "provider_definition": tool.to_dict(),
        }
    if isinstance(tool, dict):
        payload = dict(tool)
        name = _tool_name_for_sorting(tool) or "tool"
        function = payload.get("function")
        if isinstance(function, dict):
            return {
                "name": name,
                "description": function.get("description"),
                "parameters": function.get("parameters"),
                "strict": function.get("strict", payload.get("strict")),
                "provider_definition": payload,
            }
        return {
            "name": name,
            "description": payload.get("description"),
            "parameters": payload.get("parameters"),
            "strict": payload.get("strict"),
            "provider_definition": payload,
        }
    return {
        "name": type(tool).__name__ or "tool",
        "provider_definition": repr(tool),
    }


@dataclass(frozen=True)
class RequestSpec:
    provider: str
    model: str
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | dict[str, Any] | type | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None
    include: list[str] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        tools_payload = None
        if self.tools:
            tools_payload_list: list[dict[str, Any]] = [
                _serialize_tool_for_request_spec(tool)
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
            "include": list(self.include) if self.include is not None else None,
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": self.prompt_cache_retention,
            "extra": dict(self.extra),
            "stream": self.stream,
        }

    def cache_key(self) -> str:
        return request_cache_key(self)


__all__ = ["RequestContext", "RequestSpec"]
