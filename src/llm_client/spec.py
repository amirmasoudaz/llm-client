"""
Request specification and context for deterministic execution.
"""

from __future__ import annotations

import time
import uuid
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
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None
    span_id: str | None = None  # For distributed tracing
    tenant_id: str | None = None
    user_id: str | None = None  # For per-user tracking
    tags: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cancellation_token: CancellationToken = field(default_factory=_default_cancel_token)
    schema_version: int = 1

    @classmethod
    def ensure(cls, ctx: RequestContext | None) -> RequestContext:
        """Ensure a context exists - create default if None."""
        return ctx if ctx is not None else cls()

    def child(self, *, new_span: bool = True) -> RequestContext:
        """Create a child context for nested operations.
        
        Preserves request_id, trace_id, tenant_id, user_id, cancellation_token.
        Generates new span_id if new_span=True.
        """
        return RequestContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()) if new_span else self.span_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            tags=dict(self.tags),
            cancellation_token=self.cancellation_token,  # Propagate cancellation
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
            "tags": dict(self.tags),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


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
