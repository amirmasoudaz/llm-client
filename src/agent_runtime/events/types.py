"""
Runtime event types.

This module defines the RuntimeEvent schema - the unified event model
for all runtime operations. This is designed to be stable and versioned.

Event Categories:
- progress: Execution progress updates
- model.*: Model/LLM events (tokens, reasoning, done)
- tool.*: Tool execution events (start, end, error)
- action.*: Human-in-the-loop events (required, resolved, expired)
- artifact.*: Output artifacts (files, reports, diffs)
- job.*: Job lifecycle events (status changes)
- final.*: Terminal events (result, error)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeEventType(str, Enum):
    """Event type categories for runtime events."""
    
    # Progress events
    PROGRESS = "progress"
    
    # Model events (forwarded from llm-client)
    MODEL_TOKEN = "model.token"
    MODEL_REASONING = "model.reasoning"
    MODEL_DONE = "model.done"
    
    # Tool events
    TOOL_START = "tool.start"
    TOOL_END = "tool.end"
    TOOL_ERROR = "tool.error"
    
    # Action events (human-in-the-loop)
    ACTION_REQUIRED = "action.required"
    ACTION_RESOLVED = "action.resolved"
    ACTION_EXPIRED = "action.expired"
    ACTION_CANCELLED = "action.cancelled"
    
    # Artifact events
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_UPDATED = "artifact.updated"
    
    # Job lifecycle events
    JOB_STARTED = "job.started"
    JOB_STATUS_CHANGED = "job.status_changed"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    
    # Terminal events
    FINAL_RESULT = "final.result"
    FINAL_ERROR = "final.error"


@dataclass
class RuntimeEvent:
    """Unified runtime event.
    
    Every event includes correlation IDs from the execution context:
    - job_id: The job this event belongs to
    - run_id: The specific run/request
    - trace_id: For distributed tracing
    - span_id: Current span (optional)
    
    Events are designed to be:
    - Serializable to JSON
    - Streamable via SSE/WebSocket
    - Persistable for replay
    - Filterable by type
    """
    # Event identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: RuntimeEventType = RuntimeEventType.PROGRESS
    timestamp: float = field(default_factory=time.time)
    
    # Correlation
    job_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    
    # Scope (for multi-tenancy)
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    
    # Event payload
    data: dict[str, Any] = field(default_factory=dict)
    
    # Schema version for forward compatibility
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "scope_id": self.scope_id,
            "principal_id": self.principal_id,
            "session_id": self.session_id,
            "data": self.data,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeEvent:
        """Deserialize from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            type=RuntimeEventType(data["type"]),
            timestamp=data.get("timestamp", time.time()),
            job_id=data.get("job_id"),
            run_id=data.get("run_id"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            scope_id=data.get("scope_id"),
            principal_id=data.get("principal_id"),
            session_id=data.get("session_id"),
            data=dict(data.get("data", {})),
            schema_version=data.get("schema_version", 1),
        )

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        import json
        event_type = self.type.value.replace(".", "_")
        data_json = json.dumps(self.to_dict())
        return f"event: {event_type}\ndata: {data_json}\n\n"

    @classmethod
    def from_context(
        cls,
        ctx: Any,  # ExecutionContext
        type: RuntimeEventType,
        data: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        """Create an event from an ExecutionContext."""
        return cls(
            type=type,
            job_id=ctx.job_id,
            run_id=ctx.run_id,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
            session_id=ctx.session_id,
            data=data or {},
        )


# === Convenience event constructors ===


@dataclass
class ProgressEvent:
    """Progress update data."""
    progress: float  # 0.0 to 1.0
    message: str | None = None
    turn: int | None = None
    step: str | None = None

    def to_runtime_event(self, ctx: Any) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            RuntimeEventType.PROGRESS,
            {
                "progress": self.progress,
                "message": self.message,
                "turn": self.turn,
                "step": self.step,
            },
        )


@dataclass
class ModelEvent:
    """Model/LLM event data."""
    token: str | None = None
    reasoning: str | None = None
    model: str | None = None
    finish_reason: str | None = None

    def to_runtime_event(
        self,
        ctx: Any,
        type: RuntimeEventType = RuntimeEventType.MODEL_TOKEN,
    ) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            type,
            {
                "token": self.token,
                "reasoning": self.reasoning,
                "model": self.model,
                "finish_reason": self.finish_reason,
            },
        )


@dataclass
class ToolEvent:
    """Tool execution event data."""
    tool_name: str
    tool_call_id: str | None = None
    arguments: dict[str, Any] | None = None
    result: str | None = None
    success: bool = True
    error: str | None = None
    duration_ms: float | None = None

    def to_runtime_event(
        self,
        ctx: Any,
        type: RuntimeEventType = RuntimeEventType.TOOL_START,
    ) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            type,
            {
                "tool_name": self.tool_name,
                "tool_call_id": self.tool_call_id,
                "arguments": self.arguments,
                "result": self.result,
                "success": self.success,
                "error": self.error,
                "duration_ms": self.duration_ms,
            },
        )


@dataclass
class ActionEvent:
    """Human-in-the-loop action event data."""
    action_id: str
    action_type: str  # confirm, upload, reauth, apply_changes, choose
    payload: dict[str, Any] = field(default_factory=dict)
    expires_at: float | None = None
    resume_token: str | None = None
    resolution: dict[str, Any] | None = None

    def to_runtime_event(
        self,
        ctx: Any,
        type: RuntimeEventType = RuntimeEventType.ACTION_REQUIRED,
    ) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            type,
            {
                "action_id": self.action_id,
                "action_type": self.action_type,
                "payload": self.payload,
                "expires_at": self.expires_at,
                "resume_token": self.resume_token,
                "resolution": self.resolution,
            },
        )


@dataclass
class ArtifactEvent:
    """Artifact creation/update event data."""
    artifact_id: str
    artifact_type: str  # file, report, diff, image
    name: str
    mime_type: str | None = None
    size_bytes: int | None = None
    url: str | None = None
    content_preview: str | None = None

    def to_runtime_event(
        self,
        ctx: Any,
        type: RuntimeEventType = RuntimeEventType.ARTIFACT_CREATED,
    ) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            type,
            {
                "artifact_id": self.artifact_id,
                "artifact_type": self.artifact_type,
                "name": self.name,
                "mime_type": self.mime_type,
                "size_bytes": self.size_bytes,
                "url": self.url,
                "content_preview": self.content_preview,
            },
        )


@dataclass
class JobEvent:
    """Job lifecycle event data."""
    status: str
    previous_status: str | None = None
    progress: float | None = None
    error: str | None = None

    def to_runtime_event(
        self,
        ctx: Any,
        type: RuntimeEventType = RuntimeEventType.JOB_STATUS_CHANGED,
    ) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            type,
            {
                "status": self.status,
                "previous_status": self.previous_status,
                "progress": self.progress,
                "error": self.error,
            },
        )


@dataclass
class FinalEvent:
    """Final result/error event data."""
    content: str | None = None
    status: str = "success"  # success, error, cancelled, timeout
    error: str | None = None
    usage: dict[str, Any] | None = None
    turns: int | None = None

    def to_runtime_event(
        self,
        ctx: Any,
        type: RuntimeEventType = RuntimeEventType.FINAL_RESULT,
    ) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            type,
            {
                "content": self.content,
                "status": self.status,
                "error": self.error,
                "usage": self.usage,
                "turns": self.turns,
            },
        )


__all__ = [
    "RuntimeEvent",
    "RuntimeEventType",
    "ProgressEvent",
    "ModelEvent",
    "ToolEvent",
    "ActionEvent",
    "ArtifactEvent",
    "JobEvent",
    "FinalEvent",
]
