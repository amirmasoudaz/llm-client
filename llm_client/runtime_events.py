"""
Generic runtime event model and in-memory event bus.

These primitives are designed for package-level observability, replay, and
agent-runtime style event streaming without encoding product-specific policy.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class RuntimeEventType(str, Enum):
    PROGRESS = "progress"
    MODEL_TOKEN = "model.token"
    MODEL_REASONING = "model.reasoning"
    MODEL_DONE = "model.done"
    TOOL_START = "tool.start"
    TOOL_END = "tool.end"
    TOOL_ERROR = "tool.error"
    ACTION_REQUIRED = "action.required"
    ACTION_RESOLVED = "action.resolved"
    ACTION_EXPIRED = "action.expired"
    ACTION_CANCELLED = "action.cancelled"
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_UPDATED = "artifact.updated"
    JOB_STARTED = "job.started"
    JOB_STATUS_CHANGED = "job.status_changed"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    FINAL_RESULT = "final.result"
    FINAL_ERROR = "final.error"


@dataclass
class RuntimeEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: RuntimeEventType = RuntimeEventType.PROGRESS
    timestamp: float = field(default_factory=time.time)
    job_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
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
        event_type = self.type.value.replace(".", "_")
        return f"event: {event_type}\ndata: {json.dumps(self.to_dict())}\n\n"

    @classmethod
    def from_context(
        cls,
        ctx: Any,
        type: RuntimeEventType,
        data: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            type=type,
            job_id=getattr(ctx, "job_id", None),
            run_id=getattr(ctx, "run_id", None),
            trace_id=getattr(ctx, "trace_id", None),
            span_id=getattr(ctx, "span_id", None),
            scope_id=getattr(ctx, "scope_id", None),
            principal_id=getattr(ctx, "principal_id", None),
            session_id=getattr(ctx, "session_id", None),
            data=data or {},
        )


@dataclass
class ProgressEvent:
    progress: float
    message: str | None = None
    turn: int | None = None
    step: str | None = None

    def to_runtime_event(self, ctx: Any) -> RuntimeEvent:
        return RuntimeEvent.from_context(
            ctx,
            RuntimeEventType.PROGRESS,
            {"progress": self.progress, "message": self.message, "turn": self.turn, "step": self.step},
        )


@dataclass
class ModelEvent:
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
    action_id: str
    action_type: str
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
    artifact_id: str
    artifact_type: str
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
    content: str | None = None
    status: str = "success"
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


@dataclass
class EventSubscription:
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str | None = None
    event_types: set[RuntimeEventType] | None = None
    scope_id: str | None = None

    def matches(self, event: RuntimeEvent) -> bool:
        if self.job_id and event.job_id != self.job_id:
            return False
        if self.scope_id and event.scope_id != self.scope_id:
            return False
        if self.event_types and event.type not in self.event_types:
            return False
        return True


class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: RuntimeEvent) -> None: ...

    @abstractmethod
    def subscribe(
        self,
        job_id: str | None = None,
        event_types: set[RuntimeEventType] | None = None,
        scope_id: str | None = None,
    ) -> EventSubscription: ...

    @abstractmethod
    async def events(self, subscription: EventSubscription) -> AsyncIterator[RuntimeEvent]: ...

    @abstractmethod
    def unsubscribe(self, subscription: EventSubscription) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class InMemoryEventBus(EventBus):
    def __init__(
        self,
        max_queue_size: int = 1000,
        drop_policy: str = "oldest",
    ) -> None:
        self._queues: dict[str, asyncio.Queue[RuntimeEvent | None]] = {}
        self._subscriptions: dict[str, EventSubscription] = {}
        self._max_queue_size = max_queue_size
        self._drop_policy = drop_policy
        self._closed = False
        self._lock = asyncio.Lock()

    async def publish(self, event: RuntimeEvent) -> None:
        if self._closed:
            return
        async with self._lock:
            for sub_id, subscription in list(self._subscriptions.items()):
                if not subscription.matches(event):
                    continue
                queue = self._queues.get(sub_id)
                if queue is None:
                    continue
                try:
                    if queue.full():
                        if self._drop_policy == "oldest":
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        else:
                            continue
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def subscribe(
        self,
        job_id: str | None = None,
        event_types: set[RuntimeEventType] | None = None,
        scope_id: str | None = None,
    ) -> EventSubscription:
        subscription = EventSubscription(job_id=job_id, event_types=event_types, scope_id=scope_id)
        self._subscriptions[subscription.subscription_id] = subscription
        self._queues[subscription.subscription_id] = asyncio.Queue(maxsize=self._max_queue_size)
        return subscription

    async def events(self, subscription: EventSubscription) -> AsyncIterator[RuntimeEvent]:
        queue = self._queues.get(subscription.subscription_id)
        if queue is None:
            return
        while True:
            try:
                event = await queue.get()
            except asyncio.CancelledError:
                break
            if event is None:
                break
            yield event

    def unsubscribe(self, subscription: EventSubscription) -> None:
        sub_id = subscription.subscription_id
        if sub_id in self._subscriptions:
            del self._subscriptions[sub_id]
        if sub_id in self._queues:
            try:
                self._queues[sub_id].put_nowait(None)
            except asyncio.QueueFull:
                pass
            del self._queues[sub_id]

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            for queue in self._queues.values():
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            self._queues.clear()
            self._subscriptions.clear()

    async def wait_for_event(
        self,
        subscription: EventSubscription,
        timeout: float | None = None,
    ) -> RuntimeEvent | None:
        queue = self._queues.get(subscription.subscription_id)
        if queue is None:
            return None
        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None


__all__ = [
    "ActionEvent",
    "ArtifactEvent",
    "EventBus",
    "EventSubscription",
    "FinalEvent",
    "InMemoryEventBus",
    "JobEvent",
    "ModelEvent",
    "ProgressEvent",
    "RuntimeEvent",
    "RuntimeEventType",
    "ToolEvent",
]
