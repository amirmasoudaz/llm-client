# src/services/events.py
"""Event service for SSE streaming and webhook dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import httpx

from src.config import get_settings


class EventType(str, Enum):
    """SSE event types."""
    # Response streaming
    RESPONSE_START = "response_start"
    RESPONSE_TOKEN = "response_token"
    RESPONSE_END = "response_end"
    
    # Meta events
    META_ACTION = "meta_action"
    META_PROMPT = "meta_prompt"
    META_REDIRECT = "meta_redirect"
    META_REFRESH = "meta_refresh"
    
    # Progress events
    PROGRESS_UPDATE = "progress"
    PROGRESS_STAGE = "progress_stage"
    
    # Error events
    ERROR = "error"
    
    # Tool events
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"


class WebhookEvent(str, Enum):
    """Webhook event types for platform integration."""
    THREAD_CREATED = "thread.created"
    THREAD_COMPLETED = "thread.completed"
    THREAD_FAILED = "thread.failed"
    
    DOCUMENT_GENERATED = "document.generated"
    DOCUMENT_APPLIED = "document.applied"
    DOCUMENT_EXPORTED = "document.exported"
    
    EMAIL_GENERATED = "email.generated"
    EMAIL_APPLIED = "email.applied"
    
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    
    CREDITS_LOW = "credits.low"
    CREDITS_DEPLETED = "credits.depleted"


@dataclass
class SSEEvent:
    """Server-Sent Event structure."""
    event: str
    data: Any
    id: Optional[str] = None
    retry: Optional[int] = None
    
    def format(self) -> str:
        """Format as SSE string."""
        lines = []
        
        if self.id:
            lines.append(f"id: {self.id}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        
        lines.append(f"event: {self.event}")
        
        # Format data
        if isinstance(self.data, str):
            data_str = self.data
        else:
            data_str = json.dumps(self.data, default=str)
        
        # Handle multi-line data
        for line in data_str.split("\n"):
            lines.append(f"data: {line}")
        
        lines.append("")  # Empty line to end event
        return "\n".join(lines) + "\n"


@dataclass
class WebhookPayload:
    """Webhook payload structure."""
    event: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class SSEChannel:
    """
    SSE channel for a specific thread.
    
    Manages event streaming with multiple event types.
    """
    
    def __init__(self, thread_id: int):
        self.thread_id = thread_id
        self._queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
        self._closed = False
        self._event_id = 0
    
    def _next_id(self) -> str:
        self._event_id += 1
        return f"{self.thread_id}-{self._event_id}"
    
    async def send(
        self,
        event_type: EventType | str,
        data: Any,
        include_id: bool = True,
    ) -> None:
        """Send an event to the channel."""
        if self._closed:
            return
        
        event = SSEEvent(
            event=event_type.value if isinstance(event_type, EventType) else event_type,
            data=data,
            id=self._next_id() if include_id else None,
        )
        await self._queue.put(event)
    
    async def send_token(self, token: str) -> None:
        """Send a response token."""
        await self.send(EventType.RESPONSE_TOKEN, token, include_id=False)
    
    async def send_progress(self, percent: int, message: str = "") -> None:
        """Send a progress update."""
        await self.send(EventType.PROGRESS_UPDATE, {
            "percent": percent,
            "message": message,
        })
    
    async def send_meta(
        self,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a meta event (action, prompt, redirect, etc.)."""
        await self.send(EventType.META_ACTION, {
            "action": action,
            "payload": payload or {},
        })
    
    async def send_error(self, error: str, code: Optional[str] = None) -> None:
        """Send an error event."""
        await self.send(EventType.ERROR, {
            "error": error,
            "code": code,
        })
    
    async def close(self) -> None:
        """Close the channel."""
        self._closed = True
        # Send a None to unblock any waiting consumers
        await self._queue.put(None)  # type: ignore
    
    async def stream(self) -> AsyncGenerator[str, None]:
        """Stream events from the channel."""
        while not self._closed:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=30.0  # Send keepalive every 30s
                )
                
                if event is None:
                    break
                
                yield event.format()
                
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"


class EventService:
    """
    Event service for SSE streaming and webhook dispatch.
    
    Manages SSE channels per thread and dispatches webhooks to platform.
    """
    
    def __init__(self):
        self._channels: Dict[int, SSEChannel] = {}
        self._webhook_url: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        
        settings = get_settings()
        # TODO: Get webhook URL from settings
        # self._webhook_url = settings.platform_webhook_url
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for webhooks."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=100),
            )
        return self._http_client
    
    async def close(self) -> None:
        """Close all channels and HTTP client."""
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()
        
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    # =========================================================================
    # SSE Channel Management
    # =========================================================================
    
    def get_channel(self, thread_id: int) -> SSEChannel:
        """Get or create an SSE channel for a thread."""
        if thread_id not in self._channels:
            self._channels[thread_id] = SSEChannel(thread_id)
        return self._channels[thread_id]
    
    async def close_channel(self, thread_id: int) -> None:
        """Close an SSE channel."""
        if thread_id in self._channels:
            await self._channels[thread_id].close()
            del self._channels[thread_id]
    
    async def broadcast_to_thread(
        self,
        thread_id: int,
        event_type: EventType,
        data: Any,
    ) -> None:
        """Broadcast an event to a thread's channel."""
        if thread_id in self._channels:
            await self._channels[thread_id].send(event_type, data)
    
    # =========================================================================
    # Webhook Dispatch
    # =========================================================================
    
    async def dispatch_webhook(
        self,
        event: WebhookEvent | str,
        data: Dict[str, Any],
        retry_count: int = 3,
    ) -> bool:
        """
        Dispatch a webhook to the platform backend.
        
        Returns True if successful, False otherwise.
        """
        if not self._webhook_url:
            # Webhook not configured, skip silently
            return True
        
        event_str = event.value if isinstance(event, WebhookEvent) else event
        payload = WebhookPayload(event=event_str, data=data)
        
        client = await self._get_http_client()
        
        for attempt in range(retry_count):
            try:
                response = await client.post(
                    self._webhook_url,
                    json=payload.to_dict(),
                    headers={"Content-Type": "application/json"},
                )
                
                if response.status_code < 400:
                    return True
                
                # Log error but continue retrying
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Log final failure
                    pass
        
        return False
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    async def notify_thread_created(
        self,
        thread_id: int,
        student_id: int,
        funding_request_id: int,
    ) -> None:
        """Notify that a thread was created."""
        await self.dispatch_webhook(
            WebhookEvent.THREAD_CREATED,
            {
                "thread_id": thread_id,
                "student_id": student_id,
                "funding_request_id": funding_request_id,
            }
        )
    
    async def notify_thread_completed(
        self,
        thread_id: int,
        student_id: int,
    ) -> None:
        """Notify that a thread completed."""
        await self.dispatch_webhook(
            WebhookEvent.THREAD_COMPLETED,
            {
                "thread_id": thread_id,
                "student_id": student_id,
            }
        )
    
    async def notify_document_generated(
        self,
        document_id: int,
        student_id: int,
        thread_id: int,
        document_type: str,
    ) -> None:
        """Notify that a document was generated."""
        await self.dispatch_webhook(
            WebhookEvent.DOCUMENT_GENERATED,
            {
                "document_id": document_id,
                "student_id": student_id,
                "thread_id": thread_id,
                "document_type": document_type,
            }
        )
    
    async def notify_document_applied(
        self,
        document_id: int,
        funding_request_id: int,
        attachment_id: int,
    ) -> None:
        """Notify that a document was applied to a request."""
        await self.dispatch_webhook(
            WebhookEvent.DOCUMENT_APPLIED,
            {
                "document_id": document_id,
                "funding_request_id": funding_request_id,
                "attachment_id": attachment_id,
            }
        )
    
    async def notify_job_completed(
        self,
        job_id: int,
        job_type: str,
        student_id: int,
        thread_id: Optional[int],
    ) -> None:
        """Notify that a job completed."""
        await self.dispatch_webhook(
            WebhookEvent.JOB_COMPLETED,
            {
                "job_id": job_id,
                "job_type": job_type,
                "student_id": student_id,
                "thread_id": thread_id,
            }
        )
    
    async def notify_credits_low(
        self,
        student_id: int,
        remaining: float,
        threshold: float,
    ) -> None:
        """Notify that credits are running low."""
        await self.dispatch_webhook(
            WebhookEvent.CREDITS_LOW,
            {
                "student_id": student_id,
                "remaining": remaining,
                "threshold": threshold,
            }
        )





