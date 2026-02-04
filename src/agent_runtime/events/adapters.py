"""
Event adapters for different output formats and transports.

This module provides adapters to transform runtime events for:
- Server-Sent Events (SSE)
- Webhooks
- Queue systems (Kafka, Redis Streams) - future
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING
import asyncio

from .types import RuntimeEvent, RuntimeEventType
from .bus import EventBus, EventSubscription

# Optional dependency
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False


class EventAdapter(ABC):
    """Base class for event adapters."""
    
    @abstractmethod
    async def emit(self, event: RuntimeEvent) -> None:
        """Emit a single event."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close the adapter and clean up resources."""
        ...


@dataclass
class SSEEventAdapter(EventAdapter):
    """Adapter that formats events as Server-Sent Events.
    
    Use with async generators to stream events to HTTP clients.
    
    Example (FastAPI):
        ```python
        from fastapi.responses import StreamingResponse
        
        @app.get("/jobs/{job_id}/events")
        async def stream_events(job_id: str):
            subscription = event_bus.subscribe(job_id=job_id)
            adapter = SSEEventAdapter()
            
            async def generate():
                async for event in event_bus.events(subscription):
                    yield adapter.format(event)
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        ```
    """
    
    include_event_type: bool = True
    include_id: bool = True
    retry_ms: int | None = 3000  # SSE retry hint

    async def emit(self, event: RuntimeEvent) -> None:
        """Not used for SSE - use format() and stream instead."""
        pass
    
    async def close(self) -> None:
        """No cleanup needed for SSE adapter."""
        pass
    
    def format(self, event: RuntimeEvent) -> str:
        """Format a single event as SSE."""
        lines = []
        
        if self.include_id:
            lines.append(f"id: {event.event_id}")
        
        if self.include_event_type:
            # Convert event type to SSE-safe format (no dots)
            event_type = event.type.value.replace(".", "_")
            lines.append(f"event: {event_type}")
        
        if self.retry_ms is not None and event.type == RuntimeEventType.JOB_STARTED:
            lines.append(f"retry: {self.retry_ms}")
        
        # Data is JSON-encoded event payload
        data = json.dumps(event.to_dict())
        lines.append(f"data: {data}")
        
        return "\n".join(lines) + "\n\n"
    
    async def stream(
        self,
        bus: EventBus,
        subscription: EventSubscription,
    ) -> AsyncIterator[str]:
        """Stream events from the bus as SSE."""
        async for event in bus.events(subscription):
            yield self.format(event)


@dataclass
class WebhookEventAdapter(EventAdapter):
    """Adapter that sends events to a webhook URL.
    
    Features:
    - Async HTTP POST with configurable timeout
    - Automatic retries with backoff
    - Event batching (optional)
    - Signature header for verification (optional)
    
    Requires aiohttp to be installed.
    """
    
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_base: float = 1.0
    batch_size: int = 1  # >1 enables batching
    secret: str | None = None  # For HMAC signature
    
    _session: Any = field(default=None, init=False)  # aiohttp.ClientSession
    _batch: list[RuntimeEvent] = field(default_factory=list, init=False)
    _batch_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "WebhookEventAdapter requires aiohttp. "
                "Install with: pip install aiohttp"
            )

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _sign_payload(self, payload: str) -> str | None:
        """Generate HMAC signature for webhook verification."""
        if not self.secret:
            return None
        import hmac
        import hashlib
        signature = hmac.new(
            self.secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    async def emit(self, event: RuntimeEvent) -> None:
        """Send a single event to the webhook."""
        if self.batch_size > 1:
            async with self._batch_lock:
                self._batch.append(event)
                if len(self._batch) >= self.batch_size:
                    await self._flush_batch()
        else:
            await self._send_events([event])
    
    async def _flush_batch(self) -> None:
        """Flush the current batch of events."""
        if not self._batch:
            return
        events = self._batch.copy()
        self._batch.clear()
        await self._send_events(events)
    
    async def _send_events(self, events: list[RuntimeEvent]) -> None:
        """Send events to the webhook with retries."""
        session = await self._get_session()
        
        payload = json.dumps({
            "events": [e.to_dict() for e in events],
            "count": len(events),
        })
        
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        
        signature = self._sign_payload(payload)
        if signature:
            headers["X-Webhook-Signature"] = signature
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    self.url,
                    data=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    if response.status < 400:
                        return  # Success
                    last_error = f"HTTP {response.status}"
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.backoff_base * (2 ** attempt))
        
        # Log failure (don't raise to avoid blocking event processing)
        import logging
        logging.getLogger(__name__).warning(
            f"Webhook delivery failed after {self.max_retries} attempts: {last_error}"
        )
    
    async def close(self) -> None:
        """Flush any pending events and close the session."""
        if self.batch_size > 1:
            async with self._batch_lock:
                await self._flush_batch()
        
        if self._session and not self._session.closed:
            await self._session.close()


__all__ = [
    "EventAdapter",
    "SSEEventAdapter",
    "WebhookEventAdapter",
]
