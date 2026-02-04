"""
Event bus for runtime event distribution.

This module provides the EventBus abstraction and implementations
for publishing and subscribing to runtime events.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Any
import uuid

from .types import RuntimeEvent, RuntimeEventType


@dataclass
class EventSubscription:
    """Subscription to events from the event bus."""
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str | None = None
    event_types: set[RuntimeEventType] | None = None  # None = all types
    scope_id: str | None = None  # Filter by scope
    
    def matches(self, event: RuntimeEvent) -> bool:
        """Check if an event matches this subscription."""
        if self.job_id and event.job_id != self.job_id:
            return False
        if self.scope_id and event.scope_id != self.scope_id:
            return False
        if self.event_types and event.type not in self.event_types:
            return False
        return True


class EventBus(ABC):
    """Abstract event bus for runtime events.
    
    Implementations must provide:
    - publish: Send an event to all matching subscribers
    - subscribe: Create a subscription for events
    - unsubscribe: Remove a subscription
    """
    
    @abstractmethod
    async def publish(self, event: RuntimeEvent) -> None:
        """Publish an event to all matching subscribers."""
        ...
    
    @abstractmethod
    def subscribe(
        self,
        job_id: str | None = None,
        event_types: set[RuntimeEventType] | None = None,
        scope_id: str | None = None,
    ) -> EventSubscription:
        """Create a subscription and return it."""
        ...
    
    @abstractmethod
    async def events(
        self,
        subscription: EventSubscription,
    ) -> AsyncIterator[RuntimeEvent]:
        """Iterate over events for a subscription."""
        ...
    
    @abstractmethod
    def unsubscribe(self, subscription: EventSubscription) -> None:
        """Remove a subscription."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close the event bus and clean up resources."""
        ...


class InMemoryEventBus(EventBus):
    """In-memory event bus implementation.
    
    Uses asyncio.Queue for each subscription. Suitable for single-process
    deployments and testing.
    
    Features:
    - Bounded buffers to prevent memory overflow
    - Automatic cleanup of closed subscriptions
    - Support for multiple subscribers per job
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        drop_policy: str = "oldest",  # "oldest" or "newest"
    ):
        self._queues: dict[str, asyncio.Queue[RuntimeEvent | None]] = {}
        self._subscriptions: dict[str, EventSubscription] = {}
        self._max_queue_size = max_queue_size
        self._drop_policy = drop_policy
        self._closed = False
        self._lock = asyncio.Lock()
    
    async def publish(self, event: RuntimeEvent) -> None:
        """Publish an event to all matching subscribers."""
        if self._closed:
            return
        
        async with self._lock:
            for sub_id, subscription in list(self._subscriptions.items()):
                if subscription.matches(event):
                    queue = self._queues.get(sub_id)
                    if queue:
                        try:
                            if queue.full():
                                if self._drop_policy == "oldest":
                                    # Drop oldest event
                                    try:
                                        queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        pass
                                else:
                                    # Drop this event (newest)
                                    continue
                            queue.put_nowait(event)
                        except asyncio.QueueFull:
                            pass  # Shouldn't happen with above logic
    
    def subscribe(
        self,
        job_id: str | None = None,
        event_types: set[RuntimeEventType] | None = None,
        scope_id: str | None = None,
    ) -> EventSubscription:
        """Create a subscription and return it."""
        subscription = EventSubscription(
            job_id=job_id,
            event_types=event_types,
            scope_id=scope_id,
        )
        self._subscriptions[subscription.subscription_id] = subscription
        self._queues[subscription.subscription_id] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        return subscription
    
    async def events(
        self,
        subscription: EventSubscription,
    ) -> AsyncIterator[RuntimeEvent]:
        """Iterate over events for a subscription.
        
        Yields events until the subscription is closed (receives None).
        """
        queue = self._queues.get(subscription.subscription_id)
        if not queue:
            return
        
        while True:
            try:
                event = await queue.get()
                if event is None:  # Sentinel for close
                    break
                yield event
            except asyncio.CancelledError:
                break
    
    def unsubscribe(self, subscription: EventSubscription) -> None:
        """Remove a subscription."""
        sub_id = subscription.subscription_id
        if sub_id in self._subscriptions:
            del self._subscriptions[sub_id]
        if sub_id in self._queues:
            # Send sentinel to unblock any waiting consumers
            try:
                self._queues[sub_id].put_nowait(None)
            except asyncio.QueueFull:
                pass
            del self._queues[sub_id]
    
    async def close(self) -> None:
        """Close the event bus and clean up resources."""
        self._closed = True
        async with self._lock:
            for queue in self._queues.values():
                try:
                    queue.put_nowait(None)  # Unblock consumers
                except asyncio.QueueFull:
                    pass
            self._queues.clear()
            self._subscriptions.clear()

    async def wait_for_event(
        self,
        subscription: EventSubscription,
        timeout: float | None = None,
    ) -> RuntimeEvent | None:
        """Wait for a single event with optional timeout."""
        queue = self._queues.get(subscription.subscription_id)
        if not queue:
            return None
        
        try:
            if timeout:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None


__all__ = [
    "EventBus",
    "InMemoryEventBus",
    "EventSubscription",
]
