"""
Redis Streams event adapter for agent runtime.

Publishes runtime events to Redis Streams for:
- Real-time event streaming
- Consumer group processing
- Event replay and catchup

Requires redis (async): pip install redis
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

from ..events.types import RuntimeEvent, RuntimeEventType
from ..events.bus import EventBus, EventSubscription
from ..events.adapters import EventAdapter


def _require_redis() -> None:
    """Raise ImportError if redis is not available."""
    if not REDIS_AVAILABLE:
        raise ImportError(
            "Redis Streams adapter requires redis. "
            "Install with: pip install redis"
        )


@dataclass
class RedisStreamsConfig:
    """Configuration for Redis Streams adapter.
    
    Attributes:
        stream_prefix: Prefix for stream names
        default_stream: Default stream name
        max_len: Maximum stream length (trimming)
        approximate_trim: Use approximate trimming for performance
        consumer_group: Consumer group name (for reading)
        consumer_name: Consumer name (for reading)
        block_ms: Block time for reading
    """
    stream_prefix: str = "agent:events"
    default_stream: str = "all"
    
    # Stream limits
    max_len: int | None = 100000
    approximate_trim: bool = True
    
    # Consumer settings
    consumer_group: str = "default"
    consumer_name: str = "worker-1"
    block_ms: int = 5000
    
    # Stream mapping (event_type -> stream)
    stream_mapping: dict[str, str] = field(default_factory=dict)
    
    def get_stream(self, event_type: RuntimeEventType) -> str:
        """Get stream name for an event type."""
        if event_type.value in self.stream_mapping:
            return f"{self.stream_prefix}:{self.stream_mapping[event_type.value]}"
        
        # Default: use event category
        category = event_type.value.split(".")[0]
        return f"{self.stream_prefix}:{category}"


class RedisStreamsAdapter(EventAdapter):
    """Publishes runtime events to Redis Streams.
    
    Features:
    - Automatic stream routing based on event type
    - Consumer group support for distributed processing
    - Stream trimming for memory management
    - Event replay from stream history
    
    Example:
        ```python
        client = redis.from_url("redis://localhost")
        config = RedisStreamsConfig(
            stream_prefix="myapp:events",
        )
        
        adapter = RedisStreamsAdapter(client, event_bus, config)
        await adapter.start()
        
        # Events are now published to Redis Streams
        # Streams: myapp:events:job, myapp:events:tool, etc.
        
        # Read events from stream
        async for event in adapter.read_events("myapp:events:job"):
            print(event)
        
        await adapter.stop()
        ```
    """
    
    def __init__(
        self,
        client: Any,  # redis.Redis
        event_bus: EventBus | None = None,
        config: RedisStreamsConfig | None = None,
    ):
        _require_redis()
        
        self._client = client
        self._event_bus = event_bus
        self._config = config or RedisStreamsConfig()
        
        self._subscription: EventSubscription | None = None
        self._running = False
        
        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._last_error: str | None = None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    async def start(self) -> None:
        """Start publishing events."""
        if self._running:
            return
        
        # Subscribe to events if event bus provided
        if self._event_bus:
            self._subscription = await self._event_bus.subscribe(
                self._handle_event,
                event_types=None,
            )
        
        self._running = True
    
    async def stop(self) -> None:
        """Stop publishing."""
        self._running = False
        
        if self._subscription:
            await self._subscription.unsubscribe()
            self._subscription = None
    
    async def adapt(self, event: RuntimeEvent) -> None:
        """Publish a single event to Redis Streams."""
        await self._handle_event(event)
    
    async def _handle_event(self, event: RuntimeEvent) -> None:
        """Handle an event from the bus."""
        try:
            # Determine stream
            stream = self._config.get_stream(event.type)
            
            # Prepare message
            message = {
                "event_id": event.event_id,
                "type": event.type.value,
                "timestamp": str(event.timestamp),
                "job_id": event.job_id or "",
                "run_id": event.run_id or "",
                "scope_id": event.scope_id or "",
                "principal_id": event.principal_id or "",
                "data": json.dumps(event.data),
            }
            
            # Add to stream
            kwargs: dict[str, Any] = {}
            if self._config.max_len:
                kwargs["maxlen"] = self._config.max_len
                kwargs["approximate"] = self._config.approximate_trim
            
            await self._client.xadd(stream, message, **kwargs)
            
            self._messages_sent += 1
            
        except Exception as e:
            self._messages_failed += 1
            self._last_error = str(e)
    
    async def publish(self, event: RuntimeEvent) -> str | None:
        """Publish an event and return the message ID."""
        try:
            stream = self._config.get_stream(event.type)
            
            message = {
                "event_id": event.event_id,
                "type": event.type.value,
                "timestamp": str(event.timestamp),
                "job_id": event.job_id or "",
                "run_id": event.run_id or "",
                "scope_id": event.scope_id or "",
                "principal_id": event.principal_id or "",
                "data": json.dumps(event.data),
            }
            
            kwargs: dict[str, Any] = {}
            if self._config.max_len:
                kwargs["maxlen"] = self._config.max_len
                kwargs["approximate"] = self._config.approximate_trim
            
            message_id = await self._client.xadd(stream, message, **kwargs)
            self._messages_sent += 1
            return message_id
            
        except Exception as e:
            self._messages_failed += 1
            self._last_error = str(e)
            return None
    
    async def read_events(
        self,
        stream: str,
        count: int = 100,
        block_ms: int | None = None,
        start_id: str = ">",
        use_consumer_group: bool = True,
    ) -> AsyncIterator[RuntimeEvent]:
        """Read events from a stream.
        
        Args:
            stream: Stream name to read from
            count: Maximum events to read per batch
            block_ms: Block time (None = no blocking)
            start_id: Starting message ID (">" for new, "0" for all)
            use_consumer_group: Whether to use consumer groups
        
        Yields:
            RuntimeEvent objects
        """
        if use_consumer_group:
            # Ensure consumer group exists
            try:
                await self._client.xgroup_create(
                    stream,
                    self._config.consumer_group,
                    id="0",
                    mkstream=True,
                )
            except Exception:
                # Group may already exist
                pass
            
            # Read with consumer group
            while self._running:
                messages = await self._client.xreadgroup(
                    self._config.consumer_group,
                    self._config.consumer_name,
                    {stream: start_id},
                    count=count,
                    block=block_ms or self._config.block_ms,
                )
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        event = self._message_to_event(data)
                        if event:
                            yield event
                            # Acknowledge message
                            await self._client.xack(
                                stream,
                                self._config.consumer_group,
                                message_id,
                            )
        else:
            # Simple read without consumer groups
            last_id = start_id if start_id != ">" else "0"
            
            while self._running:
                messages = await self._client.xread(
                    {stream: last_id},
                    count=count,
                    block=block_ms,
                )
                
                if not messages:
                    if block_ms is None:
                        break
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        last_id = message_id
                        event = self._message_to_event(data)
                        if event:
                            yield event
    
    async def read_range(
        self,
        stream: str,
        start: str = "-",
        end: str = "+",
        count: int | None = None,
    ) -> list[RuntimeEvent]:
        """Read a range of events from a stream.
        
        Args:
            stream: Stream name
            start: Start ID ("-" for oldest)
            end: End ID ("+" for newest)
            count: Maximum events to return
        
        Returns:
            List of RuntimeEvent objects
        """
        messages = await self._client.xrange(
            stream,
            min=start,
            max=end,
            count=count,
        )
        
        events = []
        for message_id, data in messages:
            event = self._message_to_event(data)
            if event:
                events.append(event)
        
        return events
    
    async def get_stream_info(self, stream: str) -> dict[str, Any]:
        """Get information about a stream."""
        try:
            info = await self._client.xinfo_stream(stream)
            return dict(info)
        except Exception:
            return {}
    
    async def get_pending_count(self, stream: str) -> int:
        """Get count of pending messages in consumer group."""
        try:
            pending = await self._client.xpending(
                stream,
                self._config.consumer_group,
            )
            return pending.get("pending", 0) if pending else 0
        except Exception:
            return 0
    
    async def trim_stream(
        self,
        stream: str,
        max_len: int | None = None,
        approximate: bool = True,
    ) -> int:
        """Trim a stream to maximum length.
        
        Returns:
            Number of messages deleted
        """
        length = max_len or self._config.max_len or 100000
        
        if approximate:
            return await self._client.xtrim(stream, maxlen=length, approximate=True)
        else:
            return await self._client.xtrim(stream, maxlen=length)
    
    def _message_to_event(self, data: dict[bytes, bytes]) -> RuntimeEvent | None:
        """Convert Redis message to RuntimeEvent."""
        try:
            # Decode bytes to strings
            decoded = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in data.items()
            }
            
            # Parse event data
            event_data = json.loads(decoded.get("data", "{}"))
            
            return RuntimeEvent(
                event_id=decoded.get("event_id", ""),
                type=RuntimeEventType(decoded.get("type", "progress")),
                timestamp=float(decoded.get("timestamp", time.time())),
                job_id=decoded.get("job_id") or None,
                run_id=decoded.get("run_id") or None,
                scope_id=decoded.get("scope_id") or None,
                principal_id=decoded.get("principal_id") or None,
                data=event_data,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "running": self._running,
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "last_error": self._last_error,
        }


__all__ = [
    "RedisStreamsConfig",
    "RedisStreamsAdapter",
]
