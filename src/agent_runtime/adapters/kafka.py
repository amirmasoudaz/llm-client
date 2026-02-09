"""
Kafka event adapter for agent runtime.

Publishes runtime events to Apache Kafka topics for:
- Event streaming to external systems
- Analytics and monitoring pipelines
- Cross-service communication

Requires aiokafka: pip install aiokafka
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from aiokafka import AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    AIOKafkaProducer = None  # type: ignore
    KAFKA_AVAILABLE = False

from ..events.types import RuntimeEvent, RuntimeEventType
from ..events.bus import EventBus, EventSubscription
from ..events.adapters import EventAdapter


def _require_kafka() -> None:
    """Raise ImportError if aiokafka is not available."""
    if not KAFKA_AVAILABLE:
        raise ImportError(
            "Kafka adapter requires aiokafka. "
            "Install with: pip install aiokafka"
        )


@dataclass
class KafkaConfig:
    """Configuration for Kafka event adapter.
    
    Attributes:
        bootstrap_servers: Kafka broker addresses
        topic_prefix: Prefix for topic names
        default_topic: Default topic for events
        partition_by: Field to partition by (job_id, scope_id, etc.)
        acks: Producer acknowledgment level
        compression_type: Message compression
        batch_size: Batch size for sending
        linger_ms: Time to wait for batch
        key_serializer: Custom key serializer
        value_serializer: Custom value serializer
    """
    bootstrap_servers: str | list[str] = "localhost:9092"
    topic_prefix: str = "agent-runtime"
    default_topic: str = "events"
    
    # Partitioning
    partition_by: str = "job_id"
    
    # Producer settings
    acks: str = "all"  # "all", "1", "0"
    compression_type: str = "gzip"  # "gzip", "snappy", "lz4", "zstd", None
    batch_size: int = 16384
    linger_ms: int = 10
    
    # Serializers (optional)
    key_serializer: Callable[[str], bytes] | None = None
    value_serializer: Callable[[dict], bytes] | None = None
    
    # Topic mapping (event_type -> topic)
    topic_mapping: dict[str, str] = field(default_factory=dict)
    
    def get_topic(self, event_type: RuntimeEventType) -> str:
        """Get topic for an event type."""
        if event_type.value in self.topic_mapping:
            return f"{self.topic_prefix}.{self.topic_mapping[event_type.value]}"
        
        # Default: use event category
        category = event_type.value.split(".")[0]
        return f"{self.topic_prefix}.{category}"


class KafkaEventAdapter(EventAdapter):
    """Publishes runtime events to Apache Kafka.
    
    Features:
    - Automatic topic routing based on event type
    - Partitioning by job_id/scope_id for ordering
    - Async batched sending for performance
    - At-least-once delivery semantics
    
    Example:
        ```python
        config = KafkaConfig(
            bootstrap_servers="kafka:9092",
            topic_prefix="myapp.agent",
        )
        
        adapter = KafkaEventAdapter(event_bus, config)
        await adapter.start()
        
        # Events are now published to Kafka
        # Topics: myapp.agent.job, myapp.agent.tool, etc.
        
        await adapter.stop()
        ```
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: KafkaConfig | None = None,
    ):
        _require_kafka()
        
        self._event_bus = event_bus
        self._config = config or KafkaConfig()
        
        self._producer: AIOKafkaProducer | None = None
        self._subscription: EventSubscription | None = None
        self._running = False
        
        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._last_error: str | None = None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def messages_sent(self) -> int:
        return self._messages_sent
    
    @property
    def messages_failed(self) -> int:
        return self._messages_failed
    
    async def start(self) -> None:
        """Start the Kafka producer and subscribe to events."""
        if self._running:
            return
        
        # Create producer
        servers = self._config.bootstrap_servers
        if isinstance(servers, str):
            servers = servers.split(",")
        
        self._producer = AIOKafkaProducer(
            bootstrap_servers=servers,
            acks=self._config.acks,
            compression_type=self._config.compression_type,
            max_batch_size=self._config.batch_size,
            linger_ms=self._config.linger_ms,
            key_serializer=self._config.key_serializer or (lambda k: k.encode("utf-8") if k else None),
            value_serializer=self._config.value_serializer or self._default_serializer,
        )
        
        await self._producer.start()
        
        # Subscribe to events
        self._subscription = await self._event_bus.subscribe(
            self._handle_event,
            event_types=None,  # All events
        )
        
        self._running = True
    
    async def stop(self) -> None:
        """Stop the producer and unsubscribe."""
        self._running = False
        
        if self._subscription:
            await self._subscription.unsubscribe()
            self._subscription = None
        
        if self._producer:
            await self._producer.stop()
            self._producer = None
    
    async def adapt(self, event: RuntimeEvent) -> None:
        """Publish a single event to Kafka."""
        if not self._running or not self._producer:
            return
        
        await self._handle_event(event)
    
    async def _handle_event(self, event: RuntimeEvent) -> None:
        """Handle an event from the bus."""
        if not self._producer:
            return
        
        try:
            # Determine topic
            topic = self._config.get_topic(event.type)
            
            # Get partition key
            key = self._get_partition_key(event)
            
            # Serialize event
            value = event.to_dict()
            
            # Send to Kafka
            await self._producer.send_and_wait(
                topic,
                value=value,
                key=key,
            )
            
            self._messages_sent += 1
            
        except Exception as e:
            self._messages_failed += 1
            self._last_error = str(e)
            # Log but don't raise - don't break the event bus
    
    def _get_partition_key(self, event: RuntimeEvent) -> str | None:
        """Get partition key for an event."""
        if self._config.partition_by == "job_id":
            return event.job_id
        elif self._config.partition_by == "scope_id":
            return event.scope_id
        elif self._config.partition_by == "session_id":
            return event.session_id
        elif self._config.partition_by == "run_id":
            return event.run_id
        else:
            return None
    
    @staticmethod
    def _default_serializer(value: dict) -> bytes:
        """Default JSON serializer."""
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    
    async def flush(self) -> None:
        """Flush any pending messages."""
        if self._producer:
            await self._producer.flush()
    
    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "running": self._running,
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "last_error": self._last_error,
        }


__all__ = [
    "KafkaConfig",
    "KafkaEventAdapter",
]
