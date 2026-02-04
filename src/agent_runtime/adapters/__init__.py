"""
Queue adapters for agent runtime event publishing.

This module provides:
- KafkaEventAdapter: Publish events to Apache Kafka
- RedisStreamsAdapter: Publish events to Redis Streams
"""

# Kafka adapter (optional - requires aiokafka)
try:
    from .kafka import KafkaEventAdapter, KafkaConfig
    _KAFKA_EXPORTS = ["KafkaEventAdapter", "KafkaConfig"]
except ImportError:
    _KAFKA_EXPORTS = []

# Redis Streams adapter (optional - requires redis)
try:
    from .redis_streams import RedisStreamsAdapter, RedisStreamsConfig
    _REDIS_EXPORTS = ["RedisStreamsAdapter", "RedisStreamsConfig"]
except ImportError:
    _REDIS_EXPORTS = []

__all__ = [
    *_KAFKA_EXPORTS,
    *_REDIS_EXPORTS,
]
