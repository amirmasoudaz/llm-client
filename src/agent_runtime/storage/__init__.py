"""
Storage adapters for agent runtime.

This module provides persistent storage implementations for:
- Jobs (PostgresJobStore)
- Actions (PostgresActionStore)
- Ledger events (PostgresLedgerWriter)

And Redis-based components:
- RedisSignalChannel: Fast action wait/resume
- RedisJobCache: Read-through cache
- RedisActionStore: Redis-backed action store
"""

from .postgres import (
    PostgresJobStore,
    PostgresActionStore,
    PostgresLedgerWriter,
)

# Redis components (optional)
try:
    from .redis import (
        SignalMessage,
        RedisSignalChannel,
        RedisJobCache,
        RedisActionStore,
    )
    _REDIS_EXPORTS = [
        "SignalMessage",
        "RedisSignalChannel",
        "RedisJobCache",
        "RedisActionStore",
    ]
except ImportError:
    _REDIS_EXPORTS = []

__all__ = [
    "PostgresJobStore",
    "PostgresActionStore",
    "PostgresLedgerWriter",
    *_REDIS_EXPORTS,
]
