"""
Redis storage and signaling for agent runtime.

This module provides:
- RedisSignalChannel: Fast action wait/resume using pub/sub
- RedisJobCache: Optional fast read cache for jobs
- RedisActionStore: Redis-backed action store (NOT recommended as primary)

Requires redis (async): pip install redis

IMPORTANT: Durability and Persistence
-------------------------------------
Redis is designed for SPEED, not DURABILITY. By default:
- Data is stored in memory (can be lost on restart)
- TTLs cause automatic expiration (data disappears)
- Redis Cluster failover may lose recent writes
- AOF/RDB persistence has consistency tradeoffs

Recommended Architecture
------------------------
For production systems, use a TWO-TIER pattern:

    ┌─────────────┐      ┌─────────────┐
    │   Redis     │◄────►│  PostgreSQL │
    │  (fast)     │      │  (durable)  │
    └─────────────┘      └─────────────┘
         │                      │
         ▼                      ▼
    Signaling,             Source of Truth,
    Caching,               Audit Trail,
    Rate Limiting          Recovery

Use Cases:
- RedisSignalChannel: YES - signaling is ephemeral by nature
- RedisJobCache: YES - it's explicitly a cache with primary fallback
- RedisActionStore: CAUTION - only for short-lived, recoverable actions

If an action is critical (approvals, payments, compliance):
- Use PostgresActionStore as primary
- Use RedisSignalChannel for fast wakeups
- DO NOT rely on RedisActionStore as the source of truth

TTL Warning
-----------
RedisActionStore uses TTLs (default 24 hours). Actions that exceed
the TTL will disappear WITHOUT triggering expiration callbacks.
For actions that must survive longer or have reliable expiration
handling, use PostgresActionStore.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

from ..actions.types import ActionRecord, ActionStatus
from ..actions.store import ActionStore, ActionFilter


def _require_redis() -> None:
    """Raise ImportError if redis is not available."""
    if not REDIS_AVAILABLE:
        raise ImportError(
            "Redis storage requires redis. "
            "Install with: pip install redis"
        )


@dataclass
class SignalMessage:
    """A signal message for action resolution."""
    action_id: str
    resume_token: str
    status: str  # resolved, cancelled, expired
    resolution: dict[str, Any] | None = None
    error: str | None = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps({
            "action_id": self.action_id,
            "resume_token": self.resume_token,
            "status": self.status,
            "resolution": self.resolution,
            "error": self.error,
            "timestamp": self.timestamp,
        })
    
    @classmethod
    def from_json(cls, data: str) -> SignalMessage:
        parsed = json.loads(data)
        return cls(
            action_id=parsed["action_id"],
            resume_token=parsed["resume_token"],
            status=parsed["status"],
            resolution=parsed.get("resolution"),
            error=parsed.get("error"),
            timestamp=parsed.get("timestamp", time.time()),
        )


class RedisSignalChannel:
    """Fast action signaling using Redis pub/sub.
    
    Provides:
    - Low-latency action resolution notifications
    - Reduces polling overhead for action.wait()
    - Supports multiple waiters per action
    
    Channel naming:
    - action:{action_id} - Per-action channel
    - action:job:{job_id} - Per-job channel for all actions
    
    Example:
        ```python
        signal = RedisSignalChannel(redis_client)
        
        # In the action waiter
        async def wait_for_action(action_id: str) -> SignalMessage:
            return await signal.wait(action_id, timeout=300)
        
        # In the action resolver
        async def resolve_action(action_id: str, resolution: dict):
            await signal.signal(action_id, SignalMessage(
                action_id=action_id,
                resume_token="...",
                status="resolved",
                resolution=resolution,
            ))
        ```
    """
    
    def __init__(
        self,
        client: Any,  # redis.Redis
        channel_prefix: str = "agent:action",
        message_ttl_seconds: int = 3600,
    ):
        _require_redis()
        self._client = client
        self._prefix = channel_prefix
        self._ttl = message_ttl_seconds
        
        # Track active subscriptions
        self._waiters: dict[str, list[asyncio.Future]] = {}
        self._pubsub: Any = None  # redis.PubSub
        self._listener_task: asyncio.Task | None = None
    
    def _channel_name(self, action_id: str) -> str:
        """Get channel name for an action."""
        return f"{self._prefix}:{action_id}"
    
    def _job_channel_name(self, job_id: str) -> str:
        """Get channel name for all actions in a job."""
        return f"{self._prefix}:job:{job_id}"
    
    async def start(self) -> None:
        """Start the listener for incoming signals."""
        if self._listener_task is not None:
            return
        
        self._pubsub = self._client.pubsub()
        self._listener_task = asyncio.create_task(self._listen())
    
    async def stop(self) -> None:
        """Stop the listener."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None
        
        # Cancel any pending waiters
        for waiters in self._waiters.values():
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
        self._waiters.clear()
    
    async def _listen(self) -> None:
        """Background listener for pub/sub messages."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        signal = SignalMessage.from_json(message["data"])
                        await self._dispatch_signal(signal)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except asyncio.CancelledError:
            pass
    
    async def _dispatch_signal(self, signal: SignalMessage) -> None:
        """Dispatch a signal to waiting futures."""
        waiters = self._waiters.get(signal.action_id, [])
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(signal)
        
        # Clear waiters for this action
        self._waiters.pop(signal.action_id, None)
    
    async def signal(
        self,
        action_id: str,
        message: SignalMessage,
        job_id: str | None = None,
    ) -> int:
        """Send a signal for an action.
        
        Args:
            action_id: The action ID
            message: Signal message to send
            job_id: Optional job ID for job-level broadcast
        
        Returns:
            Number of subscribers that received the message
        """
        # Publish to action channel
        channel = self._channel_name(action_id)
        count = await self._client.publish(channel, message.to_json())
        
        # Also publish to job channel if provided
        if job_id:
            job_channel = self._job_channel_name(job_id)
            await self._client.publish(job_channel, message.to_json())
        
        # Store message for late subscribers (with TTL)
        key = f"{self._prefix}:msg:{action_id}"
        await self._client.setex(key, self._ttl, message.to_json())
        
        return count
    
    async def wait(
        self,
        action_id: str,
        timeout: float | None = None,
    ) -> SignalMessage | None:
        """Wait for a signal for an action.
        
        Args:
            action_id: The action ID to wait for
            timeout: Maximum time to wait (None = wait forever)
        
        Returns:
            SignalMessage if received, None if timed out
        """
        # Check for existing message first
        key = f"{self._prefix}:msg:{action_id}"
        existing = await self._client.get(key)
        if existing:
            return SignalMessage.from_json(existing)
        
        # Subscribe to channel
        channel = self._channel_name(action_id)
        await self._pubsub.subscribe(channel)
        
        # Create waiter future
        waiter: asyncio.Future[SignalMessage] = asyncio.get_event_loop().create_future()
        
        if action_id not in self._waiters:
            self._waiters[action_id] = []
        self._waiters[action_id].append(waiter)
        
        try:
            if timeout:
                return await asyncio.wait_for(waiter, timeout=timeout)
            else:
                return await waiter
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None
        finally:
            # Unsubscribe
            await self._pubsub.unsubscribe(channel)
            
            # Remove waiter
            if action_id in self._waiters:
                try:
                    self._waiters[action_id].remove(waiter)
                except ValueError:
                    pass
                if not self._waiters[action_id]:
                    del self._waiters[action_id]
    
    async def wait_any(
        self,
        action_ids: list[str],
        timeout: float | None = None,
    ) -> SignalMessage | None:
        """Wait for a signal from any of the given actions.
        
        Returns the first signal received.
        """
        # Check for existing messages first
        for action_id in action_ids:
            key = f"{self._prefix}:msg:{action_id}"
            existing = await self._client.get(key)
            if existing:
                return SignalMessage.from_json(existing)
        
        # Subscribe to all channels
        channels = [self._channel_name(aid) for aid in action_ids]
        for channel in channels:
            await self._pubsub.subscribe(channel)
        
        # Create waiters
        waiters = []
        for action_id in action_ids:
            waiter: asyncio.Future[SignalMessage] = asyncio.get_event_loop().create_future()
            if action_id not in self._waiters:
                self._waiters[action_id] = []
            self._waiters[action_id].append(waiter)
            waiters.append(waiter)
        
        try:
            done, pending = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            # Cancel pending
            for p in pending:
                p.cancel()
            
            # Return first completed
            if done:
                return list(done)[0].result()
            return None
        finally:
            # Unsubscribe and cleanup
            for channel in channels:
                await self._pubsub.unsubscribe(channel)
            
            for action_id in action_ids:
                if action_id in self._waiters:
                    self._waiters[action_id] = [
                        w for w in self._waiters[action_id]
                        if w not in waiters
                    ]
                    if not self._waiters[action_id]:
                        del self._waiters[action_id]


class RedisJobCache:
    """Fast read cache for jobs using Redis.
    
    Provides a read-through cache in front of the primary JobStore.
    Useful for high-read workloads where job status is frequently polled.
    
    Example:
        ```python
        cache = RedisJobCache(redis_client, primary_store)
        
        # Fast reads go through cache
        job = await cache.get(job_id)  # Hits Redis first
        
        # Writes go through to primary and invalidate cache
        await cache.update(job)
        ```
    """
    
    def __init__(
        self,
        client: Any,  # redis.Redis
        primary_store: Any,  # JobStore
        key_prefix: str = "agent:job",
        ttl_seconds: int = 300,
    ):
        _require_redis()
        self._client = client
        self._primary = primary_store
        self._prefix = key_prefix
        self._ttl = ttl_seconds
    
    def _cache_key(self, job_id: str) -> str:
        return f"{self._prefix}:{job_id}"
    
    async def get(self, job_id: str) -> Any | None:
        """Get job, trying cache first."""
        key = self._cache_key(job_id)
        
        # Try cache
        cached = await self._client.get(key)
        if cached:
            from ..jobs.types import JobRecord
            return JobRecord.from_dict(json.loads(cached))
        
        # Fall through to primary
        job = await self._primary.get(job_id)
        if job:
            # Cache it
            await self._client.setex(key, self._ttl, json.dumps(job.to_dict()))
        
        return job
    
    async def invalidate(self, job_id: str) -> None:
        """Invalidate cache for a job."""
        key = self._cache_key(job_id)
        await self._client.delete(key)
    
    async def warm(self, job_ids: list[str]) -> None:
        """Pre-warm cache for multiple jobs."""
        for job_id in job_ids:
            job = await self._primary.get(job_id)
            if job:
                key = self._cache_key(job_id)
                await self._client.setex(key, self._ttl, json.dumps(job.to_dict()))


class RedisActionStore(ActionStore):
    """Redis-backed action store.
    
    WARNING: NOT RECOMMENDED AS PRIMARY/SOLE STORE
    
    Redis is suitable for:
    - Short-lived actions (< TTL, typically 24h)
    - Recoverable actions (can be recreated if lost)
    - High-throughput scenarios where some loss is acceptable
    
    Redis is NOT suitable for:
    - Critical actions (approvals, payments, compliance)
    - Long-running actions (multi-day workflows)
    - Audit trails (actions may expire or be lost)
    
    Recommended pattern for critical actions:
        # Primary: durable storage
        postgres_store = PostgresActionStore(pool)
        
        # Secondary: fast signaling (NOT storage)
        signal = RedisSignalChannel(redis_client)
        
        # Create action in Postgres
        action = await postgres_store.create(action_spec)
        
        # Signal resolution via Redis (fast wakeup)
        await signal.signal(action.action_id, message)
    
    If you MUST use RedisActionStore as primary:
    - Set appropriate TTLs for your use case
    - Implement external backup/recovery
    - Accept that actions may be lost on Redis issues
    - Do NOT use for compliance-sensitive workflows
    """
    
    def __init__(
        self,
        client: Any,  # redis.Redis
        key_prefix: str = "agent:action",
        default_ttl_seconds: int = 86400,  # 24 hours
    ):
        _require_redis()
        self._client = client
        self._prefix = key_prefix
        self._ttl = default_ttl_seconds
    
    def _action_key(self, action_id: str) -> str:
        return f"{self._prefix}:{action_id}"
    
    def _resume_key(self, resume_token: str) -> str:
        return f"{self._prefix}:resume:{resume_token}"
    
    def _job_index_key(self, job_id: str) -> str:
        return f"{self._prefix}:job:{job_id}"
    
    async def create(self, action: ActionRecord) -> ActionRecord:
        key = self._action_key(action.action_id)
        
        # Check if exists
        if await self._client.exists(key):
            raise ValueError(f"Action {action.action_id} already exists")
        
        # Store action
        action_json = json.dumps(action.to_dict())
        await self._client.setex(key, self._ttl, action_json)
        
        # Store resume token index
        resume_key = self._resume_key(action.resume_token)
        await self._client.setex(resume_key, self._ttl, action.action_id)
        
        # Add to job index
        job_key = self._job_index_key(action.job_id)
        await self._client.sadd(job_key, action.action_id)
        await self._client.expire(job_key, self._ttl)
        
        return action
    
    async def get(self, action_id: str) -> ActionRecord | None:
        key = self._action_key(action_id)
        data = await self._client.get(key)
        if data:
            return ActionRecord.from_dict(json.loads(data))
        return None
    
    async def get_by_resume_token(self, resume_token: str) -> ActionRecord | None:
        resume_key = self._resume_key(resume_token)
        action_id = await self._client.get(resume_key)
        if action_id:
            return await self.get(action_id)
        return None
    
    async def update(self, action: ActionRecord) -> ActionRecord:
        key = self._action_key(action.action_id)
        
        if not await self._client.exists(key):
            raise ValueError(f"Action {action.action_id} not found")
        
        action_json = json.dumps(action.to_dict())
        await self._client.setex(key, self._ttl, action_json)
        
        return action
    
    async def delete(self, action_id: str) -> bool:
        action = await self.get(action_id)
        if not action:
            return False
        
        key = self._action_key(action_id)
        resume_key = self._resume_key(action.resume_token)
        job_key = self._job_index_key(action.job_id)
        
        await self._client.delete(key)
        await self._client.delete(resume_key)
        await self._client.srem(job_key, action_id)
        
        return True
    
    async def list(self, filter: ActionFilter | None = None) -> list[ActionRecord]:
        # For Redis, we need to iterate - not ideal for large datasets
        # In production, consider using a proper database for listing
        pattern = f"{self._prefix}:????????-????-????-????-????????????"
        
        results = []
        async for key in self._client.scan_iter(match=pattern):
            data = await self._client.get(key)
            if data:
                action = ActionRecord.from_dict(json.loads(data))
                if filter is None or filter.matches(action):
                    results.append(action)
        
        # Apply pagination
        if filter:
            results = results[filter.offset:filter.offset + filter.limit]
        
        return results
    
    async def list_pending_for_job(self, job_id: str) -> list[ActionRecord]:
        job_key = self._job_index_key(job_id)
        action_ids = await self._client.smembers(job_key)
        
        results = []
        for action_id in action_ids:
            action = await self.get(action_id)
            if action and action.status == ActionStatus.PENDING:
                results.append(action)
        
        return results
    
    async def list_expired(self) -> list[ActionRecord]:
        # This is expensive in Redis - consider using sorted sets with expiry times
        now = time.time()
        results = []
        
        pattern = f"{self._prefix}:????????-????-????-????-????????????"
        async for key in self._client.scan_iter(match=pattern):
            data = await self._client.get(key)
            if data:
                action = ActionRecord.from_dict(json.loads(data))
                if (
                    action.status == ActionStatus.PENDING
                    and action.expires_at is not None
                    and action.expires_at < now
                ):
                    results.append(action)
        
        return results


__all__ = [
    "SignalMessage",
    "RedisSignalChannel",
    "RedisJobCache",
    "RedisActionStore",
]
