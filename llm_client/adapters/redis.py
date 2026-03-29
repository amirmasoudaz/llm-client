"""
Generic and backend-specific Redis adaptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .base import (
    AdaptorCapability,
    AdaptorError,
    AdaptorExecutionOptions,
    AdaptorMetadata,
    AdaptorOperation,
    AdaptorRuntime,
    await_adaptor_timeout,
    run_adaptor_operation,
)


@dataclass(frozen=True)
class RedisGetRequest:
    key: str
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedisGetResult:
    key: str
    value: str | bytes | None
    found: bool
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(backend="redis", operation=AdaptorOperation.GET)
    )


@dataclass(frozen=True)
class RedisSetRequest:
    key: str
    value: str | bytes
    ttl_seconds: int | None = None
    nx: bool = False
    xx: bool = False
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedisSetResult:
    key: str
    written: bool
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="redis",
            operation=AdaptorOperation.SET,
            read_only=False,
        )
    )


@dataclass(frozen=True)
class RedisDeleteRequest:
    key: str
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedisDeleteResult:
    key: str
    deleted_count: int
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="redis",
            operation=AdaptorOperation.DELETE,
            read_only=False,
        )
    )


@dataclass(frozen=True)
class RedisHashGetRequest:
    key: str
    field_name: str
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedisHashGetResult:
    key: str
    field_name: str
    value: str | bytes | None
    found: bool
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(backend="redis", operation=AdaptorOperation.HASH_GET)
    )


@dataclass(frozen=True)
class RedisHashSetRequest:
    key: str
    field_name: str
    value: str | bytes
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedisHashSetResult:
    key: str
    field_name: str
    written: bool
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="redis",
            operation=AdaptorOperation.HASH_SET,
            read_only=False,
        )
    )


@dataclass(frozen=True)
class RedisHashDeleteRequest:
    key: str
    field_name: str
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedisHashDeleteResult:
    key: str
    field_name: str
    deleted_count: int
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="redis",
            operation=AdaptorOperation.HASH_DELETE,
            read_only=False,
        )
    )


class RedisAdaptor(Protocol):
    backend_name: str
    key_prefix: str | None

    async def get(self, request: RedisGetRequest) -> RedisGetResult:
        """Fetch one value by key."""

    async def set(self, request: RedisSetRequest) -> RedisSetResult:
        """Write one value by key."""

    async def delete(self, request: RedisDeleteRequest) -> RedisDeleteResult:
        """Delete one key."""

    async def hget(self, request: RedisHashGetRequest) -> RedisHashGetResult:
        """Fetch one hash field."""

    async def hset(self, request: RedisHashSetRequest) -> RedisHashSetResult:
        """Write one hash field."""

    async def hdel(self, request: RedisHashDeleteRequest) -> RedisHashDeleteResult:
        """Delete one hash field."""


class RedisSafetyError(AdaptorError):
    """Raised when a Redis operation violates adaptor safety policy."""


def _coerce_bytes_length(value: str | bytes) -> int:
    if isinstance(value, bytes):
        return len(value)
    return len(value.encode("utf-8"))


def _redis_metadata(
    *,
    operation: AdaptorOperation,
    read_only: bool,
    extra: dict[str, Any] | None = None,
) -> AdaptorMetadata:
    capabilities = (AdaptorCapability.READ,) if read_only else (AdaptorCapability.READ, AdaptorCapability.WRITE)
    return AdaptorMetadata(
        backend="redis",
        operation=operation,
        read_only=read_only,
        capabilities=capabilities,
        metadata=dict(extra or {}),
    )


@dataclass
class RedisKVAdaptor:
    client: Any
    key_prefix: str | None = None
    read_only: bool = False
    allow_delete: bool = False
    default_ttl_seconds: int | None = None
    max_value_bytes: int = 65536
    default_timeout_seconds: float | None = None
    backend_name: str = "redis"
    runtime: AdaptorRuntime = field(default_factory=AdaptorRuntime)

    def _prefixed_key(self, key: str) -> str:
        if not key:
            raise RedisSafetyError("Redis key cannot be empty", backend=self.backend_name)
        if self.key_prefix is None or key.startswith(f"{self.key_prefix}:"):
            return key
        return f"{self.key_prefix}:{key}"

    def _effective_timeout(self, options: AdaptorExecutionOptions) -> float | None:
        if options.timeout_seconds is not None:
            return options.timeout_seconds
        return self.default_timeout_seconds

    def _validate_value_size(self, value: str | bytes) -> None:
        size_bytes = _coerce_bytes_length(value)
        if size_bytes > self.max_value_bytes:
            raise RedisSafetyError(
                f"Redis value exceeds max_value_bytes={self.max_value_bytes}",
                backend=self.backend_name,
                details={"size_bytes": size_bytes, "max_value_bytes": self.max_value_bytes},
            )

    def _effective_ttl(self, ttl_seconds: int | None) -> int | None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl is not None and ttl <= 0:
            raise RedisSafetyError(
                "Redis TTL must be a positive integer when provided",
                backend=self.backend_name,
                details={"ttl_seconds": ttl},
            )
        return ttl

    async def get(self, request: RedisGetRequest) -> RedisGetResult:
        key = self._prefixed_key(request.key)
        timeout_seconds = self._effective_timeout(request.options)

        async def _run() -> RedisGetResult:
            value = await await_adaptor_timeout(
                self.client.get(key),
                backend=self.backend_name,
                operation=AdaptorOperation.GET,
                timeout_seconds=timeout_seconds,
                details={"key": key},
            )
            return RedisGetResult(
                key=key,
                value=value,
                found=value is not None,
                metadata=_redis_metadata(
                    operation=AdaptorOperation.GET,
                    read_only=True,
                    extra={"key_prefix": self.key_prefix},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.GET,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"key": key},
        )

    async def set(self, request: RedisSetRequest) -> RedisSetResult:
        if self.read_only:
            raise RedisSafetyError("Redis adaptor is configured read-only", backend=self.backend_name)
        key = self._prefixed_key(request.key)
        self._validate_value_size(request.value)
        ttl = self._effective_ttl(request.ttl_seconds)
        timeout_seconds = self._effective_timeout(request.options)

        async def _run() -> RedisSetResult:
            written = await await_adaptor_timeout(
                self.client.set(key, request.value, ex=ttl, nx=request.nx, xx=request.xx),
                backend=self.backend_name,
                operation=AdaptorOperation.SET,
                timeout_seconds=timeout_seconds,
                details={"key": key},
            )
            return RedisSetResult(
                key=key,
                written=bool(written),
                metadata=_redis_metadata(
                    operation=AdaptorOperation.SET,
                    read_only=False,
                    extra={"ttl_seconds": ttl, "key_prefix": self.key_prefix},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.SET,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"key": key, "ttl_seconds": ttl},
        )

    async def delete(self, request: RedisDeleteRequest) -> RedisDeleteResult:
        if self.read_only:
            raise RedisSafetyError("Redis adaptor is configured read-only", backend=self.backend_name)
        if not self.allow_delete:
            raise RedisSafetyError(
                "Redis delete operations require allow_delete=True on the adaptor",
                backend=self.backend_name,
            )
        key = self._prefixed_key(request.key)

        async def _run() -> RedisDeleteResult:
            deleted = await await_adaptor_timeout(
                self.client.delete(key),
                backend=self.backend_name,
                operation=AdaptorOperation.DELETE,
                timeout_seconds=self._effective_timeout(request.options),
                details={"key": key},
            )
            return RedisDeleteResult(
                key=key,
                deleted_count=int(deleted or 0),
                metadata=_redis_metadata(
                    operation=AdaptorOperation.DELETE,
                    read_only=False,
                    extra={"key_prefix": self.key_prefix},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.DELETE,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"key": key},
        )

    async def hget(self, request: RedisHashGetRequest) -> RedisHashGetResult:
        key = self._prefixed_key(request.key)

        async def _run() -> RedisHashGetResult:
            value = await await_adaptor_timeout(
                self.client.hget(key, request.field_name),
                backend=self.backend_name,
                operation=AdaptorOperation.HASH_GET,
                timeout_seconds=self._effective_timeout(request.options),
                details={"key": key, "field_name": request.field_name},
            )
            return RedisHashGetResult(
                key=key,
                field_name=request.field_name,
                value=value,
                found=value is not None,
                metadata=_redis_metadata(
                    operation=AdaptorOperation.HASH_GET,
                    read_only=True,
                    extra={"key_prefix": self.key_prefix},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.HASH_GET,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"key": key, "field_name": request.field_name},
        )

    async def hset(self, request: RedisHashSetRequest) -> RedisHashSetResult:
        if self.read_only:
            raise RedisSafetyError("Redis adaptor is configured read-only", backend=self.backend_name)
        self._validate_value_size(request.value)
        key = self._prefixed_key(request.key)

        async def _run() -> RedisHashSetResult:
            written = await await_adaptor_timeout(
                self.client.hset(key, request.field_name, request.value),
                backend=self.backend_name,
                operation=AdaptorOperation.HASH_SET,
                timeout_seconds=self._effective_timeout(request.options),
                details={"key": key, "field_name": request.field_name},
            )
            return RedisHashSetResult(
                key=key,
                field_name=request.field_name,
                written=bool(written or written == 0),
                metadata=_redis_metadata(
                    operation=AdaptorOperation.HASH_SET,
                    read_only=False,
                    extra={"key_prefix": self.key_prefix},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.HASH_SET,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"key": key, "field_name": request.field_name},
        )

    async def hdel(self, request: RedisHashDeleteRequest) -> RedisHashDeleteResult:
        if self.read_only:
            raise RedisSafetyError("Redis adaptor is configured read-only", backend=self.backend_name)
        if not self.allow_delete:
            raise RedisSafetyError(
                "Redis delete operations require allow_delete=True on the adaptor",
                backend=self.backend_name,
            )
        key = self._prefixed_key(request.key)

        async def _run() -> RedisHashDeleteResult:
            deleted = await await_adaptor_timeout(
                self.client.hdel(key, request.field_name),
                backend=self.backend_name,
                operation=AdaptorOperation.HASH_DELETE,
                timeout_seconds=self._effective_timeout(request.options),
                details={"key": key, "field_name": request.field_name},
            )
            return RedisHashDeleteResult(
                key=key,
                field_name=request.field_name,
                deleted_count=int(deleted or 0),
                metadata=_redis_metadata(
                    operation=AdaptorOperation.HASH_DELETE,
                    read_only=False,
                    extra={"key_prefix": self.key_prefix},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.HASH_DELETE,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"key": key, "field_name": request.field_name},
        )


__all__ = [
    "RedisKVAdaptor",
    "RedisSafetyError",
    "RedisAdaptor",
    "RedisDeleteRequest",
    "RedisDeleteResult",
    "RedisGetRequest",
    "RedisGetResult",
    "RedisHashDeleteRequest",
    "RedisHashDeleteResult",
    "RedisHashGetRequest",
    "RedisHashGetResult",
    "RedisHashSetRequest",
    "RedisHashSetResult",
    "RedisSetRequest",
    "RedisSetResult",
]
