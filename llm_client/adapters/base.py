"""
Base types shared by service adaptors.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any

from ..retry_policy import compute_backoff_delay
from ..runtime_events import RuntimeEvent, RuntimeEventType


class AdaptorOperation(str, Enum):
    QUERY = "query"
    EXECUTE = "execute"
    GET = "get"
    SET = "set"
    DELETE = "delete"
    HASH_GET = "hash_get"
    HASH_SET = "hash_set"
    HASH_DELETE = "hash_delete"
    UPSERT = "upsert"
    SEARCH = "search"


class AdaptorCapability(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEARCH = "search"
    UPSERT = "upsert"
    HASH = "hash"
    TTL = "ttl"


@dataclass(frozen=True)
class AdaptorExecutionOptions:
    timeout_seconds: float | None = None
    retry_attempts: int | None = None
    idempotency_key: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptorMetadata:
    backend: str
    operation: AdaptorOperation
    read_only: bool = True
    capabilities: tuple[AdaptorCapability, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptorRuntime:
    execution_context: Any | None = None
    event_bus: Any | None = None
    ledger: Any | None = None
    retry_attempts: int = 0
    base_backoff_seconds: float = 0.0
    max_backoff_seconds: float | None = None
    emit_events: bool = True
    record_usage: bool = True


class AdaptorError(Exception):
    """Base error for adaptor execution failures."""

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        operation: AdaptorOperation | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.operation = operation
        self.retryable = retryable
        self.details = dict(details or {})


class AdaptorTimeoutError(AdaptorError):
    """Raised when an adaptor operation exceeds its timeout."""

    def __init__(
        self,
        message: str = "Adaptor operation timed out",
        *,
        backend: str | None = None,
        operation: AdaptorOperation | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            backend=backend,
            operation=operation,
            retryable=True,
            details=details,
        )


async def await_adaptor_timeout(
    awaitable: Any,
    *,
    backend: str,
    operation: AdaptorOperation,
    timeout_seconds: float | None,
    details: dict[str, Any] | None = None,
) -> Any:
    try:
        if timeout_seconds is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise AdaptorTimeoutError(
            backend=backend,
            operation=operation,
            details={"timeout_seconds": timeout_seconds, **dict(details or {})},
        ) from exc


def is_retryable_adaptor_error(error: BaseException) -> bool:
    return bool(getattr(error, "retryable", False) or isinstance(error, AdaptorTimeoutError))


async def publish_adaptor_event(
    runtime: AdaptorRuntime | None,
    *,
    backend: str,
    operation: AdaptorOperation,
    phase: str,
    data: dict[str, Any] | None = None,
) -> None:
    if runtime is None or runtime.event_bus is None or not runtime.emit_events:
        return
    payload = {
        "kind": f"adaptor.{phase}",
        "backend": backend,
        "operation": operation.value,
        **dict(data or {}),
    }
    ctx = runtime.execution_context
    if ctx is not None:
        event = RuntimeEvent.from_context(ctx, RuntimeEventType.PROGRESS, payload)
    else:
        event = RuntimeEvent(type=RuntimeEventType.PROGRESS, data=payload)
    await runtime.event_bus.publish(event)


async def record_adaptor_usage(
    runtime: AdaptorRuntime | None,
    *,
    backend: str,
    operation: AdaptorOperation,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    if runtime is None or runtime.ledger is None or runtime.execution_context is None or not runtime.record_usage:
        return
    await runtime.ledger.record_connector_usage(
        runtime.execution_context,
        connector_name=backend,
        duration_ms=duration_ms,
        metadata={"operation": operation.value, **dict(metadata or {})},
    )


async def run_adaptor_operation(
    runtime: AdaptorRuntime | None,
    *,
    backend: str,
    operation: AdaptorOperation,
    retry_attempts: int | None,
    func: Any,
    metadata: dict[str, Any] | None = None,
) -> Any:
    attempts = max(0, retry_attempts if retry_attempts is not None else (runtime.retry_attempts if runtime else 0))
    last_error: BaseException | None = None
    await publish_adaptor_event(
        runtime,
        backend=backend,
        operation=operation,
        phase="start",
        data={"attempts": attempts + 1, **dict(metadata or {})},
    )
    start = time.perf_counter()
    for attempt in range(attempts + 1):
        try:
            result = await func()
        except BaseException as exc:
            last_error = exc
            should_retry = attempt < attempts and is_retryable_adaptor_error(exc)
            if should_retry:
                delay = compute_backoff_delay(
                    attempt=attempt,
                    base_backoff=runtime.base_backoff_seconds if runtime else 0.0,
                    max_backoff=runtime.max_backoff_seconds if runtime else None,
                )
                await publish_adaptor_event(
                    runtime,
                    backend=backend,
                    operation=operation,
                    phase="retry",
                    data={
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "error": str(exc),
                        **dict(metadata or {}),
                    },
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                continue
            duration_ms = (time.perf_counter() - start) * 1000.0
            await record_adaptor_usage(
                runtime,
                backend=backend,
                operation=operation,
                duration_ms=duration_ms,
                metadata={"status": "error", "error": str(exc), **dict(metadata or {})},
            )
            await publish_adaptor_event(
                runtime,
                backend=backend,
                operation=operation,
                phase="error",
                data={"attempt": attempt + 1, "error": str(exc), **dict(metadata or {})},
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000.0
        await record_adaptor_usage(
            runtime,
            backend=backend,
            operation=operation,
            duration_ms=duration_ms,
            metadata={"status": "ok", "attempt": attempt + 1, **dict(metadata or {})},
        )
        await publish_adaptor_event(
            runtime,
            backend=backend,
            operation=operation,
            phase="finish",
            data={"attempt": attempt + 1, "duration_ms": duration_ms, **dict(metadata or {})},
        )
        return result
    assert last_error is not None
    raise last_error


__all__ = [
    "AdaptorCapability",
    "AdaptorError",
    "AdaptorExecutionOptions",
    "AdaptorMetadata",
    "AdaptorOperation",
    "AdaptorRuntime",
    "AdaptorTimeoutError",
    "await_adaptor_timeout",
    "is_retryable_adaptor_error",
    "publish_adaptor_event",
    "record_adaptor_usage",
    "run_adaptor_operation",
]
