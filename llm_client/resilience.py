"""
Resilience primitives (circuit breaker).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .retry_policy import DEFAULT_RETRYABLE_STATUSES


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_successes: int = 2
    trip_statuses: tuple[int, ...] = DEFAULT_RETRYABLE_STATUSES
    trip_on_exceptions: bool = True


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0
        self._half_open_in_flight = False
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self._state == "open":
                if time.time() - self._opened_at >= self.config.recovery_timeout:
                    self._state = "half_open"
                    self._half_open_in_flight = False
                    self._failure_count = 0
                    self._success_count = 0
                else:
                    return False

            if self._state == "half_open":
                if self._half_open_in_flight:
                    return False
                self._half_open_in_flight = True
                return True

            return True

    async def on_success(self) -> None:
        async with self._lock:
            if self._state == "half_open":
                self._success_count += 1
                self._half_open_in_flight = False
                if self._success_count >= self.config.half_open_successes:
                    self._state = "closed"
                    self._failure_count = 0
                    self._success_count = 0
                return

            self._failure_count = 0

    async def on_failure(self, *, status: int | None = None) -> None:
        async with self._lock:
            if not self._should_trip(status):
                if self._state == "half_open":
                    self._state = "closed"
                    self._half_open_in_flight = False
                    self._failure_count = 0
                    self._success_count = 0
                return

            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.time()
                self._half_open_in_flight = False
                self._failure_count = 0
                self._success_count = 0
                return

            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._state = "open"
                self._opened_at = time.time()
                self._success_count = 0

    def _should_trip(self, status: int | None) -> bool:
        if status is None:
            return self.config.trip_on_exceptions
        return status in self.config.trip_statuses

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self._state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "is_open": self._state == "open",
            "is_half_open": self._state == "half_open",
        }


__all__ = ["CircuitBreakerConfig", "CircuitBreaker"]
