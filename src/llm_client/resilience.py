"""
Resilience primitives (circuit breaker).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_successes: int = 2


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

    async def on_failure(self) -> None:
        async with self._lock:
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


__all__ = ["CircuitBreakerConfig", "CircuitBreaker"]
