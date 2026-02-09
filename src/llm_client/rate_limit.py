import asyncio
import time

from .models import ModelProfile


class TokenBucket:
    def __init__(self, size: int = 0) -> None:
        self._maximum_size = max(0, int(size))
        self._current_size = self._maximum_size
        self._consume_per_second = (self._maximum_size / 60) if self._maximum_size > 0 else 0.0
        self._last_fill_time = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, amount: int = 0) -> None:
        # A bucket size of 0 means "unlimited/disabled" (no rate limiting).
        if amount == 0 or self._maximum_size == 0:
            return

        async with self._lock:
            if amount > self._maximum_size:
                raise ValueError("Amount exceeds bucket size.")

            self._refill()

            while amount > self._current_size:
                await asyncio.sleep(0.05)
                self._refill()

            self._current_size -= amount

    def _refill(self) -> None:
        if self._maximum_size == 0:
            return
        now = time.time()
        elapsed = now - self._last_fill_time
        refilled_tokens = int(elapsed * self._consume_per_second)
        self._current_size = min(self._maximum_size, self._current_size + refilled_tokens)
        self._last_fill_time = now


class Limiter:
    DEFAULT_RATE_LIMITS = {"tkn_per_min": 0, "req_per_min": 0}

    def __init__(self, model_specs: type["ModelProfile"] | None = None) -> None:
        rate_limits = getattr(model_specs, "rate_limits", None) if model_specs else None
        if not isinstance(rate_limits, dict):
            rate_limits = self.DEFAULT_RATE_LIMITS

        tpm = float(rate_limits.get("tkn_per_min", 0) or 0)
        rpm = float(rate_limits.get("req_per_min", 0) or 0)

        self.tkn_limiter = TokenBucket(size=int(tpm * 0.75))
        self.req_limiter = TokenBucket(size=int(rpm * 0.95))

    def limit(self, tokens: int = 0, requests: int = 0):
        return self._LimitContextManager(self, tokens, requests)

    class _LimitContextManager:
        def __init__(self, limiter, tokens, requests):
            self.limiter = limiter
            self.tokens = tokens
            self.requests = requests
            self.output_tokens = 0

        async def __aenter__(self):
            await asyncio.gather(
                self.limiter.tkn_limiter.consume(self.tokens),
                self.limiter.req_limiter.consume(self.requests),
            )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.output_tokens > 0:
                await self.limiter.tkn_limiter.consume(self.output_tokens)


__all__ = ["Limiter", "TokenBucket"]
