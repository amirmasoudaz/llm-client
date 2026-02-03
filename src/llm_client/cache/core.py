"""
Core cache logic and orchestration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .base import CacheBackend


@dataclass
class CacheStats:
    """Statistics for cache operations.

    Thread-safe counters for cache hit/miss/write operations with latency tracking.
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    total_read_ms: float = 0.0
    total_write_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_read_ms(self) -> float:
        """Average read latency in milliseconds."""
        reads = self.hits + self.misses
        return self.total_read_ms / reads if reads > 0 else 0.0

    @property
    def avg_write_ms(self) -> float:
        """Average write latency in milliseconds."""
        return self.total_write_ms / self.writes if self.writes > 0 else 0.0

    def record_hit(self, latency_ms: float = 0.0) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.total_read_ms += latency_ms

    def record_miss(self, latency_ms: float = 0.0) -> None:
        """Record a cache miss."""
        self.misses += 1
        self.total_read_ms += latency_ms

    def record_write(self, latency_ms: float = 0.0) -> None:
        """Record a cache write."""
        self.writes += 1
        self.total_write_ms += latency_ms

    def record_error(self) -> None:
        """Record a cache error."""
        self.errors += 1

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "avg_read_ms": self.avg_read_ms,
            "avg_write_ms": self.avg_write_ms,
            "total_read_ms": self.total_read_ms,
            "total_write_ms": self.total_write_ms,
        }

    def reset(self) -> CacheStats:
        """Reset statistics and return a copy of the old values."""
        old = CacheStats(
            hits=self.hits,
            misses=self.misses,
            writes=self.writes,
            errors=self.errors,
            total_read_ms=self.total_read_ms,
            total_write_ms=self.total_write_ms,
        )
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self.total_read_ms = 0.0
        self.total_write_ms = 0.0
        return old


class CacheCore:
    """Core cache orchestration with statistics tracking."""

    def __init__(self, backend: CacheBackend | None, default_collection: str | None = None) -> None:
        self.backend = backend
        self.default_collection = default_collection
        self.stats = CacheStats()

    def _resolve_collection(self, collection: str | None) -> str | None:
        """Resolve effective collection: explicit > backend default > core default."""
        if collection:
            return collection
        if self.backend and hasattr(self.backend, "default_collection"):
            return self.backend.default_collection
        return self.default_collection

    async def ensure_ready(self) -> None:
        if self.backend:
            await self.backend.ensure_ready()

    async def warm(self) -> None:
        if self.backend:
            await self.backend.warm()

    async def close(self) -> None:
        if self.backend:
            await self.backend.close()

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self.stats

    def reset_stats(self) -> CacheStats:
        """Reset statistics and return the old values."""
        return self.stats.reset()

    async def get_cached(
        self,
        identifier: str,
        *,
        rewrite_cache: bool,
        regen_cache: bool,
        only_ok: bool = True,
        collection: str | None = None,
    ) -> tuple[dict | None, str]:
        if not self.backend:
            return None, identifier

        start_time = time.perf_counter()
        eff_collection = self._resolve_collection(collection)
        eff, can_read = await self.backend.resolve_key(identifier, rewrite_cache, regen_cache, eff_collection)

        if not can_read:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.record_miss(latency_ms)
            return None, eff

        try:
            resp = await self.backend.read(eff, eff_collection)
        except Exception:
            # On read failure, treat as cache miss but don't crash
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.record_error()
            self.stats.record_miss(latency_ms)
            return None, eff

        latency_ms = (time.perf_counter() - start_time) * 1000

        if not resp:
            self.stats.record_miss(latency_ms)
            return None, eff

        if only_ok and resp.get("error") != "OK":
            self.stats.record_miss(latency_ms)
            return None, eff

        self.stats.record_hit(latency_ms)
        return resp, eff

    async def put_cached(
        self,
        identifier: str,
        *,
        rewrite_cache: bool,
        regen_cache: bool,
        response: dict,
        model_name: str,
        log_errors: bool,
        collection: str | None = None,
    ) -> str:
        if not self.backend:
            return identifier

        start_time = time.perf_counter()
        eff_collection = self._resolve_collection(collection)
        eff, _ = await self.backend.resolve_key(identifier, rewrite_cache, regen_cache, eff_collection)

        try:
            if response.get("error") == "OK" or (response.get("error") != "OK" and log_errors):
                await self.backend.write(eff, response, model_name=model_name, collection=eff_collection)
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.stats.record_write(latency_ms)
        except Exception:
            # On write failure, record error and continue
            self.stats.record_error()

        return eff


__all__ = ["CacheCore", "CacheStats"]
