"""
Hybrid Redis + PostgreSQL cache backend.
"""

from __future__ import annotations

import json
import os
import re
import time
import zlib
from dataclasses import dataclass
from typing import Any

import asyncpg
import redis.asyncio as redis_lib
from redis.exceptions import ConnectionError as RedisConnectionError

from ..persistence import PostgresRepository
from .base import CacheBackendName


def _sanitize_table_name(name: str) -> str:
    if not name:
        raise ValueError("table_name cannot be empty")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name


@dataclass
class HybridCacheConfig:
    default_table: str
    client_type: str

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_prefix: str = "resp_cache"
    redis_ttl_seconds: int = 60 * 60 * 24

    lock_ttl_seconds: int = 30

    pg_dsn: str = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")

    compress: bool = True
    compression_level: int = 6


class HybridRedisPostgreSQLCache:
    name: CacheBackendName = "pg_redis"

    def __init__(self, cfg: HybridCacheConfig) -> None:
        self.cfg = cfg
        self.default_collection = _sanitize_table_name(cfg.default_table)
        self.client_type = cfg.client_type

        self._redis: redis_lib.Redis | None = None
        self._pg_pool: asyncpg.Pool | None = None
        self._repo: PostgresRepository | None = None

    def _get_table(self, collection: str | None) -> str:
        table = collection or self.default_collection
        return _sanitize_table_name(table)

    async def _redis_ok(self) -> bool:
        if self._redis is None:
            return False
        try:
            pong = self._redis.ping()
            if isinstance(pong, bool):
                return pong
            return await pong
        except RedisConnectionError:
            return False
        except Exception:
            raise

    async def ensure_ready(self) -> None:
        # Postgres must work
        if self._pg_pool is None:
            # noinspection PyUnresolvedReferences
            self._pg_pool = await asyncpg.create_pool(dsn=self.cfg.pg_dsn, min_size=1, max_size=20)
            # Initialize repository with the pool
            self._repo = PostgresRepository(
                self._pg_pool, compress=self.cfg.compress, compression_level=self.cfg.compression_level
            )

        # Redis is optional
        if self._redis is None:
            self._redis = redis_lib.from_url(self.cfg.redis_url, decode_responses=False)

        if self._repo:
            await self._repo.ensure_table(self.default_collection)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
        if self._pg_pool is not None:
            await self._pg_pool.close()

    async def warm(self) -> None:
        return

    def _encode(self, obj: dict[str, Any]) -> bytes:
        raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if not self.cfg.compress:
            return raw
        return zlib.compress(raw, level=self.cfg.compression_level)

    def _decode(self, data: bytes) -> dict[str, Any]:
        if self.cfg.compress:
            data = zlib.decompress(data)
        return json.loads(data.decode("utf-8"))

    def _rk(self, key: str, collection: str | None = None) -> str:
        table = self._get_table(collection)
        return f"{self.cfg.redis_prefix}:{table}:{self.client_type}:{key}"

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        await self.ensure_ready()
        table = self._get_table(collection)

        # Ensure table (using repo cache)
        if self._repo:
            await self._repo.ensure_table(table)

        if await self._redis_ok():
            redis_client = self._redis
            assert redis_client is not None
            try:
                if await redis_client.exists(self._rk(effective_key, collection)):
                    return True
            except Exception:
                pass

        # Postgres fallback via Repository
        if self._repo:
            return (await self._repo.read(table, effective_key, self.client_type)) is not None
        return False

    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: str | None = None,
    ) -> tuple[str, bool]:
        if self.client_type == "completions" and rewrite_cache and not regen_cache:
            for i in range(0, 1000):
                eff = f"{identifier}_{i}"
                if not await self.exists(eff, collection):
                    return eff, False
            return f"{identifier}_{int(time.time())}", False
        return identifier, (not regen_cache)

    async def read(self, effective_key: str, collection: str | None = None) -> dict[str, Any] | None:
        await self.ensure_ready()
        table = self._get_table(collection)
        if self._repo:
            await self._repo.ensure_table(table)

        rk = self._rk(effective_key, collection)

        # Try Redis first, but never die if it's down.
        if await self._redis_ok():
            redis_client = self._redis
            assert redis_client is not None
            try:
                b = await redis_client.get(rk)
                if b:
                    try:
                        return self._decode(b)
                    except Exception:
                        # corrupt entry
                        await redis_client.delete(rk)
            except RedisConnectionError:
                pass
            except Exception:
                pass

        # Postgres fallback via Repository
        if not self._repo:
            return None

        row = await self._repo.read(table, effective_key, self.client_type)
        if row is None:
            return None

        # Best-effort repopulate Redis
        if await self._redis_ok():
            redis_client = self._redis
            assert redis_client is not None
            try:
                await redis_client.set(rk, self._encode(row), ex=self.cfg.redis_ttl_seconds)
            except Exception:
                pass

        return row

    async def write(
        self,
        effective_key: str,
        response: dict[str, Any],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        await self.ensure_ready()
        table = self._get_table(collection)
        if self._repo:
            await self._repo.ensure_table(table)

        # Durable always via Repository
        if self._repo:
            await self._repo.upsert(table, effective_key, self.client_type, model_name, response)

        # Best-effort hot cache
        if await self._redis_ok():
            redis_client = self._redis
            assert redis_client is not None
            try:
                await redis_client.set(
                    self._rk(effective_key, collection), self._encode(response), ex=self.cfg.redis_ttl_seconds
                )
            except Exception:
                pass
