"""
Cache factory and settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .base import CacheBackendName
from .core import CacheCore
from .fs import FSCache, FSCacheConfig
from .postgres_redis import HybridCacheConfig, HybridRedisPostgreSQLCache
from .qdrant import QdrantCache


@dataclass
class CacheSettings:
    backend: CacheBackendName
    client_type: str
    default_collection: str | None = None
    cache_dir: Path | None = None

    # qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # pg_redis
    pg_dsn: str | None = None
    redis_url: str | None = None
    redis_ttl_seconds: int = 60 * 60 * 24
    compress: bool = True


def build_cache_core(settings: CacheSettings) -> CacheCore:
    backend = settings.backend
    default_coll = settings.default_collection or f"{settings.client_type}_cache"

    if backend == "none" or backend is None:
        return CacheCore(None, default_collection=default_coll)

    if backend == "fs":
        if not settings.cache_dir:
            raise ValueError("cache_dir is required for fs cache backend")
        fs = FSCache(
            FSCacheConfig(
                dir=settings.cache_dir,
                client_type=settings.client_type,
                default_collection=default_coll,
            )
        )
        return CacheCore(fs, default_collection=default_coll)

    if backend == "qdrant":
        q = QdrantCache(
            default_collection=default_coll,
            client_type=settings.client_type,
            base_url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        return CacheCore(q, default_collection=default_coll)

    if backend == "pg_redis":
        pg_dsn = settings.pg_dsn or os.getenv("PG_DSN") or ""
        redis_url = settings.redis_url or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        h = HybridRedisPostgreSQLCache(
            HybridCacheConfig(
                default_table=default_coll,
                client_type=settings.client_type,
                pg_dsn=pg_dsn,
                redis_url=redis_url,
                redis_ttl_seconds=settings.redis_ttl_seconds,
                compress=settings.compress,
            )
        )
        return CacheCore(h, default_collection=default_coll)

    raise ValueError(f"Unknown cache backend: {backend!r}")
