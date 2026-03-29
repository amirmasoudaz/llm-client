"""
Cache configuration classes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .base import CacheBackendType


@dataclass
class CacheConfig:
    """Configuration for caching."""

    # Backend selection
    backend: CacheBackendType = "none"
    enabled: bool = True

    # Collection/namespace
    default_collection: str | None = None

    # TTL settings
    ttl_seconds: int | None = None

    # Behavior
    cache_errors: bool = False
    only_cache_ok: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.backend not in ("none", "fs", "pg_redis", "qdrant"):
            raise ValueError(f"Invalid cache backend: {self.backend}")


@dataclass
class FSCacheConfig(CacheConfig):
    """Filesystem cache configuration."""

    backend: CacheBackendType = "fs"
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class RedisPGCacheConfig(CacheConfig):
    """PostgreSQL + Redis hybrid cache configuration."""

    backend: CacheBackendType = "pg_redis"

    # PostgreSQL settings
    pg_dsn: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
    )

    # Redis settings
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    redis_ttl_seconds: int = 86400  # 24 hours

    # Compression
    compress: bool = True
    compression_level: int = 6

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        if not self.pg_dsn:
            raise ValueError("pg_dsn is required for pg_redis backend")
        if not self.pg_dsn.startswith(("postgresql://", "postgres://")):
            raise ValueError("pg_dsn must be a valid PostgreSQL connection string")
        if not self.redis_url:
            raise ValueError("redis_url is required for pg_redis backend")
        if not self.redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("redis_url must be a valid Redis connection string")
        if self.redis_ttl_seconds <= 0:
            raise ValueError("redis_ttl_seconds must be positive")
        if not (0 <= self.compression_level <= 9):
            raise ValueError("compression_level must be between 0 and 9")


@dataclass
class QdrantCacheConfig(CacheConfig):
    """Qdrant vector cache configuration."""

    backend: CacheBackendType = "qdrant"

    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        if not self.qdrant_url:
            raise ValueError("qdrant_url is required for qdrant backend")
        if not self.qdrant_url.startswith(("http://", "https://")):
            raise ValueError("qdrant_url must be a valid HTTP(S) URL")


__all__ = ["CacheConfig", "FSCacheConfig", "RedisPGCacheConfig", "QdrantCacheConfig"]
