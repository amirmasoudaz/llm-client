"""
Unified Cache System for LLM Client.
"""

from .base import CacheBackend, CacheBackendName, CacheEntry
from .core import CacheCore, CacheLookupResult, CacheStats, CacheWriteResult
from .factory import CacheSettings, build_cache_core
from .fs import FSCache, FSCacheConfig
from .policy import CacheInvalidationMode, CachePolicy
from .postgres_redis import HybridCacheConfig, HybridRedisPostgreSQLCache
from .qdrant import QdrantCache
from .stores import MetadataCacheRecord, MetadataCacheStore, SummaryCacheStore
from ..cache_keys import (
    CACHE_KEY_SCHEMA_VERSION,
    CacheKeyDescriptor,
    build_cache_key,
    embedding_cache_key,
    metadata_cache_key,
    request_cache_key,
    summary_cache_key,
)

__all__ = [
    "CacheBackend",
    "CacheBackendName",
    "CacheEntry",
    "CacheCore",
    "CacheLookupResult",
    "CacheStats",
    "CacheWriteResult",
    "CacheInvalidationMode",
    "CachePolicy",
    "CACHE_KEY_SCHEMA_VERSION",
    "CacheKeyDescriptor",
    "CacheSettings",
    "build_cache_key",
    "build_cache_core",
    "request_cache_key",
    "embedding_cache_key",
    "metadata_cache_key",
    "summary_cache_key",
    "FSCache",
    "FSCacheConfig",
    "QdrantCache",
    "HybridRedisPostgreSQLCache",
    "HybridCacheConfig",
    "MetadataCacheRecord",
    "MetadataCacheStore",
    "SummaryCacheStore",
]
