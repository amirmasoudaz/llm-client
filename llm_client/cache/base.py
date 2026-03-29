"""
Base classes and protocols for the cache system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

CacheBackendName = Literal["fs", "qdrant", "pg_redis", "none"]


@runtime_checkable
class CacheBackend(Protocol):
    """
    Protocol defining the interface for cache backends.

    All cache backends must implement these methods for
    compatibility with CacheCore.
    """

    name: CacheBackendName
    default_collection: str

    async def ensure_ready(self) -> None:
        """Initialize backend and ensure it's ready for use."""
        ...

    async def close(self) -> None:
        """Clean up and close backend connections."""
        ...

    async def warm(self) -> None:
        """Warm up cache (optional optimization)."""
        ...

    async def exists(
        self,
        effective_key: str,
        collection: str | None = None,
    ) -> bool:
        """Check if a key exists in the cache."""
        ...

    async def read(
        self,
        effective_key: str,
        collection: str | None = None,
    ) -> dict[str, Any] | None:
        """Read a cached response."""
        ...

    async def write(
        self,
        effective_key: str,
        response: dict[str, Any],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        """Write a response to cache."""
        ...

    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: str | None = None,
    ) -> tuple[str, bool]:
        """
        Resolve an identifier to an effective cache key.

        Returns (effective_key, can_read_existing).
        If rewrite_cache=True (and not regen_cache), should pick a new unused suffix and set can_read_existing=False.
        Otherwise, returns identifier and can_read_existing = not regen_cache.
        """
        ...


@dataclass
class CacheEntry:
    """Metadata about a cached entry."""

    key: str
    collection: str
    model: str
    status: int | None = None
    error: str | None = None
    version: int = 1
    created_at: str | None = None

    def is_ok(self) -> bool:
        """Check if this is a successful cache entry."""
        if self.error == "OK":
            return True
        if self.status is None:
            return False
        return 200 <= self.status < 300


class BaseCacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Provides default implementations for optional methods.
    """

    name: CacheBackendName = "none"  # Default, override in subclasses
    default_collection: str = "default"

    @abstractmethod
    async def ensure_ready(self) -> None:
        pass

    async def close(self) -> None:
        """Default no-op close."""
        return None

    async def warm(self) -> None:
        """Default no-op warm."""
        return None

    @abstractmethod
    async def exists(
        self,
        effective_key: str,
        collection: str | None = None,
    ) -> bool:
        pass

    @abstractmethod
    async def read(
        self,
        effective_key: str,
        collection: str | None = None,
    ) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def write(
        self,
        effective_key: str,
        response: dict[str, Any],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        pass

    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: str | None = None,
    ) -> tuple[str, bool]:
        """
        Default key resolution logic.
        """
        if regen_cache:
            return identifier, False
        return identifier, True

    def _get_collection(self, collection: str | None) -> str:
        """Helper to resolve collection name."""
        return collection or self.default_collection
