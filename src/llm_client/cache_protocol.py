"""
Cache Backend Protocol and Utilities.

This module defines the cache protocol/interface that all backends 
must implement, along with shared utilities.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable


@runtime_checkable
class CacheBackendProtocol(Protocol):
    """
    Protocol defining the interface for cache backends.
    
    All cache backends must implement these methods for
    compatibility with CacheCore.
    """
    
    name: str
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
        collection: Optional[str] = None,
    ) -> bool:
        """Check if a key exists in the cache."""
        ...
    
    async def read(
        self, 
        effective_key: str, 
        collection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read a cached response."""
        ...
    
    async def write(
        self,
        effective_key: str,
        response: Dict[str, Any],
        model_name: str,
        collection: Optional[str] = None,
    ) -> None:
        """Write a response to cache."""
        ...
    
    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Resolve an identifier to an effective cache key.
        
        Returns (effective_key, can_read_existing).
        """
        ...


@dataclass
class CacheEntry:
    """Metadata about a cached entry."""
    
    key: str
    collection: str
    model: str
    status: Optional[int] = None
    error: Optional[str] = None
    version: int = 1
    created_at: Optional[str] = None
    
    def is_ok(self) -> bool:
        """Check if this is a successful cache entry."""
        return self.error == "OK" or (self.status and 200 <= self.status < 300)


class BaseCacheBackend(ABC):
    """
    Abstract base class for cache backends.
    
    Provides default implementations for optional methods.
    """
    
    name: str = "base"
    default_collection: str = "default"
    
    @abstractmethod
    async def ensure_ready(self) -> None:
        pass
    
    async def close(self) -> None:
        """Default no-op close."""
        pass
    
    async def warm(self) -> None:
        """Default no-op warm."""
        pass
    
    @abstractmethod
    async def exists(
        self, 
        effective_key: str, 
        collection: Optional[str] = None,
    ) -> bool:
        pass
    
    @abstractmethod
    async def read(
        self, 
        effective_key: str, 
        collection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def write(
        self,
        effective_key: str,
        response: Dict[str, Any],
        model_name: str,
        collection: Optional[str] = None,
    ) -> None:
        pass
    
    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Default key resolution logic.
        
        Override for backends that need special handling (e.g., rewrite_cache).
        """
        if regen_cache:
            return identifier, False
        return identifier, True
    
    def _get_collection(self, collection: Optional[str]) -> str:
        """Helper to resolve collection name."""
        return collection or self.default_collection


# Convenience export of cache types
__all__ = [
    "CacheBackendProtocol",
    "BaseCacheBackend",
    "CacheEntry",
]
