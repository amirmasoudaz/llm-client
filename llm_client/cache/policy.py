"""
Cache policy and invalidation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheInvalidationMode(str, Enum):
    """How reads/writes should interact with existing cached entries."""

    USE_EXISTING = "use_existing"
    REGENERATE = "regenerate"
    REWRITE = "rewrite"


@dataclass(frozen=True)
class CachePolicy:
    """
    Canonical cache policy for engine and helper paths.

    This centralizes the old `rewrite_cache` / `regen_cache` booleans and
    keeps the collection / error-caching rules attached to one object.
    """

    enabled: bool = True
    collection: str | None = None
    invalidation: CacheInvalidationMode = CacheInvalidationMode.USE_EXISTING
    only_ok: bool = True
    cache_errors: bool = False
    scope: str | None = None

    @property
    def rewrite_cache(self) -> bool:
        return self.invalidation is CacheInvalidationMode.REWRITE

    @property
    def regen_cache(self) -> bool:
        return self.invalidation is CacheInvalidationMode.REGENERATE

    def should_cache_status(self, status: int | None) -> bool:
        if status is None:
            return self.cache_errors
        if 200 <= status < 400:
            return True
        return self.cache_errors

    @classmethod
    def default_response(cls, *, collection: str | None = None) -> "CachePolicy":
        return cls(enabled=True, collection=collection)

    @classmethod
    def embeddings(cls, *, collection: str | None = None) -> "CachePolicy":
        return cls(enabled=True, collection=collection, only_ok=True)

    @classmethod
    def metadata(cls, *, collection: str | None = None) -> "CachePolicy":
        return cls(enabled=True, collection=collection, only_ok=True)

    @classmethod
    def summaries(cls, *, collection: str | None = None) -> "CachePolicy":
        return cls(enabled=True, collection=collection, only_ok=True)


__all__ = ["CacheInvalidationMode", "CachePolicy"]
