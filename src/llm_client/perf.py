"""
Performance utilities for fingerprinting and caching.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from blake3 import blake3

from .serialization import stable_json_dumps


@lru_cache(maxsize=2048)
def _hash_string(s: str) -> str:
    """Cache hash computation for strings."""
    return blake3(s.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    """
    Generate a stable fingerprint/hash for any object.

    Uses stable JSON serialization for deterministic output.
    Results are cached for repeated calls with the same string representation.

    Args:
        obj: Any JSON-serializable object

    Returns:
        64-character hex digest
    """
    json_str = stable_json_dumps(obj)
    return _hash_string(json_str)


def fingerprint_messages(messages: list[dict[str, Any]]) -> str:
    """
    Generate a fingerprint for a list of messages.

    Optimized for chat message lists - common in LLM requests.

    Args:
        messages: List of message dicts

    Returns:
        64-character hex digest
    """
    # Use tuple of (role, content) for faster comparison
    key_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            key_parts.append(f"{role}:{content[:1000]}")  # Truncate for speed
        else:
            key_parts.append(f"{role}:{stable_json_dumps(content)[:1000]}")

    combined = "\n".join(key_parts)
    return _hash_string(combined)


class FingerprintCache:
    """
    A secondary cache layer that tracks fingerprints for objects.

    Useful for avoiding redundant hash computations when the same
    object is fingerprinted multiple times.
    """

    def __init__(self, maxsize: int = 1024):
        self._cache: dict[int, str] = {}
        self._maxsize = maxsize

    def get_or_compute(self, obj: Any) -> str:
        """Get cached fingerprint or compute and cache it."""
        obj_id = id(obj)

        if obj_id in self._cache:
            return self._cache[obj_id]

        fp = fingerprint(obj)

        # Evict oldest entries if at capacity
        if len(self._cache) >= self._maxsize:
            # Remove first 25% of entries
            remove_count = self._maxsize // 4
            keys_to_remove = list(self._cache.keys())[:remove_count]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[obj_id] = fp
        return fp

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


# Global fingerprint cache for convenience
_global_fp_cache = FingerprintCache()


def get_fingerprint(obj: Any) -> str:
    """
    Get fingerprint using global cache.

    For high-frequency fingerprinting, use a dedicated FingerprintCache instance.
    """
    return _global_fp_cache.get_or_compute(obj)


def clear_fingerprint_cache() -> None:
    """Clear the global fingerprint cache and LRU caches."""
    _global_fp_cache.clear()
    _hash_string.cache_clear()


__all__ = [
    "fingerprint",
    "fingerprint_messages",
    "FingerprintCache",
    "get_fingerprint",
    "clear_fingerprint_cache",
]
