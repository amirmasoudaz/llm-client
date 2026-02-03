"""
Unified hashing utilities for llm-client.

This module consolidates all hashing operations used throughout the package,
providing consistent, high-performance hashing for cache keys, content 
addressing, and database identifiers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from blake3 import blake3

from .serialization import stable_json_dumps

HashAlgorithm = Literal["blake3", "sha256", "md5"]


def compute_hash(
    data: str | bytes,
    algorithm: HashAlgorithm = "blake3",
    truncate: int | None = None,
) -> str:
    """
    Compute a hash using the specified algorithm.

    Args:
        data: Input data to hash (string or bytes)
        algorithm: Hash algorithm to use:
            - "blake3": Fast, modern hash (default, preferred for performance)
            - "sha256": Widely compatible, cryptographic
            - "md5": Legacy compatibility only
        truncate: Truncate output to N characters (for shorter keys)

    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if algorithm == "blake3":
        result = blake3(data).hexdigest()
    elif algorithm == "sha256":
        result = hashlib.sha256(data).hexdigest()
    elif algorithm == "md5":
        result = hashlib.md5(data).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return result[:truncate] if truncate else result


def content_hash(obj: Any) -> str:
    """
    Generate a deterministic content hash for any JSON-serializable object.

    Uses blake3 for speed combined with stable JSON serialization
    for determinism (consistent output regardless of dict key order).

    Args:
        obj: Any JSON-serializable object

    Returns:
        64-character hexadecimal hash
    """
    return blake3(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def cache_key(api: str, params: dict[str, Any]) -> str:
    """
    Generate a cache key from an API endpoint and parameters.

    This is the standard format used throughout llm-client for
    identifying cached responses.

    Args:
        api: API endpoint identifier (e.g., "chat.completions", "responses")
        params: Request parameters dictionary

    Returns:
        64-character hexadecimal cache key
    """
    payload = {"api": api, "params": params}
    return content_hash(payload)


def int_hash(s: str) -> int:
    """
    Generate an integer hash from a string.

    Used for systems requiring integer identifiers (e.g., Qdrant point IDs).
    Produces a consistent unsigned 64-bit integer from the input.

    Args:
        s: Input string

    Returns:
        Unsigned 64-bit integer hash
    """
    return int.from_bytes(blake3(s.encode("utf-8")).digest()[:8], "big", signed=False)


__all__ = [
    "HashAlgorithm",
    "compute_hash",
    "content_hash",
    "cache_key",
    "int_hash",
]
