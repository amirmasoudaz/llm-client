"""
Shared retry classification and backoff helpers.
"""

from __future__ import annotations

import random
from typing import Any


DEFAULT_RETRYABLE_STATUSES: tuple[int, ...] = (0, 408, 409, 425, 429, 500, 502, 503, 504)


def is_retryable_status(status: int | None, *, retryable_statuses: tuple[int, ...] = DEFAULT_RETRYABLE_STATUSES) -> bool:
    if status is None:
        return False
    return int(status) in retryable_statuses


def extract_retry_after_seconds(source: Any) -> float | None:
    response = getattr(source, "response", None)
    headers = None
    if response is not None and hasattr(response, "headers"):
        headers = response.headers
    elif hasattr(source, "raw_response") and hasattr(source.raw_response, "headers"):
        headers = source.raw_response.headers
    elif hasattr(source, "headers"):
        headers = source.headers
    if headers is None:
        return None
    try:
        raw_value = headers.get("Retry-After")
    except Exception:
        return None
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def compute_backoff_delay(
    *,
    attempt: int,
    base_backoff: float,
    max_backoff: float | None = None,
    retry_after: float | None = None,
    jitter_ratio: float = 0.2,
) -> float:
    if retry_after is not None:
        return max(0.0, retry_after)
    delay = max(0.0, float(base_backoff)) * (2 ** max(0, int(attempt)))
    if max_backoff is not None:
        delay = min(delay, max(0.0, float(max_backoff)))
    if delay <= 0:
        return 0.0
    jitter = max(0.0, float(jitter_ratio))
    return delay * random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)


__all__ = [
    "DEFAULT_RETRYABLE_STATUSES",
    "compute_backoff_delay",
    "extract_retry_after_seconds",
    "is_retryable_status",
]
