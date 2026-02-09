"""
Idempotency Keys for Request Deduplication.

This module provides:
- Idempotency key generation and validation
- Request deduplication logic
- Tracking of in-flight requests

Idempotency Boundaries
----------------------
There are TWO levels of idempotency when using llm-client with agent-runtime:

1. ENGINE IDEMPOTENCY (this module, llm-client layer):
   - Scope: "Don't send the same LLM request twice"
   - Granularity: Per-completion call
   - Use case: Prevent duplicate API calls during retries, concurrent requests
   - Key format: Typically content-hash or caller-provided unique ID
   - Lifetime: Short-lived (seconds to minutes)

2. RUNTIME IDEMPOTENCY (agent-runtime layer):
   - Scope: "Don't start the same job twice"
   - Granularity: Per-job lifecycle
   - Use case: Prevent duplicate job creation from retried webhooks, queue replays
   - Key format: Business-level identifier (e.g., "order-123:process")
   - Lifetime: Long-lived (hours to days, persisted in JobStore)

Key Format Recommendations
--------------------------
To prevent accidental key collisions between unrelated calls within the same job,
agent-runtime should generate structured idempotency keys for engine calls:

    {job_id}:{run_id}:{turn}:{operation}
    
Examples:
    - "job-abc:run-123:turn-0:completion"  # First model call
    - "job-abc:run-123:turn-1:completion"  # Second turn
    - "job-abc:run-123:turn-1:tool:search" # Tool call within turn

This ensures that:
- Each model call within a job has a unique key
- Retries of the same call get deduplicated
- Different turns/tools don't accidentally merge

Integration
-----------
When ExecutionEngine has an IdempotencyTracker:
- Caller provides idempotency_key via engine.complete(..., idempotency_key=...)
- Or via spec.extra["idempotency_key"] or context.tags["idempotency_key"]
- Engine checks for in-flight/completed requests before calling provider
- Returns cached result for completed keys (within timeout)
- Returns 409 status for in-flight duplicate keys
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def generate_idempotency_key(
    prefix: str = "idem",
    include_timestamp: bool = True,
) -> str:
    """
    Generate a unique idempotency key.

    Args:
        prefix: Key prefix
        include_timestamp: Include timestamp for traceability

    Returns:
        Unique idempotency key string
    """
    unique = uuid.uuid4().hex[:12]
    if include_timestamp:
        ts = int(time.time())
        return f"{prefix}_{ts}_{unique}"
    return f"{prefix}_{unique}"


def compute_request_hash(
    messages: Any,
    model: str,
    tools: list | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> str:
    """
    Compute a content-based hash for a request.

    This can be used as an idempotency key for identical requests.

    Args:
        messages: Request messages
        model: Model name
        tools: Tool definitions
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        **kwargs: Additional parameters

    Returns:
        SHA256 hash of the request content
    """
    # Normalize messages
    if hasattr(messages, "to_dict"):
        messages = messages.to_dict()
    elif isinstance(messages, list):
        messages = [m.to_dict() if hasattr(m, "to_dict") else m for m in messages]

    # Normalize tools
    if tools:
        tools = [t.to_openai_format() if hasattr(t, "to_openai_format") else t for t in tools]

    # Build canonical representation
    canonical = {
        "messages": messages,
        "model": model,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    canonical.update(kwargs)

    # Sort keys for consistent hashing
    canonical_json = json.dumps(canonical, sort_keys=True, ensure_ascii=True)

    return hashlib.sha256(canonical_json.encode()).hexdigest()[:32]


@dataclass
class PendingRequest:
    """Tracks a pending request for deduplication."""

    key: str
    started_at: float = field(default_factory=time.time)
    request_hash: str | None = None

    def is_expired(self, timeout: float = 60.0) -> bool:
        """Check if this pending request has expired."""
        return (time.time() - self.started_at) > timeout


class IdempotencyTracker:
    """
    Tracks in-flight requests for deduplication.

    Prevents duplicate requests with the same idempotency key
    from being processed simultaneously.

    Example:
        ```python
        tracker = IdempotencyTracker()

        # Check if we can process this request
        if tracker.can_start(idempotency_key):
            tracker.start_request(idempotency_key)
            try:
                result = await process_request()
                tracker.complete_request(idempotency_key, result)
            except Exception:
                tracker.fail_request(idempotency_key)
        else:
            # Duplicate in flight - wait or return cached result
            pass
        ```
    """

    def __init__(self, request_timeout: float = 60.0):
        self._pending: dict[str, PendingRequest] = {}
        self._completed: dict[str, Any] = {}
        self._timeout = request_timeout

    def can_start(self, key: str) -> bool:
        """
        Check if a request with this key can be started.

        Returns False if a request is already in flight.
        Returns True if no request exists or previous timed out.
        """
        self._cleanup_expired()

        if key in self._pending:
            return False

        return True

    def start_request(
        self,
        key: str,
        request_hash: str | None = None,
    ) -> bool:
        """
        Mark a request as started.

        Returns True if started, False if already in flight.
        """
        self._cleanup_expired()

        if key in self._pending:
            return False

        self._pending[key] = PendingRequest(
            key=key,
            request_hash=request_hash,
        )
        return True

    def complete_request(self, key: str, result: Any | None = None) -> None:
        """Mark a request as completed."""
        if key in self._pending:
            del self._pending[key]

        if result is not None:
            self._completed[key] = result

    def fail_request(self, key: str) -> None:
        """Mark a request as failed."""
        if key in self._pending:
            del self._pending[key]

    def get_result(self, key: str) -> Any | None:
        """Get a completed result if available."""
        return self._completed.get(key)

    def has_result(self, key: str) -> bool:
        """Check if a completed result exists."""
        return key in self._completed

    def is_pending(self, key: str) -> bool:
        """Check if a request is currently pending."""
        self._cleanup_expired()
        return key in self._pending

    def _cleanup_expired(self) -> None:
        """Remove expired pending requests."""
        expired = [k for k, v in self._pending.items() if v.is_expired(self._timeout)]
        for k in expired:
            del self._pending[k]

    def clear(self) -> None:
        """Clear all tracked requests."""
        self._pending.clear()
        self._completed.clear()

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        self._cleanup_expired()
        return len(self._pending)

    @property
    def completed_count(self) -> int:
        """Number of cached completed results."""
        return len(self._completed)


# Global tracker instance
_global_tracker: IdempotencyTracker | None = None


def get_tracker() -> IdempotencyTracker:
    """Get the global idempotency tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = IdempotencyTracker()
    return _global_tracker


__all__ = [
    "generate_idempotency_key",
    "compute_request_hash",
    "PendingRequest",
    "IdempotencyTracker",
    "get_tracker",
]
