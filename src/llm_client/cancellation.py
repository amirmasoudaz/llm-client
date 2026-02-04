"""Cancellation tokens for cooperative task interruption.

This module provides a CancellationToken class that enables cooperative
cancellation of long-running operations like streaming, agent loops,
and tool execution.

Key design:
- Token uses asyncio.Event internally for async-friendly waiting
- Safe to embed in frozen RequestContext (mutable object inside frozen container)
- Cooperative cancellation: components check token at natural breakpoints
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Any


class CancelledError(Exception):
    """Canonical cancellation exception for llm-client.
    
    Raised when an operation is cancelled via CancellationToken.
    This is a library-specific exception that wraps the concept
    of asyncio.CancelledError for llm-client operations.
    """
    pass


@dataclass
class CancellationToken:
    """Mutable token for cooperative cancellation.
    
    Safe to embed in frozen RequestContext since internal state is mutable.
    Uses asyncio.Event for async-friendly waiting.
    
    Usage:
        token = CancellationToken()
        
        # In long-running operation:
        for chunk in stream:
            token.raise_if_cancelled()
            process(chunk)
        
        # To cancel:
        token.cancel()
    """
    
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _callbacks: list[Callable[[], Any]] = field(default_factory=list, init=False)
    _noop: bool = field(default=False, init=False, repr=False)
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        if self._noop:
            return False
        return self._event.is_set()
    
    def cancel(self) -> None:
        """Request cancellation (idempotent).
        
        Can be called multiple times safely. Triggers all registered
        callbacks on first call.
        """
        if self._noop:
            return
        if not self._event.is_set():
            self._event.set()
            for cb in self._callbacks:
                try:
                    cb()
                except Exception:
                    # Don't let callback errors prevent cancellation
                    pass
    
    async def wait(self) -> None:
        """Await cancellation (for cleanup tasks).
        
        Blocks until cancel() is called. Useful for cleanup coroutines
        that need to wait for cancellation signal.
        """
        if self._noop:
            # Never resolves: this token never cancels.
            await asyncio.Future()
            return
        await self._event.wait()
    
    def on_cancel(self, callback: Callable[[], Any]) -> None:
        """Register callback for cancellation.
        
        Callback is invoked immediately if already cancelled.
        
        Args:
            callback: Zero-argument callable to invoke on cancellation.
        """
        self._callbacks.append(callback)
        if self._noop:
            return
        if self._event.is_set():
            try:
                callback()
            except Exception:
                pass
    
    def raise_if_cancelled(self) -> None:
        """Raise CancelledError if cancelled.
        
        This is the primary method for cooperative cancellation.
        Call this at natural breakpoints in long-running operations.
        
        Raises:
            CancelledError: If cancellation was requested.
        """
        if self._noop:
            return
        if self._event.is_set():
            raise CancelledError("Operation was cancelled")
    
    @classmethod
    def none(cls) -> CancellationToken:
        """Return a no-op token (never cancels).
        
        Use this for operations that don't support cancellation
        or when no token is provided.
        """
        return _NEVER_CANCEL


# Singleton for operations that don't support cancellation
# This is created lazily to avoid issues with event loop not being available
_NEVER_CANCEL: CancellationToken | None = None


def _get_never_cancel() -> CancellationToken:
    """Get or create the singleton no-op token."""
    global _NEVER_CANCEL
    if _NEVER_CANCEL is None:
        token = CancellationToken()
        token._noop = True
        _NEVER_CANCEL = token
    return _NEVER_CANCEL


# Patch the classmethod to use the lazy initialization
CancellationToken.none = classmethod(lambda cls: _get_never_cancel())


__all__ = ["CancellationToken", "CancelledError"]
