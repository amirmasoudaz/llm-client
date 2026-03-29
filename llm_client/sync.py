"""Thin sync wrappers for async-first APIs.

This module provides synchronous wrappers for async-first APIs in
``llm_client``. Use with caution - these are primarily for scripting
and testing contexts where async is not available.

Key design:
- Uses a dedicated background event loop for sync calls
- Raises RuntimeError if called inside an existing event loop
- Clear error messages guide users to the async alternative
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Awaitable
from concurrent.futures import Future
import inspect
import threading
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .conversation import Conversation
    from .models import ModelProfile
    from .providers.types import Message

T = TypeVar("T")


class _SyncLoopRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return self._loop

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="llm-client-sync-loop",
                daemon=True,
            )
            self._thread.start()

        self._ready.wait()
        if self._loop is None:
            raise RuntimeError("Failed to start sync event loop runner.")
        return self._loop

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def run(self, awaitable: Awaitable[T]) -> T:
        loop = self._ensure_loop()
        future: Future[T] = asyncio.run_coroutine_threadsafe(awaitable, loop)
        return future.result()

    def shutdown(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)


_SYNC_LOOP_RUNNER = _SyncLoopRunner()
atexit.register(_SYNC_LOOP_RUNNER.shutdown)


def _ensure_not_in_async_context(function_name: str, guidance: str) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError(
            f"{function_name}() cannot be called inside an async context. "
            f"{guidance}"
        )


def run_async_sync(awaitable: Awaitable[T]) -> T:
    """Run an awaitable from synchronous code on a dedicated background loop."""
    try:
        _ensure_not_in_async_context(
            "run_async_sync",
            "Use 'await ...' instead.",
        )
    except RuntimeError:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise
    return _SYNC_LOOP_RUNNER.run(awaitable)


def get_messages_sync(
    conversation: Conversation,
    model: type[ModelProfile] | None = None,
    include_system: bool = True,
) -> list[Message]:
    """Sync wrapper for Conversation.get_messages_async.
    
    This is a convenience wrapper for contexts where async is not available.
    Prefer using the async version when possible.
    
    Args:
        conversation: The Conversation instance.
        model: Optional model profile for token counting.
        include_system: Whether to include system message.
    
    Returns:
        List of messages, potentially truncated/summarized.
    
    Raises:
        RuntimeError: If called inside an existing async event loop.
    """
    _ensure_not_in_async_context(
        "get_messages_sync",
        "Use 'await conversation.get_messages_async()' instead.",
    )

    return run_async_sync(
        conversation.get_messages_async(model=model, include_system=include_system)
    )


def summarize_sync(
    summarizer: any,  # Summarizer protocol
    messages: list[Message],
    max_tokens: int,
) -> str:
    """Sync wrapper for Summarizer.summarize.
    
    Args:
        summarizer: A Summarizer implementation.
        messages: Messages to summarize.
        max_tokens: Maximum tokens for summary.
    
    Returns:
        Summary string.
    
    Raises:
        RuntimeError: If called inside an existing async event loop.
    """
    _ensure_not_in_async_context(
        "summarize_sync",
        "Use 'await summarizer.summarize()' instead.",
    )

    return run_async_sync(summarizer.summarize(messages, max_tokens))


__all__ = ["get_messages_sync", "run_async_sync", "summarize_sync"]
