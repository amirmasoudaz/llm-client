"""
Async concurrency helpers.

This library is async-first, but some integrations (e.g. sync tools, filesystem I/O)
need to run synchronous code without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


def _default_max_workers() -> int:
    # Mirrors ThreadPoolExecutor's default sizing heuristics.
    return min(32, (os.cpu_count() or 1) + 4)


_EXECUTOR = ThreadPoolExecutor(max_workers=_default_max_workers())


async def run_sync(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """
    Run a synchronous callable in a shared thread pool.

    We intentionally use a dedicated executor (vs. asyncio's default) so the library
    has predictable behavior across environments and tests.
    """
    # NOTE: We intentionally avoid `loop.run_in_executor()`/`asyncio.to_thread()` here.
    # In some environments (including certain sandboxes and test harnesses),
    # cross-thread wakeups via `call_soon_threadsafe()` can be unreliable and cause
    # awaits to hang even though the worker thread finished. Polling the concurrent
    # future avoids that failure mode while keeping the event loop responsive.
    future = _EXECUTOR.submit(partial(func, *args, **kwargs))
    try:
        while True:
            if future.done():
                return future.result()
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        future.cancel()
        raise


__all__ = ["run_sync"]
