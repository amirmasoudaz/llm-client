from __future__ import annotations

import asyncio
import time

import pytest

from llm_client.rate_limit import Limiter


@pytest.mark.asyncio
async def test_limiter_supports_explicit_window_budgets() -> None:
    limiter = Limiter(tokens_per_window=50, requests_per_window=2, window_seconds=0.3)

    assert limiter.tkn_limiter.maximum_size == 50
    assert limiter.req_limiter.maximum_size == 2
    assert limiter.window_seconds == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_limiter_refills_over_custom_window() -> None:
    limiter = Limiter(tokens_per_window=0, requests_per_window=2, window_seconds=0.3)
    started = time.monotonic()

    async def _one_request() -> float:
        queued_at = time.monotonic()
        async with limiter.limit(requests=1):
            return time.monotonic() - queued_at

    waits = await asyncio.gather(_one_request(), _one_request(), _one_request())
    elapsed = time.monotonic() - started

    assert waits[0] < 0.05
    assert waits[1] < 0.05
    assert waits[2] >= 0.1
    assert elapsed >= 0.1
