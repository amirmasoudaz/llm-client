import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_client import RequestSpec
from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.providers.base import Provider
from llm_client.providers.types import CompletionResult, Message


class MockProvider(Provider):
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0
        self._model = MagicMock()
        self._model.name = "mock-model"

    async def complete(self, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return CompletionResult(content="mock", status=200)

    @property
    def model_name(self):
        return "mock-model"


@pytest.mark.asyncio
async def test_batch_completes_concurrently():
    # Setup: Provider with 0.1s delay
    delay = 0.1
    provider = MockProvider(delay=delay)

    # Engine with concurrency 10
    engine = ExecutionEngine(provider=provider, max_concurrency=10)

    # Create 10 requests
    # Create 10 requests
    specs = [RequestSpec(provider="mock", model="mock-model", messages=[Message.user(f"msg {i}")]) for i in range(10)]

    start = time.monotonic()
    results = await engine.batch_complete(specs)
    duration = time.monotonic() - start
    # Should take roughly 0.1s (plus overhead), definitely less than 1.0s
    assert len(results) == 10
    assert provider.calls == 10
    # Allow some buffer, but if sequential it would be > 1.0s
    assert duration < 0.3
    assert duration >= delay


@pytest.mark.asyncio
async def test_batch_respects_concurrency_limit():
    # Setup: 10 requests, limit 2
    # Should take 5 batches * 0.05s = 0.25s
    delay = 0.05
    provider = MockProvider(delay=delay)
    engine = ExecutionEngine(provider=provider, max_concurrency=2)

    specs = [RequestSpec(provider="mock", model="mock-model", messages=[Message.user(f"msg {i}")]) for i in range(10)]

    start = time.monotonic()
    results = await engine.batch_complete(specs)
    duration = time.monotonic() - start

    assert len(results) == 10
    # Expected: 5 rounds of parallel requests. 5 * 0.05 = 0.25s
    # Sequential would be 0.5s.
    # Parallel (unlimited) would be 0.05s.
    assert duration >= 0.20  # Lower bound for 5 rounds
    assert duration < 0.45  # Upper bound (allow overhead)


@pytest.mark.asyncio
async def test_batch_handles_errors_gracefully():
    provider = MockProvider(delay=0.01)

    # Make every other call fail
    async def side_effect(*args, **kwargs):
        provider.calls += 1
        if provider.calls % 2 == 0:
            raise ValueError("Boom")
        return CompletionResult(content="ok", status=200)

    provider.complete = AsyncMock(side_effect=side_effect)

    engine = ExecutionEngine(provider=provider, max_concurrency=5, retry=RetryConfig(attempts=1))
    specs = [RequestSpec(provider="mock", model="mock-model", messages=[Message.user(f"msg {i}")]) for i in range(4)]

    results = await engine.batch_complete(specs)

    assert len(results) == 4
    # Check statuses
    failures = [r for r in results if r.status == 500]
    successes = [r for r in results if r.status == 200]

    assert len(successes) == 2
    assert len(failures) == 2
    assert "Boom" in failures[0].error
