from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.providers.base import BaseProvider
from llm_client.providers.types import CompletionResult, Usage
from llm_client.retry_policy import (
    DEFAULT_RETRYABLE_STATUSES,
    compute_backoff_delay,
    extract_retry_after_seconds,
    is_retryable_status,
)


def test_retry_policy_default_statuses_cover_engine_and_structured_cases() -> None:
    assert DEFAULT_RETRYABLE_STATUSES == (0, 408, 409, 425, 429, 500, 502, 503, 504)
    assert is_retryable_status(408) is True
    assert is_retryable_status(425) is True
    assert is_retryable_status(404) is False


def test_retry_policy_extracts_retry_after_from_result_or_error() -> None:
    result = CompletionResult(
        status=429,
        error="rate limited",
        raw_response=SimpleNamespace(headers={"Retry-After": "2.5"}),
    )
    error = RuntimeError("boom")
    error.response = SimpleNamespace(headers={"Retry-After": "1"})  # type: ignore[attr-defined]

    assert extract_retry_after_seconds(result) == 2.5
    assert extract_retry_after_seconds(error) == 1.0


def test_retry_policy_compute_backoff_delay_obeys_retry_after_and_max() -> None:
    assert compute_backoff_delay(attempt=0, base_backoff=1.0, retry_after=3.0) == 3.0
    delay = compute_backoff_delay(attempt=3, base_backoff=1.0, max_backoff=4.0, jitter_ratio=0.0)
    assert delay == 4.0


@pytest.mark.asyncio
async def test_provider_base_retry_wrapper_retries_standardized_statuses() -> None:
    calls: list[int] = []

    async def _operation():
        calls.append(len(calls))
        if len(calls) == 1:
            return CompletionResult(status=425, error="too early", usage=Usage())
        return CompletionResult(status=200, content="ok", usage=Usage(total_tokens=1))

    result = await BaseProvider._with_retry(_operation, attempts=2, backoff=0.0)

    assert result.ok is True
    assert len(calls) == 2
