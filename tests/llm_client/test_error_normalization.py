from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from llm_client.engine import ExecutionEngine, FailoverPolicy, RetryConfig
from llm_client.errors import (
    FailureCategory,
    normalize_exception,
    normalize_provider_failure,
    normalize_structured_failure,
    normalize_tool_failure,
)
from llm_client.hooks import HookManager
from llm_client.providers.types import Message, StreamEvent, StreamEventType
from llm_client.spec import RequestContext, RequestSpec
from llm_client.structured import StructuredExecutionFailure
from llm_client.tools.runtime import StructuredToolLoopError
from tests.llm_client.fakes import ScriptedProvider


@dataclass
class _StatusError(Exception):
    status: int
    message: str

    def __str__(self) -> str:
        return self.message


class _CollectingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, event: str, payload: dict, context: RequestContext) -> None:
        _ = context
        self.events.append((event, dict(payload)))


def _spec() -> RequestSpec:
    return RequestSpec(
        provider="openai",
        model="gpt-5-mini",
        messages=[Message.user("hello")],
    )


def test_normalize_provider_failure_maps_status_and_retryability() -> None:
    failure = normalize_provider_failure(
        status=429,
        message="rate limited",
        provider="openai",
        model="gpt-5-mini",
        operation="complete",
        request_id="req-1",
    )

    assert failure.category is FailureCategory.RATE_LIMIT
    assert failure.retryable is True
    assert failure.status == 429
    assert failure.remediation is not None


def test_normalize_exception_maps_timeout_and_status_exceptions() -> None:
    timeout_failure = normalize_exception(
        asyncio.TimeoutError(),
        provider="openai",
        model="gpt-5-mini",
        operation="complete",
    )
    status_failure = normalize_exception(
        _StatusError(status=401, message="bad auth"),
        provider="openai",
        model="gpt-5-mini",
        operation="complete",
    )

    assert timeout_failure.category is FailureCategory.TIMEOUT
    assert timeout_failure.retryable is True
    assert status_failure.category is FailureCategory.AUTHENTICATION
    assert status_failure.retryable is False


def test_normalize_tool_and_structured_failures() -> None:
    tool_failure = normalize_tool_failure(
        StructuredToolLoopError(
            code="tool_not_allowed",
            message="tool denied",
            category="policy_denied",
            retryable=False,
            details={"tool": "search"},
        ),
        operation="tools",
    )
    structured_failure = normalize_structured_failure(
        StructuredExecutionFailure(
            code="structured_output_failed",
            message="schema mismatch",
            category="validation",
            retryable=True,
            details={"field": "value"},
        ),
        operation="structured",
        validation_errors=["field is required"],
    )

    assert tool_failure.category is FailureCategory.TOOL_POLICY
    assert tool_failure.details["tool"] == "search"
    assert structured_failure.category is FailureCategory.STRUCTURED_OUTPUT
    assert structured_failure.retryable is True
    assert structured_failure.details["validation_errors"] == ["field is required"]


@pytest.mark.asyncio
async def test_engine_request_error_emits_normalized_failure_payload() -> None:
    hook = _CollectingHook()
    provider = ScriptedProvider(complete_script=[RuntimeError("boom")])
    engine = ExecutionEngine(
        provider=provider,
        hooks=HookManager([hook]),
        retry=RetryConfig(attempts=1, backoff=0.0),
    )

    result = await engine.complete(_spec())

    assert result.ok is False
    request_error = next(payload for name, payload in hook.events if name == "request.error")
    normalized = request_error["normalized_failure"]
    assert normalized["category"] == FailureCategory.INTERNAL.value
    assert normalized["message"] == "boom"


@pytest.mark.asyncio
async def test_engine_stream_error_emits_normalized_failure_payload() -> None:
    hook = _CollectingHook()
    provider = ScriptedProvider(stream_script=[RuntimeError("stream boom")])
    engine = ExecutionEngine(
        provider=provider,
        hooks=HookManager([hook]),
        failover_policy=FailoverPolicy(fallback_on_exceptions=False),
    )

    events = [event async for event in engine.stream(_spec())]

    assert len(events) == 1
    assert events[0].type is StreamEventType.ERROR
    normalized = events[0].data["normalized_failure"]
    assert normalized["category"] == FailureCategory.INTERNAL.value
    stream_error = next(payload for name, payload in hook.events if name == "stream.error")
    assert stream_error["category"] == FailureCategory.INTERNAL.value
