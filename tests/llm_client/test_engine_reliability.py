from __future__ import annotations

import asyncio

import pytest

from llm_client.cancellation import CancellationToken, CancelledError
from llm_client.engine import ExecutionEngine, FailoverPolicy, RetryConfig
from llm_client.hooks import HookManager
from llm_client.idempotency import IdempotencyTracker
from llm_client.provider_registry import ProviderCapabilities, ProviderDescriptor, ProviderRegistry
from llm_client.providers.types import Message
from llm_client.providers.types import CompletionResult, StreamEvent, StreamEventType, Usage
from llm_client.resilience import CircuitBreakerConfig
from llm_client.routing import RegistryRouter, StaticRouter
from llm_client.spec import RequestContext, RequestSpec
from tests.llm_client.fakes import ScriptedProvider, error_result, ok_result


def _spec() -> RequestSpec:
    return RequestSpec(
        provider="openai",
        model="gpt-5-mini",
        messages=[Message.user("hello")],
    )


class _CollectingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, event: str, payload: dict, context: RequestContext) -> None:
        self.events.append((event, dict(payload)))


class _SlowProvider(ScriptedProvider):
    def __init__(self, *, delay: float, model_name: str = "gpt-5-mini") -> None:
        super().__init__(model_name=model_name)
        self._delay = delay

    async def complete(self, messages, **kwargs):
        _ = (messages, kwargs)
        await asyncio.sleep(self._delay)
        return ok_result("slow-ok", model=self.model_name)

    async def stream(self, messages, **kwargs):
        _ = (messages, kwargs)
        await asyncio.sleep(self._delay)
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=ok_result("slow-ok", model=self.model_name),
        )


@pytest.mark.asyncio
async def test_engine_retries_transient_complete_failure() -> None:
    provider = ScriptedProvider(complete_script=[error_result(500, "temporary"), ok_result("ok")])
    engine = ExecutionEngine(provider=provider, retry=RetryConfig(attempts=2, backoff=0.0, max_backoff=0.0))

    result = await engine.complete(_spec())

    assert result.ok is True
    assert result.content == "ok"
    assert len(provider.complete_calls) == 2


@pytest.mark.asyncio
async def test_engine_falls_back_to_second_provider() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(router=StaticRouter([first, second]), retry=RetryConfig(attempts=1, backoff=0.0))

    result = await engine.complete(_spec())

    assert result.ok is True
    assert result.content == "from fallback"
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 1


@pytest.mark.asyncio
async def test_engine_stream_falls_back_before_tokens_seen() -> None:
    first = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.ERROR, data={"status": 503, "error": "temporary"}),
        ]]
    )
    second = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(
                type=StreamEventType.DONE,
                data=CompletionResult(content="hi", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2), model="gpt-5-mini", status=200),
            ),
        ]]
    )
    engine = ExecutionEngine(router=StaticRouter([first, second]))

    events = [event async for event in engine.stream(_spec())]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert events[-1].data.content == "hi"
    assert len(first.stream_calls) == 1
    assert len(second.stream_calls) == 1


@pytest.mark.asyncio
async def test_engine_complete_honors_cancellation_before_attempt() -> None:
    provider = ScriptedProvider(complete_script=[ok_result("should not happen")])
    engine = ExecutionEngine(provider=provider)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        await engine.complete(_spec(), context=RequestContext(cancellation_token=token))

    assert provider.complete_calls == []


@pytest.mark.asyncio
async def test_engine_complete_standardizes_timeout_status() -> None:
    provider = _SlowProvider(delay=0.05)
    engine = ExecutionEngine(provider=provider, retry=RetryConfig(attempts=1, backoff=0.0))

    result = await engine.complete(_spec(), timeout=0.01)

    assert result.ok is False
    assert result.status == 408
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_engine_retries_standardized_retryable_statuses() -> None:
    provider = ScriptedProvider(complete_script=[error_result(425, "too early"), ok_result("ok")])
    engine = ExecutionEngine(provider=provider, retry=RetryConfig(attempts=2, backoff=0.0, max_backoff=0.0))

    result = await engine.complete(_spec())

    assert result.ok is True
    assert len(provider.complete_calls) == 2


@pytest.mark.asyncio
async def test_engine_complete_reuses_cached_idempotent_result() -> None:
    provider = ScriptedProvider(complete_script=[ok_result("once")])
    tracker = IdempotencyTracker()
    engine = ExecutionEngine(provider=provider, idempotency_tracker=tracker)

    first = await engine.complete(_spec(), idempotency_key="idem-complete")
    second = await engine.complete(_spec(), idempotency_key="idem-complete")

    assert first.ok is True
    assert second.ok is True
    assert second.content == "once"
    assert len(provider.complete_calls) == 1
    assert tracker.completed_count == 1


@pytest.mark.asyncio
async def test_engine_stream_reuses_cached_idempotent_terminal_result() -> None:
    provider = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(type=StreamEventType.DONE, data=ok_result("hi")),
        ]]
    )
    tracker = IdempotencyTracker()
    engine = ExecutionEngine(provider=provider, idempotency_tracker=tracker)

    first_events = [event async for event in engine.stream(_spec(), idempotency_key="idem-stream")]
    second_events = [event async for event in engine.stream(_spec(), idempotency_key="idem-stream")]

    assert [event.type for event in first_events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert [event.type for event in second_events] == [StreamEventType.DONE]
    assert second_events[0].data.content == "hi"
    assert len(provider.stream_calls) == 1
    assert tracker.completed_count == 1


@pytest.mark.asyncio
async def test_engine_stream_timeout_falls_back_before_tokens_seen() -> None:
    first = _SlowProvider(delay=0.05)
    second = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(
                type=StreamEventType.DONE,
                data=CompletionResult(content="hi", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2), model="gpt-5-mini", status=200),
            ),
        ]]
    )
    engine = ExecutionEngine(router=StaticRouter([first, second]))

    events = [event async for event in engine.stream(_spec(), timeout=0.01)]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert events[-1].data.content == "hi"


@pytest.mark.asyncio
async def test_engine_emits_router_selection_and_fallback_events() -> None:
    hook = _CollectingHook()
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        hooks=HookManager([hook]),
        retry=RetryConfig(attempts=1, backoff=0.0),
    )

    result = await engine.complete(_spec())

    assert result.ok is True
    event_names = [name for name, _ in hook.events]
    assert "router.selection" in event_names
    assert "router.fallback" in event_names
    selection_payload = next(payload for name, payload in hook.events if name == "router.selection")
    assert selection_payload["requested_model"] == "gpt-5-mini"
    assert selection_payload["selected_count"] == 2


@pytest.mark.asyncio
async def test_engine_request_end_payload_includes_usage_provider_and_model() -> None:
    hook = _CollectingHook()
    provider = ScriptedProvider(complete_script=[ok_result("ok", model="gpt-5-mini")])
    engine = ExecutionEngine(provider=provider, hooks=HookManager([hook]))

    result = await engine.complete(_spec())

    assert result.ok is True
    request_end = next(payload for name, payload in hook.events if name == "request.end")
    assert request_end["provider"] == "openai"
    assert request_end["model"] == "gpt-5-mini"
    assert request_end["usage"]["total_tokens"] == result.usage.total_tokens


@pytest.mark.asyncio
async def test_engine_idempotent_request_end_payload_preserves_provider_model_and_usage() -> None:
    hook = _CollectingHook()
    provider = ScriptedProvider(complete_script=[ok_result("once", model="gpt-5-mini")])
    tracker = IdempotencyTracker()
    engine = ExecutionEngine(provider=provider, idempotency_tracker=tracker, hooks=HookManager([hook]))

    await engine.complete(_spec(), idempotency_key="idem-complete")
    second = await engine.complete(_spec(), idempotency_key="idem-complete")

    assert second.ok is True
    request_end_payloads = [payload for name, payload in hook.events if name == "request.end"]
    replay_end = request_end_payloads[-1]
    assert replay_end["provider"] == "openai"
    assert replay_end["model"] == "gpt-5-mini"
    assert replay_end["usage"]["total_tokens"] == second.usage.total_tokens


@pytest.mark.asyncio
async def test_engine_emits_request_lifecycle_hooks_and_diagnostics() -> None:
    hook = _CollectingHook()
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        hooks=HookManager([hook]),
        retry=RetryConfig(attempts=1, backoff=0.0),
    )

    result = await engine.complete(_spec())

    assert result.ok is True
    event_names = [name for name, _ in hook.events]
    assert event_names.count("request.pre_dispatch") == 2
    assert event_names.count("request.post_response") == 2
    assert "request.diagnostics" in event_names
    diagnostics = next(payload for name, payload in hook.events if name == "request.diagnostics")
    assert diagnostics["attempts"] == 2
    assert diagnostics["fallbacks"] == 1
    assert len(diagnostics["providers_selected"]) == 2
    assert len(diagnostics["providers_dispatched"]) == 2
    assert diagnostics["providers_tried"] == diagnostics["providers_dispatched"]
    assert diagnostics["final_status"] == 200
    assert diagnostics["cache_hit"] is False


@pytest.mark.asyncio
async def test_engine_emits_stream_lifecycle_hooks_and_diagnostics() -> None:
    hook = _CollectingHook()
    first = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.ERROR, data={"status": 503, "error": "temporary"}),
        ]]
    )
    second = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(type=StreamEventType.DONE, data=ok_result("hi")),
        ]]
    )
    engine = ExecutionEngine(router=StaticRouter([first, second]), hooks=HookManager([hook]))

    events = [event async for event in engine.stream(_spec())]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    event_names = [name for name, _ in hook.events]
    assert event_names.count("stream.pre_dispatch") == 2
    assert "stream.diagnostics" in event_names
    diagnostics = next(payload for name, payload in hook.events if name == "stream.diagnostics")
    assert diagnostics["fallbacks"] == 1
    assert len(diagnostics["providers_selected"]) == 2
    assert len(diagnostics["providers_dispatched"]) == 2
    assert diagnostics["providers_tried"] == diagnostics["providers_dispatched"]
    assert diagnostics["token_seen"] is True
    assert diagnostics["final_status"] == 200


@pytest.mark.asyncio
async def test_engine_updates_registry_router_health_on_fallback() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")], model_name="gpt-5-mini")
    second = ScriptedProvider(complete_script=[ok_result("from fallback")], model_name="gemini-2.0-flash")
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=lambda model, **kwargs: first,
            default_model="gpt-5-mini",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=lambda model, **kwargs: second,
            default_model="gemini-2.0-flash",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=20,
        )
    )
    router = RegistryRouter(registry=registry, unhealthy_after=1)
    engine = ExecutionEngine(router=router, retry=RetryConfig(attempts=1, backoff=0.0))

    result = await engine.complete(
        RequestSpec(
            provider="unknown",
            model=None,
            messages=[Message.user("hello")],
        )
    )

    assert result.ok is True
    assert router.get_provider_health("openai").degraded is True
    assert router.get_provider_health("google").successes == 1


@pytest.mark.asyncio
async def test_circuit_breaker_ignores_non_trip_statuses() -> None:
    provider = ScriptedProvider(complete_script=[error_result(400, "bad request"), error_result(400, "bad request"), ok_result("ok")])
    engine = ExecutionEngine(
        provider=provider,
        retry=RetryConfig(attempts=1, backoff=0.0),
        breaker_config=CircuitBreakerConfig(failure_threshold=1),
    )

    first = await engine.complete(_spec())
    second = await engine.complete(_spec())

    assert first.status == 400
    assert second.status == 400
    assert len(provider.complete_calls) == 2
    assert engine._get_breaker(engine._provider_id(provider)).get_state()["is_open"] is False


@pytest.mark.asyncio
async def test_engine_falls_back_when_primary_circuit_is_open() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")], model_name="gpt-5-mini")
    second = ScriptedProvider(
        complete_script=[ok_result("from fallback"), ok_result("from fallback")],
        model_name="gemini-2.0-flash",
    )
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        retry=RetryConfig(attempts=1, backoff=0.0),
        breaker_config=CircuitBreakerConfig(failure_threshold=1),
    )

    initial = await engine.complete(_spec())
    repeated = await engine.complete(_spec())

    assert initial.ok is True
    assert repeated.ok is True
    assert repeated.content == "from fallback"
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 2


@pytest.mark.asyncio
async def test_engine_failover_policy_can_limit_provider_attempts() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        retry=RetryConfig(attempts=1, backoff=0.0),
        failover_policy=FailoverPolicy(max_providers=1),
    )

    result = await engine.complete(_spec())

    assert result.ok is False
    assert result.status == 503
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 0
