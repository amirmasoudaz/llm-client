from __future__ import annotations

import pytest

from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager
from llm_client.providers.types import Message, StreamEvent, StreamEventType
from llm_client.spec import RequestContext, RequestSpec
from tests.llm_client.fakes import ScriptedProvider, error_result, ok_result


def _spec() -> RequestSpec:
    return RequestSpec(
        provider="openai",
        model="gpt-5-mini",
        messages=[Message.user("hello")],
    )


@pytest.mark.asyncio
async def test_engine_diagnostics_recorder_captures_request_summary() -> None:
    recorder = EngineDiagnosticsRecorder()
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=HooklessStaticRouter([first, second]),
        hooks=HookManager([recorder]),
        retry=RetryConfig(attempts=1, backoff=0.0),
    )
    context = RequestContext(request_id="request-123")

    result = await engine.complete(_spec(), context=context)

    assert result.ok is True
    snapshot = recorder.latest_request("request-123")
    assert snapshot is not None
    assert snapshot.kind == "request"
    assert snapshot.payload["attempts"] == 2
    assert snapshot.payload["fallbacks"] == 1
    assert len(snapshot.payload["providers_selected"]) == 2
    assert len(snapshot.payload["providers_dispatched"]) == 2
    assert snapshot.payload["providers_tried"] == snapshot.payload["providers_dispatched"]
    assert snapshot.payload["final_status"] == 200


@pytest.mark.asyncio
async def test_engine_diagnostics_recorder_captures_stream_summary() -> None:
    recorder = EngineDiagnosticsRecorder()
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
    engine = ExecutionEngine(
        router=HooklessStaticRouter([first, second]),
        hooks=HookManager([recorder]),
    )
    context = RequestContext(request_id="stream-123")

    events = [event async for event in engine.stream(_spec(), context=context)]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    snapshot = recorder.latest_stream("stream-123")
    assert snapshot is not None
    assert snapshot.kind == "stream"
    assert snapshot.payload["fallbacks"] == 1
    assert len(snapshot.payload["providers_selected"]) == 2
    assert len(snapshot.payload["providers_dispatched"]) == 2
    assert snapshot.payload["providers_tried"] == snapshot.payload["providers_dispatched"]
    assert snapshot.payload["token_seen"] is True
    assert snapshot.payload["final_status"] == 200


class HooklessStaticRouter:
    def __init__(self, providers):
        self._providers = list(providers)

    def select(self, spec):  # type: ignore[no-untyped-def]
        _ = spec
        return list(self._providers)
