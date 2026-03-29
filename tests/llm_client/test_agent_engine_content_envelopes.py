from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.agent.core import Agent
from llm_client.agent.result import AgentResult
from llm_client.content import ContentRequestEnvelope, ContentResponseEnvelope
from llm_client.providers.types import CompletionResult, StreamEvent, StreamEventType, ToolCall, Usage
from llm_client.spec import RequestContext
from llm_client.tools.decorators import tool


class _FakeProvider:
    model_name = "gpt-5-mini"
    model = SimpleNamespace(key="gpt-5-mini")


class _FakeEngine:
    def __init__(self) -> None:
        self.completed: list[ContentRequestEnvelope] = []
        self.streamed: list[ContentRequestEnvelope] = []
        self.complete_kwargs: list[dict] = []
        self.stream_kwargs: list[dict] = []

    async def complete_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        self.completed.append(request)
        self.complete_kwargs.append(dict(kwargs))
        return ContentResponseEnvelope.from_completion_result(
            CompletionResult(
                content="agent engine completion",
                usage=Usage(total_tokens=6),
                model=request.model,
                status=200,
            )
        )

    async def stream_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        self.streamed.append(request)
        self.stream_kwargs.append(dict(kwargs))
        yield StreamEvent(type=StreamEventType.TOKEN, data="agent ")
        yield StreamEvent(type=StreamEventType.TOKEN, data="engine")
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=ContentResponseEnvelope.from_completion_result(
                CompletionResult(
                    content="agent engine",
                    usage=Usage(total_tokens=7),
                    model=request.model,
                    status=200,
                )
            ),
        )


class _MultiTurnEngine(_FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self._complete_count = 0

    async def complete_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        self.completed.append(request)
        self.complete_kwargs.append(dict(kwargs))
        if self._complete_count == 0:
            self._complete_count += 1
            return ContentResponseEnvelope.from_completion_result(
                CompletionResult(
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments='{"text":"hello"}')],
                    usage=Usage(total_tokens=4),
                    model=request.model,
                    status=200,
                )
            )
        return ContentResponseEnvelope.from_completion_result(
            CompletionResult(
                content="agent finished",
                usage=Usage(total_tokens=5),
                model=request.model,
                status=200,
            )
        )


@pytest.mark.asyncio
async def test_agent_run_uses_content_envelope_engine_path() -> None:
    engine = _FakeEngine()
    agent = Agent(provider=_FakeProvider(), engine=engine)

    result = await agent.run("hello")

    assert result.status == "success"
    assert result.content == "agent engine completion"
    assert len(engine.completed) == 1
    assert isinstance(engine.completed[0], ContentRequestEnvelope)
    assert engine.completed[0].model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_agent_stream_uses_content_envelope_engine_path() -> None:
    engine = _FakeEngine()
    agent = Agent(provider=_FakeProvider(), engine=engine)

    events = [event async for event in agent.stream("hello")]

    assert len(engine.streamed) == 1
    assert isinstance(engine.streamed[0], ContentRequestEnvelope)
    assert engine.streamed[0].model == "gpt-5-mini"
    assert [event.data for event in events if event.type == StreamEventType.TOKEN] == ["agent ", "engine"]
    assert isinstance(events[-1].data, AgentResult)
    assert events[-1].data.content == "agent engine"


@tool
async def echo(text: str) -> str:
    return text


@pytest.mark.asyncio
async def test_agent_run_derives_per_turn_idempotency_keys_from_context() -> None:
    engine = _MultiTurnEngine()
    agent = Agent(provider=_FakeProvider(), engine=engine, tools=[echo])

    result = await agent.run("hello", context=RequestContext(request_id="run-123"))

    assert result.status == "success"
    assert [call["idempotency_key"] for call in engine.complete_kwargs] == [
        "run-123:turn-0:completion",
        "run-123:turn-1:completion",
    ]
    assert [call["context"].request_id for call in engine.complete_kwargs] == ["run-123", "run-123"]


@pytest.mark.asyncio
async def test_agent_stream_uses_explicit_per_turn_idempotency_key_base() -> None:
    engine = _FakeEngine()
    agent = Agent(provider=_FakeProvider(), engine=engine)

    events = [event async for event in agent.stream("hello", idempotency_key="agent-stream-base")]

    assert isinstance(events[-1].data, AgentResult)
    assert [call["idempotency_key"] for call in engine.stream_kwargs] == ["agent-stream-base:turn-0:stream"]
