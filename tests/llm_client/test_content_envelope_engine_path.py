from __future__ import annotations

import pytest

from llm_client.content import ContentMessage, ContentRequestEnvelope, ContentResponseEnvelope, TextBlock
from llm_client.engine import ExecutionEngine
from llm_client.providers.types import CompletionResult, Message, StreamEvent, StreamEventType, Usage
from llm_client.request_builders import build_content_request_envelope
from tests.llm_client.fakes import ScriptedProvider


def test_build_content_request_envelope_from_messages() -> None:
    envelope = build_content_request_envelope(
        messages=[Message.user("hello")],
        provider=type("OpenAIProviderFake", (), {"model_name": "gpt-5-mini", "model": type("Model", (), {"key": "gpt-5-mini"})()})(),
        request_kwargs={"temperature": 0.2, "response_format": "json_object"},
        model="gpt-5-mini",
        stream=True,
    )

    assert envelope.provider == "openaiproviderfake"
    assert envelope.model == "gpt-5-mini"
    assert envelope.stream is True
    assert envelope.messages[0].blocks[0] == TextBlock("hello")


def test_build_content_request_envelope_from_content_messages() -> None:
    envelope = build_content_request_envelope(
        messages=[ContentMessage(role=Message.user("hello").role, blocks=(TextBlock("hello"),))],
        provider=type("OpenAIProvider", (), {"model_name": "gpt-5-mini", "model": type("Model", (), {"key": "gpt-5-mini"})()})(),
        request_kwargs={"max_tokens": 42},
    )

    assert envelope.provider == "openai"
    assert envelope.max_tokens == 42
    assert envelope.messages[0].blocks[0] == TextBlock("hello")


@pytest.mark.asyncio
async def test_engine_complete_content_returns_content_response_envelope() -> None:
    engine = ExecutionEngine(
        provider=ScriptedProvider(
            complete_script=[
                CompletionResult(
                    content="hello",
                    usage=Usage(total_tokens=3),
                    model="gpt-5-mini",
                    status=200,
                )
            ]
        )
    )
    envelope = ContentRequestEnvelope(
        provider="scripted",
        model="gpt-5-mini",
        messages=(ContentMessage(role=Message.user("hello").role, blocks=(TextBlock("hello"),)),),
    )

    result = await engine.complete_content(envelope)

    assert isinstance(result, ContentResponseEnvelope)
    assert result.message.blocks[0] == TextBlock("hello")
    assert result.model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_engine_stream_content_converts_done_event_to_content_response_envelope() -> None:
    engine = ExecutionEngine(
        provider=ScriptedProvider(
            stream_script=[
                [
                    StreamEvent(type=StreamEventType.TOKEN, data="hel"),
                    StreamEvent(type=StreamEventType.TOKEN, data="lo"),
                    StreamEvent(
                        type=StreamEventType.DONE,
                        data=CompletionResult(
                            content="hello",
                            usage=Usage(total_tokens=3),
                            model="gpt-5-mini",
                            status=200,
                        ),
                    ),
                ]
            ]
        )
    )
    envelope = ContentRequestEnvelope(
        provider="scripted",
        model="gpt-5-mini",
        messages=(ContentMessage(role=Message.user("hello").role, blocks=(TextBlock("hello"),)),),
        stream=True,
    )

    events = [event async for event in engine.stream_content(envelope)]

    assert events[0].type == StreamEventType.META or events[0].type == StreamEventType.TOKEN
    assert events[-1].type == StreamEventType.DONE
    assert isinstance(events[-1].data, ContentResponseEnvelope)
    assert events[-1].data.message.blocks[0] == TextBlock("hello")
