from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.content import ContentRequestEnvelope, ContentResponseEnvelope
from llm_client.providers.openai import OpenAIProvider
from llm_client.providers.types import CompletionResult, Message, StreamEvent, StreamEventType, Usage
from llm_client.structured import (
    StructuredOutputConfig,
    StructuredStreamEventType,
    build_structured_response_format,
    extract_structured,
    stream_structured,
)
from tests.llm_client.fakes import ScriptedProvider, ok_result


class OpenAIProviderFake(ScriptedProvider):
    pass


class _StreamEngine:
    def __init__(self, events: list[StreamEvent]) -> None:
        self.provider = OpenAIProviderFake(model_name="gpt-5-mini")
        self.calls: list[dict[str, object]] = []
        self._events = list(events)

    async def stream_content(self, envelope, *, context=None, **kwargs):
        self.calls.append({"envelope": envelope, "context": context, "kwargs": kwargs})
        for event in self._events:
            if event.type == StreamEventType.DONE and isinstance(event.data, CompletionResult):
                yield StreamEvent(
                    type=event.type,
                    data=ContentResponseEnvelope.from_completion_result(event.data),
                )
                continue
            yield event


def test_build_structured_response_format_normalizes_openai_json_schema() -> None:
    response_format = build_structured_response_format(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:test",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "child": {
                    "type": "object",
                    "$id": "urn:child",
                    "properties": {"value": {"type": "string"}},
                },
            },
            "required": ["name"],
        },
        provider="openai",
        name="bad schema name v1!",
    )

    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["name"] == "bad_schema_name_v1"
    assert "$schema" not in json_schema["schema"]
    assert "$id" not in json_schema["schema"]
    assert "$id" not in json_schema["schema"]["properties"]["child"]


def test_openai_provider_normalizes_json_schema_response_format_dict() -> None:
    response_format = OpenAIProvider._normalize_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "bad schema name!",
                "strict": True,
                "schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        [{"role": "user", "content": "Return json please"}],
    )

    assert isinstance(response_format, dict)
    assert response_format["json_schema"]["name"] == "bad_schema_name"
    assert "$schema" not in response_format["json_schema"]["schema"]


@pytest.mark.asyncio
async def test_extract_structured_uses_json_schema_for_openai_like_providers() -> None:
    provider = OpenAIProviderFake(
        complete_script=[ok_result('{"name":"Ada"}')]
    )

    result = await extract_structured(
        provider,
        [Message.user("extract a name")],
        StructuredOutputConfig(
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            max_repair_attempts=0,
        ),
    )

    assert result.valid is True
    response_format = provider.complete_calls[0]["kwargs"]["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"]["properties"]["name"]["type"] == "string"
    assert "$schema" not in response_format["json_schema"]["schema"]


@pytest.mark.asyncio
async def test_stream_structured_emits_result_and_done_events() -> None:
    provider = ScriptedProvider(
        stream_script=[
            [
                StreamEvent(type=StreamEventType.META, data={"model": "gpt-5-mini"}),
                StreamEvent(type=StreamEventType.TOKEN, data='{"name":'),
                StreamEvent(type=StreamEventType.TOKEN, data='"Ada"}'),
                StreamEvent(
                    type=StreamEventType.DONE,
                    data=CompletionResult(
                        content='{"name":"Ada"}',
                        usage=Usage(total_tokens=4),
                        model="gpt-5-mini",
                        status=200,
                    ),
                ),
            ]
        ]
    )

    events = [
        event
        async for event in stream_structured(
            provider,
            [Message.user("extract a name")],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                max_repair_attempts=0,
            ),
        )
    ]

    assert [event.type for event in events] == [
        StructuredStreamEventType.RAW_EVENT,
        StructuredStreamEventType.CONTENT_DELTA,
        StructuredStreamEventType.CONTENT_DELTA,
        StructuredStreamEventType.RESULT,
        StructuredStreamEventType.DONE,
    ]
    assert events[-1].data.valid is True
    assert events[-1].data.data == {"name": "Ada"}
    assert provider.stream_calls[0]["kwargs"]["response_format"] == "json_object"


@pytest.mark.asyncio
async def test_stream_structured_prefers_engine_and_preserves_response_format() -> None:
    engine = _StreamEngine(
        [
            StreamEvent(type=StreamEventType.TOKEN, data='{"name":"Ada"}'),
            StreamEvent(
                type=StreamEventType.DONE,
                data=CompletionResult(
                    content='{"name":"Ada"}',
                    usage=Usage(total_tokens=3),
                    model="gpt-5-mini",
                    status=200,
                ),
            ),
        ]
    )

    events = [
        event
        async for event in stream_structured(
            provider=SimpleNamespace(model_name="gpt-5-mini", model=SimpleNamespace(key="gpt-5-mini")),
            messages=[Message.user("extract a name")],
            config=StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                max_repair_attempts=0,
            ),
            engine=engine,
            model="gpt-5-mini",
        )
    ]

    assert events[-1].type == StructuredStreamEventType.DONE
    assert events[-1].data.valid is True
    assert len(engine.calls) == 1
    envelope = engine.calls[0]["envelope"]
    assert isinstance(envelope, ContentRequestEnvelope)
    assert isinstance(envelope.response_format, dict)
    assert envelope.response_format["type"] == "json_schema"
