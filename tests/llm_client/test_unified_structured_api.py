from __future__ import annotations

import pytest

from llm_client.content import TextBlock
from llm_client.providers.base import BaseProvider
from llm_client.providers.types import CompletionResult, Message, StreamEvent, StreamEventType, Usage
from llm_client.structured import (
    StructuredExecutionMode,
    StructuredOutputConfig,
    StructuredStreamEventType,
    structured,
)
from tests.llm_client.fakes import ScriptedProvider, ok_result


class _MinimalBaseProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self) -> None:
        super().__init__("gpt-5-mini")

    async def complete(self, messages, **kwargs):
        return CompletionResult(content='{"name":"Ada"}', usage=Usage(total_tokens=2), model="gpt-5-mini", status=200)

    async def stream(self, messages, **kwargs):
        yield StreamEvent(type=StreamEventType.TOKEN, data='{"name":"Ada"}')
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=CompletionResult(content='{"name":"Ada"}', usage=Usage(total_tokens=2), model="gpt-5-mini", status=200),
        )


@pytest.mark.asyncio
async def test_structured_dispatcher_complete_mode() -> None:
    provider = ScriptedProvider(complete_script=[ok_result('{"name":"Ada"}')])

    result = await structured(
        provider=provider,
        messages=[Message.user("extract a name")],
        config=StructuredOutputConfig(
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            max_repair_attempts=0,
        ),
        mode=StructuredExecutionMode.COMPLETE,
    )

    assert result.valid is True
    assert result.data == {"name": "Ada"}


@pytest.mark.asyncio
async def test_structured_dispatcher_validate_mode_accepts_content_blocks() -> None:
    result = await structured(
        content=[TextBlock('{"name":"Ada"}')],
        config=StructuredOutputConfig(
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        mode="validate",
    )

    assert result.valid is True
    assert result.data == {"name": "Ada"}


@pytest.mark.asyncio
async def test_structured_dispatcher_auto_mode_streams_when_requested() -> None:
    provider = ScriptedProvider(
        stream_script=[
            [
                StreamEvent(type=StreamEventType.TOKEN, data='{"name":"Ada"}'),
                StreamEvent(
                    type=StreamEventType.DONE,
                    data=CompletionResult(
                        content='{"name":"Ada"}',
                        usage=Usage(total_tokens=2),
                        model="gpt-5-mini",
                        status=200,
                    ),
                ),
            ]
        ]
    )

    events = [
        event
        async for event in structured(
            provider=provider,
            messages=[Message.user("extract a name")],
            config=StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
            stream=True,
        )
    ]

    assert events[-1].type == StructuredStreamEventType.DONE
    assert events[-1].data.valid is True


@pytest.mark.asyncio
async def test_base_provider_helpers_use_unified_structured_api() -> None:
    provider = _MinimalBaseProvider()

    complete_result = await provider.complete_structured(
        [Message.user("extract a name")],
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        max_repair_attempts=0,
    )

    stream_events = [
        event
        async for event in provider.stream_structured(
            [Message.user("extract a name")],
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            max_repair_attempts=0,
        )
    ]

    assert complete_result.valid is True
    assert stream_events[-1].type == StructuredStreamEventType.DONE
    assert stream_events[-1].data.valid is True
