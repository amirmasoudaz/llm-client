from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_client.client import OpenAIClient
from llm_client.content import ContentRequestEnvelope, ContentResponseEnvelope
from llm_client.providers.types import CompletionResult, StreamEvent, StreamEventType, Usage


class _FakeEngine:
    def __init__(self) -> None:
        self.completed: list[ContentRequestEnvelope] = []
        self.streamed: list[ContentRequestEnvelope] = []

    async def complete_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        self.completed.append(request)
        return ContentResponseEnvelope.from_completion_result(
            CompletionResult(
                content="legacy envelope completion",
                usage=Usage(total_tokens=7),
                status=200,
                model=request.model,
            )
        )

    async def stream_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        self.streamed.append(request)
        yield StreamEvent(type=StreamEventType.TOKEN, data="legacy ")
        yield StreamEvent(type=StreamEventType.TOKEN, data="stream")
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=ContentResponseEnvelope.from_completion_result(
                CompletionResult(
                    content="legacy stream",
                    usage=Usage(total_tokens=9),
                    status=200,
                    model=request.model,
                )
            ),
        )


def _build_client() -> OpenAIClient:
    client = object.__new__(OpenAIClient)
    client.model = SimpleNamespace(category="completions", model_name="gpt-5-mini")
    client.engine = _FakeEngine()
    client.default_cache_collection = None
    client.cache = None
    return client


@pytest.mark.asyncio
async def test_openai_client_completion_uses_content_envelopes() -> None:
    client = _build_client()

    result = await client.get_response(messages=[{"role": "user", "content": "hello"}])

    assert isinstance(result, dict)
    assert result["output"] == "legacy envelope completion"
    assert result["body"]["completion"] == "legacy envelope completion"
    assert result["usage"]["total_tokens"] == 7
    assert len(client.engine.completed) == 1
    assert isinstance(client.engine.completed[0], ContentRequestEnvelope)
    assert client.engine.completed[0].model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_openai_client_streaming_uses_content_envelopes() -> None:
    client = _build_client()

    stream = await client.get_response(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_mode="sse",
    )

    payloads = [item async for item in stream]

    assert len(payloads) == 4
    assert len(client.engine.streamed) == 1
    assert isinstance(client.engine.streamed[0], ContentRequestEnvelope)
    assert client.engine.streamed[0].model == "gpt-5-mini"
    done_payload = json.loads(payloads[-1].split("data: ", 1)[1])
    assert done_payload["output"] == "legacy stream"
    assert done_payload["usage"]["total_tokens"] == 9
