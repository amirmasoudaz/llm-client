from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm_client.providers.google as google_mod
from llm_client.providers.anthropic import AnthropicProvider
from llm_client.providers.google import GoogleProvider
from llm_client.providers.openai import OpenAIProvider
from llm_client.providers.types import Message, StreamEventType
from tests.llm_client.fakes import FakeModel
from tests.llm_client.stream_transcripts import (
    AnthropicStreamManager,
    AsyncSequence,
    anthropic_text_events,
    anthropic_tool_call_events,
    google_text_chunks,
    google_tool_call_chunks,
    openai_text_chunks,
    openai_tool_call_chunks,
)


class _LimitContext:
    output_tokens: int = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _NoopLimiter:
    def limit(self, **kwargs):
        _ = kwargs
        return _LimitContext()


def _openai_provider(model_name: str = "gpt-5-mini") -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._model = FakeModel(key=model_name, model_name=model_name)
    provider.limiter = _NoopLimiter()
    return provider


def _google_provider(monkeypatch: pytest.MonkeyPatch, model_name: str = "gemini-2.0-flash") -> GoogleProvider:
    class _DummyPart:
        def __init__(self, *, text=None, function_call=None, function_response=None):
            self.text = text
            self.function_call = function_call
            self.function_response = function_response

    monkeypatch.setattr(
        google_mod,
        "types",
        SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            AutomaticFunctionCallingConfig=lambda disable: SimpleNamespace(disable=disable),
            Content=lambda role, parts: SimpleNamespace(role=role, parts=parts),
            Part=SimpleNamespace(
                from_text=lambda text: _DummyPart(text=text),
                from_function_call=lambda name, args: _DummyPart(function_call=SimpleNamespace(name=name, args=args)),
                from_function_response=lambda name, response: _DummyPart(
                    function_response=SimpleNamespace(name=name, response=response)
                ),
            ),
        ),
    )
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._model = FakeModel(key=model_name, model_name=model_name)
    provider.limiter = _NoopLimiter()
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace()))
    return provider


def _anthropic_provider(model_name: str = "claude-4-5-sonnet") -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._model = FakeModel(key=model_name, model_name=model_name)
    provider.limiter = _NoopLimiter()
    provider.max_tokens = 256
    provider.default_temperature = None
    provider.client = SimpleNamespace(messages=SimpleNamespace())
    return provider


@pytest.mark.asyncio
async def test_openai_stream_transcript_parses_text_and_tool_calls() -> None:
    provider = _openai_provider()

    async def _streaming_create(**kwargs):
        _ = kwargs
        return AsyncSequence(openai_tool_call_chunks())

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_streaming_create)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: None))),
    )

    events = [event async for event in provider.stream([Message.user("call a tool")], tools=[])]

    assert [event.type for event in events] == [
        StreamEventType.META,
        StreamEventType.TOOL_CALL_START,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOOL_CALL_END,
        StreamEventType.USAGE,
        StreamEventType.DONE,
    ]
    assert events[-1].data.tool_calls[0].name == "lookup"
    assert events[-1].data.tool_calls[0].arguments == '{"q":"x"}'


@pytest.mark.asyncio
async def test_google_stream_transcript_parses_text_and_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _google_provider(monkeypatch)

    async def _generate_content_stream(**kwargs):
        _ = kwargs
        return AsyncSequence(google_tool_call_chunks())

    provider._client.aio.models.generate_content_stream = _generate_content_stream

    events = [event async for event in provider.stream([Message.user("call a tool")], tools=[])]

    assert events[0].type is StreamEventType.META
    assert StreamEventType.TOOL_CALL_START in [event.type for event in events]
    assert events[-1].type is StreamEventType.DONE
    assert events[-1].data.tool_calls[0].name == "lookup"


@pytest.mark.asyncio
async def test_anthropic_stream_transcript_parses_text_and_tool_calls() -> None:
    provider = _anthropic_provider()

    def _stream(**kwargs):
        _ = kwargs
        return AnthropicStreamManager(anthropic_tool_call_events())

    provider.client.messages.stream = _stream

    events = [event async for event in provider.stream([Message.user("call a tool")], tools=[])]

    assert events[0].type is StreamEventType.META
    assert StreamEventType.TOOL_CALL_START in [event.type for event in events]
    assert events[-1].type is StreamEventType.DONE
    assert events[-1].data.tool_calls[0].arguments == '{"q":"x"}'


def test_openai_complete_content_parsing_coerces_part_lists() -> None:
    content = OpenAIProvider._coerce_chat_message_content(
        [
            {"text": "hello "},
            {"content": "world"},
        ]
    )

    assert content == "hello world"


def test_anthropic_complete_helpers_parse_tool_calls_and_usage() -> None:
    provider = _anthropic_provider()

    text_content, tool_calls = provider._extract_tool_calls_from_response(
        [
            SimpleNamespace(type="text", text="anthropic-ok"),
            SimpleNamespace(type="tool_use", id="toolu_1", name="lookup", input={"q": "x"}),
        ]
    )
    usage = provider._parse_anthropic_usage(SimpleNamespace(input_tokens=2, output_tokens=1))

    assert text_content == "anthropic-ok"
    assert tool_calls is not None
    assert tool_calls[0].name == "lookup"
    assert usage.total_tokens == 3


@pytest.mark.asyncio
async def test_google_complete_parses_usage_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _google_provider(monkeypatch)

    async def _generate_content(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            parts=[SimpleNamespace(text="google-ok")],
            usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3),
        )

    provider._client.aio.models.generate_content = _generate_content

    result = await provider.complete([Message.user("hello")])

    assert result.content == "google-ok"
    assert result.usage.total_tokens == 3


def test_stream_transcript_fixture_module_exposes_provider_shaped_sequences() -> None:
    assert openai_text_chunks()
    assert google_text_chunks()
    assert anthropic_text_events()
