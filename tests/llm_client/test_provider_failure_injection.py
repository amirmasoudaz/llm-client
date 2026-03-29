from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.providers.types import Message, StreamEventType
from tests.llm_client.stream_transcripts import AnthropicStreamManager
from tests.llm_client.test_provider_overlap_contracts import (
    _build_anthropic_provider,
    _build_google_provider,
    _build_openai_provider,
)


@pytest.mark.asyncio
async def test_openai_provider_complete_normalizes_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.openai as openai_mod

    monkeypatch.setattr(
        openai_mod,
        "openai",
        SimpleNamespace(
            APIConnectionError=type("APIConnectionError", (Exception,), {}),
            RateLimitError=type("RateLimitError", (Exception,), {}),
            APIStatusError=type("APIStatusError", (Exception,), {}),
        ),
    )

    provider = _build_openai_provider()

    async def _bad_create(**kwargs):
        _ = kwargs
        return SimpleNamespace(choices=None, usage=None, model="gpt-5-mini")

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_bad_create)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=_bad_create))),
    )

    result = await provider.complete([Message.user("hi")])

    assert result.ok is False
    assert result.status == 500
    assert result.raw_response["normalized_failure"]["category"] in {"internal", "provider"}


@pytest.mark.asyncio
async def test_google_provider_stream_normalizes_5xx_interruption(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    class _APIError(Exception):
        def __init__(self, code: int, message: str) -> None:
            super().__init__(message)
            self.code = code
            self.status_code = code
            self.message = message

    provider = _build_google_provider(monkeypatch)
    monkeypatch.setattr(google_mod, "genai_errors", SimpleNamespace(APIError=_APIError))

    async def _stream_fail(**kwargs):
        _ = kwargs
        raise _APIError(503, "upstream unavailable")

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_stream_fail)))

    events = [event async for event in provider.stream([Message.user("hi")])]

    assert events[0].type is StreamEventType.META
    assert events[-1].type is StreamEventType.ERROR
    assert events[-1].data["normalized_failure"]["status"] == 503
    assert events[-1].data["normalized_failure"]["retryable"] is True


@pytest.mark.asyncio
async def test_anthropic_provider_stream_normalizes_malformed_event_payload() -> None:
    provider = _build_anthropic_provider()

    def _stream(**kwargs):
        _ = kwargs
        return AnthropicStreamManager(
            [
                SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=2))),
                SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta")),
            ]
        )

    provider.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: None, stream=_stream))

    events = [event async for event in provider.stream([Message.user("hi")])]

    assert events[0].type is StreamEventType.META
    assert events[-1].type is StreamEventType.ERROR
    assert events[-1].data["normalized_failure"]["category"] in {"internal", "provider"}


@pytest.mark.asyncio
async def test_openai_provider_stream_normalizes_midstream_interruption(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.openai as openai_mod

    monkeypatch.setattr(
        openai_mod,
        "openai",
        SimpleNamespace(
            APIConnectionError=type("APIConnectionError", (Exception,), {}),
            RateLimitError=type("RateLimitError", (Exception,), {}),
            APIStatusError=type("APIStatusError", (Exception,), {}),
        ),
    )

    provider = _build_openai_provider()

    class _BrokenStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("stream interrupted")

    async def _streaming_create(**kwargs):
        _ = kwargs
        return _BrokenStream()

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_streaming_create)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: None))),
    )

    events = [event async for event in provider.stream([Message.user("hi")])]

    assert events[0].type is StreamEventType.META
    assert events[-1].type is StreamEventType.ERROR
    assert events[-1].data["normalized_failure"]["category"] in {"internal", "provider"}
