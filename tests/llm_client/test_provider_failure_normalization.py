from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.providers.types import Message, StreamEventType
from tests.llm_client.test_provider_overlap_contracts import (
    _build_anthropic_provider,
    _build_google_provider,
    _build_openai_provider,
)


@pytest.mark.asyncio
async def test_openai_provider_complete_attaches_normalized_rate_limit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.openai as openai_mod

    class _RateLimitError(Exception):
        status_code = 429

    monkeypatch.setattr(
        openai_mod,
        "openai",
        SimpleNamespace(
            APIConnectionError=type("APIConnectionError", (Exception,), {}),
            RateLimitError=_RateLimitError,
            APIStatusError=type("APIStatusError", (Exception,), {}),
        ),
    )

    provider = _build_openai_provider()

    async def _raise(**kwargs):
        _ = kwargs
        raise _RateLimitError("too many requests")

    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_raise)))

    result = await provider.complete([Message.user("hi")])

    assert result.status == 429
    assert result.raw_response["normalized_failure"]["category"] == "rate_limit"
    assert result.raw_response["normalized_failure"]["retryable"] is True


@pytest.mark.asyncio
async def test_anthropic_provider_stream_attaches_normalized_availability_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.anthropic as anthropic_mod

    class _APIConnectionError(Exception):
        pass

    monkeypatch.setattr(
        anthropic_mod,
        "anthropic",
        SimpleNamespace(
            APIConnectionError=_APIConnectionError,
            RateLimitError=type("RateLimitError", (Exception,), {}),
            APIStatusError=type("APIStatusError", (Exception,), {}),
        ),
    )

    provider = _build_anthropic_provider()

    def _raise(**kwargs):
        _ = kwargs
        raise _APIConnectionError("network down")

    provider.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: None, stream=_raise))

    events = [event async for event in provider.stream([Message.user("hi")])]

    assert len(events) == 2
    assert events[-1].type is StreamEventType.ERROR
    assert events[-1].data["normalized_failure"]["category"] == "availability"
    assert events[-1].data["normalized_failure"]["status"] == 503


@pytest.mark.asyncio
async def test_google_provider_complete_attaches_normalized_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    class _APIError(Exception):
        def __init__(self, code: int, message: str) -> None:
            super().__init__(message)
            self.code = code
            self.status_code = code
            self.message = message

    provider = _build_google_provider(monkeypatch)
    monkeypatch.setattr(google_mod, "genai_errors", SimpleNamespace(APIError=_APIError))

    async def _raise(**kwargs):
        _ = kwargs
        raise _APIError(401, "bad key")

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=_raise)))

    result = await provider.complete([Message.user("hi")])

    assert result.status == 401
    assert result.raw_response["normalized_failure"]["category"] == "authentication"
    assert result.raw_response["normalized_failure"]["retryable"] is False


@pytest.mark.asyncio
async def test_openai_provider_complete_attaches_normalized_quota_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.openai as openai_mod

    class _APIStatusError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    monkeypatch.setattr(
        openai_mod,
        "openai",
        SimpleNamespace(
            APIConnectionError=type("APIConnectionError", (Exception,), {}),
            RateLimitError=type("RateLimitError", (Exception,), {}),
            APIStatusError=_APIStatusError,
        ),
    )

    provider = _build_openai_provider()

    async def _raise(**kwargs):
        _ = kwargs
        raise _APIStatusError(402, "quota exceeded")

    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_raise)))

    result = await provider.complete([Message.user("hi")])

    assert result.status == 402
    assert result.raw_response["normalized_failure"]["category"] == "quota"
    assert result.raw_response["normalized_failure"]["retryable"] is False


@pytest.mark.asyncio
async def test_anthropic_provider_complete_attaches_normalized_content_filter_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.anthropic as anthropic_mod

    class _APIStatusError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code
            self.message = message

    monkeypatch.setattr(
        anthropic_mod,
        "anthropic",
        SimpleNamespace(
            APIConnectionError=type("APIConnectionError", (Exception,), {}),
            RateLimitError=type("RateLimitError", (Exception,), {}),
            APIStatusError=_APIStatusError,
        ),
    )

    provider = _build_anthropic_provider()

    async def _raise(**kwargs):
        _ = kwargs
        raise _APIStatusError(400, "content filter triggered by safety policy")

    provider.client = SimpleNamespace(messages=SimpleNamespace(create=_raise))

    result = await provider.complete([Message.user("hi")])

    assert result.status == 400
    assert result.raw_response["normalized_failure"]["category"] == "content_filter"
    assert result.raw_response["normalized_failure"]["retryable"] is False


@pytest.mark.asyncio
async def test_google_provider_complete_attaches_normalized_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    class _APIError(Exception):
        def __init__(self, code: int, message: str) -> None:
            super().__init__(message)
            self.code = code
            self.status_code = code
            self.message = message

    provider = _build_google_provider(monkeypatch)
    monkeypatch.setattr(google_mod, "genai_errors", SimpleNamespace(APIError=_APIError))

    async def _raise(**kwargs):
        _ = kwargs
        raise _APIError(400, "invalid request: schema validation failed")

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=_raise)))

    result = await provider.complete([Message.user("hi")])

    assert result.status == 400
    assert result.raw_response["normalized_failure"]["category"] == "validation"
    assert result.raw_response["normalized_failure"]["retryable"] is False


@pytest.mark.asyncio
async def test_google_provider_complete_attaches_normalized_context_length_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    class _APIError(Exception):
        def __init__(self, code: int, message: str) -> None:
            super().__init__(message)
            self.code = code
            self.status_code = code
            self.message = message

    provider = _build_google_provider(monkeypatch)
    monkeypatch.setattr(google_mod, "genai_errors", SimpleNamespace(APIError=_APIError))

    async def _raise(**kwargs):
        _ = kwargs
        raise _APIError(400, "maximum context length exceeded")

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=_raise)))

    result = await provider.complete([Message.user("hi")])

    assert result.status == 400
    assert result.raw_response["normalized_failure"]["category"] == "request"
    assert result.raw_response["normalized_failure"]["code"] == "ERR_1005"
