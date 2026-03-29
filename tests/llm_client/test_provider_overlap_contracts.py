from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from llm_client.providers.anthropic import AnthropicProvider
from llm_client.providers.base import BaseProvider
from llm_client.providers.google import GoogleProvider
from llm_client.providers.openai import OpenAIProvider

from tests.llm_client.provider_contracts import (
    ProviderContractSpec,
    assert_provider_embedding_contract,
    assert_provider_stream_tool_call_contract,
    assert_provider_tool_call_contract,
    run_provider_contract_suite,
)


class _DummyLimiter:
    class _Ctx:
        def __init__(self) -> None:
            self.output_tokens = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def limit(self, *, tokens: int, requests: int):
        _ = (tokens, requests)
        return self._Ctx()


class _NullCache:
    async def close(self) -> None:
        return None

    async def warm(self) -> None:
        return None


def _tool(name: str = "lookup") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description="Lookup data",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        strict=False,
    )


def _build_openai_provider(model: str = "gpt-5-mini") -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    BaseProvider.__init__(provider, model)
    provider.use_responses_api = False
    provider.limiter = _DummyLimiter()  # type: ignore[assignment]
    provider.cache = _NullCache()  # type: ignore[assignment]
    provider.default_cache_collection = None
    return provider


def _build_google_types_module() -> SimpleNamespace:
    class _Content:
        def __init__(self, role: str, parts: list[object]) -> None:
            self.role = role
            self.parts = parts

    class _Part:
        @staticmethod
        def from_text(*, text: str) -> SimpleNamespace:
            return SimpleNamespace(text=text)

        @staticmethod
        def from_function_call(*, name: str, args: dict) -> SimpleNamespace:
            return SimpleNamespace(function_call=SimpleNamespace(name=name, args=args))

        @staticmethod
        def from_function_response(*, name: str, response: dict) -> SimpleNamespace:
            return SimpleNamespace(function_response=SimpleNamespace(name=name, response=response))

    class _FunctionDeclaration:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Tool:
        def __init__(self, function_declarations: list[object]) -> None:
            self.function_declarations = function_declarations

    class _GenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _AutomaticFunctionCallingConfig:
        def __init__(self, *, disable: bool) -> None:
            self.disable = disable

    class _EmbedContentConfig:
        def __init__(self, *, output_dimensionality: int) -> None:
            self.output_dimensionality = output_dimensionality

    return SimpleNamespace(
        Content=_Content,
        Part=_Part,
        FunctionDeclaration=_FunctionDeclaration,
        Tool=_Tool,
        GenerateContentConfig=_GenerateContentConfig,
        AutomaticFunctionCallingConfig=_AutomaticFunctionCallingConfig,
        EmbedContentConfig=_EmbedContentConfig,
    )


class _AsyncListIterator:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self) -> _AsyncListIterator:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _build_google_provider(monkeypatch: pytest.MonkeyPatch) -> GoogleProvider:
    import llm_client.providers.google as google_mod

    provider = GoogleProvider.__new__(GoogleProvider)
    BaseProvider.__init__(provider, "gemini-2.0-flash")
    provider.limiter = _DummyLimiter()  # type: ignore[assignment]
    provider.cache = _NullCache()  # type: ignore[assignment]
    provider.default_cache_collection = None

    fake_types = _build_google_types_module()
    monkeypatch.setattr(google_mod, "types", fake_types)
    monkeypatch.setattr(google_mod, "genai_errors", SimpleNamespace(APIError=RuntimeError))

    async def _generate_content(**kwargs):
        config = kwargs["config"]
        if getattr(config, "kwargs", {}).get("tools"):
            return SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(name="lookup", args={"q": "x"}),
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3),
            )
        return SimpleNamespace(
            parts=[SimpleNamespace(text="google-ok")],
            usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3),
        )

    async def _generate_content_stream(**kwargs):
        config = kwargs["config"]
        if getattr(config, "kwargs", {}).get("tools"):
            return _AsyncListIterator(
                [
                    SimpleNamespace(
                        candidates=[
                            SimpleNamespace(
                                content=SimpleNamespace(
                                    parts=[
                                        SimpleNamespace(
                                            function_call=SimpleNamespace(name="lookup", args={"q": "x"})
                                        )
                                    ]
                                )
                            )
                        ],
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=2,
                            candidates_token_count=1,
                            total_token_count=3,
                        ),
                    )
                ]
            )
        return _AsyncListIterator(
            [
                SimpleNamespace(
                    candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="google-ok")]))],
                    usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3),
                )
            ]
        )

    async def _embed_content(**kwargs):
        _ = kwargs
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0, 3.0])])

    provider._client = SimpleNamespace(  # type: ignore[attr-defined]
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=_generate_content,
                generate_content_stream=_generate_content_stream,
                embed_content=_embed_content,
            ),
            aclose=lambda: None,
        )
    )
    return provider


class _AnthropicStreamManager:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def __aenter__(self) -> _AsyncListIterator:
        return _AsyncListIterator(self._events)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _build_anthropic_provider() -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    BaseProvider.__init__(provider, "claude-4-5-sonnet")
    provider.max_tokens = 4096
    provider.default_temperature = None
    provider.limiter = _DummyLimiter()  # type: ignore[assignment]
    provider.cache = _NullCache()  # type: ignore[assignment]
    provider.default_cache_collection = None

    async def _create(**kwargs):
        if kwargs.get("tools"):
            content = [SimpleNamespace(type="tool_use", id="toolu_1", name="lookup", input={"q": "x"})]
        else:
            content = [SimpleNamespace(type="text", text="anthropic-ok")]
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
            stop_reason="end_turn",
        )

    def _stream(**kwargs):
        if kwargs.get("tools"):
            events = [
                SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=2))),
                SimpleNamespace(
                    type="content_block_start",
                    index=0,
                    content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="lookup"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    index=0,
                    delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":"x"}'),
                ),
                SimpleNamespace(type="content_block_stop", index=0),
                SimpleNamespace(
                    type="message_delta",
                    usage=SimpleNamespace(output_tokens=1),
                    delta=SimpleNamespace(stop_reason="tool_use"),
                ),
            ]
        else:
            events = [
                SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=2))),
                SimpleNamespace(
                    type="content_block_delta",
                    index=0,
                    delta=SimpleNamespace(type="text_delta", text="anthropic-ok"),
                ),
                SimpleNamespace(
                    type="message_delta",
                    usage=SimpleNamespace(output_tokens=1),
                    delta=SimpleNamespace(stop_reason="end_turn"),
                ),
            ]
        return _AnthropicStreamManager(events)

    provider.client = SimpleNamespace(  # type: ignore[assignment]
        messages=SimpleNamespace(create=_create, stream=_stream),
        close=lambda: None,
    )
    return provider


@pytest.mark.asyncio
async def test_openai_provider_overlap_contracts_complete_and_stream() -> None:
    provider = _build_openai_provider("gpt-5-mini")

    async def _fake_create(**kwargs):
        if kwargs.get("stream"):
            async def _iterator() -> AsyncIterator[object]:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="openai-ok", tool_calls=None), finish_reason=None)],
                    usage=None,
                )
                yield SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}),
                )

            return _iterator()
        message = SimpleNamespace(content="openai-ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
        return SimpleNamespace(choices=[choice], usage=usage, model="gpt-5-mini")

    provider.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=_fake_create))),
    )

    await run_provider_contract_suite(
        spec=ProviderContractSpec(name="openai-overlap", supports_embeddings=False),
        provider_factory=lambda: provider,
    )


@pytest.mark.asyncio
async def test_openai_provider_embedding_overlap_contract() -> None:
    provider = _build_openai_provider("text-embedding-3-small")

    async def _fake_embeddings_create(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])],
            usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "total_tokens": 2}),
        )

    provider.client = SimpleNamespace(embeddings=SimpleNamespace(create=_fake_embeddings_create))  # type: ignore[assignment]

    result = await assert_provider_embedding_contract(provider, inputs="embed me")

    assert result.model == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_google_provider_overlap_contracts_complete_stream_and_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _build_google_provider(monkeypatch)

    await run_provider_contract_suite(
        spec=ProviderContractSpec(name="google-overlap", supports_embeddings=True),
        provider_factory=lambda: provider,
    )


@pytest.mark.asyncio
async def test_anthropic_provider_overlap_contracts_complete_and_stream() -> None:
    provider = _build_anthropic_provider()

    await run_provider_contract_suite(
        spec=ProviderContractSpec(name="anthropic-overlap", supports_embeddings=False),
        provider_factory=lambda: provider,
    )


@pytest.mark.asyncio
async def test_tool_call_overlap_contracts_across_supported_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool()

    openai_provider = _build_openai_provider("gpt-5-mini")

    async def _openai_complete(**kwargs):
        _ = kwargs
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="lookup", arguments='{"q":"x"}'))],
        )
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
        return SimpleNamespace(choices=[choice], usage=usage, model="gpt-5-mini")

    async def _openai_stream(**kwargs):
        _ = kwargs

        async def _iterator() -> AsyncIterator[object]:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="lookup", arguments='{"q":'))],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments='"x"}'))],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}))

        return _iterator()

    openai_provider.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_openai_complete)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=_openai_complete))),
    )
    await assert_provider_tool_call_contract(
        openai_provider,
        messages=[{"role": "user", "content": "call a tool"}],
        kwargs={"tools": [tool]},
    )
    openai_provider.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_openai_stream)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=_openai_complete))),
    )
    await assert_provider_stream_tool_call_contract(
        openai_provider,
        messages=[{"role": "user", "content": "call a tool"}],
        kwargs={"tools": [tool]},
    )

    google_provider = _build_google_provider(monkeypatch)
    await assert_provider_tool_call_contract(
        google_provider,
        messages=[{"role": "user", "content": "call a tool"}],
        kwargs={"tools": [tool]},
    )
    await assert_provider_stream_tool_call_contract(
        google_provider,
        messages=[{"role": "user", "content": "call a tool"}],
        kwargs={"tools": [tool]},
    )

    anthropic_provider = _build_anthropic_provider()
    await assert_provider_tool_call_contract(
        anthropic_provider,
        messages=[{"role": "user", "content": "call a tool"}],
        kwargs={"tools": [tool]},
    )
    await assert_provider_stream_tool_call_contract(
        anthropic_provider,
        messages=[{"role": "user", "content": "call a tool"}],
        kwargs={"tools": [tool]},
    )
