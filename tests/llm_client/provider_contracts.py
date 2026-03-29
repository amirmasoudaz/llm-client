from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from llm_client.providers.base import BaseProvider, Provider
from llm_client.providers.types import (
    CompletionResult,
    EmbeddingResult,
    StreamEvent,
    StreamEventType,
    Usage,
)


@dataclass(frozen=True)
class ProviderContractSpec:
    name: str
    supports_embeddings: bool = True


class ContractFakeProvider(BaseProvider):
    """Fake provider used to validate the shared contract harness itself."""

    def __init__(self, *, supports_embeddings: bool = True) -> None:
        super().__init__("gpt-5-mini")
        self._supports_embeddings = supports_embeddings

    async def complete(self, messages, **kwargs: Any) -> CompletionResult:
        _ = (messages, kwargs)
        return CompletionResult(
            content="contract-ok",
            usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
            model="gpt-5-mini",
            status=200,
        )

    async def stream(self, messages, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        _ = (messages, kwargs)
        yield StreamEvent(type=StreamEventType.META, data={"provider": "fake"})
        yield StreamEvent(type=StreamEventType.TOKEN, data="contract")
        yield StreamEvent(type=StreamEventType.TOKEN, data="-ok")
        yield StreamEvent(
            type=StreamEventType.USAGE,
            data=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
        )
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=CompletionResult(
                content="contract-ok",
                usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
                model="gpt-5-mini",
                status=200,
            ),
        )

    async def embed(self, inputs: str | list[str], **kwargs: Any) -> EmbeddingResult:
        _ = kwargs
        if not self._supports_embeddings:
            raise NotImplementedError("Embeddings disabled for this fake provider")
        values = [inputs] if isinstance(inputs, str) else list(inputs)
        return EmbeddingResult(
            embeddings=[[float(len(text or "")), 1.0] for text in values],
            usage=Usage(input_tokens=sum(len(text or "") for text in values)),
            model="text-embedding-3-small",
            status=200,
        )


def assert_completion_contract(result: CompletionResult) -> None:
    assert isinstance(result, CompletionResult)
    assert isinstance(result.ok, bool)
    assert isinstance(result.status, int)
    if result.ok:
        assert result.error is None
    if result.usage is not None:
        assert isinstance(result.usage, Usage)
        assert result.usage.total_tokens >= 0
        assert result.usage.input_tokens >= 0
        assert result.usage.output_tokens >= 0
    if result.tool_calls:
        for tool_call in result.tool_calls:
            assert tool_call.id
            assert tool_call.name
            assert isinstance(tool_call.arguments, str)


async def assert_provider_complete_contract(
    provider: Provider,
    *,
    messages: Any,
    kwargs: dict[str, Any] | None = None,
) -> CompletionResult:
    result = await provider.complete(messages, **(kwargs or {}))
    assert_completion_contract(result)
    return result


async def assert_provider_stream_contract(
    provider: Provider,
    *,
    messages: Any,
    kwargs: dict[str, Any] | None = None,
) -> list[StreamEvent]:
    events = [event async for event in provider.stream(messages, **(kwargs or {}))]

    assert events, "stream() must yield at least one event"
    assert events[-1].type in {StreamEventType.DONE, StreamEventType.ERROR}

    for event in events:
        assert isinstance(event, StreamEvent)
        assert isinstance(event.type, StreamEventType)
        if event.type == StreamEventType.USAGE:
            assert isinstance(event.data, Usage)
        if event.type == StreamEventType.DONE:
            assert isinstance(event.data, CompletionResult)
            assert_completion_contract(event.data)

    return events


async def assert_provider_embedding_contract(
    provider: Provider,
    *,
    inputs: str | list[str],
) -> EmbeddingResult:
    result = await provider.embed(inputs)
    assert isinstance(result, EmbeddingResult)
    assert isinstance(result.ok, bool)
    assert result.ok is True
    assert isinstance(result.embeddings, list)
    assert result.embeddings
    assert all(isinstance(vector, list) for vector in result.embeddings)
    assert all(all(isinstance(value, float) for value in vector) for vector in result.embeddings)
    if result.usage is not None:
        assert isinstance(result.usage, Usage)
    return result


async def assert_provider_tool_call_contract(
    provider: Provider,
    *,
    messages: Any,
    kwargs: dict[str, Any] | None = None,
) -> CompletionResult:
    result = await assert_provider_complete_contract(provider, messages=messages, kwargs=kwargs)
    assert result.tool_calls, "provider completion should expose tool calls"
    for tool_call in result.tool_calls or []:
        assert tool_call.id
        assert tool_call.name
        assert isinstance(tool_call.arguments, str)
        assert tool_call.arguments
    return result


async def assert_provider_stream_tool_call_contract(
    provider: Provider,
    *,
    messages: Any,
    kwargs: dict[str, Any] | None = None,
) -> list[StreamEvent]:
    events = await assert_provider_stream_contract(provider, messages=messages, kwargs=kwargs)
    assert any(event.type == StreamEventType.TOOL_CALL_START for event in events)
    assert any(event.type == StreamEventType.TOOL_CALL_DELTA for event in events)
    assert any(event.type == StreamEventType.TOOL_CALL_END for event in events)
    return events


async def run_provider_contract_suite(
    *,
    spec: ProviderContractSpec,
    provider_factory: Callable[[], Provider | Awaitable[Provider]],
) -> None:
    provider = provider_factory()
    if isinstance(provider, Awaitable):
        provider = await provider

    completion = await assert_provider_complete_contract(
        provider,
        messages=[{"role": "user", "content": f"contract test for {spec.name}"}],
    )
    assert completion.ok is True

    await assert_provider_stream_contract(
        provider,
        messages=[{"role": "user", "content": f"stream contract test for {spec.name}"}],
    )

    if spec.supports_embeddings:
        await assert_provider_embedding_contract(provider, inputs="embed me")
