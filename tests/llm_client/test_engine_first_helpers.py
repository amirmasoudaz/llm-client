from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.content import ContentRequestEnvelope, ContentResponseEnvelope, TextBlock
from llm_client.providers.types import CompletionResult, Message, Usage
from llm_client.providers.types import ToolCall
from llm_client.structured import StructuredOutputConfig, extract_structured
from llm_client.summarization import LLMSummarizer, LLMSummarizerConfig


class _FailIfCalledProvider:
    model_name = "gpt-5-mini"
    model = SimpleNamespace(key="gpt-5-mini")

    async def complete(self, *args, **kwargs):
        raise AssertionError("provider.complete should not be called when engine is provided")


class _ProviderOnlyForEngine:
    model_name = "gpt-5-mini"
    model = SimpleNamespace(key="gpt-5-mini")

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return CompletionResult(content="Provider summary", usage=Usage(total_tokens=4), model="gpt-5-mini", status=200)


class _FakeEngine:
    def __init__(self, *results: CompletionResult) -> None:
        self.provider = SimpleNamespace(model_name="gpt-5-mini", model=SimpleNamespace(key="gpt-5-mini"))
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    async def complete_content(self, envelope, *, context=None, **kwargs):
        self.calls.append({"envelope": envelope, "context": context, "kwargs": kwargs})
        if not self._results:
            raise AssertionError("unexpected engine.complete_content call")
        return ContentResponseEnvelope.from_completion_result(self._results.pop(0))


class _FakeEngineWithDirectComplete(_FakeEngine):
    async def complete(self, spec, *, context=None, **kwargs):
        self.calls.append({"spec": spec, "context": context, "kwargs": kwargs})
        if not self._results:
            raise AssertionError("unexpected engine.complete call")
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_extract_structured_prefers_engine_when_provided() -> None:
    engine = _FakeEngine(
        CompletionResult(content='{"wrong": 1}', usage=Usage(total_tokens=5), model="gpt-5-mini", status=200),
        CompletionResult(content='{"name": "Ada"}', usage=Usage(total_tokens=6), model="gpt-5-mini", status=200),
    )

    result = await extract_structured(
        _FailIfCalledProvider(),
        [Message.user("extract a name")],
        StructuredOutputConfig(
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            max_repair_attempts=1,
        ),
        engine=engine,
        model="gpt-5-mini",
    )

    assert result.valid is True
    assert result.data == {"name": "Ada"}
    assert result.repair_attempts == 1
    assert len(engine.calls) == 2
    first_envelope = engine.calls[0]["envelope"]
    assert isinstance(first_envelope, ContentRequestEnvelope)
    assert first_envelope.response_format == "json_object"
    assert first_envelope.model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_llm_summarizer_prefers_engine_when_provided() -> None:
    engine = _FakeEngine(
        CompletionResult(content="Short summary", usage=Usage(total_tokens=4), model="gpt-5-mini", status=200),
    )
    summarizer = LLMSummarizer(
        engine=engine,
        config=LLMSummarizerConfig(model_override="gpt-5-mini"),
    )

    summary = await summarizer.summarize(
        [
            Message.user("hello"),
            Message.assistant("hi"),
        ],
        max_tokens=120,
    )

    assert summary == "Short summary"
    assert len(engine.calls) == 1
    envelope = engine.calls[0]["envelope"]
    assert isinstance(envelope, ContentRequestEnvelope)
    assert envelope.model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_llm_summarizer_prefers_direct_engine_completion_when_available() -> None:
    engine = _FakeEngineWithDirectComplete(
        CompletionResult(content="Direct summary", usage=Usage(total_tokens=4), model="gpt-5-mini", status=200),
    )
    summarizer = LLMSummarizer(
        engine=engine,
        config=LLMSummarizerConfig(model_override="gpt-5-mini"),
    )

    summary = await summarizer.summarize(
        [
            Message.user("hello"),
            Message.assistant("hi"),
        ],
        max_tokens=120,
    )

    assert summary == "Direct summary"
    assert len(engine.calls) == 1
    assert "spec" in engine.calls[0]


@pytest.mark.asyncio
async def test_llm_summarizer_creates_internal_engine_for_provider_only() -> None:
    provider = _ProviderOnlyForEngine()
    summarizer = LLMSummarizer(
        provider=provider,
        config=LLMSummarizerConfig(model_override="gpt-5-mini"),
    )

    summary = await summarizer.summarize(
        [
            Message.user("hello"),
            Message.assistant("hi"),
        ],
        max_tokens=120,
    )

    assert summary == "Provider summary"
    assert summarizer.engine is not None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_extract_structured_rejects_mixed_content_and_tool_calls() -> None:
    engine = _FakeEngine(
        CompletionResult(
            content='{"name":"Ada"}',
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments="{}")],
            usage=Usage(total_tokens=5),
            model="gpt-5-mini",
            status=200,
        ),
    )

    result = await extract_structured(
        _FailIfCalledProvider(),
        [Message.user("extract a name")],
        StructuredOutputConfig(
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

    assert result.valid is False
    assert result.response_kind == "mixed_content_and_tools"
    assert result.validation_errors == ["Structured output cannot mix tool calls with JSON content."]


@pytest.mark.asyncio
async def test_extract_structured_accepts_json_from_content_blocks() -> None:
    engine = _FakeEngine(
        CompletionResult(
            content=[TextBlock('{"name":"Ada"}')],
            usage=Usage(total_tokens=3),
            model="gpt-5-mini",
            status=200,
        ),
    )

    result = await extract_structured(
        _FailIfCalledProvider(),
        [Message.user("extract a name")],
        StructuredOutputConfig(
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

    assert result.valid is True
    assert result.data == {"name": "Ada"}
    assert result.response_kind == "content_blocks"
