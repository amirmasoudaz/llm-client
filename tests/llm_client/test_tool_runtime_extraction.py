from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.content import ContentRequestEnvelope, ContentResponseEnvelope
from llm_client.providers.types import CompletionResult, StreamEvent, StreamEventType, Usage
from llm_client.tools.runtime import (
    RuntimeToolError,
    StructuredToolLoopError,
    StructuredToolRuntime,
    complete_with_tools,
    execute_runtime_tool,
    tool_to_provider_definition,
    validate_runtime_tool_arguments,
)


class _ToolWithContracts:
    def __init__(self, *, contracts: object, schema: dict):
        self._contracts = contracts
        self.parameters = schema


class _FakeStreamProvider:
    async def stream(self, messages, **kwargs):
        _ = (messages, kwargs)
        yield StreamEvent(type=StreamEventType.META, data={"model": "gpt-5-mini"})
        yield StreamEvent(type=StreamEventType.TOKEN, data='{"result":{"outcome":{"assistant_message":{"text":"Hel')
        yield StreamEvent(type=StreamEventType.TOKEN, data='lo world"}}}}')
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=CompletionResult(
                content='{"result":{"outcome":{"assistant_message":{"text":"Hello world"}}}}',
                usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
                model="gpt-5-mini",
                status=200,
            ),
        )


class _FakeStreamProviderWithInternalIds:
    async def stream(self, messages, **kwargs):
        _ = (messages, kwargs)
        text = (
            '{"result":{"outcome":{"assistant_message":{"text":"'
            "Attach funding_request_id and professor_id to thread 407. "
            'Open /v1/threads/407/context if you need IDs."}}}}'
        )
        mid = len(text) // 2
        yield StreamEvent(type=StreamEventType.TOKEN, data=text[:mid])
        yield StreamEvent(type=StreamEventType.TOKEN, data=text[mid:])
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=CompletionResult(content=text, usage=Usage(), model="gpt-5-mini", status=200),
        )


class _FakeEnvelopeEngine:
    def __init__(self) -> None:
        self.completed: list[ContentRequestEnvelope] = []
        self.streamed: list[ContentRequestEnvelope] = []

    async def complete_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        self.completed.append(request)
        return ContentResponseEnvelope.from_completion_result(
            CompletionResult(
                content='{"result":{"outcome":{"assistant_message":{"text":"Engine completion"}}}}',
                usage=Usage(total_tokens=4),
                model=request.model,
                status=200,
            )
        )

    async def stream_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        self.streamed.append(request)
        yield StreamEvent(type=StreamEventType.TOKEN, data='{"result":{"outcome":{"assistant_message":{"text":"Eng')
        yield StreamEvent(type=StreamEventType.TOKEN, data='ine stream"}}}}')
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=ContentResponseEnvelope.from_completion_result(
                CompletionResult(
                    content='{"result":{"outcome":{"assistant_message":{"text":"Engine stream"}}}}',
                    usage=Usage(total_tokens=5),
                    model=request.model,
                    status=200,
                )
            ),
        )


class _FakeTool:
    name = "Conversation.Profile.Requirements"
    description = "Test tool"
    parameters = {
        "type": "object",
        "properties": {"thread_id": {"type": "integer"}},
        "required": ["thread_id"],
    }

    def to_openai_format(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


class _SlowRuntimeTool:
    name = "slow_tool"
    description = "Slow test tool"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    async def execute(self, value: str = "") -> str:
        import asyncio

        await asyncio.sleep(0.2)
        return value


class _FakeToolCallProvider:
    def __init__(self, completion: CompletionResult) -> None:
        self._completion = completion

    async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        _ = (messages, kwargs)
        return self._completion


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_streaming_extracts_assistant_text_token_deltas() -> None:
    callbacks: list[str] = []

    async def _on_delta(chunk: str) -> None:
        callbacks.append(chunk)

    completion, error, messages_out = await complete_with_tools(
        engine=None,
        provider=_FakeStreamProvider(),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"response_format": "json_object", "model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(tools_by_name={}, provider_tools=[], max_tool_calls=0, max_tool_call_depth=0),
        tool_timeout_ms=1000,
        token_delta_callback=_on_delta,
        token_delta_mode="conversation_assistant_text",
    )

    assert error is None
    assert completion is not None
    assert completion.ok is True
    assert completion.content is not None
    assert "".join(callbacks) == "Hello world"
    assert messages_out == [{"role": "user", "content": "json"}]


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_streaming_redacts_internal_ids_from_assistant_text() -> None:
    callbacks: list[str] = []

    async def _on_delta(chunk: str) -> None:
        callbacks.append(chunk)

    completion, error, _ = await complete_with_tools(
        engine=None,
        provider=_FakeStreamProviderWithInternalIds(),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"response_format": "json_object", "model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(tools_by_name={}, provider_tools=[], max_tool_calls=0, max_tool_call_depth=0),
        tool_timeout_ms=1000,
        token_delta_callback=_on_delta,
        token_delta_mode="conversation_assistant_text",
    )

    assert error is None
    assert completion is not None
    streamed = "".join(callbacks)
    assert "funding_request_id" not in streamed
    assert "professor_id" not in streamed
    assert "thread 407" not in streamed.lower()
    assert "/v1/threads/407/context" not in streamed
    assert "funding request" in streamed


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_engine_completion_uses_content_envelopes() -> None:
    engine = _FakeEnvelopeEngine()

    completion, error, messages_out = await complete_with_tools(
        engine=engine,
        provider=SimpleNamespace(),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"response_format": "json_object", "model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(tools_by_name={}, provider_tools=[], max_tool_calls=0, max_tool_call_depth=0),
        tool_timeout_ms=1000,
    )

    assert error is None
    assert completion is not None
    assert completion.content is not None
    assert "Engine completion" in completion.content
    assert messages_out == [{"role": "user", "content": "json"}]
    assert len(engine.completed) == 1
    assert isinstance(engine.completed[0], ContentRequestEnvelope)
    assert engine.completed[0].model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_engine_streaming_uses_content_envelopes() -> None:
    engine = _FakeEnvelopeEngine()
    callbacks: list[str] = []

    async def _on_delta(chunk: str) -> None:
        callbacks.append(chunk)

    completion, error, _ = await complete_with_tools(
        engine=engine,
        provider=SimpleNamespace(),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"response_format": "json_object", "model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(tools_by_name={}, provider_tools=[], max_tool_calls=0, max_tool_call_depth=0),
        tool_timeout_ms=1000,
        token_delta_callback=_on_delta,
        token_delta_mode="conversation_assistant_text",
    )

    assert error is None
    assert completion is not None
    assert completion.content is not None
    assert "".join(callbacks) == "Engine stream"
    assert len(engine.streamed) == 1
    assert isinstance(engine.streamed[0], ContentRequestEnvelope)
    assert engine.streamed[0].model == "gpt-5-mini"


def test_llm_client_tool_runtime_converts_tools_to_provider_defs() -> None:
    tool_def = tool_to_provider_definition(_FakeTool())

    assert isinstance(tool_def, dict)
    assert tool_def["function"]["name"] == "Conversation.Profile.Requirements"


def test_llm_client_tool_runtime_argument_validation_accepts_contract_registries() -> None:
    contracts = SimpleNamespace(registry_for=lambda schema: None)
    tool = _ToolWithContracts(contracts=contracts, schema={"type": "object", "properties": {"thread_id": {"type": "integer"}}})

    errors = validate_runtime_tool_arguments(tool, {"thread_id": 123})

    assert errors == []


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_rejects_malformed_tool_calls() -> None:
    completion, error, messages_out = await complete_with_tools(
        engine=None,
        provider=_FakeToolCallProvider(
            CompletionResult(
                content="",
                tool_calls=[{"id": "call_1", "name": "Conversation.Profile.Requirements"}],
                model="gpt-5-mini",
                status=200,
            )
        ),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(tools_by_name={}, provider_tools=[], max_tool_calls=0, max_tool_call_depth=0),
        tool_timeout_ms=1000,
    )

    assert completion is None
    assert error is not None
    assert error.code == "tool_call_invalid"
    assert "missing required fields: arguments" in error.message
    assert error.to_normalized_failure()["category"] == "tool"
    assert messages_out == [{"role": "user", "content": "json"}]


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_enforces_depth_limits() -> None:
    provider_tool = _FakeTool().to_openai_format()

    completion, error, messages_out = await complete_with_tools(
        engine=None,
        provider=_FakeToolCallProvider(
            CompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "Conversation.Profile.Requirements",
                        "arguments": '{"thread_id": 123}',
                    }
                ],
                model="gpt-5-mini",
                status=200,
            )
        ),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(
            tools_by_name={"Conversation.Profile.Requirements": _FakeTool()},
            provider_tools=[provider_tool],
            max_tool_calls=1,
            max_tool_call_depth=0,
        ),
        tool_timeout_ms=1000,
    )

    assert completion is None
    assert error is not None
    assert error.code == "tool_call_depth_exceeded"
    assert error.category == "rate_limited"
    assert error.to_normalized_failure()["category"] == "tool"
    assert any(message.get("role") == "user" for message in messages_out)


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_enforces_allowlist_for_declared_tools() -> None:
    provider_tool = _FakeTool().to_openai_format()

    completion, error, messages_out = await complete_with_tools(
        engine=None,
        provider=_FakeToolCallProvider(
            CompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "Conversation.Profile.Load",
                        "arguments": '{"thread_id": 123}',
                    }
                ],
                model="gpt-5-mini",
                status=200,
            )
        ),
        messages=[{"role": "user", "content": "json"}],
        completion_kwargs={"model": "gpt-5-mini"},
        runtime=StructuredToolRuntime(
            tools_by_name={"Conversation.Profile.Requirements": _FakeTool()},
            provider_tools=[provider_tool],
            max_tool_calls=1,
            max_tool_call_depth=2,
        ),
        tool_timeout_ms=1000,
    )

    assert completion is None
    assert error is not None
    assert error.code == "tool_not_allowed"
    assert error.category == "policy_denied"
    assert error.to_normalized_failure()["category"] == "tool_policy"
    assert "undeclared tool" in error.message
    assert any(message.get("role") == "assistant" for message in messages_out)


@pytest.mark.asyncio
async def test_llm_client_tool_runtime_timeout_normalizes_failure() -> None:
    with pytest.raises(RuntimeToolError) as exc_info:
        await execute_runtime_tool(_SlowRuntimeTool(), {"value": "x"}, timeout_ms=1)

    failure = exc_info.value
    normalized = StructuredToolLoopError(
        code=failure.code,
        message=failure.message,
        category=failure.category,
        retryable=failure.retryable,
        details=failure.details,
    ).to_normalized_failure()

    assert failure.code == "tool_call_timeout"
    assert normalized["category"] == "tool"
    assert normalized["retryable"] is False
