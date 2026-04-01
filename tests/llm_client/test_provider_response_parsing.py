from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import llm_client.providers.google as google_mod
from llm_client.content import ContentResponseEnvelope
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


class _AsyncResponseStreamManager:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)
        self._index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        item = self._events[self._index]
        self._index += 1
        return item


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


@pytest.mark.asyncio
async def test_openai_responses_complete_parses_text_tools_and_usage() -> None:
    provider = _openai_provider()
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _responses_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model="gpt-5-mini",
            status="completed",
            output_text="done",
            output=[
                SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="thought", type="summary_text")], content=[]),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    id="fc_1",
                    name="lookup",
                    arguments='{"q":"x"}',
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="done")],
                ),
            ],
            usage=SimpleNamespace(
                to_dict=lambda: {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 1},
                    "output_tokens_details": {"reasoning_tokens": 4},
                }
            ),
            incomplete_details=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=_responses_create, parse=None),
    )

    result = await provider.complete(
        [Message.user("hello")],
        tools=[],
        tool_choice="none",
        max_tokens=55,
        previous_response_id="resp_prev",
        store=True,
        parallel_tool_calls=False,
    )

    assert captured["input"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
    assert captured["max_output_tokens"] == 55
    assert captured["previous_response_id"] == "resp_prev"
    assert captured["store"] is True
    assert captured["parallel_tool_calls"] is False
    assert result.content == "done"
    assert result.tool_calls is not None
    assert result.tool_calls[0].name == "lookup"
    assert result.reasoning == "thought"
    assert result.provider_items is not None
    assert result.provider_items[0]["type"] == "reasoning"
    assert result.output_items is not None
    assert result.output_items[0].type == "reasoning"
    assert result.usage.input_tokens == 2
    assert result.usage.output_tokens == 1
    assert result.usage.input_tokens_cached == 1
    assert result.usage.output_tokens_reasoning == 4
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_openai_responses_stream_parses_text_reasoning_and_tool_calls() -> None:
    provider = _openai_provider()
    provider.use_responses_api = True

    completed_response = SimpleNamespace(
        model="gpt-5-mini",
        status="completed",
        output_text="hello world",
        output=[
            SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="plan it", type="summary_text")], content=[]),
            SimpleNamespace(type="function_call", call_id="call_1", id="fc_1", name="lookup", arguments='{"q":"x"}'),
            SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="hello world")]),
        ],
        usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}),
        incomplete_details=None,
    )

    def _responses_stream(**kwargs):
        _ = kwargs
        return _AsyncResponseStreamManager(
            [
                SimpleNamespace(type="response.output_item.added", output_index=0, item=SimpleNamespace(type="function_call", call_id="call_1", id="fc_1", name="lookup", arguments="")),
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="plan ", item_id="rs_1", output_index=1, summary_index=0),
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="it", item_id="rs_1", output_index=1, summary_index=0),
                SimpleNamespace(type="response.output_text.delta", delta="hello ", item_id="msg_1", output_index=2, content_index=0),
                SimpleNamespace(type="response.function_call_arguments.delta", delta='{"q":', item_id="fc_1", output_index=0),
                SimpleNamespace(type="response.output_text.delta", delta="world", item_id="msg_1", output_index=2, content_index=0),
                SimpleNamespace(type="response.function_call_arguments.delta", delta='"x"}', item_id="fc_1", output_index=0),
                SimpleNamespace(type="response.output_item.done", output_index=0, item=SimpleNamespace(type="function_call", call_id="call_1", id="fc_1", name="lookup", arguments='{"q":"x"}')),
                SimpleNamespace(type="response.completed", response=completed_response),
            ]
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(stream=_responses_stream),
    )

    events = [event async for event in provider.stream([Message.user("hello")], tools=[])]

    assert [event.type for event in events] == [
        StreamEventType.META,
        StreamEventType.TOOL_CALL_START,
        StreamEventType.REASONING,
        StreamEventType.REASONING,
        StreamEventType.TOKEN,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOKEN,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOOL_CALL_END,
        StreamEventType.USAGE,
        StreamEventType.DONE,
    ]
    assert events[-1].data.content == "hello world"
    assert events[-1].data.reasoning == "plan it"
    assert events[-1].data.provider_items is not None
    assert events[-1].data.provider_items[0]["type"] == "reasoning"
    assert events[-1].data.output_items is not None
    assert events[-1].data.output_items[0].type == "reasoning"
    assert events[-1].data.tool_calls is not None
    assert events[-1].data.tool_calls[0].arguments == '{"q":"x"}'
    assert events[-2].data.total_tokens == 4


@pytest.mark.asyncio
async def test_openai_responses_complete_replays_preserved_reasoning_items_into_followup_tool_loop() -> None:
    provider = _openai_provider()
    provider.use_responses_api = True

    captured_calls: list[dict[str, object]] = []

    async def _responses_create(**kwargs):
        captured_calls.append(dict(kwargs))
        if len(captured_calls) == 1:
            return SimpleNamespace(
                model="gpt-5-mini",
                status="completed",
                output_text="",
                output=[
                    SimpleNamespace(type="reasoning", id="rs_1", summary=[SimpleNamespace(text="plan", type="summary_text")], content=[]),
                    SimpleNamespace(type="function_call", call_id="call_1", id="fc_1", name="lookup", arguments='{"q":"x"}'),
                ],
                usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}),
                incomplete_details=None,
            )
        return SimpleNamespace(
            model="gpt-5-mini",
            status="completed",
            output_text="done",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="done")])],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}),
            incomplete_details=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=_responses_create, parse=None),
    )

    first = await provider.complete(
        [Message.user("hello")],
        tools=[],
    )
    followup_messages = [
        Message.user("hello"),
        first.to_message(),
        Message.tool_result("call_1", '{"ok":true}', name="lookup"),
    ]
    second = await provider.complete(followup_messages, tools=[])

    assert second.content == "done"
    assert len(captured_calls) == 2
    assert captured_calls[1]["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "content": [],
        },
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        },
    ]


@pytest.mark.asyncio
async def test_openai_responses_complete_normalizes_refusal_and_hosted_tool_outputs() -> None:
    provider = _openai_provider()
    provider.use_responses_api = True

    async def _responses_create(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            model="gpt-5-mini",
            status="completed",
            output_text="",
            output=[
                SimpleNamespace(
                    type="message",
                    id="msg_1",
                    status="completed",
                    content=[SimpleNamespace(type="refusal", refusal="I can't comply with that request.")],
                ),
                SimpleNamespace(
                    type="web_search_call",
                    id="ws_1",
                    status="completed",
                    action=SimpleNamespace(type="search", query="orbital mechanics"),
                ),
                SimpleNamespace(
                    type="file_search_call",
                    id="fs_1",
                    status="completed",
                    queries=["orbital mechanics"],
                    results=[SimpleNamespace(file_id="file_1", filename="notes.md")],
                ),
                SimpleNamespace(
                    type="code_interpreter_call",
                    id="ci_1",
                    status="completed",
                    code="print('orbit')",
                    container_id="ctr_1",
                    outputs=[
                        SimpleNamespace(type="logs", logs="orbit"),
                        SimpleNamespace(type="image", url="https://example.com/plot.png"),
                    ],
                ),
                SimpleNamespace(
                    type="image_generation_call",
                    id="img_1",
                    status="completed",
                    result="https://example.com/generated.png",
                ),
            ],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}),
            incomplete_details=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=_responses_create, parse=None),
    )

    result = await provider.complete([Message.user("hello")], tools=[])

    assert result.refusal == "I can't comply with that request."
    assert result.output_items is not None
    assert [item.type for item in result.output_items] == [
        "refusal",
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
    ]
    assert result.output_items[0].text == "I can't comply with that request."
    assert result.output_items[1].details["action"]["query"] == "orbital mechanics"
    assert result.output_items[2].details["queries"] == ["orbital mechanics"]
    assert result.output_items[3].text == "orbit"
    assert result.output_items[3].url == "https://example.com/plot.png"
    assert result.output_items[4].url == "https://example.com/generated.png"

    roundtrip = ContentResponseEnvelope.from_completion_result(result).to_completion_result()
    assert roundtrip.refusal == result.refusal
    assert roundtrip.output_items is not None
    assert [item.to_dict() for item in roundtrip.output_items] == [item.to_dict() for item in result.output_items]


def test_openai_responses_finish_reason_normalizes_incomplete_statuses() -> None:
    provider = _openai_provider()

    max_tokens_response = SimpleNamespace(
        status="incomplete",
        output=[],
        output_text="partial",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    content_filter_response = SimpleNamespace(
        status="incomplete",
        output=[],
        output_text="partial",
        incomplete_details=SimpleNamespace(reason="content_filter"),
    )
    queued_response = SimpleNamespace(
        status="queued",
        output=[],
        output_text="",
        incomplete_details=None,
    )

    _, _, _, max_tokens_finish = provider._extract_responses_output(max_tokens_response, alias_to_original={})
    _, _, _, content_filter_finish = provider._extract_responses_output(content_filter_response, alias_to_original={})
    _, _, _, queued_finish = provider._extract_responses_output(queued_response, alias_to_original={})

    assert max_tokens_finish == "length"
    assert content_filter_finish == "content_filter"
    assert queued_finish is None


@pytest.mark.asyncio
async def test_openai_background_response_retrieve_and_cancel_parse_lifecycle_state() -> None:
    provider = _openai_provider("gpt-5.2")
    provider.use_responses_api = True

    retrieved_calls: list[tuple[str, dict[str, object]]] = []

    async def _retrieve(response_id: str, **kwargs):
        retrieved_calls.append((response_id, dict(kwargs)))
        return SimpleNamespace(
            id=response_id,
            model="gpt-5.2",
            status="completed",
            output_text="finished",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="finished")])],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}),
            incomplete_details=None,
            error=None,
        )

    async def _cancel(response_id: str, **kwargs):
        _ = kwargs
        return SimpleNamespace(
            id=response_id,
            model="gpt-5.2",
            status="cancelled",
            output_text="",
            output=[],
            usage=None,
            incomplete_details=None,
            error=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(retrieve=_retrieve, cancel=_cancel),
    )

    retrieved = await provider.retrieve_background_response("resp_1", include=["reasoning.encrypted_content"])
    cancelled = await provider.cancel_background_response("resp_2")

    assert retrieved_calls == [("resp_1", {"include": ["reasoning.encrypted_content"]})]
    assert retrieved.response_id == "resp_1"
    assert retrieved.lifecycle_status == "completed"
    assert retrieved.completion is not None
    assert retrieved.completion.content == "finished"
    assert cancelled.lifecycle_status == "cancelled"
    assert cancelled.is_terminal is True


@pytest.mark.asyncio
async def test_openai_background_response_wait_polls_until_terminal_state() -> None:
    provider = _openai_provider("gpt-5.2")
    provider.use_responses_api = True

    statuses = iter(
        [
            SimpleNamespace(id="resp_wait", model="gpt-5.2", status="queued", output=[], output_text="", usage=None, incomplete_details=None, error=None),
            SimpleNamespace(id="resp_wait", model="gpt-5.2", status="in_progress", output=[], output_text="", usage=None, incomplete_details=None, error=None),
            SimpleNamespace(
                id="resp_wait",
                model="gpt-5.2",
                status="completed",
                output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="done")])],
                output_text="done",
                usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
                incomplete_details=None,
                error=None,
            ),
        ]
    )

    async def _retrieve(response_id: str, **kwargs):
        _ = (response_id, kwargs)
        return next(statuses)

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(retrieve=_retrieve),
    )

    result = await provider.wait_background_response("resp_wait", poll_interval=0.0)

    assert result.lifecycle_status == "completed"
    assert result.completion is not None
    assert result.completion.content == "done"


@pytest.mark.asyncio
async def test_openai_background_response_stream_resume_uses_starting_after_and_emits_sequence_numbers() -> None:
    provider = _openai_provider("gpt-5.2")
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _retrieve(response_id: str, **kwargs):
        captured["response_id"] = response_id
        captured.update(kwargs)
        completed_response = SimpleNamespace(
            id=response_id,
            model="gpt-5.2",
            status="completed",
            output_text="background done",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="background done")])],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}),
            incomplete_details=None,
            error=None,
        )
        return AsyncSequence(
            [
                SimpleNamespace(type="response.output_text.delta", delta="background ", sequence_number=43),
                SimpleNamespace(type="response.output_text.delta", delta="done", sequence_number=44),
                SimpleNamespace(type="response.completed", response=completed_response, sequence_number=45),
            ]
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(retrieve=_retrieve),
    )

    events = [event async for event in provider.stream_background_response("resp_bg", starting_after=42)]

    assert captured["response_id"] == "resp_bg"
    assert captured["stream"] is True
    assert captured["starting_after"] == 42
    assert events[0].type is StreamEventType.META
    assert events[1].sequence_number == 43
    assert events[2].sequence_number == 44
    assert events[-1].sequence_number == 45
    assert events[-1].data.content == "background done"


@pytest.mark.asyncio
async def test_openai_responses_complete_passes_explicit_prompt_cache_and_include_controls() -> None:
    provider = _openai_provider("gpt-5")
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _responses_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="resp_controls",
            model="gpt-5",
            status="completed",
            output_text="ok",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="ok")])],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            incomplete_details=None,
            error=None,
        )

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=_responses_create))

    result = await provider.complete(
        [Message.user("hello")],
        include=["reasoning.encrypted_content"],
        prompt_cache_key="tenant-a",
        prompt_cache_retention="24h",
    )

    assert result.content == "ok"
    assert captured["include"] == ["reasoning.encrypted_content"]
    assert captured["prompt_cache_key"] == "tenant-a"
    assert captured["prompt_cache_retention"] == "24h"


@pytest.mark.asyncio
async def test_openai_conversation_lifecycle_and_compaction_are_supported() -> None:
    provider = _openai_provider("gpt-5")
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _create_conversation(**kwargs):
        captured["create"] = dict(kwargs)
        return SimpleNamespace(id="conv_1", created_at=123, metadata={"scope": "a"})

    async def _retrieve_conversation(conversation_id: str, **kwargs):
        captured["retrieve"] = {"conversation_id": conversation_id, **dict(kwargs)}
        return SimpleNamespace(id=conversation_id, created_at=123, metadata={"scope": "a"})

    async def _update_conversation(conversation_id: str, **kwargs):
        captured["update"] = {"conversation_id": conversation_id, **dict(kwargs)}
        return SimpleNamespace(id=conversation_id, created_at=123, metadata=dict(kwargs.get("metadata") or {}))

    async def _delete_conversation(conversation_id: str, **kwargs):
        captured["delete"] = {"conversation_id": conversation_id, **dict(kwargs)}
        return SimpleNamespace(id=conversation_id, deleted=True, object="conversation.deleted")

    async def _compact(**kwargs):
        captured["compact"] = dict(kwargs)
        return SimpleNamespace(
            id="cmp_1",
            created_at=456,
            output=[SimpleNamespace(type="compaction", id="cmp_item_1", encrypted_content="enc_123", created_by="resp_prev")],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}),
        )

    provider.client = SimpleNamespace(
        conversations=SimpleNamespace(
            create=_create_conversation,
            retrieve=_retrieve_conversation,
            update=_update_conversation,
            delete=_delete_conversation,
        ),
        responses=SimpleNamespace(compact=_compact),
    )

    created = await provider.create_conversation(items=[Message.user("hello")], metadata={"scope": "a"})
    retrieved = await provider.retrieve_conversation("conv_1")
    updated = await provider.update_conversation("conv_1", metadata={"scope": "b"})
    deleted = await provider.delete_conversation("conv_1")
    compacted = await provider.compact_response_context(
        messages=[Message.user("hello again")],
        instructions="Preserve context",
        previous_response_id="resp_prev",
    )

    assert created.conversation_id == "conv_1"
    assert retrieved.conversation_id == "conv_1"
    assert updated.metadata == {"scope": "b"}
    assert deleted.deleted is True
    assert captured["create"]["items"][0]["type"] == "message"
    assert captured["compact"]["previous_response_id"] == "resp_prev"
    assert compacted.compaction_id == "cmp_1"
    assert compacted.output_items is not None
    assert compacted.output_items[0].type == "compaction"
    assert compacted.output_items[0].details["encrypted_content"] == "enc_123"


@pytest.mark.asyncio
async def test_openai_submit_mcp_approval_response_continues_response() -> None:
    provider = _openai_provider("gpt-5")
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _responses_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="resp_approved",
            model="gpt-5",
            status="completed",
            output_text="approved",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="approved")])],
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}),
            incomplete_details=None,
            error=None,
        )

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=_responses_create))

    result = await provider.submit_mcp_approval_response(
        previous_response_id="resp_prev",
        approval_request_id="mcpr_1",
        approve=True,
    )

    assert result.content == "approved"
    assert captured["previous_response_id"] == "resp_prev"
    assert captured["input"] == [
        {
            "type": "mcp_approval_response",
            "approval_request_id": "mcpr_1",
            "approve": True,
        }
    ]


@pytest.mark.asyncio
async def test_openai_delete_response_and_conversation_items_are_supported() -> None:
    provider = _openai_provider("gpt-5-mini")
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _delete_response(response_id: str, **kwargs):
        captured["delete_response"] = (response_id, dict(kwargs))
        return None

    async def _create_items(conversation_id: str, **kwargs):
        captured["create_items"] = (conversation_id, dict(kwargs))
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="item_msg_1",
                    type="message",
                    role="user",
                    status="completed",
                    content=[SimpleNamespace(type="input_text", text="hello")],
                )
            ],
            first_id="item_msg_1",
            last_id="item_msg_1",
            has_more=False,
            object="list",
        )

    class _ListPaginator:
        async def _get_page(self):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id="item_fc_1",
                        type="function_call",
                        call_id="call_1",
                        name="lookup",
                        arguments='{"q":"x"}',
                        status="completed",
                    ),
                    SimpleNamespace(
                        id="item_fco_1",
                        type="function_call_output",
                        call_id="call_1",
                        output='{"ok":true}',
                        status="completed",
                    ),
                    SimpleNamespace(
                        id="item_custom_out_1",
                        type="custom_tool_call_output",
                        call_id="custom_1",
                        output="done",
                        status="completed",
                    ),
                    SimpleNamespace(
                        id="item_mcp_resp_1",
                        type="mcp_approval_response",
                        approval_request_id="apr_1",
                        approve=True,
                        reason="approved",
                    ),
                ],
                first_id="item_fc_1",
                last_id="item_mcp_resp_1",
                has_more=True,
                object="list",
            )

    async def _list_items(conversation_id: str, **kwargs):
        captured["list_items"] = (conversation_id, dict(kwargs))
        return _ListPaginator()

    async def _retrieve_item(item_id: str, **kwargs):
        captured["retrieve_item"] = (item_id, dict(kwargs))
        return SimpleNamespace(
            id=item_id,
            type="computer_call_output",
            call_id="comp_1",
            output={"image_url": "https://example.com/screenshot.png"},
            acknowledged_safety_checks=[{"code": "safe"}],
            status="completed",
        )

    async def _delete_item(item_id: str, *, conversation_id: str, **kwargs):
        captured["delete_item"] = (item_id, conversation_id, dict(kwargs))
        return SimpleNamespace(
            id=conversation_id,
            created_at=123,
            metadata={"topic": "support"},
            deleted=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(delete=_delete_response),
        conversations=SimpleNamespace(
            items=SimpleNamespace(
                create=_create_items,
                list=_list_items,
                retrieve=_retrieve_item,
                delete=_delete_item,
            )
        ),
    )

    deleted = await provider.delete_response("resp_delete_1", reason="cleanup")
    created_page = await provider.create_conversation_items(
        "conv_1",
        items=[Message.user("hello")],
        include=["reasoning.encrypted_content"],
    )
    listed_page = await provider.list_conversation_items(
        "conv_1",
        after="item_0",
        include=["reasoning.encrypted_content"],
        limit=20,
        order="asc",
    )
    retrieved_item = await provider.retrieve_conversation_item(
        "conv_1",
        "item_comp_1",
        include=["reasoning.encrypted_content"],
    )
    conversation_after_delete = await provider.delete_conversation_item("conv_1", "item_comp_1")

    assert deleted.resource_id == "resp_delete_1"
    assert deleted.deleted is True
    assert captured["delete_response"] == ("resp_delete_1", {"reason": "cleanup"})

    assert captured["create_items"][0] == "conv_1"
    assert captured["create_items"][1]["include"] == ["reasoning.encrypted_content"]
    assert created_page.first_id == "item_msg_1"
    assert created_page.items[0].content == "hello"

    assert captured["list_items"] == (
        "conv_1",
        {
            "after": "item_0",
            "include": ["reasoning.encrypted_content"],
            "limit": 20,
            "order": "asc",
        },
    )
    assert listed_page.has_more is True
    assert [item.item_type for item in listed_page.items] == [
        "function_call",
        "function_call_output",
        "custom_tool_call_output",
        "mcp_approval_response",
    ]
    assert listed_page.items[0].output_items[0].type == "function_call"
    assert listed_page.items[1].output_items[0].type == "function_call_output"
    assert listed_page.items[1].output_items[0].text == '{"ok":true}'
    assert listed_page.items[2].output_items[0].text == "done"
    assert listed_page.items[3].output_items[0].details["approval_request_id"] == "apr_1"

    assert captured["retrieve_item"] == (
        "item_comp_1",
        {"conversation_id": "conv_1", "include": ["reasoning.encrypted_content"]},
    )
    assert retrieved_item.item_type == "computer_call_output"
    assert retrieved_item.output_items[0].details["acknowledged_safety_checks"] == [{"code": "safe"}]
    assert retrieved_item.output_items[0].url == "https://example.com/screenshot.png"
    assert retrieved_item.output_items[0].details["output"] == {"image_url": "https://example.com/screenshot.png"}

    assert captured["delete_item"] == ("item_comp_1", "conv_1", {})
    assert conversation_after_delete.conversation_id == "conv_1"
    assert conversation_after_delete.metadata == {"topic": "support"}


@pytest.mark.asyncio
async def test_openai_responses_complete_uses_parse_for_pydantic_structured_outputs() -> None:
    class EventPayload(BaseModel):
        answer: str

    provider = _openai_provider()
    provider.use_responses_api = True

    captured: dict[str, object] = {}

    async def _responses_parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model="gpt-5-mini",
            status="completed",
            output_text="",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="",
                            parsed=EventPayload(answer="ok"),
                        )
                    ],
                )
            ],
            output_parsed=EventPayload(answer="ok"),
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            incomplete_details=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=None, parse=_responses_parse),
    )

    result = await provider.complete(
        [Message.user("Return JSON.")],
        response_format=EventPayload,
    )

    assert captured["text_format"] is EventPayload
    assert "text" not in captured
    assert result.content == '{"answer": "ok"}'
    assert result.usage.total_tokens == 5
