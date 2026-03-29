from __future__ import annotations

from llm_client.content import (
    ContentResponseEnvelope,
    TextBlock,
    completion_stream_event_to_content_event,
    content_stream_event_to_completion_event,
    ensure_completion_result,
    ensure_content_response_envelope,
)
from llm_client.providers.types import CompletionResult, StreamEvent, StreamEventType, ToolCall, Usage


def test_content_completion_roundtrip_preserves_terminal_payload() -> None:
    completion = CompletionResult(
        content="hello",
        tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"hello"}')],
        usage=Usage(total_tokens=4),
        model="gpt-5-mini",
        status=200,
    )

    envelope = ensure_content_response_envelope(completion)
    rebuilt = ensure_completion_result(envelope)

    assert isinstance(envelope, ContentResponseEnvelope)
    assert envelope.message.blocks[0] == TextBlock("hello")
    assert rebuilt.content == "hello"
    assert rebuilt.tool_calls is not None
    assert rebuilt.tool_calls[0].name == "lookup"
    assert rebuilt.usage is not None
    assert rebuilt.usage.total_tokens == 4


def test_done_stream_event_roundtrip_preserves_terminal_result() -> None:
    completion = CompletionResult(
        content="hello",
        usage=Usage(total_tokens=3),
        model="gpt-5-mini",
        status=200,
    )
    event = StreamEvent(type=StreamEventType.DONE, data=completion)

    as_content = completion_stream_event_to_content_event(event)
    rebuilt = content_stream_event_to_completion_event(as_content)

    assert as_content.type is StreamEventType.DONE
    assert isinstance(as_content.data, ContentResponseEnvelope)
    assert rebuilt.type is StreamEventType.DONE
    assert isinstance(rebuilt.data, CompletionResult)
    assert rebuilt.data.content == "hello"
    assert rebuilt.data.usage is not None
    assert rebuilt.data.usage.total_tokens == 3
