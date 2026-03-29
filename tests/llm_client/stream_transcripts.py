from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class AsyncSequence:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class AnthropicStreamManager:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def __aiter__(self):
        return AsyncSequence(list(self._events))


def openai_text_chunks() -> list[Any]:
    return [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="openai-", tool_calls=None), finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}),
        ),
    ]


def openai_tool_call_chunks() -> list[Any]:
    return [
        SimpleNamespace(
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
        ),
        SimpleNamespace(
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
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}),
        ),
    ]


def google_text_chunks() -> list[Any]:
    return [
        SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="google-ok")]))],
            usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3),
        )
    ]


def google_tool_call_chunks() -> list[Any]:
    return [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(function_call=SimpleNamespace(name="lookup", args={"q": "x"}))]
                    )
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3),
        )
    ]


def anthropic_text_events() -> list[Any]:
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=2))),
        SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="anthropic-ok")),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=1),
            delta=SimpleNamespace(stop_reason="end_turn"),
        ),
    ]


def anthropic_tool_call_events() -> list[Any]:
    return [
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


__all__ = [
    "AnthropicStreamManager",
    "AsyncSequence",
    "openai_text_chunks",
    "openai_tool_call_chunks",
    "google_text_chunks",
    "google_tool_call_chunks",
    "anthropic_text_events",
    "anthropic_tool_call_events",
]
