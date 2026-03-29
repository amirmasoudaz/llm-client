from __future__ import annotations

import asyncio

import pytest

from llm_client.conversation import Conversation
from llm_client.models import GPT5Nano
from llm_client.providers.types import Message
from llm_client.summarization import NoOpSummarizer
from llm_client.sync import get_messages_sync, run_async_sync, summarize_sync


async def _loop_identity() -> int:
    return id(asyncio.get_running_loop())


def test_run_async_sync_reuses_background_event_loop() -> None:
    first = run_async_sync(_loop_identity())
    second = run_async_sync(_loop_identity())

    assert first == second


def test_sync_wrappers_support_conversation_and_summarizer() -> None:
    conversation = Conversation(
        system_message="You are concise.",
        max_tokens=120,
        reserve_tokens=10,
        truncation_strategy="summarize",
        summarizer=NoOpSummarizer(),
    )
    conversation.add_user("alpha")
    conversation.add_assistant("beta")
    conversation.add_user("gamma")

    messages = get_messages_sync(conversation, GPT5Nano)
    summary = summarize_sync(NoOpSummarizer(), [Message.user("x"), Message.assistant("y")], max_tokens=32)

    assert [message.role.value for message in messages] == ["system", "user", "assistant", "user"]
    assert summary == "[Summary of 2 earlier messages]"


@pytest.mark.asyncio
async def test_sync_wrappers_reject_async_contexts() -> None:
    conversation = Conversation()
    conversation.add_user("hello")

    with pytest.raises(RuntimeError, match="get_messages_sync\\(\\) cannot be called inside an async context"):
        get_messages_sync(conversation, GPT5Nano)

    with pytest.raises(RuntimeError, match="summarize_sync\\(\\) cannot be called inside an async context"):
        summarize_sync(NoOpSummarizer(), [Message.user("hello")], max_tokens=16)

    with pytest.raises(RuntimeError, match="run_async_sync\\(\\) cannot be called inside an async context"):
        run_async_sync(_loop_identity())
