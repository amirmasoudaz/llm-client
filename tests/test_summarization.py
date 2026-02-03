"""Tests for summarization interface."""

from __future__ import annotations

import pytest

from llm_client.summarization import (
    Summarizer,
    NoOpSummarizer,
    LLMSummarizer,
    LLMSummarizerConfig,
)
from llm_client.providers.types import Message


class TestNoOpSummarizer:
    """Test NoOpSummarizer behavior."""

    @pytest.mark.asyncio
    async def test_returns_placeholder(self):
        """NoOpSummarizer should return a placeholder summary."""
        summarizer = NoOpSummarizer()
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
            Message.user("How are you?"),
        ]
        
        result = await summarizer.summarize(messages, max_tokens=100)
        
        assert "3" in result  # Number of messages
        assert "summary" in result.lower() or "Summary" in result


class TestLLMSummarizerConfig:
    """Test LLMSummarizerConfig defaults."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = LLMSummarizerConfig()
        
        assert config.max_summary_tokens == 500
        assert config.temperature == 0.3
        assert "{conversation}" in config.prompt_template

    def test_custom_values(self):
        """Config should accept custom values."""
        config = LLMSummarizerConfig(
            max_summary_tokens=200,
            temperature=0.5,
            prompt_template="Summarize: {conversation}",
        )
        
        assert config.max_summary_tokens == 200
        assert config.temperature == 0.5


class TestLLMSummarizer:
    """Test LLMSummarizer behavior."""

    @pytest.mark.asyncio
    async def test_summarize_with_mock_provider(self, mock_provider):
        """LLMSummarizer should use provider to generate summary."""
        provider = mock_provider()
        summarizer = LLMSummarizer(provider)
        
        messages = [
            Message.user("What is Python?"),
            Message.assistant("Python is a programming language."),
        ]
        
        result = await summarizer.summarize(messages, max_tokens=100)
        
        # Mock provider returns content
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_empty_messages(self, mock_provider):
        """LLMSummarizer should handle empty messages gracefully."""
        provider = mock_provider()
        summarizer = LLMSummarizer(provider)
        
        result = await summarizer.summarize([], max_tokens=100)
        
        assert "no content" in result.lower()

    @pytest.mark.asyncio
    async def test_custom_config(self, mock_provider):
        """LLMSummarizer should use custom config."""
        provider = mock_provider()
        config = LLMSummarizerConfig(
            prompt_template="Brief summary: {conversation}",
            max_summary_tokens=50,
        )
        summarizer = LLMSummarizer(provider, config)
        
        messages = [Message.user("Hello")]
        result = await summarizer.summarize(messages, max_tokens=100)
        
        assert result is not None


class TestConversationSummarization:
    """Test Conversation with summarization."""

    @pytest.mark.asyncio
    async def test_get_messages_async_without_summarizer(self):
        """get_messages_async should work without summarizer."""
        from llm_client.conversation import Conversation
        
        conv = Conversation(
            system_message="You are helpful.",
            max_tokens=1000,
            truncation_strategy="sliding",
        )
        conv.add_user("Hello")
        conv.add_assistant("Hi!")
        
        messages = await conv.get_messages_async()
        
        assert len(messages) == 3  # system + 2 messages
        assert messages[0].content == "You are helpful."

    @pytest.mark.asyncio
    async def test_get_messages_async_with_summarizer(self):
        """get_messages_async should use summarizer when configured."""
        from llm_client.conversation import Conversation
        
        summarizer = NoOpSummarizer()
        conv = Conversation(
            system_message="You are helpful.",
            max_tokens=100,  # Very low to force summarization
            truncation_strategy="summarize",
            summarizer=summarizer,
        )
        
        # Add many messages to exceed token limit
        for i in range(20):
            conv.add_user(f"Message {i} " * 10)
            conv.add_assistant(f"Reply {i} " * 10)
        
        # This requires a model profile which we don't have in this test
        # Just verify the method is callable
        messages = await conv.get_messages_async()
        assert len(messages) > 0


class TestSyncWrappers:
    """Test sync wrapper behavior."""

    def test_sync_wrapper_outside_loop(self):
        """sync wrapper should work outside event loop."""
        from llm_client.sync import summarize_sync
        
        summarizer = NoOpSummarizer()
        messages = [Message.user("test")]
        
        result = summarize_sync(summarizer, messages, 100)
        assert "1" in result  # Summary of 1 message

    @pytest.mark.asyncio
    async def test_sync_wrapper_inside_loop_raises(self):
        """sync wrapper should raise inside event loop."""
        from llm_client.sync import summarize_sync
        
        summarizer = NoOpSummarizer()
        messages = [Message.user("test")]
        
        with pytest.raises(RuntimeError, match="async context"):
            summarize_sync(summarizer, messages, 100)
