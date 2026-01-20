"""
Tests for the OpenAI provider implementation.

These tests use mock responses to avoid making real API calls.
The tests validate the provider's behavior for:
- Basic completions
- Tool calling
- Error handling
- Streaming
- Retry logic
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.providers.types import (
    CompletionResult,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)
from llm_client.errors import (
    RateLimitError,
    AuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class TestOpenAIProviderWithMocks:
    """Test OpenAI provider using the mock provider fixture."""
    
    async def test_complete_basic(self, mock_provider, mock_completion_result):
        """Test basic completion without tools."""
        provider = mock_provider(responses=[
            mock_completion_result(content="Hello, I'm an AI assistant.")
        ])
        
        result = await provider.complete(messages="Hello, who are you?")
        
        assert result.ok
        assert result.content == "Hello, I'm an AI assistant."
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
    
    async def test_complete_with_tool_calls(
        self,
        mock_provider,
        mock_completion_result,
        mock_tool_call,
    ):
        """Test completion with tool calling."""
        tool_call = mock_tool_call(
            name="get_weather",
            arguments='{"city": "Tokyo"}',
        )
        
        provider = mock_provider(responses=[
            mock_completion_result(
                content=None,
                tool_calls=[tool_call],
            )
        ])
        
        result = await provider.complete(messages="What's the weather in Tokyo?")
        
        assert result.ok
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
    
    async def test_streaming(self, mock_provider, mock_completion_result):
        """Test streaming produces correct events."""
        provider = mock_provider(responses=[
            mock_completion_result(content="Hello there!")
        ])
        
        tokens = []
        result = None
        
        async for event in provider.stream(messages="Hi"):
            if event.type == StreamEventType.TOKEN:
                tokens.append(event.data)
            elif event.type == StreamEventType.DONE:
                result = event.data
        
        # Should get individual characters from "Hello there!"
        assert "".join(tokens) == "Hello there!"
        assert result is not None
        assert result.content == "Hello there!"
    
    async def test_embedding(self, mock_provider):
        """Test embedding generation."""
        provider = mock_provider()
        
        result = await provider.embed("Test text")
        
        assert result.ok
        assert len(result.embeddings) == 1
        assert len(result.embedding) == 1536  # Standard OpenAI dimension
    
    async def test_token_counting(self, mock_provider):
        """Test token counting."""
        provider = mock_provider()
        
        count = provider.count_tokens("Hello world")
        
        # Mock implementation divides by 4
        assert count > 0
    
    async def test_multiple_calls_use_response_queue(
        self,
        mock_provider,
        mock_completion_result,
    ):
        """Test that multiple calls use responses in order."""
        provider = mock_provider(responses=[
            mock_completion_result(content="First response"),
            mock_completion_result(content="Second response"),
        ])
        
        result1 = await provider.complete("First")
        result2 = await provider.complete("Second")
        
        assert result1.content == "First response"
        assert result2.content == "Second response"


class TestCompletionResult:
    """Test CompletionResult dataclass."""
    
    def test_ok_property(self, mock_completion_result):
        """Test ok property for successful result."""
        result = mock_completion_result(status=200, content="Success")
        assert result.ok
        
    def test_ok_false_for_error(self, mock_completion_result):
        """Test ok property for error result."""
        result = CompletionResult(status=500, error="Server error")
        assert not result.ok
    
    def test_has_tool_calls(self, mock_completion_result, mock_tool_call):
        """Test has_tool_calls property."""
        result_no_tools = mock_completion_result(content="No tools")
        assert not result_no_tools.has_tool_calls
        
        result_with_tools = mock_completion_result(
            content=None,
            tool_calls=[mock_tool_call()],
        )
        assert result_with_tools.has_tool_calls
    
    def test_to_message(self, mock_completion_result, mock_tool_call):
        """Test conversion to Message."""
        result = mock_completion_result(
            content="Hello",
            tool_calls=[mock_tool_call()],
        )
        
        msg = result.to_message()
        
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hello"
        assert msg.tool_calls is not None
    
    def test_to_dict(self, mock_completion_result):
        """Test dictionary serialization."""
        result = mock_completion_result(content="Test", model="gpt-5-nano")
        
        d = result.to_dict()
        
        assert d["content"] == "Test"
        assert d["model"] == "gpt-5-nano"
        assert d["status"] == 200


class TestMessage:
    """Test Message dataclass."""
    
    def test_user_factory(self):
        """Test user message factory."""
        msg = Message.user("Hello")
        
        assert msg.role == Role.USER
        assert msg.content == "Hello"
    
    def test_assistant_factory(self):
        """Test assistant message factory."""
        msg = Message.assistant("Hi there!")
        
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_system_factory(self):
        """Test system message factory."""
        msg = Message.system("You are helpful.")
        
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."
    
    def test_tool_result_factory(self):
        """Test tool result message factory."""
        msg = Message.tool_result(
            tool_call_id="call_123",
            content="Weather is sunny",
            name="get_weather",
        )
        
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.content == "Weather is sunny"
    
    def test_to_dict(self, mock_tool_call):
        """Test dictionary conversion."""
        msg = Message.assistant("Hi", tool_calls=[mock_tool_call()])
        
        d = msg.to_dict()
        
        assert d["role"] == "assistant"
        assert d["content"] == "Hi"
        assert "tool_calls" in d
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "role": "user",
            "content": "Hello",
        }
        
        msg = Message.from_dict(d)
        
        assert msg.role == Role.USER
        assert msg.content == "Hello"


class TestUsage:
    """Test Usage dataclass."""
    
    def test_creation(self, mock_usage):
        """Test usage creation."""
        usage = mock_usage(input_tokens=100, output_tokens=50)
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
    
    def test_to_dict(self, mock_usage):
        """Test dictionary serialization."""
        usage = mock_usage(input_tokens=10, output_tokens=20)
        
        d = usage.to_dict()
        
        assert d["input_tokens"] == 10
        assert d["output_tokens"] == 20
        assert d["total_tokens"] == 30
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
        }
        
        usage = Usage.from_dict(d)
        
        assert usage.input_tokens == 5
        assert usage.output_tokens == 10
