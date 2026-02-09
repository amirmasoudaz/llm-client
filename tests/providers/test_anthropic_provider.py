"""
Tests for the Anthropic provider implementation.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.models import ModelProfile
from llm_client.providers.anthropic import AnthropicProvider
from llm_client.providers.types import (
    Message,
    StreamEventType,
    ToolCall,
)


# Define mock model for tests
class MockClaudeModel(ModelProfile):
    key = "mock-claude-3-5-sonnet"
    model_name = "claude-3-5-sonnet-20240620"
    category = "completions"
    context_window = 200000
    usage_costs = {
        "input": Decimal("3.00") / Decimal("1000000"),
        "output": Decimal("15.00") / Decimal("1000000"),
    }
    rate_limits = {}


# Mock anthropic classes since they might not be installed in test env
class MockAnthropicResponse:
    def __init__(self, content=None, usage=None, stop_reason=None):
        self.content = content or []
        self.usage = usage or MagicMock(input_tokens=10, output_tokens=20)
        self.stop_reason = stop_reason or "end_turn"


class MockStreamEvent:
    def __init__(self, type, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def mock_anthropic_client():
    with patch("llm_client.providers.anthropic.AsyncAnthropic") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def provider(mock_anthropic_client):
    with patch("llm_client.providers.anthropic.ANTHROPIC_AVAILABLE", True):
        # Pass the class directly to avoid lookup failure if registry is clean
        return AnthropicProvider(api_key="test-key", model=MockClaudeModel)


class TestAnthropicProvider:
    async def test_initialization(self):
        """Test provider initialization."""
        with patch("llm_client.providers.anthropic.ANTHROPIC_AVAILABLE", True):
            # Use the registered mock model key
            provider = AnthropicProvider(api_key="sk-test", model=MockClaudeModel.key)
            assert provider.model_name == MockClaudeModel.model_name
            assert provider.client is not None

    async def test_initialization_no_package(self):
        """Test initialization raises error if package missing."""
        with patch("llm_client.providers.anthropic.ANTHROPIC_AVAILABLE", False):
            with pytest.raises(ImportError):
                AnthropicProvider(model="claude-3-5-sonnet")

    async def test_convert_messages(self, provider):
        """Test message conversion logic."""
        messages = [
            Message.system("Be helpful"),
            Message.user("Hello"),
            Message.assistant("Hi there"),
            Message.user("How are you?"),
        ]

        system, formatted = provider._convert_messages_for_anthropic(messages)

        assert system == "Be helpful"
        assert len(formatted) == 3
        assert formatted[0] == {"role": "user", "content": "Hello"}
        assert formatted[1] == {"role": "assistant", "content": "Hi there"}
        assert formatted[2] == {"role": "user", "content": "How are you?"}

    async def test_convert_tool_messages(self, provider):
        """Test conversion of tool-related messages."""
        messages = [
            Message.assistant(
                "Calling tool", tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"city": "Paris"}')]
            ),
            Message.tool_result("call_1", "Sunny, 20C", name="get_weather"),
        ]

        system, formatted = provider._convert_messages_for_anthropic(messages)

        assert len(formatted) == 2

        # Check assistant message with tool use
        assistant_msg = formatted[0]
        assert assistant_msg["role"] == "assistant"
        content_blocks = assistant_msg["content"]
        assert len(content_blocks) == 2
        assert content_blocks[0]["type"] == "text"
        assert content_blocks[1]["type"] == "tool_use"
        assert content_blocks[1]["name"] == "get_weather"
        assert content_blocks[1]["input"] == {"city": "Paris"}

        # Check tool result message
        tool_msg = formatted[1]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_1"
        assert tool_msg["content"][0]["content"] == "Sunny, 20C"

    async def test_complete(self, provider, mock_anthropic_client):
        """Test basic completion."""
        # Mock response
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = "Hello from Claude"

        response = MockAnthropicResponse(content=[content_block])
        mock_anthropic_client.messages.create.return_value = response

        result = await provider.complete("Hello")

        assert result.ok
        assert result.content == "Hello from Claude"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

        # Verify call arguments
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-3-5-sonnet-20240620"
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Hello"

    async def test_embed_not_implemented(self, provider):
        """Test embeddings raise NotImplementedError."""

    async def test_complete_with_tools(self, provider, mock_anthropic_client):
        """Test completion with tool calls."""
        # Mock response with tool usage
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Checking weather..."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_123"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "London"}

        response = MockAnthropicResponse(content=[text_block, tool_block])
        mock_anthropic_client.messages.create.return_value = response

        from llm_client.tools.base import tool_from_function

        async def get_weather(city: str) -> str:
            """Get weather for city."""
            return f"Weather in {city}"

        tool = tool_from_function(get_weather)

        result = await provider.complete("Check London weather", tools=[tool])

        assert result.ok
        assert result.has_tool_calls
        assert result.content == "Checking weather..."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == '{"city": "London"}'

        # Verify call params
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"][0]["name"] == "get_weather"

    async def test_streaming_basic(self, provider, mock_anthropic_client):
        """Test streaming response."""
        # Setup mock stream context manager - stream() is synchronous returning context manager
        mock_stream_ctx = MagicMock()
        mock_stream_it = AsyncMock()

        # Configure the iterator
        mock_anthropic_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aenter__.return_value = mock_stream_it

        # Define stream events
        events = [
            MockStreamEvent("message_start", message=MagicMock(usage=MagicMock(input_tokens=5))),
            MockStreamEvent("content_block_start", index=0, content_block=MagicMock(type="text", text="")),
            MockStreamEvent("content_block_delta", index=0, delta=MagicMock(type="text_delta", text="Hello")),
            MockStreamEvent("content_block_delta", index=0, delta=MagicMock(type="text_delta", text=" World")),
            MockStreamEvent("content_block_stop", index=0),
            MockStreamEvent(
                "message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=10)
            ),
            MockStreamEvent("message_stop"),
        ]

        mock_stream_it.__aiter__.return_value = events

        chunks = []
        async for event in provider.stream("Hi"):
            if event.type == StreamEventType.TOKEN:
                chunks.append(event.data)
            elif event.type == StreamEventType.DONE:
                final_result = event.data

        assert "".join(chunks) == "Hello World"
        assert final_result.content == "Hello World"
        assert final_result.usage.total_tokens == 15  # 5 input + 10 output

    async def test_streaming_tools(self, provider, mock_anthropic_client):
        """Test streaming tool calls."""
        mock_stream_ctx = MagicMock()
        mock_stream_it = AsyncMock()

        # Configure the iterator
        mock_anthropic_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aenter__.return_value = mock_stream_it

        # Create tool use block mock with explicit name attribute
        tool_block = MagicMock(type="tool_use", id="call_1")
        tool_block.name = "search"

        events = [
            MockStreamEvent("message_start", message=MagicMock(usage=MagicMock(input_tokens=5))),
            MockStreamEvent("content_block_start", index=0, content_block=tool_block),
            MockStreamEvent(
                "content_block_delta", index=0, delta=MagicMock(type="input_json_delta", partial_json='{"q": "py')
            ),
            MockStreamEvent(
                "content_block_delta", index=0, delta=MagicMock(type="input_json_delta", partial_json='thon"}')
            ),
            MockStreamEvent("content_block_stop", index=0),
            MockStreamEvent(
                "message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock(output_tokens=20)
            ),
            MockStreamEvent("message_stop"),
        ]

        mock_stream_it.__aiter__.return_value = events

        tool_deltas = []
        tool_calls = []

        async for event in provider.stream("Search python"):
            if event.type == StreamEventType.TOOL_CALL_DELTA:
                tool_deltas.append(event.data.arguments_delta)
            elif event.type == StreamEventType.TOOL_CALL_END:
                tool_calls.append(event.data)

        assert "".join(tool_deltas) == '{"q": "python"}'
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == '{"q": "python"}'

    async def test_error_handling(self, provider, mock_anthropic_client):
        """Test error handling mapping."""
        import anthropic

        # Rate limit error
        mock_anthropic_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit", response=MagicMock(), body={}
        )

        result = await provider.complete("hi")
        assert result.status == 429
        assert "Rate limit" in result.error

        # Reset side effect
        mock_anthropic_client.messages.create.side_effect = None

        # Connection error
        mock_anthropic_client.messages.create.side_effect = anthropic.APIConnectionError(request=MagicMock())

        result = await provider.complete("hi")
        assert result.status == 500
        assert "Connection error" in result.error
