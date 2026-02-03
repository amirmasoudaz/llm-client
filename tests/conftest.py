"""
Shared test fixtures and mocks for llm-client tests.

This module provides:
- Mock OpenAI/Anthropic responses
- Fake cache backends (in-memory)
- Sample conversations and messages
- Tool fixtures for testing
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import project types
from llm_client.providers.types import (
    CompletionResult,
    EmbeddingResult,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)
from llm_client.tools.base import Tool

# =============================================================================
# Mock Response Factories
# =============================================================================


def make_completion_result(
    content: str = "Test response",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
    model: str = "gpt-5-nano",
    finish_reason: str = "stop",
    status: int = 200,
) -> CompletionResult:
    """Create a mock CompletionResult."""
    return CompletionResult(
        content=content,
        tool_calls=tool_calls,
        usage=usage or Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        model=model,
        finish_reason=finish_reason,
        status=status,
    )


def make_tool_call(
    id: str = "call_test123",
    name: str = "test_tool",
    arguments: str = '{"arg1": "value1"}',
) -> ToolCall:
    """Create a mock ToolCall."""
    return ToolCall(id=id, name=name, arguments=arguments)


def make_usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    total_tokens: int | None = None,
) -> Usage:
    """Create a mock Usage."""
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or (input_tokens + output_tokens),
    )


# =============================================================================
# In-Memory Cache Backend (for testing)
# =============================================================================


@dataclass
class InMemoryCacheBackend:
    """Simple in-memory cache for testing."""

    name: str = "memory"
    default_collection: str = "test"
    _store: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def ensure_ready(self) -> None:
        pass

    async def close(self) -> None:
        self._store.clear()

    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: str | None = None,
    ) -> tuple[str, bool]:
        col = collection or self.default_collection
        if rewrite_cache and not regen_cache:
            # Find new suffix
            suffix = 1
            while f"{identifier}_{suffix}" in self._store.get(col, {}):
                suffix += 1
            return f"{identifier}_{suffix}", False
        return identifier, not regen_cache

    async def read(self, effective_key: str, collection: str | None = None) -> dict[str, Any] | None:
        col = collection or self.default_collection
        return self._store.get(col, {}).get(effective_key)

    async def write(
        self,
        effective_key: str,
        response: dict[str, Any],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        col = collection or self.default_collection
        if col not in self._store:
            self._store[col] = {}
        self._store[col][effective_key] = response

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        col = collection or self.default_collection
        return effective_key in self._store.get(col, {})

    async def warm(self) -> None:
        pass

    # Methods expected by ExecutionEngine
    async def get_cached(
        self,
        cache_key: str,
        only_ok: bool = True,
        collection: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Get cached response by key."""
        col = collection or self.default_collection
        cached = self._store.get(col, {}).get(cache_key)
        return cached, cache_key if cached else None

    async def put_cached(
        self,
        cache_key: str,
        response: dict[str, Any],
        collection: str | None = None,
    ) -> None:
        """Store response in cache."""
        col = collection or self.default_collection
        if col not in self._store:
            self._store[col] = {}
        self._store[col][cache_key] = response


# =============================================================================
# Mock Providers
# =============================================================================


class MockOpenAIProvider:
    """Mock OpenAI provider for testing."""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        responses: list[CompletionResult] | None = None,
    ):
        self._model_name = model
        self._responses = responses or [make_completion_result()]
        self._call_count = 0
        self._call_history: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages,
        tools=None,
        **kwargs,
    ) -> CompletionResult:
        self._call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        result = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return result

    async def stream(
        self,
        messages,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1

        # Yield tokens character by character
        if response.content:
            for char in response.content:
                yield StreamEvent(type=StreamEventType.TOKEN, data=char)

        # Yield usage
        if response.usage:
            yield StreamEvent(type=StreamEventType.USAGE, data=response.usage)

        # Yield done
        yield StreamEvent(type=StreamEventType.DONE, data=response)

    async def embed(
        self,
        inputs,
        **kwargs,
    ) -> EmbeddingResult:
        if isinstance(inputs, str):
            inputs = [inputs]
        if not hasattr(self, '_embed_count'):
            self._embed_count = 0
        self._embed_count += 1
        return EmbeddingResult(
            embeddings=[[0.1] * 1536 for _ in inputs],
            usage=make_usage(input_tokens=len(inputs) * 10, output_tokens=0),
            model="text-embedding-3-small",
        )

    def count_tokens(self, content: Any) -> int:
        if isinstance(content, str):
            return len(content) // 4
        return 100

    async def close(self) -> None:
        pass


# =============================================================================
# Test Tools
# =============================================================================


async def simple_tool_handler(message: str) -> str:
    """Simple test tool that echoes the message."""
    return f"Echo: {message}"


async def error_tool_handler(should_fail: bool = False) -> str:
    """Tool that can optionally fail for testing."""
    if should_fail:
        raise ValueError("Tool intentionally failed")
    return "Success"


async def slow_tool_handler(delay: float = 0.1) -> str:
    """Tool that simulates slow execution."""
    await asyncio.sleep(delay)
    return f"Completed after {delay}s"


def make_test_tool(
    name: str = "test_tool",
    description: str = "A test tool",
    handler=None,
) -> Tool:
    """Create a test tool with the given handler."""
    if handler is None:
        handler = simple_tool_handler

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "A message"}},
            "required": ["message"],
        },
        handler=handler,
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def mock_completion_result():
    """Fixture providing a factory for CompletionResults."""
    return make_completion_result


@pytest.fixture
def mock_tool_call():
    """Fixture providing a factory for ToolCalls."""
    return make_tool_call


@pytest.fixture
def mock_usage():
    """Fixture providing a factory for Usage."""
    return make_usage


@pytest.fixture
def memory_cache():
    """Fixture providing an in-memory cache backend."""
    return InMemoryCacheBackend()


@pytest.fixture
def mock_provider():
    """Fixture providing a factory for mock providers."""

    def _factory(responses=None, model="gpt-5-nano"):
        return MockOpenAIProvider(model=model, responses=responses)

    return _factory


@pytest.fixture
def sample_messages():
    """Fixture providing sample conversation messages."""
    return [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="Hello, how are you?"),
        Message(role=Role.ASSISTANT, content="I'm doing well, thank you!"),
        Message(role=Role.USER, content="What can you help me with?"),
    ]


@pytest.fixture
def test_tool():
    """Fixture providing a simple test tool."""
    return make_test_tool()


@pytest.fixture
def error_tool():
    """Fixture providing a tool that can fail."""
    return Tool(
        name="error_tool",
        description="A tool that can fail",
        parameters={
            "type": "object",
            "properties": {"should_fail": {"type": "boolean", "default": False}},
        },
        handler=error_tool_handler,
    )


@pytest.fixture
def slow_tool():
    """Fixture providing a slow tool for timeout testing."""
    return Tool(
        name="slow_tool",
        description="A slow tool",
        parameters={
            "type": "object",
            "properties": {"delay": {"type": "number", "default": 0.1}},
        },
        handler=slow_tool_handler,
    )


# =============================================================================
# Mock Patching Helpers
# =============================================================================


@pytest.fixture
def mock_openai_client():
    """Fixture that patches the OpenAI AsyncOpenAI client."""
    with patch("openai.AsyncOpenAI") as mock_class:
        mock_instance = AsyncMock()
        mock_class.return_value = mock_instance

        # Setup default responses
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.choices[0].message.tool_calls = None
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = MagicMock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_completion.usage.total_tokens = 30
        mock_completion.model = "gpt-5-nano"

        mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)

        yield mock_instance


@pytest.fixture
def mock_anthropic_client():
    """Fixture that patches the Anthropic client."""
    with patch("anthropic.AsyncAnthropic") as mock_class:
        mock_instance = AsyncMock()
        mock_class.return_value = mock_instance

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.model = "claude-3-5-sonnet-20241022"

        mock_instance.messages.create = AsyncMock(return_value=mock_response)

        yield mock_instance
