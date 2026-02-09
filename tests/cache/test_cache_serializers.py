"""Tests for cache serializers."""

import pytest

from llm_client.cache.serializers import cache_dict_to_result, result_to_cache_dict
from llm_client.providers.types import CompletionResult, ToolCall, Usage


class TestResultToCacheDict:
    """Tests for result_to_cache_dict function."""

    def test_basic_serialization(self) -> None:
        """Basic result serializes correctly."""
        result = CompletionResult(
            content="Hello world",
            status=200,
            model="gpt-5",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

        cached = result_to_cache_dict(result, {"model": "gpt-5"})

        assert cached["output"] == "Hello world"
        assert cached["status"] == 200
        assert cached["model"] == "gpt-5"
        assert cached["params"] == {"model": "gpt-5"}
        assert cached["error"] == "OK"

    def test_with_tool_calls(self) -> None:
        """Tool calls are serialized correctly."""
        result = CompletionResult(
            content=None,
            status=200,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments='{"city": "Tokyo"}'),
                ToolCall(id="call_2", name="get_time", arguments='{"tz": "UTC"}'),
            ],
        )

        cached = result_to_cache_dict(result, {})

        assert len(cached["tool_calls"]) == 2
        assert cached["tool_calls"][0]["name"] == "get_weather"
        assert cached["tool_calls"][1]["id"] == "call_2"

    def test_with_error(self) -> None:
        """Error is preserved."""
        result = CompletionResult(
            status=500,
            error="API error",
        )

        cached = result_to_cache_dict(result, {})

        assert cached["error"] == "API error"
        assert cached["status"] == 500

    def test_with_reasoning(self) -> None:
        """Reasoning content is preserved."""
        result = CompletionResult(
            content="Answer",
            status=200,
            reasoning="Let me think step by step...",
        )

        cached = result_to_cache_dict(result, {})

        assert cached["reasoning"] == "Let me think step by step..."


class TestCacheDictToResult:
    """Tests for cache_dict_to_result function."""

    def test_basic_deserialization(self) -> None:
        """Basic cached dict deserializes correctly."""
        cached = {
            "output": "Hello world",
            "status": 200,
            "model": "gpt-5",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "error": "OK",
            "tool_calls": [],
        }

        result = cache_dict_to_result(cached)

        assert result.content == "Hello world"
        assert result.status == 200
        assert result.model == "gpt-5"
        assert result.ok
        assert result.error is None

    def test_with_tool_calls(self) -> None:
        """Tool calls are deserialized correctly."""
        cached = {
            "output": None,
            "status": 200,
            "error": "OK",
            "tool_calls": [
                {"id": "call_1", "name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            ],
        }

        result = cache_dict_to_result(cached)

        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

    def test_with_error(self) -> None:
        """Error is deserialized correctly."""
        cached = {
            "output": None,
            "status": 500,
            "error": "Server error",
        }

        result = cache_dict_to_result(cached)

        assert not result.ok
        assert result.error == "Server error"

    def test_round_trip(self) -> None:
        """Round trip preserves data."""
        original = CompletionResult(
            content="Test content",
            status=200,
            model="gpt-5-nano",
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            tool_calls=[
                ToolCall(id="call_abc", name="test_tool", arguments="{}"),
            ],
            finish_reason="stop",
            reasoning="Some reasoning",
        )

        cached = result_to_cache_dict(original, {"test": "params"})
        restored = cache_dict_to_result(cached)

        assert restored.content == original.content
        assert restored.status == original.status
        assert restored.model == original.model
        assert restored.finish_reason == original.finish_reason
        assert restored.reasoning == original.reasoning
        assert len(restored.tool_calls) == len(original.tool_calls)
        assert restored.tool_calls[0].name == original.tool_calls[0].name
