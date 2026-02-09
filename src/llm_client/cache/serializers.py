"""
Cache serialization utilities for CompletionResult.

This module provides functions to convert CompletionResult objects
to/from dictionaries for cache storage, eliminating duplicate
implementations across providers.
"""

from __future__ import annotations

from typing import Any

from ..providers.types import CompletionResult, ToolCall, Usage


def result_to_cache_dict(result: CompletionResult, params: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a CompletionResult to a dictionary suitable for cache storage.

    Args:
        result: The completion result to serialize
        params: The original request parameters (stored for debugging/analysis)

    Returns:
        Dictionary with all relevant result data
    """
    return {
        "params": params,
        "output": result.content,
        "usage": result.usage.to_dict() if result.usage else {},
        "status": result.status,
        "error": result.error or "OK",
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in (result.tool_calls or [])
        ],
        "model": result.model,
        "finish_reason": result.finish_reason,
        "reasoning": result.reasoning,
    }


def cache_dict_to_result(cached: dict[str, Any]) -> CompletionResult:
    """
    Convert a cached dictionary back to a CompletionResult.

    Args:
        cached: Dictionary from cache storage

    Returns:
        Reconstructed CompletionResult
    """
    tool_calls = None
    raw_tcs = cached.get("tool_calls")
    if raw_tcs:
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in raw_tcs
        ]

    error = cached.get("error")
    if error == "OK":
        error = None

    return CompletionResult(
        content=cached.get("output"),
        tool_calls=tool_calls,
        usage=Usage.from_dict(cached.get("usage", {})),
        status=cached.get("status", 200),
        error=error,
        model=cached.get("model"),
        finish_reason=cached.get("finish_reason"),
        reasoning=cached.get("reasoning"),
    )


__all__ = ["result_to_cache_dict", "cache_dict_to_result"]
