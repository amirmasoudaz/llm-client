"""
Tool system for agent function calling.

Provides tools, registry, and decorators for defining callable functions
that LLM agents can use.
"""
from .base import Tool, ToolRegistry, ToolResult, tool_from_function
from .decorators import sync_tool, tool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool_from_function",
    "tool",
    "sync_tool",
]

