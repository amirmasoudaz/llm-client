"""
Tool system for agent function calling.

Provides tools, registry, decorators, and middleware for defining callable functions
that LLM agents can use.
"""

from .base import Tool, ToolRegistry, ToolResult, tool_from_function
from .decorators import sync_tool, tool
from .middleware import (
    ToolMiddleware,
    ToolExecutionContext,
    MiddlewareChain,
    LoggingMiddleware,
    TimeoutMiddleware,
    RetryMiddleware,
    PolicyMiddleware,
    BudgetMiddleware,
    ConcurrencyLimitMiddleware,
    CircuitBreakerMiddleware,
    ResultSizeMiddleware,
    RedactionMiddleware,
    TelemetryMiddleware,
)

__all__ = [
    # Core
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool_from_function",
    # Decorators
    "tool",
    "sync_tool",
    # Middleware
    "ToolMiddleware",
    "ToolExecutionContext",
    "MiddlewareChain",
    "LoggingMiddleware",
    "TimeoutMiddleware",
    "RetryMiddleware",
    "PolicyMiddleware",
    "BudgetMiddleware",
    "ConcurrencyLimitMiddleware",
    "CircuitBreakerMiddleware",
    "ResultSizeMiddleware",
    "RedactionMiddleware",
    "TelemetryMiddleware",
]
