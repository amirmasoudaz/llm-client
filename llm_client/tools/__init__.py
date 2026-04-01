"""
Tool system for agent function calling.

Provides tools, registry, decorators, and middleware for defining callable functions
that LLM agents can use.
"""

from .base import (
    ResponsesBuiltinTool,
    ResponsesConnectorId,
    ResponsesMCPApprovalPolicy,
    ResponsesMCPTool,
    ResponsesMCPToolFilter,
    ResponsesCustomTool,
    ResponsesGrammar,
    Tool,
    ToolExecutionMetadata,
    ToolRegistry,
    ToolResult,
    ensure_function_tools_only,
    is_provider_native_tool,
    tool_from_function,
)
from .decorators import sync_tool, tool
from .execution_engine import ToolExecutionBatch, ToolExecutionEngine, ToolExecutionEnvelope, ToolExecutionStatus
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
    ToolOutputPolicyMiddleware,
)

__all__ = [
    # Core
    "Tool",
    "ToolExecutionBatch",
    "ToolExecutionEngine",
    "ToolExecutionEnvelope",
    "ToolExecutionMetadata",
    "ToolExecutionStatus",
    "ToolResult",
    "ToolRegistry",
    "ResponsesGrammar",
    "ResponsesBuiltinTool",
    "ResponsesConnectorId",
    "ResponsesMCPToolFilter",
    "ResponsesMCPApprovalPolicy",
    "ResponsesMCPTool",
    "ResponsesCustomTool",
    "is_provider_native_tool",
    "ensure_function_tools_only",
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
    "ToolOutputPolicyMiddleware",
]
