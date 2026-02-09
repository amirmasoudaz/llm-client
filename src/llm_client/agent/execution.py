"""
Tool execution helpers for the agent.

This module provides functions for executing tools during agent runs.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..providers.types import ToolCall
    from ..spec import RequestContext
    from ..tools.base import ToolRegistry, ToolResult
    from ..tools.middleware import MiddlewareChain


async def execute_tools(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    config: AgentConfig,
    *,
    middleware_chain: MiddlewareChain | None = None,
    request_context: RequestContext | None = None,
) -> list[ToolResult]:
    """
    Execute tool calls, optionally in parallel.

    Args:
        tool_calls: List of tool calls to execute
        registry: Tool registry containing the tools
        config: Agent configuration with execution settings
        middleware_chain: Optional middleware chain for tool execution
        request_context: Optional request context for correlation

    Returns:
        List of tool results
    """
    if not tool_calls:
        return []

    # Limit tool calls per turn
    tool_calls = tool_calls[: config.max_tool_calls_per_turn]

    # Get effective middleware chain
    effective_chain = middleware_chain or config.get_middleware_chain()

    if config.parallel_tool_execution:
        # Check for cancellation before starting parallel execution
        if request_context and request_context.cancellation_token:
            request_context.cancellation_token.raise_if_cancelled()

        # Execute all tools in parallel
        tasks = [
            execute_single_tool(
                tc,
                registry,
                config.tool_timeout,
                config.tool_retry_attempts,
                middleware_chain=effective_chain,
                request_context=request_context,
            )
            for tc in tool_calls
        ]
        return await asyncio.gather(*tasks)
    else:
        # Execute sequentially
        results = []
        for tc in tool_calls:
            # Check for cancellation before each tool execution
            if request_context and request_context.cancellation_token:
                request_context.cancellation_token.raise_if_cancelled()

            result = await execute_single_tool(
                tc,
                registry,
                config.tool_timeout,
                config.tool_retry_attempts,
                middleware_chain=effective_chain,
                request_context=request_context,
            )
            results.append(result)
        return results


async def execute_single_tool(
    tool_call: ToolCall,
    registry: ToolRegistry,
    timeout: float,
    retries: int = 0,
    *,
    middleware_chain: MiddlewareChain | None = None,
    request_context: RequestContext | None = None,
) -> ToolResult:
    """
    Execute a single tool call with optional middleware, timeout and retries.

    When middleware is provided, the middleware chain handles timeout and retries
    (via TimeoutMiddleware and RetryMiddleware). When no middleware is provided,
    this function handles them directly for backward compatibility.

    Args:
        tool_call: The tool call to execute
        registry: Tool registry containing the tool
        timeout: Timeout in seconds (used when no middleware)
        retries: Number of retry attempts on failure (used when no middleware)
        middleware_chain: Optional middleware chain for execution
        request_context: Optional request context for correlation

    Returns:
        ToolResult with success or error
    """
    from ..tools.base import ToolResult

    # If middleware chain is provided, delegate to it (it handles timeout/retries)
    if middleware_chain is not None:
        return await registry.execute_with_middleware(
            tool_call.name,
            tool_call.arguments,
            middleware_chain=middleware_chain,
            context=request_context,
        )

    # Legacy execution path without middleware
    last_error = None

    for attempt in range(retries + 1):
        # Check for cancellation before each retry attempt
        if request_context and request_context.cancellation_token:
            request_context.cancellation_token.raise_if_cancelled()

        try:
            result = await asyncio.wait_for(
                registry.execute(tool_call.name, tool_call.arguments),
                timeout=timeout,
            )
            if result.success:
                return result
            # Treat tool failure as error to trigger retry
            last_error = result.error or "Tool execution failed"
        except asyncio.TimeoutError:
            last_error = f"Tool '{tool_call.name}' timed out after {timeout}s"
        except Exception as e:
            last_error = f"Tool execution error: {e}"

        # If we have retries left, wait a bit
        if attempt < retries:
            await asyncio.sleep(1.0)

    return ToolResult.error_result(last_error or "Unknown error")


def apply_tool_output_limit(
    result: ToolResult,
    tool_name: str,
    limit: int | None,
) -> ToolResult:
    """
    Apply output character limit to a tool result.

    Args:
        result: The tool result to truncate
        tool_name: Name of the tool (for metadata)
        limit: Maximum output characters, or None for no limit

    Returns:
        Original result if within limit, truncated result otherwise
    """
    from ..tools.base import ToolResult

    if not limit:
        return result

    output = result.to_string()
    if len(output) <= limit:
        return result

    metadata = dict(result.metadata)
    metadata.update(
        {
            "truncated": True,
            "original_size": len(output),
            "limit": limit,
            "tool": tool_name,
        }
    )
    return ToolResult(
        content=output[:limit],
        success=result.success,
        error=result.error,
        metadata=metadata,
    )


__all__ = ["execute_tools", "execute_single_tool", "apply_tool_output_limit"]
