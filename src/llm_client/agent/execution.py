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
    from ..tools.base import ToolRegistry, ToolResult


async def execute_tools(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    config: AgentConfig,
) -> list[ToolResult]:
    """
    Execute tool calls, optionally in parallel.

    Args:
        tool_calls: List of tool calls to execute
        registry: Tool registry containing the tools
        config: Agent configuration with execution settings

    Returns:
        List of tool results
    """
    if not tool_calls:
        return []

    # Limit tool calls per turn
    tool_calls = tool_calls[: config.max_tool_calls_per_turn]

    if config.parallel_tool_execution:
        # Execute all tools in parallel
        tasks = [
            execute_single_tool(tc, registry, config.tool_timeout, config.tool_retry_attempts) for tc in tool_calls
        ]
        return await asyncio.gather(*tasks)
    else:
        # Execute sequentially
        results = []
        for tc in tool_calls:
            result = await execute_single_tool(tc, registry, config.tool_timeout, config.tool_retry_attempts)
            results.append(result)
        return results


async def execute_single_tool(
    tool_call: ToolCall,
    registry: ToolRegistry,
    timeout: float,
    retries: int = 0,
) -> ToolResult:
    """
    Execute a single tool call with timeout and retries.

    Args:
        tool_call: The tool call to execute
        registry: Tool registry containing the tool
        timeout: Timeout in seconds
        retries: Number of retry attempts on failure

    Returns:
        ToolResult with success or error
    """
    from ..tools.base import ToolResult

    last_error = None

    for attempt in range(retries + 1):
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
