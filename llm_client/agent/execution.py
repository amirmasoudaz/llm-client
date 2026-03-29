"""
Compatibility wrappers for agent-side tool execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tools.execution_engine import ToolExecutionEngine
from .definition import ToolExecutionMode

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
    execution_mode: ToolExecutionMode | None = None,
) -> list[ToolResult]:
    """Execute tool calls via the canonical tool execution engine."""
    engine = ToolExecutionEngine.from_agent_config(
        registry,
        config,
        middleware_chain=middleware_chain,
    )
    mode = execution_mode or (
        ToolExecutionMode.PARALLEL if config.parallel_tool_execution else ToolExecutionMode.SEQUENTIAL
    )
    batch = await engine.execute_calls(
        tool_calls,
        mode=mode,
        request_context=request_context,
        max_tool_calls=config.max_tool_calls_per_turn,
    )
    return [envelope.to_tool_result() for envelope in batch.results]


def apply_tool_output_limit(
    result: ToolResult,
    tool_name: str,
    limit: int | None,
) -> ToolResult:
    """
    Apply output character limit to a tool result.

    Returns the original result if within limit, truncated result otherwise.
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


__all__ = ["execute_tools", "apply_tool_output_limit"]
