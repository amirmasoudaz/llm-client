"""
Agent session persistence.

This module provides functions for saving and loading agent sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tools.base import Tool, ToolRegistry


def save_agent_session(
    path: str | Path,
    conversation: Any,
    config: Any,
    tools: ToolRegistry | None,
) -> None:
    """
    Save an agent session to a file.

    Args:
        path: File path to save the session to (JSON format)
        conversation: The conversation object
        config: The agent configuration
        tools: The tool registry
    """
    path = Path(path)

    session_data = {
        "version": "1.0",
        "conversation": conversation.to_dict(),
        "config": {
            "max_turns": config.max_turns,
            "max_tool_calls_per_turn": config.max_tool_calls_per_turn,
            "parallel_tool_execution": config.parallel_tool_execution,
            "tool_timeout": config.tool_timeout,
            "max_tool_output_chars": config.max_tool_output_chars,
            "max_tokens": config.max_tokens,
            "reserve_tokens": config.reserve_tokens,
            "stop_on_tool_error": config.stop_on_tool_error,
            "include_tool_errors_in_context": config.include_tool_errors_in_context,
            "stream_tool_calls": config.stream_tool_calls,
        },
        "tool_names": tools.names if tools else [],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session_data, indent=2))


def load_agent_session(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load an agent session from a file.

    Args:
        path: File path to load the session from

    Returns:
        Dictionary with session data containing:
        - conversation: Conversation dict
        - config: AgentConfig dict
        - tool_names: List of tool names used in the session
    """
    path = Path(path)
    return json.loads(path.read_text())


async def quick_agent(
    prompt: str,
    *,
    model: str = "gpt-5-mini",
    tools: list[Tool] | None = None,
    system_message: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Quick one-shot agent call.

    Convenience function for simple agent interactions without
    explicitly creating provider and agent instances.

    Args:
        prompt: User message
        model: Model key
        tools: Optional tools
        system_message: Optional system message
        **kwargs: Additional arguments

    Returns:
        Agent's response text

    Example:
        ```python
        response = await quick_agent(
            "What's 2 + 2?",
            model="gpt-5-nano",
        )
        ```
    """
    from ..providers.openai import OpenAIProvider
    from .core import Agent

    async with OpenAIProvider(model=model) as provider:
        agent = Agent(
            provider=provider,
            tools=tools,
            system_message=system_message,
        )
        result = await agent.run(prompt, **kwargs)
        return result.content or ""


__all__ = ["save_agent_session", "load_agent_session", "quick_agent"]
