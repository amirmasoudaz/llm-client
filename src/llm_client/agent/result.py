"""
Agent result types.

This module defines the result types for agent operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..conversation import Conversation
    from ..providers.types import CompletionResult, ToolCall, Usage
    from ..tools.base import ToolResult


@dataclass
class TurnResult:
    """Result of a single agent turn."""

    completion: CompletionResult
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    turn_number: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def content(self) -> str | None:
        return self.completion.content


@dataclass
class AgentResult:
    """Final result of an agent run."""

    content: str | None = None
    turns: list[TurnResult] = field(default_factory=list)
    conversation: Conversation | None = None

    # Aggregated usage
    total_usage: Usage | None = None

    # Status
    status: Literal["success", "max_turns", "error"] = "success"
    error: str | None = None

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls across all turns."""
        calls = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "num_turns": self.num_turns,
            "status": self.status,
            "error": self.error,
            "total_usage": self.total_usage.to_dict() if self.total_usage else None,
        }


__all__ = ["TurnResult", "AgentResult"]
