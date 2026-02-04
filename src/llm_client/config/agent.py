"""
Agent configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.middleware import MiddlewareChain


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

    # Turn limits
    max_turns: int = 10
    max_tool_calls_per_turn: int = 10

    # Tool execution
    parallel_tool_execution: bool = True
    tool_timeout: float = 30.0
    max_tool_output_chars: int | None = None
    tool_retry_attempts: int = 0

    # Context management
    max_tokens: int | None = None
    reserve_tokens: int = 2000

    # Behavior
    stop_on_tool_error: bool = False
    include_tool_errors_in_context: bool = True
    stream_tool_calls: bool = True
    batch_concurrency: int = 20

    # Middleware support
    middleware_chain: MiddlewareChain | None = None
    use_default_middleware: bool = False  # If True, uses MiddlewareChain.production_defaults()

    def __post_init__(self):
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        if self.tool_retry_attempts < 0:
            raise ValueError("tool_retry_attempts cannot be negative")

    def get_middleware_chain(self) -> MiddlewareChain | None:
        """Get the effective middleware chain."""
        if self.middleware_chain is not None:
            return self.middleware_chain
        if self.use_default_middleware:
            from ..tools.middleware import MiddlewareChain
            return MiddlewareChain.production_defaults()
        return None


__all__ = ["AgentConfig"]
