"""
Agent orchestration with ReAct loop and tool calling.

This package provides the Agent class that composes providers, tools,
and conversations into an autonomous agent capable of multi-turn
reasoning and action.
"""

# Re-export AgentConfig from config for backward compatibility
from ..config import AgentConfig
from .core import Agent
from .definition import (
    AgentDefinition,
    AgentExecutionPolicy,
    AgentMemoryPolicy,
    AgentOutputPolicy,
    AgentRuntimeState,
    PromptTemplateReference,
    ToolExecutionMode,
)
from .result import AgentResult, TurnResult
from .session import quick_agent

__all__ = [
    "Agent",
    "AgentConfig",
    "ToolExecutionMode",
    "PromptTemplateReference",
    "AgentExecutionPolicy",
    "AgentOutputPolicy",
    "AgentMemoryPolicy",
    "AgentDefinition",
    "AgentRuntimeState",
    "AgentResult",
    "TurnResult",
    "quick_agent",
]
