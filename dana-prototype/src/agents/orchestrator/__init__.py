# src/agents/orchestrator/__init__.py
"""Dana Orchestrator - Hybrid AI agent for Dana Copilot.

Implements a token-efficient hybrid approach:
- DIRECT mode: Single tool, no reasoning (most efficient)
- GUIDED mode: Predefined sequences with minimal synthesis  
- AGENTIC mode: Full ReAct with CoT for complex tasks
"""

from src.agents.orchestrator.engine import DanaOrchestrator
from src.agents.orchestrator.context import ContextBuilder
from src.agents.orchestrator.tools import ToolRegistry, tool
from src.agents.orchestrator.router import IntentRouter, ProcessingMode, route_request
from src.agents.orchestrator.helpers import FollowUpAgent, TitleAgent, SummarizationAgent

__all__ = [
    "DanaOrchestrator",
    "ContextBuilder",
    "ToolRegistry",
    "tool",
    "IntentRouter",
    "ProcessingMode",
    "route_request",
    "FollowUpAgent",
    "TitleAgent",
    "SummarizationAgent",
]

