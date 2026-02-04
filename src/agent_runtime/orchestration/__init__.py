"""
Multi-agent orchestration for agent runtime.

This module provides:
- Operator: Abstract workflow that can spawn child jobs
- Router: Selects which operator handles a request
- GraphExecutor: DAG-based multi-agent flows
- Common patterns (map-reduce, planner-executor, debate)
"""

from .types import (
    Operator,
    OperatorResult,
    OperatorContext,
    AgentRole,
    RoutingDecision,
    OrchestrationConfig,
)
from .router import Router, RoutingRule, DefaultRouter
from .executor import (
    GraphExecutor,
    GraphNode,
    GraphEdge,
    ExecutionGraph,
    NodeResult,
)
from .patterns import (
    MapReduceOperator,
    PlannerExecutorOperator,
    DebateOperator,
    ChainOperator,
    ParallelOperator,
)

__all__ = [
    # Types
    "Operator",
    "OperatorResult",
    "OperatorContext",
    "AgentRole",
    "RoutingDecision",
    "OrchestrationConfig",
    # Router
    "Router",
    "RoutingRule",
    "DefaultRouter",
    # Executor
    "GraphExecutor",
    "GraphNode",
    "GraphEdge",
    "ExecutionGraph",
    "NodeResult",
    # Patterns
    "MapReduceOperator",
    "PlannerExecutorOperator",
    "DebateOperator",
    "ChainOperator",
    "ParallelOperator",
]
