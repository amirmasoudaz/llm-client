"""
Types for multi-agent orchestration.

This module defines the core abstractions for multi-agent systems:
- Operator: Abstract workflow unit that processes requests
- AgentRole: Defines agent capabilities and constraints
- RoutingDecision: Result of routing to an operator
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..jobs.types import JobRecord


class AgentRole(str, Enum):
    """Standard agent roles in multi-agent systems."""
    
    PLANNER = "planner"          # Plans tasks and creates subtasks
    EXECUTOR = "executor"        # Executes specific tasks
    REVIEWER = "reviewer"        # Reviews and validates results
    COORDINATOR = "coordinator"  # Coordinates multiple agents
    SPECIALIST = "specialist"    # Domain-specific expert
    CRITIC = "critic"            # Provides criticism/feedback
    SYNTHESIZER = "synthesizer"  # Combines outputs from multiple agents
    RESEARCHER = "researcher"    # Gathers and analyzes information
    VERIFIER = "verifier"        # Verifies facts and outputs
    CUSTOM = "custom"            # Custom role


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration behavior.
    
    Attributes:
        max_child_jobs: Maximum number of child jobs per parent
        max_depth: Maximum nesting depth of child jobs
        timeout_seconds: Default timeout for operators
        inherit_budgets: Whether child jobs inherit parent budgets
        propagate_cancellation: Whether cancellation propagates to children
        parallel_execution: Whether to run child jobs in parallel
        max_parallel: Maximum concurrent child jobs
    """
    max_child_jobs: int = 10
    max_depth: int = 5
    timeout_seconds: float = 300.0
    inherit_budgets: bool = True
    propagate_cancellation: bool = True
    parallel_execution: bool = True
    max_parallel: int = 5


@dataclass
class OperatorContext:
    """Context for operator execution.
    
    Contains:
    - Execution context from parent
    - Configuration for this operator
    - Depth tracking for nested operations
    - Reference to parent job (if any)
    """
    execution_ctx: ExecutionContext
    config: OrchestrationConfig
    depth: int = 0
    parent_job: JobRecord | None = None
    
    def child(self, new_job: JobRecord | None = None) -> OperatorContext:
        """Create a child context."""
        return OperatorContext(
            execution_ctx=self.execution_ctx.child(),
            config=self.config,
            depth=self.depth + 1,
            parent_job=new_job or self.parent_job,
        )
    
    @property
    def can_spawn_children(self) -> bool:
        """Check if we can spawn more child jobs."""
        return self.depth < self.config.max_depth


@dataclass
class OperatorResult:
    """Result from an operator execution.
    
    Operators may return:
    - Direct output content
    - Child job results
    - Artifacts generated
    - Metadata about execution
    """
    # Main output
    content: str | None = None
    output_data: dict[str, Any] = field(default_factory=dict)
    
    # Status
    success: bool = True
    error: str | None = None
    
    # Child job results
    child_results: list[OperatorResult] = field(default_factory=list)
    
    # Metadata
    operator_name: str = ""
    role: AgentRole | None = None
    execution_time_ms: float = 0.0
    turn_count: int = 0
    
    # Artifacts
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    
    # For chaining
    should_continue: bool = False
    next_input: dict[str, Any] | None = None


class Operator(ABC):
    """Abstract base class for operators.
    
    An operator is a reusable workflow unit that can:
    - Process requests using one or more agents
    - Spawn child jobs for subtasks
    - Aggregate results from multiple executions
    
    Example:
        ```python
        class SummarizerOperator(Operator):
            async def execute(
                self,
                input_data: dict[str, Any],
                context: OperatorContext,
            ) -> OperatorResult:
                # Use the agent to summarize
                result = await self.agent.run(
                    f"Summarize: {input_data['text']}"
                )
                return OperatorResult(
                    content=result.content,
                    success=True,
                    operator_name=self.name,
                )
        ```
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this operator."""
        ...
    
    @property
    def role(self) -> AgentRole:
        """Default role for this operator."""
        return AgentRole.EXECUTOR
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return f"{self.name} operator"
    
    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> OperatorResult:
        """Execute the operator with given input.
        
        Args:
            input_data: Input payload for the operator
            context: Execution context with config and parent info
        
        Returns:
            OperatorResult with output and status
        """
        ...
    
    async def validate_input(
        self,
        input_data: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Validate input before execution.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None
    
    async def on_child_complete(
        self,
        child_result: OperatorResult,
        context: OperatorContext,
    ) -> None:
        """Called when a child operator completes.
        
        Override to implement custom aggregation logic.
        """
        pass


@dataclass
class RoutingDecision:
    """Result of routing a request to operators.
    
    Contains:
    - Selected operator(s)
    - Confidence scores
    - Reasoning for selection
    """
    operator_name: str
    confidence: float = 1.0  # 0.0 to 1.0
    reasoning: str | None = None
    fallback_operators: list[str] = field(default_factory=list)
    
    # For multi-operator routing
    secondary_operators: list[str] = field(default_factory=list)
    parallel: bool = False  # Whether to run in parallel
    
    # Transformation instructions
    input_transform: dict[str, Any] | None = None
    output_transform: dict[str, Any] | None = None


__all__ = [
    "AgentRole",
    "OrchestrationConfig",
    "OperatorContext",
    "OperatorResult",
    "Operator",
    "RoutingDecision",
]
