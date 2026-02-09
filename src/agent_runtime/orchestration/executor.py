"""
Graph executor for DAG-based multi-agent workflows.

This module provides:
- GraphNode: Node in the execution graph
- GraphEdge: Edge connecting nodes
- ExecutionGraph: The DAG structure
- GraphExecutor: Executes graphs with topological ordering
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .types import Operator, OperatorResult, OperatorContext


class NodeStatus(str, Enum):
    """Status of a graph node."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class GraphNode:
    """A node in the execution graph.
    
    Each node wraps an operator and defines how it connects
    to other nodes in the graph.
    
    Attributes:
        node_id: Unique identifier
        operator: The operator to execute
        input_transform: Transform input before passing to operator
        output_transform: Transform output after operator completes
        condition: Condition function; node skipped if returns False
        retry_count: Number of retries on failure
        timeout_seconds: Timeout for this node
    """
    node_id: str
    operator: Operator
    
    # Transformations
    input_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    output_transform: Callable[[OperatorResult], dict[str, Any]] | None = None
    
    # Control flow
    condition: Callable[[dict[str, Any]], bool] | None = None
    retry_count: int = 0
    timeout_seconds: float | None = None
    
    # Metadata
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = self.operator.name


@dataclass
class GraphEdge:
    """An edge connecting two nodes.
    
    Edges define data flow and dependencies between nodes.
    
    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        data_mapping: How to map source output to target input
        condition: Condition for this edge; if False, skip
    """
    source_id: str
    target_id: str
    data_mapping: dict[str, str] | None = None  # target_key -> source_key
    condition: Callable[[OperatorResult], bool] | None = None


@dataclass
class NodeResult:
    """Result of executing a single node."""
    node_id: str
    status: NodeStatus
    result: OperatorResult | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    attempts: int = 1


@dataclass
class ExecutionGraph:
    """A directed acyclic graph (DAG) of operators.
    
    The graph defines the workflow structure:
    - Nodes are operators to execute
    - Edges define data flow and dependencies
    
    Example:
        ```python
        graph = ExecutionGraph()
        
        # Add nodes
        graph.add_node(GraphNode("planner", planner_op))
        graph.add_node(GraphNode("executor1", executor_op))
        graph.add_node(GraphNode("executor2", executor_op))
        graph.add_node(GraphNode("synthesizer", synth_op))
        
        # Define edges (data flow)
        graph.add_edge(GraphEdge("planner", "executor1"))
        graph.add_edge(GraphEdge("planner", "executor2"))
        graph.add_edge(GraphEdge("executor1", "synthesizer"))
        graph.add_edge(GraphEdge("executor2", "synthesizer"))
        
        # Execute
        executor = GraphExecutor(graph)
        results = await executor.execute(input_data, context)
        ```
    """
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    
    # Entry and exit points
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)
    
    # Metadata
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found")
        self.edges.append(edge)
    
    def get_dependencies(self, node_id: str) -> list[str]:
        """Get all nodes that must complete before this node."""
        return [e.source_id for e in self.edges if e.target_id == node_id]
    
    def get_dependents(self, node_id: str) -> list[str]:
        """Get all nodes that depend on this node."""
        return [e.target_id for e in self.edges if e.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target_id == node_id]
    
    def topological_sort(self) -> list[list[str]]:
        """Get nodes in topological order, grouped by level.
        
        Returns:
            List of lists, where each inner list contains nodes
            that can be executed in parallel.
        """
        # Calculate in-degree for each node
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] += 1
        
        # Find nodes with no dependencies (entry points)
        levels: list[list[str]] = []
        current_level = [
            node_id for node_id, degree in in_degree.items()
            if degree == 0
        ]
        
        while current_level:
            levels.append(current_level)
            next_level = []
            
            for node_id in current_level:
                for dep_id in self.get_dependents(node_id):
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        next_level.append(dep_id)
            
            current_level = next_level
        
        # Check for cycles
        total_nodes = sum(len(level) for level in levels)
        if total_nodes != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        
        return levels
    
    def validate(self) -> list[str]:
        """Validate the graph structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for orphan nodes
        referenced = set()
        for edge in self.edges:
            referenced.add(edge.source_id)
            referenced.add(edge.target_id)
        
        # Entry nodes must have no incoming edges
        for node_id in self.entry_nodes:
            if self.get_dependencies(node_id):
                errors.append(f"Entry node {node_id} has incoming edges")
        
        # Exit nodes must have no outgoing edges
        for node_id in self.exit_nodes:
            if self.get_dependents(node_id):
                errors.append(f"Exit node {node_id} has outgoing edges")
        
        # Check for cycles (done in topological_sort)
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))
        
        return errors


class GraphExecutor:
    """Executes an execution graph.
    
    Features:
    - Parallel execution of independent nodes
    - Data flow between nodes via edges
    - Conditional execution
    - Retry handling
    - Timeout management
    """
    
    def __init__(
        self,
        graph: ExecutionGraph,
        max_parallel: int = 5,
    ):
        self._graph = graph
        self._max_parallel = max_parallel
        self._results: dict[str, NodeResult] = {}
        self._semaphore: asyncio.Semaphore | None = None
    
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> dict[str, NodeResult]:
        """Execute the entire graph.
        
        Args:
            input_data: Initial input for entry nodes
            context: Execution context
        
        Returns:
            Dict mapping node_id to NodeResult
        """
        self._results.clear()
        self._semaphore = asyncio.Semaphore(self._max_parallel)
        
        # Get execution order
        levels = self._graph.topological_sort()
        
        # Track data flowing through the graph
        node_outputs: dict[str, dict[str, Any]] = {}
        
        # Execute level by level
        for level in levels:
            # Execute all nodes in this level in parallel
            tasks = []
            for node_id in level:
                # Build input for this node
                node_input = self._build_node_input(
                    node_id, input_data, node_outputs
                )
                tasks.append(
                    self._execute_node(node_id, node_input, context)
                )
            
            # Wait for all nodes in this level
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for node_id, result in zip(level, results, strict=False):
                if isinstance(result, Exception):
                    self._results[node_id] = NodeResult(
                        node_id=node_id,
                        status=NodeStatus.FAILED,
                        error=str(result),
                    )
                else:
                    self._results[node_id] = result
                    if result.status == NodeStatus.COMPLETED and result.result:
                        # Store output for downstream nodes
                        node = self._graph.nodes[node_id]
                        if node.output_transform:
                            node_outputs[node_id] = node.output_transform(result.result)
                        else:
                            node_outputs[node_id] = result.result.output_data
        
        return self._results
    
    def _build_node_input(
        self,
        node_id: str,
        initial_input: dict[str, Any],
        node_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build input for a node from initial input and upstream outputs."""
        # Start with initial input for entry nodes
        deps = self._graph.get_dependencies(node_id)
        if not deps:
            node_input = dict(initial_input)
        else:
            node_input = {}
        
        # Merge in outputs from upstream nodes
        for edge in self._graph.get_edges_to(node_id):
            source_output = node_outputs.get(edge.source_id, {})
            
            # Check edge condition
            if edge.condition:
                source_result = self._results.get(edge.source_id)
                if source_result and source_result.result:
                    if not edge.condition(source_result.result):
                        continue
            
            # Apply data mapping
            if edge.data_mapping:
                for target_key, source_key in edge.data_mapping.items():
                    if source_key in source_output:
                        node_input[target_key] = source_output[source_key]
            else:
                # Default: merge all output
                node_input.update(source_output)
        
        # Apply node's input transform
        node = self._graph.nodes[node_id]
        if node.input_transform:
            node_input = node.input_transform(node_input)
        
        return node_input
    
    async def _execute_node(
        self,
        node_id: str,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> NodeResult:
        """Execute a single node with retry handling."""
        node = self._graph.nodes[node_id]
        
        # Check condition
        if node.condition and not node.condition(input_data):
            return NodeResult(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
            )
        
        # Execute with retries
        attempts = 0
        last_error: str | None = None
        
        while attempts <= node.retry_count:
            attempts += 1
            start_time = time.perf_counter()
            
            try:
                async with self._semaphore:
                    # Apply timeout if configured
                    if node.timeout_seconds:
                        coro = node.operator.execute(input_data, context)
                        result = await asyncio.wait_for(
                            coro,
                            timeout=node.timeout_seconds,
                        )
                    else:
                        result = await node.operator.execute(input_data, context)
                
                execution_time = (time.perf_counter() - start_time) * 1000
                
                if result.success:
                    return NodeResult(
                        node_id=node_id,
                        status=NodeStatus.COMPLETED,
                        result=result,
                        execution_time_ms=execution_time,
                        attempts=attempts,
                    )
                else:
                    last_error = result.error
            
            except asyncio.TimeoutError:
                last_error = f"Timeout after {node.timeout_seconds}s"
            except asyncio.CancelledError:
                return NodeResult(
                    node_id=node_id,
                    status=NodeStatus.CANCELLED,
                    error="Execution cancelled",
                    attempts=attempts,
                )
            except Exception as e:
                last_error = str(e)
        
        return NodeResult(
            node_id=node_id,
            status=NodeStatus.FAILED,
            error=last_error,
            attempts=attempts,
        )
    
    def get_output(self, node_id: str | None = None) -> dict[str, Any] | None:
        """Get output from a specific node or exit nodes.
        
        If node_id is None, returns combined output from all exit nodes.
        """
        if node_id:
            result = self._results.get(node_id)
            if result and result.result:
                return result.result.output_data
            return None
        
        # Combine exit node outputs
        combined = {}
        for exit_id in self._graph.exit_nodes:
            result = self._results.get(exit_id)
            if result and result.result:
                combined[exit_id] = result.result.output_data
        
        return combined if combined else None
    
    def get_final_content(self) -> str | None:
        """Get final content from exit nodes."""
        contents = []
        for exit_id in self._graph.exit_nodes:
            result = self._results.get(exit_id)
            if result and result.result and result.result.content:
                contents.append(result.result.content)
        
        return "\n\n".join(contents) if contents else None


__all__ = [
    "NodeStatus",
    "GraphNode",
    "GraphEdge",
    "NodeResult",
    "ExecutionGraph",
    "GraphExecutor",
]
