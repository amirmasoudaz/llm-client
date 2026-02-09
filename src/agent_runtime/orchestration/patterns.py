"""
Common multi-agent orchestration patterns.

This module provides pre-built operators for common patterns:
- MapReduceOperator: Parallel map + reduce aggregation
- PlannerExecutorOperator: Plan then execute workflow
- DebateOperator: Multiple agents debate to consensus
- ChainOperator: Sequential chaining of operators
- ParallelOperator: Parallel execution with aggregation
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import (
    Operator,
    OperatorResult,
    OperatorContext,
    AgentRole,
)


class ChainOperator(Operator):
    """Chains multiple operators in sequence.
    
    Each operator's output becomes the next operator's input.
    The chain continues until all operators complete or an error occurs.
    
    Example:
        ```python
        chain = ChainOperator(
            operators=[preprocessor, analyzer, formatter],
            name="analysis_chain",
        )
        result = await chain.execute({"text": "..."}, context)
        ```
    """
    
    def __init__(
        self,
        operators: list[Operator],
        name: str = "chain",
        stop_on_error: bool = True,
    ):
        self._operators = operators
        self._name = name
        self._stop_on_error = stop_on_error
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.COORDINATOR
    
    @property
    def description(self) -> str:
        op_names = [op.name for op in self._operators]
        return f"Chain of operators: {' -> '.join(op_names)}"
    
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> OperatorResult:
        start_time = time.perf_counter()
        current_input = input_data
        child_results: list[OperatorResult] = []
        total_turns = 0
        
        for operator in self._operators:
            result = await operator.execute(current_input, context)
            child_results.append(result)
            total_turns += result.turn_count
            
            if not result.success:
                if self._stop_on_error:
                    return OperatorResult(
                        content=result.content,
                        output_data=result.output_data,
                        success=False,
                        error=f"Chain failed at {operator.name}: {result.error}",
                        child_results=child_results,
                        operator_name=self._name,
                        role=self.role,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        turn_count=total_turns,
                    )
            
            # Pass output as input to next operator
            current_input = result.output_data
        
        # Return final result
        final_result = child_results[-1] if child_results else None
        return OperatorResult(
            content=final_result.content if final_result else None,
            output_data=final_result.output_data if final_result else {},
            success=True,
            child_results=child_results,
            operator_name=self._name,
            role=self.role,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            turn_count=total_turns,
        )


class ParallelOperator(Operator):
    """Executes multiple operators in parallel.
    
    All operators receive the same input and run concurrently.
    Results are aggregated using a configurable strategy.
    
    Example:
        ```python
        parallel = ParallelOperator(
            operators=[analyst1, analyst2, analyst3],
            aggregator=lambda results: {"combined": [r.content for r in results]},
        )
        result = await parallel.execute({"data": "..."}, context)
        ```
    """
    
    def __init__(
        self,
        operators: list[Operator],
        name: str = "parallel",
        aggregator: Callable[[list[OperatorResult]], dict[str, Any]] | None = None,
        require_all_success: bool = False,
        timeout_seconds: float | None = None,
    ):
        self._operators = operators
        self._name = name
        self._aggregator = aggregator or self._default_aggregator
        self._require_all_success = require_all_success
        self._timeout = timeout_seconds
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.COORDINATOR
    
    @property
    def description(self) -> str:
        return f"Parallel execution of {len(self._operators)} operators"
    
    @staticmethod
    def _default_aggregator(results: list[OperatorResult]) -> dict[str, Any]:
        """Default aggregation: collect all outputs."""
        return {
            "results": [r.output_data for r in results if r.success],
            "contents": [r.content for r in results if r.content],
        }
    
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> OperatorResult:
        start_time = time.perf_counter()
        
        # Create tasks for all operators
        tasks = [
            op.execute(input_data, context)
            for op in self._operators
        ]
        
        # Execute in parallel
        if self._timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                return OperatorResult(
                    success=False,
                    error=f"Parallel execution timed out after {self._timeout}s",
                    operator_name=self._name,
                    role=self.role,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        child_results: list[OperatorResult] = []
        errors: list[str] = []
        total_turns = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{self._operators[i].name}: {str(result)}")
                child_results.append(OperatorResult(
                    success=False,
                    error=str(result),
                    operator_name=self._operators[i].name,
                ))
            else:
                child_results.append(result)
                total_turns += result.turn_count
                if not result.success:
                    errors.append(f"{self._operators[i].name}: {result.error}")
        
        # Check success criteria
        success = not self._require_all_success or len(errors) == 0
        successful_results = [r for r in child_results if r.success]
        
        # Aggregate results
        aggregated = self._aggregator(successful_results)
        
        # Combine contents
        contents = [r.content for r in successful_results if r.content]
        combined_content = "\n\n---\n\n".join(contents) if contents else None
        
        return OperatorResult(
            content=combined_content,
            output_data=aggregated,
            success=success,
            error="; ".join(errors) if errors else None,
            child_results=child_results,
            operator_name=self._name,
            role=self.role,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            turn_count=total_turns,
        )


class MapReduceOperator(Operator):
    """Map-reduce pattern for parallel processing.
    
    1. Map: Apply operator to each item in input
    2. Reduce: Aggregate all results
    
    Example:
        ```python
        map_reduce = MapReduceOperator(
            map_operator=summarizer,
            reduce_operator=synthesizer,
            input_key="documents",
        )
        result = await map_reduce.execute(
            {"documents": [doc1, doc2, doc3]},
            context,
        )
        ```
    """
    
    def __init__(
        self,
        map_operator: Operator,
        reduce_operator: Operator | None = None,
        name: str = "map_reduce",
        input_key: str = "items",
        max_parallel: int = 5,
    ):
        self._map_op = map_operator
        self._reduce_op = reduce_operator
        self._name = name
        self._input_key = input_key
        self._max_parallel = max_parallel
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.COORDINATOR
    
    @property
    def description(self) -> str:
        return f"Map-reduce using {self._map_op.name}"
    
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> OperatorResult:
        start_time = time.perf_counter()
        
        # Get items to map over
        items = input_data.get(self._input_key, [])
        if not items:
            return OperatorResult(
                success=False,
                error=f"No items found in '{self._input_key}'",
                operator_name=self._name,
                role=self.role,
            )
        
        # Map phase: process items in parallel
        semaphore = asyncio.Semaphore(self._max_parallel)
        
        async def map_item(item: Any) -> OperatorResult:
            async with semaphore:
                item_input = {"item": item, **{k: v for k, v in input_data.items() if k != self._input_key}}
                return await self._map_op.execute(item_input, context)
        
        map_tasks = [map_item(item) for item in items]
        map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
        
        # Process map results
        child_results: list[OperatorResult] = []
        successful_outputs: list[dict[str, Any]] = []
        total_turns = 0
        
        for result in map_results:
            if isinstance(result, Exception):
                child_results.append(OperatorResult(
                    success=False,
                    error=str(result),
                    operator_name=self._map_op.name,
                ))
            else:
                child_results.append(result)
                total_turns += result.turn_count
                if result.success:
                    successful_outputs.append(result.output_data)
        
        # Reduce phase (if reducer provided)
        if self._reduce_op:
            reduce_input = {
                "mapped_results": successful_outputs,
                "original_items": items,
                **{k: v for k, v in input_data.items() if k != self._input_key},
            }
            reduce_result = await self._reduce_op.execute(reduce_input, context)
            child_results.append(reduce_result)
            total_turns += reduce_result.turn_count
            
            return OperatorResult(
                content=reduce_result.content,
                output_data=reduce_result.output_data,
                success=reduce_result.success,
                error=reduce_result.error,
                child_results=child_results,
                operator_name=self._name,
                role=self.role,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                turn_count=total_turns,
            )
        
        # No reducer - return map results
        return OperatorResult(
            content="\n\n".join(
                r.content for r in child_results if r.success and r.content
            ),
            output_data={"mapped_results": successful_outputs},
            success=len(successful_outputs) > 0,
            child_results=child_results,
            operator_name=self._name,
            role=self.role,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            turn_count=total_turns,
        )


class PlannerExecutorOperator(Operator):
    """Planner-executor pattern for complex tasks.
    
    1. Planner creates a plan with subtasks
    2. Executor runs each subtask
    3. Results are synthesized
    
    Example:
        ```python
        planner_executor = PlannerExecutorOperator(
            planner=planner_agent,
            executor=executor_agent,
            synthesizer=synthesizer_agent,
        )
        result = await planner_executor.execute(
            {"task": "Research and summarize AI trends"},
            context,
        )
        ```
    """
    
    def __init__(
        self,
        planner: Operator,
        executor: Operator,
        synthesizer: Operator | None = None,
        name: str = "planner_executor",
        max_subtasks: int = 10,
        parallel_execution: bool = True,
    ):
        self._planner = planner
        self._executor = executor
        self._synthesizer = synthesizer
        self._name = name
        self._max_subtasks = max_subtasks
        self._parallel = parallel_execution
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.COORDINATOR
    
    @property
    def description(self) -> str:
        return f"Planner-executor workflow"
    
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> OperatorResult:
        start_time = time.perf_counter()
        child_results: list[OperatorResult] = []
        total_turns = 0
        
        # Planning phase
        plan_result = await self._planner.execute(input_data, context)
        child_results.append(plan_result)
        total_turns += plan_result.turn_count
        
        if not plan_result.success:
            return OperatorResult(
                success=False,
                error=f"Planning failed: {plan_result.error}",
                child_results=child_results,
                operator_name=self._name,
                role=self.role,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                turn_count=total_turns,
            )
        
        # Extract subtasks from plan
        subtasks = plan_result.output_data.get("subtasks", [])
        if not subtasks:
            # Try extracting from content
            subtasks = plan_result.output_data.get("tasks", [])
        
        if not subtasks:
            # No subtasks - just return plan result
            return OperatorResult(
                content=plan_result.content,
                output_data=plan_result.output_data,
                success=True,
                child_results=child_results,
                operator_name=self._name,
                role=self.role,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                turn_count=total_turns,
            )
        
        # Limit subtasks
        subtasks = subtasks[:self._max_subtasks]
        
        # Execution phase
        if self._parallel:
            exec_tasks = [
                self._executor.execute(
                    {"subtask": task, "plan": plan_result.output_data, **input_data},
                    context,
                )
                for task in subtasks
            ]
            exec_results = await asyncio.gather(*exec_tasks, return_exceptions=True)
        else:
            exec_results = []
            for task in subtasks:
                result = await self._executor.execute(
                    {"subtask": task, "plan": plan_result.output_data, **input_data},
                    context,
                )
                exec_results.append(result)
        
        # Process execution results
        successful_outputs: list[dict[str, Any]] = []
        for result in exec_results:
            if isinstance(result, Exception):
                child_results.append(OperatorResult(
                    success=False,
                    error=str(result),
                    operator_name=self._executor.name,
                ))
            else:
                child_results.append(result)
                total_turns += result.turn_count
                if result.success:
                    successful_outputs.append(result.output_data)
        
        # Synthesis phase
        if self._synthesizer:
            synth_input = {
                "plan": plan_result.output_data,
                "execution_results": successful_outputs,
                "subtasks": subtasks,
                **input_data,
            }
            synth_result = await self._synthesizer.execute(synth_input, context)
            child_results.append(synth_result)
            total_turns += synth_result.turn_count
            
            return OperatorResult(
                content=synth_result.content,
                output_data=synth_result.output_data,
                success=synth_result.success,
                error=synth_result.error,
                child_results=child_results,
                operator_name=self._name,
                role=self.role,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                turn_count=total_turns,
            )
        
        # No synthesizer - combine results
        combined_content = "\n\n".join(
            r.content for r in child_results if r.success and r.content
        )
        
        return OperatorResult(
            content=combined_content,
            output_data={
                "plan": plan_result.output_data,
                "execution_results": successful_outputs,
            },
            success=len(successful_outputs) > 0,
            child_results=child_results,
            operator_name=self._name,
            role=self.role,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            turn_count=total_turns,
        )


class DebateOperator(Operator):
    """Debate pattern for consensus-building.
    
    Multiple agents debate a topic, with optional moderator
    to determine consensus.
    
    Example:
        ```python
        debate = DebateOperator(
            debaters=[agent1, agent2, agent3],
            moderator=moderator_agent,
            max_rounds=3,
        )
        result = await debate.execute(
            {"topic": "Best approach for X?"},
            context,
        )
        ```
    """
    
    def __init__(
        self,
        debaters: list[Operator],
        moderator: Operator | None = None,
        name: str = "debate",
        max_rounds: int = 3,
        stop_on_consensus: bool = True,
    ):
        self._debaters = debaters
        self._moderator = moderator
        self._name = name
        self._max_rounds = max_rounds
        self._stop_on_consensus = stop_on_consensus
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.COORDINATOR
    
    @property
    def description(self) -> str:
        return f"Debate between {len(self._debaters)} participants"
    
    async def execute(
        self,
        input_data: dict[str, Any],
        context: OperatorContext,
    ) -> OperatorResult:
        start_time = time.perf_counter()
        child_results: list[OperatorResult] = []
        total_turns = 0
        
        debate_history: list[dict[str, Any]] = []
        topic = input_data.get("topic", input_data.get("prompt", ""))
        
        for round_num in range(self._max_rounds):
            round_positions: list[dict[str, Any]] = []
            
            # Each debater contributes
            for debater in self._debaters:
                debate_input = {
                    "topic": topic,
                    "round": round_num + 1,
                    "history": debate_history,
                    **input_data,
                }
                
                result = await debater.execute(debate_input, context)
                child_results.append(result)
                total_turns += result.turn_count
                
                if result.success:
                    round_positions.append({
                        "debater": debater.name,
                        "position": result.content,
                        "data": result.output_data,
                    })
            
            debate_history.append({
                "round": round_num + 1,
                "positions": round_positions,
            })
            
            # Check for consensus (if moderator provided)
            if self._moderator and self._stop_on_consensus:
                consensus_input = {
                    "topic": topic,
                    "history": debate_history,
                    "current_round": round_num + 1,
                    "check_consensus": True,
                }
                
                mod_result = await self._moderator.execute(consensus_input, context)
                child_results.append(mod_result)
                total_turns += mod_result.turn_count
                
                if mod_result.output_data.get("consensus_reached", False):
                    return OperatorResult(
                        content=mod_result.content,
                        output_data={
                            "debate_history": debate_history,
                            "consensus": mod_result.output_data,
                            "rounds_completed": round_num + 1,
                        },
                        success=True,
                        child_results=child_results,
                        operator_name=self._name,
                        role=self.role,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        turn_count=total_turns,
                    )
        
        # Final moderation / summary
        if self._moderator:
            final_input = {
                "topic": topic,
                "history": debate_history,
                "summarize": True,
            }
            final_result = await self._moderator.execute(final_input, context)
            child_results.append(final_result)
            total_turns += final_result.turn_count
            
            return OperatorResult(
                content=final_result.content,
                output_data={
                    "debate_history": debate_history,
                    "summary": final_result.output_data,
                    "rounds_completed": self._max_rounds,
                },
                success=final_result.success,
                child_results=child_results,
                operator_name=self._name,
                role=self.role,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                turn_count=total_turns,
            )
        
        # No moderator - return history
        final_positions = debate_history[-1]["positions"] if debate_history else []
        combined = "\n\n---\n\n".join(
            f"**{p['debater']}**: {p['position']}"
            for p in final_positions
        )
        
        return OperatorResult(
            content=combined,
            output_data={
                "debate_history": debate_history,
                "rounds_completed": self._max_rounds,
            },
            success=True,
            child_results=child_results,
            operator_name=self._name,
            role=self.role,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            turn_count=total_turns,
        )


__all__ = [
    "ChainOperator",
    "ParallelOperator",
    "MapReduceOperator",
    "PlannerExecutorOperator",
    "DebateOperator",
]
