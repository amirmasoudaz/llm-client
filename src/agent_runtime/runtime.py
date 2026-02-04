"""
Runtime kernel - the main orchestrator for agent executions.

This module provides the RuntimeKernel that ties together all
runtime components and provides the main execution API.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .context import ExecutionContext, BudgetSpec, PolicyRef
from .events import (
    EventBus,
    InMemoryEventBus,
    RuntimeEvent,
    RuntimeEventType,
    EventSubscription,
    FinalEvent,
    ProgressEvent,
)
from .jobs import (
    JobManager,
    JobRecord,
    JobStatus,
    JobStore,
    InMemoryJobStore,
    JobSpec,
)
from .actions import (
    ActionManager,
    ActionRecord,
    ActionStore,
    InMemoryActionStore,
    ActionSpec,
)
from .policy import (
    PolicyEngine,
    PolicyContext,
    PolicyDenied,
)
from .ledger import (
    Ledger,
    BudgetExceededError,
)
from .plugins import (
    PluginRegistry,
)

if TYPE_CHECKING:
    from llm_client import ExecutionEngine, Agent
    from llm_client.providers.types import CompletionResult


@dataclass
class ExecutionRequest:
    """Request specification for starting an execution.
    
    This is the main input to RuntimeKernel.execute().
    """
    # Query/prompt
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None  # Alternative to prompt
    
    # Identity
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    
    # Configuration
    operator_id: str | None = None  # Which operator/workflow to use
    tool_names: list[str] | None = None  # Specific tools to use
    
    # Policy and budgets
    policy_ref: PolicyRef | None = None
    budgets: BudgetSpec | None = None
    
    # Execution options
    max_turns: int = 10
    deadline_seconds: float | None = None
    idempotency_key: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Final result of an execution."""
    job_id: str
    content: str | None = None
    status: str = "success"  # success, error, cancelled, timeout
    error: str | None = None
    turns: int = 0
    usage: dict[str, Any] | None = None


class ExecutionHandle:
    """Handle for an ongoing or completed execution.
    
    Provides access to:
    - Event stream
    - Final result
    - Cancellation
    - Action management
    """
    
    def __init__(
        self,
        job_id: str,
        kernel: RuntimeKernel,
        subscription: EventSubscription,
    ):
        self._job_id = job_id
        self._kernel = kernel
        self._subscription = subscription
        self._result: ExecutionResult | None = None
    
    @property
    def job_id(self) -> str:
        return self._job_id
    
    async def events(self) -> AsyncIterator[RuntimeEvent]:
        """Stream events for this execution."""
        async for event in self._kernel._event_bus.events(self._subscription):
            yield event
            if event.type in {
                RuntimeEventType.FINAL_RESULT,
                RuntimeEventType.FINAL_ERROR,
                RuntimeEventType.JOB_CANCELLED,
            }:
                break
    
    async def result(self, timeout: float | None = None) -> ExecutionResult:
        """Wait for and return the final result.
        
        Args:
            timeout: Maximum time to wait (None = wait indefinitely)
            
        Returns:
            ExecutionResult with final content and status
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        if self._result is not None:
            return self._result
        
        # Wait for completion event
        start = time.time()
        async for event in self.events():
            if event.type == RuntimeEventType.FINAL_RESULT:
                self._result = ExecutionResult(
                    job_id=self._job_id,
                    content=event.data.get("content"),
                    status=event.data.get("status", "success"),
                    turns=event.data.get("turns", 0),
                    usage=event.data.get("usage"),
                )
                return self._result
            
            if event.type == RuntimeEventType.FINAL_ERROR:
                self._result = ExecutionResult(
                    job_id=self._job_id,
                    status="error",
                    error=event.data.get("error"),
                )
                return self._result
            
            if event.type == RuntimeEventType.JOB_CANCELLED:
                self._result = ExecutionResult(
                    job_id=self._job_id,
                    status="cancelled",
                    error=event.data.get("error"),
                )
                return self._result
            
            if timeout and (time.time() - start) > timeout:
                raise asyncio.TimeoutError("Execution timeout")
        
        # Should not reach here
        job = await self._kernel._job_manager.get(self._job_id)
        return ExecutionResult(
            job_id=self._job_id,
            status=job.status.value if job else "unknown",
            error=job.error if job else None,
        )
    
    async def cancel(self, reason: str | None = None) -> None:
        """Cancel this execution."""
        await self._kernel.cancel(self._job_id, reason)
    
    async def get_pending_actions(self) -> list[ActionRecord]:
        """Get pending actions for this job."""
        return await self._kernel._action_manager.list_pending_for_job(self._job_id)
    
    async def resolve_action(
        self,
        action_id: str,
        resolution: dict[str, Any],
    ) -> ActionRecord:
        """Resolve an action and resume execution."""
        return await self._kernel._action_manager.resolve(action_id, resolution)


class RuntimeKernel:
    """The main runtime orchestrator.
    
    The RuntimeKernel:
    - Assembles all runtime components
    - Provides the execute() entry point
    - Coordinates job lifecycle, policies, and events
    - Manages the execution loop with action support
    
    Example:
        ```python
        from agent_runtime import RuntimeKernel, ExecutionRequest
        from llm_client import ExecutionEngine, OpenAIProvider, Agent
        
        # Create kernel with default components
        kernel = RuntimeKernel.create(
            engine=ExecutionEngine(provider=OpenAIProvider(model="gpt-5")),
        )
        
        # Or with custom components
        kernel = RuntimeKernel(
            job_manager=JobManager(store=PostgresJobStore(...)),
            action_manager=ActionManager(...),
            policy_engine=PolicyEngine.default(),
            ledger=Ledger(...),
            event_bus=InMemoryEventBus(),
            plugins=PluginRegistry(),
        )
        
        # Execute
        handle = await kernel.execute(ExecutionRequest(
            prompt="Hello!",
            scope_id="tenant-123",
        ))
        
        # Stream events
        async for event in handle.events():
            print(event.type, event.data)
        
        # Or just get result
        result = await handle.result()
        ```
    """
    
    def __init__(
        self,
        job_manager: JobManager,
        action_manager: ActionManager,
        policy_engine: PolicyEngine,
        ledger: Ledger,
        event_bus: EventBus,
        plugins: PluginRegistry,
        *,
        engine: Any = None,  # ExecutionEngine
        agent_factory: Any = None,  # Callable to create Agent
    ):
        self._job_manager = job_manager
        self._action_manager = action_manager
        self._policy_engine = policy_engine
        self._ledger = ledger
        self._event_bus = event_bus
        self._plugins = plugins
        self._engine = engine
        self._agent_factory = agent_factory
    
    @classmethod
    def create(
        cls,
        *,
        engine: Any = None,
        agent_factory: Any = None,
        job_store: JobStore | None = None,
        action_store: ActionStore | None = None,
        event_bus: EventBus | None = None,
        policy_engine: PolicyEngine | None = None,
        ledger: Ledger | None = None,
        plugins: PluginRegistry | None = None,
    ) -> RuntimeKernel:
        """Create a kernel with default or provided components."""
        event_bus = event_bus or InMemoryEventBus()
        job_store = job_store or InMemoryJobStore()
        action_store = action_store or InMemoryActionStore()
        plugins = plugins or PluginRegistry()
        
        job_manager = JobManager(store=job_store, event_bus=event_bus)
        action_manager = ActionManager(
            store=action_store,
            job_manager=job_manager,
            event_bus=event_bus,
        )
        
        return cls(
            job_manager=job_manager,
            action_manager=action_manager,
            policy_engine=policy_engine or PolicyEngine.default(),
            ledger=ledger or Ledger(),
            event_bus=event_bus,
            plugins=plugins,
            engine=engine,
            agent_factory=agent_factory,
        )
    
    async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
        """Start an execution and return a handle.
        
        This is the main entry point for running agent executions.
        The execution runs asynchronously and can be monitored via
        the returned handle.
        
        Args:
            request: Execution request specification
            
        Returns:
            ExecutionHandle for monitoring and control
            
        Raises:
            PolicyDenied: If policy check fails
            BudgetExceededError: If budget is exceeded
        """
        # Create job
        job = await self._job_manager.start(JobSpec(
            scope_id=request.scope_id,
            principal_id=request.principal_id,
            session_id=request.session_id,
            idempotency_key=request.idempotency_key,
            budgets=request.budgets,
            policy_ref=request.policy_ref,
            deadline_seconds=request.deadline_seconds,
            metadata=dict(request.metadata),
            tags=dict(request.tags),
        ))
        
        # Create execution context
        ctx = self._job_manager.create_context(job)
        
        # Check policy
        policy_result = self._policy_engine.evaluate(
            PolicyContext.from_execution_context(ctx)
        )
        if not policy_result.allowed:
            await self._job_manager.transition(
                job.job_id,
                JobStatus.FAILED,
                error=policy_result.reason,
                error_code="POLICY_DENIED",
            )
            raise PolicyDenied(
                policy_result.reason or "Policy denied",
                policy_result.policy_name,
            )
        
        # Check budget
        try:
            await self._ledger.require_budget(ctx)
        except BudgetExceededError as e:
            await self._job_manager.transition(
                job.job_id,
                JobStatus.FAILED,
                error=str(e),
                error_code="BUDGET_EXCEEDED",
            )
            raise
        
        # Subscribe to events for this job
        subscription = self._event_bus.subscribe(job_id=job.job_id)
        
        # Start the execution in background
        asyncio.create_task(self._run_execution(job, ctx, request))
        
        return ExecutionHandle(job.job_id, self, subscription)
    
    async def _run_execution(
        self,
        job: JobRecord,
        ctx: ExecutionContext,
        request: ExecutionRequest,
    ) -> None:
        """Run the actual execution (called in background task)."""
        try:
            # Transition to running
            job = await self._job_manager.transition(job.job_id, JobStatus.RUNNING)
            
            # Create agent
            agent = await self._create_agent(ctx, request)
            if agent is None:
                raise RuntimeError("No agent factory configured")
            
            # Run agent
            prompt = request.prompt or (
                request.messages[-1]["content"]
                if request.messages
                else ""
            )
            
            # Use llm_client's request context
            llm_ctx = ctx.to_request_context()
            
            # Run the agent
            result = await agent.run(
                prompt,
                max_turns=request.max_turns,
                context=llm_ctx,
            )
            
            # Record usage
            if result.total_usage:
                await self._ledger.record_usage(
                    ctx,
                    result.total_usage,
                    provider=agent.provider.__class__.__name__,
                    model=agent.provider.model_name,
                )
            
            # Complete the job
            await self._job_manager.transition(
                job.job_id,
                JobStatus.SUCCEEDED,
                progress=1.0,
            )
            
            # Emit final event
            final = FinalEvent(
                content=result.content,
                status=result.status,
                error=result.error,
                usage=result.total_usage.to_dict() if hasattr(result.total_usage, 'to_dict') else None,
                turns=len(result.turns),
            )
            await self._event_bus.publish(final.to_runtime_event(ctx))
            
        except PolicyDenied as e:
            await self._handle_error(job, ctx, str(e), "POLICY_DENIED")
        except BudgetExceededError as e:
            await self._handle_error(job, ctx, str(e), "BUDGET_EXCEEDED")
        except asyncio.CancelledError:
            await self._job_manager.transition(
                job.job_id,
                JobStatus.CANCELLED,
                error="Execution cancelled",
            )
        except Exception as e:
            await self._handle_error(job, ctx, str(e), "EXECUTION_ERROR")
    
    async def _handle_error(
        self,
        job: JobRecord,
        ctx: ExecutionContext,
        error: str,
        error_code: str,
    ) -> None:
        """Handle execution error."""
        await self._job_manager.transition(
            job.job_id,
            JobStatus.FAILED,
            error=error,
            error_code=error_code,
        )
        
        final = FinalEvent(
            status="error",
            error=error,
        )
        await self._event_bus.publish(
            final.to_runtime_event(ctx, RuntimeEventType.FINAL_ERROR)
        )
    
    async def _create_agent(
        self,
        ctx: ExecutionContext,
        request: ExecutionRequest,
    ) -> Any:
        """Create an agent for the execution."""
        if self._agent_factory:
            return await self._agent_factory(ctx, request)
        
        # Default: use engine directly if available
        if self._engine:
            from llm_client import Agent
            
            # Get tools from plugins
            tools = self._plugins.get_tools()
            
            # Filter by request if specified
            if request.tool_names:
                tools = [t for t in tools if t.name in request.tool_names]
            
            return Agent(
                provider=self._engine._provider,
                tools=tools,
                use_middleware=True,
            )
        
        return None
    
    async def cancel(self, job_id: str, reason: str | None = None) -> JobRecord:
        """Cancel an execution."""
        return await self._job_manager.cancel(job_id, reason)
    
    async def get_job(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        return await self._job_manager.get(job_id)
    
    async def require_action(
        self,
        job_id: str,
        action_type: str,
        payload: dict[str, Any] | None = None,
        expires_in_seconds: float = 300.0,
    ) -> ActionRecord:
        """Request an action for a job.
        
        This pauses the job and waits for external resolution.
        """
        return await self._action_manager.require_action(ActionSpec(
            job_id=job_id,
            type=action_type,
            payload=payload,
            expires_in_seconds=expires_in_seconds,
        ))
    
    async def resolve_action(
        self,
        action_id: str,
        resolution: dict[str, Any],
        resume_token: str | None = None,
    ) -> ActionRecord:
        """Resolve an action."""
        return await self._action_manager.resolve(
            action_id,
            resolution,
            resume_token=resume_token,
        )
    
    async def close(self) -> None:
        """Close the kernel and clean up resources."""
        await self._event_bus.close()
        await self._plugins.unload_all()


__all__ = [
    "RuntimeKernel",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionHandle",
]
