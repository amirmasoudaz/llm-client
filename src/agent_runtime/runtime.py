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

try:
    from llm_client.cancellation import CancelledError as LlmCancelledError
except Exception:  # pragma: no cover
    LlmCancelledError = None  # type: ignore


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
    run_id: str | None = None  # workflow_id/query_id (canonical execution id)
    
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
        self._cancel_by_job_id: dict[str, Any] = {}
        self._task_by_job_id: dict[str, asyncio.Task[None]] = {}
    
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
        # Apply request-scoped runtime budget to the job deadline (v0.1):
        # - budgets.max_runtime_seconds constrains deadline_seconds
        effective_deadline_seconds = request.deadline_seconds
        if request.budgets and request.budgets.max_runtime_seconds is not None:
            max_runtime = float(request.budgets.max_runtime_seconds)
            if effective_deadline_seconds is None or effective_deadline_seconds > max_runtime:
                effective_deadline_seconds = max_runtime

        # Create job
        job = await self._job_manager.start(
            JobSpec(
            scope_id=request.scope_id,
            principal_id=request.principal_id,
            session_id=request.session_id,
            idempotency_key=request.idempotency_key,
            run_id=request.run_id,
            budgets=request.budgets,
            policy_ref=request.policy_ref,
            deadline_seconds=effective_deadline_seconds,
            metadata=dict(request.metadata),
            tags=dict(request.tags),
            )
        )
        
        # Create execution context
        ctx = self._job_manager.create_context(job)
        # Track per-job cancellation token so cancel() can stop in-flight work.
        self._cancel_by_job_id[job.job_id] = ctx.cancel

        # Subscribe to events for this job early so callers can observe immediate failures.
        subscription = self._event_bus.subscribe(job_id=job.job_id)

        # Apply request-scoped budget hints (v0.1):
        # - budgets.max_turns constrains request.max_turns
        if request.budgets and request.budgets.max_turns is not None:
            if request.max_turns > int(request.budgets.max_turns):
                await self._job_manager.transition(
                    job.job_id,
                    JobStatus.FAILED,
                    error=f"Budget exceeded: max_turns={request.budgets.max_turns}",
                    error_code="BUDGET_EXCEEDED",
                )
                final = FinalEvent(status="error", error="BUDGET_EXCEEDED")
                await self._event_bus.publish(final.to_runtime_event(ctx, RuntimeEventType.FINAL_ERROR))
                self._cancel_by_job_id.pop(job.job_id, None)
                return ExecutionHandle(job.job_id, self, subscription)

        # Check policy
        if request.policy_ref and request.policy_ref.name == "deny_all":
            await self._job_manager.transition(
                job.job_id,
                JobStatus.FAILED,
                error="Policy denied (deny_all)",
                error_code="POLICY_DENIED",
            )
            final = FinalEvent(status="error", error="POLICY_DENIED")
            await self._event_bus.publish(final.to_runtime_event(ctx, RuntimeEventType.FINAL_ERROR))
            self._cancel_by_job_id.pop(job.job_id, None)
            return ExecutionHandle(job.job_id, self, subscription)

        policy_result = self._policy_engine.evaluate(PolicyContext.from_execution_context(ctx))
        if not policy_result.allowed:
            await self._job_manager.transition(
                job.job_id,
                JobStatus.FAILED,
                error=policy_result.reason,
                error_code="POLICY_DENIED",
            )
            final = FinalEvent(status="error", error="POLICY_DENIED")
            await self._event_bus.publish(final.to_runtime_event(ctx, RuntimeEventType.FINAL_ERROR))
            self._cancel_by_job_id.pop(job.job_id, None)
            return ExecutionHandle(job.job_id, self, subscription)

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
            final = FinalEvent(status="error", error="BUDGET_EXCEEDED")
            await self._event_bus.publish(final.to_runtime_event(ctx, RuntimeEventType.FINAL_ERROR))
            self._cancel_by_job_id.pop(job.job_id, None)
            return ExecutionHandle(job.job_id, self, subscription)

        # Start the execution in background
        task = asyncio.create_task(self._run_execution(job, ctx, request))
        self._task_by_job_id[job.job_id] = task

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

            # Contract operators (test-only / deterministic operators)
            if request.operator_id and request.operator_id.startswith("contract."):
                await self._run_contract_operator(job, ctx, request)
                return
            
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

            # Stream the agent so Layer 2 can project tokens live via RuntimeEvents.
            content_parts: list[str] = []
            total_usage: Any = None
            turns: int = 0
            final_content: str | None = None
            stream_error: str | None = None

            try:
                async for ev in agent.stream(prompt, max_turns=request.max_turns, context=llm_ctx):
                    if ctx.cancel is not None:
                        ctx.cancel.raise_if_cancelled()

                    ev_type = getattr(ev, "type", None)
                    if ev_type is None:
                        continue

                    # llm-client StreamEventType values are strings like "token", "done", etc.
                    if ev_type.value == "token":
                        token = str(ev.data)
                        content_parts.append(token)
                        await self._event_bus.publish(
                            RuntimeEvent.from_context(ctx, RuntimeEventType.MODEL_TOKEN, {"token": token})
                        )
                        continue

                    if ev_type.value == "reasoning":
                        await self._event_bus.publish(
                            RuntimeEvent.from_context(ctx, RuntimeEventType.MODEL_REASONING, {"reasoning": str(ev.data)})
                        )
                        continue

                    if ev_type.value == "error":
                        # Provider-level streaming error. Agent.stream will typically follow up with a DONE carrying
                        # an AgentResult(status="error"), but we record it in case the DONE never arrives.
                        try:
                            stream_error = str(ev.data.get("error"))  # type: ignore[union-attr]
                        except Exception:
                            stream_error = str(ev.data)
                        continue

                    if ev_type.value == "usage":
                        total_usage = ev.data
                        continue

                    if ev_type.value == "done":
                        done = ev.data
                        # Agent.stream() yields DONE with an AgentResult, not a CompletionResult.
                        status = getattr(done, "status", None)
                        if status == "success":
                            final_content = getattr(done, "content", None)
                            total_usage = getattr(done, "total_usage", total_usage)
                            turns = int(getattr(done, "num_turns", 0))
                            break
                        if status == "max_turns":
                            final_content = getattr(done, "content", None)
                            total_usage = getattr(done, "total_usage", total_usage)
                            turns = int(getattr(done, "num_turns", 0))
                            break
                        if status == "error":
                            err = getattr(done, "error", None) or getattr(done, "content", None) or stream_error
                            raise RuntimeError(err or "Agent stream failed")
                        # Unknown DONE payload type; treat as best-effort final content.
                        final_content = getattr(done, "content", None)
                        break

            except RuntimeError:
                # A semantic error from the streamed AgentResult (or provider ERROR mapped into DONE).
                # Do not re-run non-streaming; surface it as an execution failure.
                raise
            except Exception as stream_exc:
                # Fall back to non-streaming if streaming is unsupported or raises unexpectedly.
                # IMPORTANT: Reset the conversation before fallback! The stream may have partially
                # modified the conversation (added user message, assistant message with tool_calls,
                # or partial tool results). Running agent.run() without resetting would add a duplicate
                # user message and corrupt the conversation structure, causing OpenAI API errors like:
                # "An assistant message with 'tool_calls' must be followed by tool messages..."
                import logging
                logging.getLogger("agent_runtime").warning(
                    f"Streaming failed with {type(stream_exc).__name__}: {stream_exc}. "
                    f"Resetting conversation and falling back to non-streaming."
                )
                agent.conversation.clear()
                
                result = await agent.run(prompt, max_turns=request.max_turns, context=llm_ctx)
                turns = int(getattr(result, "num_turns", len(getattr(result, "turns", []) or [])))
                total_usage = getattr(result, "total_usage", None)
                if getattr(result, "status", None) == "error":
                    raise RuntimeError(getattr(result, "error", None) or getattr(result, "content", None) or "Agent error")
                final_content = getattr(result, "content", None)

            if final_content is None:
                final_content = "".join(content_parts)

            # Record usage
            if total_usage:
                await self._ledger.record_usage(
                    ctx,
                    total_usage,
                    provider=agent.provider.__class__.__name__,
                    model=agent.provider.model_name,
                )
            
            # Complete the job
            try:
                await self._job_manager.transition(
                    job.job_id,
                    JobStatus.SUCCEEDED,
                    progress=1.0,
                )
            except Exception:
                # Most likely cancelled/terminal already.
                return
            
            # Emit final event
            final = FinalEvent(
                content=final_content,
                status="success",
                error=None,
                usage=total_usage.to_dict() if hasattr(total_usage, "to_dict") else None,
                turns=turns,
            )
            await self._event_bus.publish(final.to_runtime_event(ctx))
            
        except PolicyDenied as e:
            await self._handle_error(job, ctx, str(e), "POLICY_DENIED")
        except BudgetExceededError as e:
            await self._handle_error(job, ctx, str(e), "BUDGET_EXCEEDED")
        except asyncio.CancelledError:
            try:
                await self._job_manager.transition(
                    job.job_id,
                    JobStatus.CANCELLED,
                    error="Execution cancelled",
                )
            except Exception:
                pass
        except Exception as e:
            if LlmCancelledError is not None and isinstance(e, LlmCancelledError):
                try:
                    await self._job_manager.transition(
                        job.job_id,
                        JobStatus.CANCELLED,
                        error="Execution cancelled",
                    )
                except Exception:
                    pass
            else:
                await self._handle_error(job, ctx, str(e), "EXECUTION_ERROR")
        finally:
            # Best-effort cleanup.
            self._cancel_by_job_id.pop(job.job_id, None)
            self._task_by_job_id.pop(job.job_id, None)
    
    async def _handle_error(
        self,
        job: JobRecord,
        ctx: ExecutionContext,
        error: str,
        error_code: str,
    ) -> None:
        """Handle execution error."""
        try:
            await self._job_manager.transition(
                job.job_id,
                JobStatus.FAILED,
                error=error,
                error_code=error_code,
            )
        except Exception:
            # Ignore invalid transitions (e.g. job cancelled already).
            return
        
        final = FinalEvent(
            status="error",
            error=error,
        )
        await self._event_bus.publish(
            final.to_runtime_event(ctx, RuntimeEventType.FINAL_ERROR)
        )

    async def _run_contract_operator(
        self,
        job: JobRecord,
        ctx: ExecutionContext,
        request: ExecutionRequest,
    ) -> None:
        """Run a deterministic contract operator (for integration tests).

        These operators bypass the LLM and emit RuntimeEvents directly so we can
        test cancellation, retries, and SSE plumbing deterministically.

        Supported operator_id values:
        - contract.stream_slow: emit model.token events with delay
        - contract.tool_sleep: emit tool.start/tool.end with a cancellable sleep
        - contract.retry_backoff: emit progress events with exponential backoff sleeps
        """
        operator_id = request.operator_id or ""
        meta = dict(request.metadata or {})

        async def _check_cancel() -> None:
            if ctx.cancel is not None:
                ctx.cancel.raise_if_cancelled()

        if operator_id == "contract.stream_slow":
            text = str(meta.get("text") or "x" * 200)
            delay_ms = float(meta.get("delay_ms") or 25.0)
            delay_s = max(0.0, delay_ms / 1000.0)

            for ch in text:
                await _check_cancel()
                await self._event_bus.publish(
                    RuntimeEvent.from_context(ctx, RuntimeEventType.MODEL_TOKEN, {"token": ch})
                )
                if delay_s:
                    await asyncio.sleep(delay_s)

            try:
                await self._job_manager.transition(job.job_id, JobStatus.SUCCEEDED, progress=1.0)
            except Exception:
                return
            final = FinalEvent(content=text, status="success", error=None, usage=None, turns=0)
            await self._event_bus.publish(final.to_runtime_event(ctx))
            return

        if operator_id == "contract.tool_sleep":
            seconds = float(meta.get("seconds") or 10.0)
            tool_name = str(meta.get("tool_name") or "debug_sleep")
            await self._event_bus.publish(
                RuntimeEvent.from_context(ctx, RuntimeEventType.TOOL_START, {"tool_name": tool_name})
            )
            end = time.monotonic() + max(0.0, seconds)
            while True:
                await _check_cancel()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(0.25, remaining))
            await self._event_bus.publish(
                RuntimeEvent.from_context(ctx, RuntimeEventType.TOOL_END, {"tool_name": tool_name, "success": True})
            )
            try:
                await self._job_manager.transition(job.job_id, JobStatus.SUCCEEDED, progress=1.0)
            except Exception:
                return
            final = FinalEvent(content="ok", status="success", error=None, usage=None, turns=0)
            await self._event_bus.publish(final.to_runtime_event(ctx))
            return

        if operator_id == "contract.retry_backoff":
            attempts = int(meta.get("attempts") or 5)
            base_backoff_ms = float(meta.get("base_backoff_ms") or 500.0)
            base_backoff_s = max(0.0, base_backoff_ms / 1000.0)
            for attempt in range(max(1, attempts)):
                await _check_cancel()
                await self._event_bus.publish(
                    ProgressEvent(progress=min(0.99, (attempt + 1) / max(1, attempts)), message=f"attempt {attempt+1}").to_runtime_event(ctx)
                )
                if attempt < attempts - 1 and base_backoff_s:
                    await asyncio.sleep(base_backoff_s * (2 ** attempt))

            try:
                await self._job_manager.transition(job.job_id, JobStatus.SUCCEEDED, progress=1.0)
            except Exception:
                return
            final = FinalEvent(content="ok", status="success", error=None, usage=None, turns=0)
            await self._event_bus.publish(final.to_runtime_event(ctx))
            return

        # Unknown contract operator
        await self._handle_error(job, ctx, f"Unknown operator_id: {operator_id}", "OPERATOR_NOT_FOUND")
    
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
                provider=self._engine.provider,
                tools=tools,
                use_middleware=True,
            )
        
        return None
    
    async def cancel(self, job_id: str, reason: str | None = None) -> JobRecord:
        """Cancel an execution."""
        task = self._task_by_job_id.get(job_id)
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
        token = self._cancel_by_job_id.get(job_id)
        if token is not None:
            try:
                token.cancel()
            except Exception:
                pass
        try:
            return await self._job_manager.cancel(job_id, reason)
        except Exception:
            job = await self._job_manager.get(job_id)
            if job is None:
                raise
            return job
    
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
