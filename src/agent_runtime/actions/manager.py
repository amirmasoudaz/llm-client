"""
Action manager for human-in-the-loop protocol.

This module provides the ActionManager that orchestrates action
creation, resolution, and job coordination.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from ..context import ExecutionContext
from ..events import EventBus, RuntimeEvent, RuntimeEventType
from ..jobs import JobManager, JobStatus
from .types import ActionRecord, ActionStatus, ActionType
from .store import ActionStore


class ActionRequiredError(Exception):
    """Raised when an action is required to continue execution."""
    def __init__(self, action: ActionRecord):
        self.action = action
        super().__init__(f"Action required: {action.type}")


@dataclass
class ActionSpec:
    """Specification for creating a new action."""
    job_id: str
    type: str = ActionType.CONFIRM.value
    payload: dict[str, Any] | None = None
    expires_in_seconds: float | None = 300.0  # 5 minutes default
    metadata: dict[str, Any] | None = None


class ActionManager:
    """Manages action lifecycle and job coordination.
    
    The ActionManager handles:
    - Creating actions and transitioning jobs to WAITING_ACTION
    - Resolving actions and resuming jobs
    - Expiring stale actions
    - Event emission for UI integration
    """
    
    def __init__(
        self,
        store: ActionStore,
        job_manager: JobManager,
        event_bus: EventBus | None = None,
    ):
        self._store = store
        self._job_manager = job_manager
        self._event_bus = event_bus
        self._waiters: dict[str, asyncio.Event] = {}  # action_id -> event
    
    async def require_action(
        self,
        spec: ActionSpec,
        ctx: ExecutionContext | None = None,
    ) -> ActionRecord:
        """Create an action and transition the job to WAITING_ACTION.
        
        This is the main entry point for requesting human input.
        After calling this, the execution should pause until
        the action is resolved.
        
        Returns:
            The created action record
        """
        # Calculate expiry
        expires_at = None
        if spec.expires_in_seconds:
            expires_at = time.time() + spec.expires_in_seconds
        
        # Create action
        action = ActionRecord(
            job_id=spec.job_id,
            type=spec.type,
            payload=dict(spec.payload or {}),
            status=ActionStatus.PENDING,
            expires_at=expires_at,
            metadata=dict(spec.metadata or {}),
        )
        
        action = await self._store.create(action)
        
        # Transition job to waiting
        await self._job_manager.transition(spec.job_id, JobStatus.WAITING_ACTION)
        
        # Create waiter for this action
        self._waiters[action.action_id] = asyncio.Event()
        
        # Emit event
        await self._emit_action_event(action, RuntimeEventType.ACTION_REQUIRED, ctx)
        
        return action
    
    async def resolve(
        self,
        action_id: str,
        resolution: dict[str, Any],
        *,
        resume_token: str | None = None,
    ) -> ActionRecord:
        """Resolve an action and resume the job.
        
        Args:
            action_id: The action to resolve
            resolution: The resolution payload from the user/UI
            resume_token: Optional security token (must match if provided)
        
        Returns:
            The resolved action record
            
        Raises:
            ValueError: If action not found, already resolved, or token mismatch
        """
        action = await self._store.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")
        
        if action.status != ActionStatus.PENDING:
            raise ValueError(f"Action {action_id} is not pending: {action.status.value}")
        
        if resume_token and action.resume_token != resume_token:
            raise ValueError("Invalid resume token")
        
        if action.is_expired:
            # Mark as expired instead of resolving
            action = action.expire()
            await self._store.update(action)
            await self._emit_action_event(action, RuntimeEventType.ACTION_EXPIRED)
            raise ValueError(f"Action {action_id} has expired")
        
        # Resolve the action
        action = action.resolve(resolution)
        await self._store.update(action)
        
        # Resume the job
        await self._job_manager.transition(action.job_id, JobStatus.RUNNING)
        
        # Signal any waiters
        if action.action_id in self._waiters:
            self._waiters[action.action_id].set()
        
        # Emit event
        await self._emit_action_event(action, RuntimeEventType.ACTION_RESOLVED)
        
        return action
    
    async def resolve_by_token(
        self,
        resume_token: str,
        resolution: dict[str, Any],
    ) -> ActionRecord:
        """Resolve an action using just the resume token.
        
        This is useful for webhook-based resolution where the
        action_id might not be known.
        """
        action = await self._store.get_by_resume_token(resume_token)
        if not action:
            raise ValueError(f"No action found for resume token")
        
        return await self.resolve(action.action_id, resolution, resume_token=resume_token)
    
    async def cancel(
        self,
        action_id: str,
        reason: str | None = None,
    ) -> ActionRecord:
        """Cancel a pending action.
        
        Also cancels the associated job.
        """
        action = await self._store.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")
        
        if action.status != ActionStatus.PENDING:
            raise ValueError(f"Action {action_id} is not pending")
        
        action = action.cancel(reason)
        await self._store.update(action)
        
        # Cancel the job
        await self._job_manager.cancel(action.job_id, reason or "Action cancelled")
        
        # Signal any waiters
        if action.action_id in self._waiters:
            self._waiters[action.action_id].set()
        
        # Emit event
        await self._emit_action_event(action, RuntimeEventType.ACTION_CANCELLED)
        
        return action
    
    async def wait_for_resolution(
        self,
        action_id: str,
        timeout: float | None = None,
    ) -> ActionRecord:
        """Wait for an action to be resolved.
        
        This is used internally by the runtime to pause execution
        until an action is resolved.
        
        Args:
            action_id: The action to wait for
            timeout: Maximum time to wait (None = use action's expires_at)
            
        Returns:
            The resolved/expired/cancelled action
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        action = await self._store.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")
        
        if action.status != ActionStatus.PENDING:
            return action
        
        # Calculate effective timeout
        if timeout is None and action.expires_at:
            timeout = max(0, action.expires_at - time.time())
        
        # Wait for signal
        event = self._waiters.get(action_id)
        if not event:
            event = asyncio.Event()
            self._waiters[action_id] = event
        
        try:
            if timeout:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
        except asyncio.TimeoutError:
            # Check if expired and mark accordingly
            action = await self._store.get(action_id)
            if action and action.status == ActionStatus.PENDING:
                action = action.expire()
                await self._store.update(action)
                await self._emit_action_event(action, RuntimeEventType.ACTION_EXPIRED)
            raise
        finally:
            # Clean up waiter
            self._waiters.pop(action_id, None)
        
        # Return final state
        return await self._store.get(action_id)  # type: ignore
    
    async def expire_stale_actions(self) -> list[ActionRecord]:
        """Find and expire all stale pending actions.
        
        This should be called periodically to clean up actions
        that were never resolved.
        
        Returns:
            List of expired actions
        """
        expired = await self._store.list_expired()
        results = []
        
        for action in expired:
            action = action.expire()
            await self._store.update(action)
            
            # Fail the associated job
            try:
                await self._job_manager.transition(
                    action.job_id,
                    JobStatus.TIMED_OUT,
                    error="Action expired",
                    error_code="ACTION_EXPIRED",
                )
            except ValueError:
                pass  # Job might already be in terminal state
            
            # Signal any waiters
            if action.action_id in self._waiters:
                self._waiters[action.action_id].set()
            
            await self._emit_action_event(action, RuntimeEventType.ACTION_EXPIRED)
            results.append(action)
        
        return results
    
    async def get(self, action_id: str) -> ActionRecord | None:
        """Get an action by ID."""
        return await self._store.get(action_id)
    
    async def list_pending_for_job(self, job_id: str) -> list[ActionRecord]:
        """List pending actions for a job."""
        return await self._store.list_pending_for_job(job_id)
    
    async def _emit_action_event(
        self,
        action: ActionRecord,
        event_type: RuntimeEventType,
        ctx: ExecutionContext | None = None,
    ) -> None:
        """Emit an action event."""
        if not self._event_bus:
            return
        
        # Get job for additional context
        job = await self._job_manager.get(action.job_id)
        
        event = RuntimeEvent(
            type=event_type,
            job_id=action.job_id,
            run_id=job.run_id if job else None,
            scope_id=job.scope_id if job else None,
            principal_id=job.principal_id if job else None,
            session_id=job.session_id if job else None,
            data=action.to_event_payload(),
        )
        
        await self._event_bus.publish(event)


__all__ = [
    "ActionManager",
    "ActionSpec",
    "ActionRequiredError",
]
