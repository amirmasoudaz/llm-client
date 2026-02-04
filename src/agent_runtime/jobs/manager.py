"""
Job manager for lifecycle operations.

This module provides the JobManager that orchestrates job lifecycle
operations, including state transitions, cancellation, and event emission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ..context import ExecutionContext, BudgetSpec, PolicyRef
from ..events import EventBus, RuntimeEvent, RuntimeEventType, JobEvent
from .types import JobRecord, JobStatus
from .store import JobStore, JobFilter


@dataclass
class JobSpec:
    """Specification for creating a new job."""
    # Scope
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    
    # Correlation
    parent_job_id: str | None = None
    idempotency_key: str | None = None
    
    # Configuration
    budgets: BudgetSpec | None = None
    policy_ref: PolicyRef | None = None
    deadline_seconds: float | None = None
    
    # Metadata
    metadata: dict[str, Any] | None = None
    tags: dict[str, str] | None = None


class JobManager:
    """Manages job lifecycle operations.
    
    The JobManager is responsible for:
    - Creating and starting jobs
    - State transitions with validation
    - Cancellation handling
    - Event emission for observability
    - Idempotency enforcement
    """
    
    def __init__(
        self,
        store: JobStore,
        event_bus: EventBus | None = None,
    ):
        self._store = store
        self._event_bus = event_bus
    
    async def start(self, spec: JobSpec) -> JobRecord:
        """Create and start a new job.
        
        If idempotency_key is provided and a matching job exists,
        returns the existing job instead of creating a new one.
        
        Returns:
            The job record (either new or existing)
        """
        # Check idempotency
        if spec.idempotency_key:
            existing = await self._store.get_by_idempotency_key(
                spec.idempotency_key,
                spec.scope_id,
            )
            if existing:
                return existing
        
        # Calculate deadline
        deadline = None
        if spec.deadline_seconds:
            deadline = time.time() + spec.deadline_seconds
        
        # Create job record
        job = JobRecord(
            scope_id=spec.scope_id,
            principal_id=spec.principal_id,
            session_id=spec.session_id,
            parent_job_id=spec.parent_job_id,
            idempotency_key=spec.idempotency_key,
            status=JobStatus.QUEUED,
            deadline=deadline,
            budgets=spec.budgets,
            policy_ref=spec.policy_ref,
            metadata=dict(spec.metadata or {}),
            tags=dict(spec.tags or {}),
        )
        
        job = await self._store.create(job)
        
        # Emit event
        await self._emit_job_event(job, RuntimeEventType.JOB_STARTED)
        
        return job
    
    async def get(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        return await self._store.get(job_id)
    
    async def transition(
        self,
        job_id: str,
        new_status: JobStatus,
        *,
        error: str | None = None,
        error_code: str | None = None,
        result_ref: str | None = None,
        progress: float | None = None,
    ) -> JobRecord:
        """Transition a job to a new status.
        
        Validates the transition and updates timestamps appropriately.
        
        Raises:
            ValueError: If job doesn't exist or transition is invalid
        """
        job = await self._store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        previous_status = job.status
        
        # Apply transition
        job = job.transition_to(new_status)
        
        # Apply additional updates
        if error:
            job = job.with_error(error, error_code)
        if result_ref:
            job = job.with_result(result_ref)
        if progress is not None:
            job = job.with_progress(progress)
        
        job = await self._store.update(job)
        
        # Emit appropriate event
        if new_status == JobStatus.SUCCEEDED:
            await self._emit_job_event(job, RuntimeEventType.JOB_COMPLETED, previous_status)
        elif new_status == JobStatus.FAILED:
            await self._emit_job_event(job, RuntimeEventType.JOB_FAILED, previous_status)
        elif new_status == JobStatus.CANCELLED:
            await self._emit_job_event(job, RuntimeEventType.JOB_CANCELLED, previous_status)
        else:
            await self._emit_job_event(job, RuntimeEventType.JOB_STATUS_CHANGED, previous_status)
        
        return job
    
    async def cancel(self, job_id: str, reason: str | None = None) -> JobRecord:
        """Cancel a job.
        
        Can only cancel jobs that are not already in a terminal state.
        
        Raises:
            ValueError: If job doesn't exist or cannot be cancelled
        """
        job = await self._store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status.is_terminal:
            raise ValueError(f"Cannot cancel job in terminal state: {job.status.value}")
        
        return await self.transition(
            job_id,
            JobStatus.CANCELLED,
            error=reason or "Job cancelled",
            error_code="CANCELLED",
        )
    
    async def update_progress(
        self,
        job_id: str,
        progress: float,
        turn: int | None = None,
    ) -> JobRecord:
        """Update job progress without changing status."""
        job = await self._store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        job = job.with_progress(progress, turn)
        job = await self._store.update(job)
        
        return job
    
    async def check_deadline(self, job_id: str) -> bool:
        """Check if job has exceeded its deadline.
        
        If deadline is exceeded, transitions job to TIMED_OUT.
        
        Returns:
            True if job timed out, False otherwise
        """
        job = await self._store.get(job_id)
        if not job or not job.deadline:
            return False
        
        if job.status.is_terminal:
            return False
        
        if time.time() > job.deadline:
            await self.transition(
                job_id,
                JobStatus.TIMED_OUT,
                error="Job exceeded deadline",
                error_code="TIMED_OUT",
            )
            return True
        
        return False
    
    async def list(self, filter: JobFilter | None = None) -> list[JobRecord]:
        """List jobs matching the filter."""
        return await self._store.list(filter)
    
    async def list_active(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
    ) -> list[JobRecord]:
        """List active (non-terminal) jobs."""
        return await self._store.list(JobFilter(
            scope_id=scope_id,
            principal_id=principal_id,
            status={JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.WAITING_ACTION},
        ))
    
    async def _emit_job_event(
        self,
        job: JobRecord,
        event_type: RuntimeEventType,
        previous_status: JobStatus | None = None,
    ) -> None:
        """Emit a job lifecycle event."""
        if not self._event_bus:
            return
        
        event = RuntimeEvent(
            type=event_type,
            job_id=job.job_id,
            run_id=job.run_id,
            scope_id=job.scope_id,
            principal_id=job.principal_id,
            session_id=job.session_id,
            data={
                "status": job.status.value,
                "previous_status": previous_status.value if previous_status else None,
                "progress": job.progress,
                "error": job.error,
            },
        )
        
        await self._event_bus.publish(event)
    
    def create_context(self, job: JobRecord) -> ExecutionContext:
        """Create an ExecutionContext from a job record."""
        return ExecutionContext(
            scope_id=job.scope_id,
            principal_id=job.principal_id,
            session_id=job.session_id,
            run_id=job.run_id,
            job_id=job.job_id,
            budgets=job.budgets,
            policy_ref=job.policy_ref,
            versions=job.versions,
            metadata=dict(job.metadata),
        )


__all__ = [
    "JobManager",
    "JobSpec",
]
