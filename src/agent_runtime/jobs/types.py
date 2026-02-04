"""
Job types for agent runtime.

This module defines the JobStatus enum and JobRecord dataclass
that form the core of the job lifecycle system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..context import BudgetSpec, PolicyRef, RunVersions


class JobStatus(str, Enum):
    """Job lifecycle states.
    
    State transitions:
    - QUEUED -> RUNNING (job starts)
    - RUNNING -> WAITING_ACTION (needs human input)
    - WAITING_ACTION -> RUNNING (action resolved)
    - RUNNING -> SUCCEEDED (completed successfully)
    - RUNNING -> FAILED (error occurred)
    - * -> CANCELLED (explicitly cancelled)
    - RUNNING -> TIMED_OUT (deadline exceeded)
    """
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_ACTION = "waiting_action"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        }

    @property
    def is_active(self) -> bool:
        """Check if the job is still active."""
        return self in {
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.WAITING_ACTION,
        }


# Valid state transitions
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    # A job can fail before it ever starts running (e.g. policy/budget denied at admission time).
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.RUNNING: {
        JobStatus.WAITING_ACTION,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
    },
    JobStatus.WAITING_ACTION: {
        JobStatus.RUNNING,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
    },
    # Terminal states have no valid transitions
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.TIMED_OUT: set(),
}


@dataclass
class JobRecord:
    """Persistent record of a job execution.
    
    Contains all state needed to track, resume, and audit a job.
    """
    # Identity
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Scope (multi-tenancy)
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Correlation
    parent_job_id: str | None = None  # For child jobs
    idempotency_key: str | None = None
    
    # Status
    status: JobStatus = JobStatus.QUEUED
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    deadline: float | None = None  # Absolute timeout
    
    # Progress
    progress: float | None = None  # 0.0 to 1.0
    current_turn: int = 0
    total_turns: int | None = None
    
    # Configuration
    budgets: BudgetSpec | None = None
    policy_ref: PolicyRef | None = None
    versions: RunVersions | None = None
    
    # Results
    result_ref: str | None = None  # Pointer to stored result
    error: str | None = None
    error_code: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    
    # Schema version
    schema_version: int = 1

    def can_transition_to(self, new_status: JobStatus) -> bool:
        """Check if transition to new_status is valid."""
        return new_status in VALID_TRANSITIONS.get(self.status, set())

    def transition_to(self, new_status: JobStatus) -> JobRecord:
        """Create a new JobRecord with updated status.
        
        Raises:
            ValueError: If the transition is invalid
        """
        if not self.can_transition_to(new_status):
            raise ValueError(
                f"Invalid transition: {self.status.value} -> {new_status.value}"
            )
        
        now = time.time()
        updates: dict[str, Any] = {
            "status": new_status,
            "updated_at": now,
        }
        
        # Set timestamps based on transition
        if new_status == JobStatus.RUNNING and self.started_at is None:
            updates["started_at"] = now
        
        if new_status.is_terminal:
            updates["completed_at"] = now
        
        return JobRecord(
            job_id=self.job_id,
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=self.session_id,
            run_id=self.run_id,
            parent_job_id=self.parent_job_id,
            idempotency_key=self.idempotency_key,
            status=updates.get("status", self.status),
            created_at=self.created_at,
            updated_at=updates.get("updated_at", self.updated_at),
            started_at=updates.get("started_at", self.started_at),
            completed_at=updates.get("completed_at", self.completed_at),
            deadline=self.deadline,
            progress=self.progress,
            current_turn=self.current_turn,
            total_turns=self.total_turns,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            result_ref=self.result_ref,
            error=self.error,
            error_code=self.error_code,
            metadata=dict(self.metadata),
            tags=dict(self.tags),
            schema_version=self.schema_version,
        )

    def with_progress(self, progress: float, turn: int | None = None) -> JobRecord:
        """Create a new JobRecord with updated progress."""
        return JobRecord(
            job_id=self.job_id,
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=self.session_id,
            run_id=self.run_id,
            parent_job_id=self.parent_job_id,
            idempotency_key=self.idempotency_key,
            status=self.status,
            created_at=self.created_at,
            updated_at=time.time(),
            started_at=self.started_at,
            completed_at=self.completed_at,
            deadline=self.deadline,
            progress=progress,
            current_turn=turn if turn is not None else self.current_turn,
            total_turns=self.total_turns,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            result_ref=self.result_ref,
            error=self.error,
            error_code=self.error_code,
            metadata=dict(self.metadata),
            tags=dict(self.tags),
            schema_version=self.schema_version,
        )

    def with_error(self, error: str, error_code: str | None = None) -> JobRecord:
        """Create a new JobRecord with error set."""
        return JobRecord(
            job_id=self.job_id,
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=self.session_id,
            run_id=self.run_id,
            parent_job_id=self.parent_job_id,
            idempotency_key=self.idempotency_key,
            status=self.status,
            created_at=self.created_at,
            updated_at=time.time(),
            started_at=self.started_at,
            completed_at=self.completed_at,
            deadline=self.deadline,
            progress=self.progress,
            current_turn=self.current_turn,
            total_turns=self.total_turns,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            result_ref=self.result_ref,
            error=error,
            error_code=error_code,
            metadata=dict(self.metadata),
            tags=dict(self.tags),
            schema_version=self.schema_version,
        )

    def with_result(self, result_ref: str) -> JobRecord:
        """Create a new JobRecord with result reference set."""
        return JobRecord(
            job_id=self.job_id,
            scope_id=self.scope_id,
            principal_id=self.principal_id,
            session_id=self.session_id,
            run_id=self.run_id,
            parent_job_id=self.parent_job_id,
            idempotency_key=self.idempotency_key,
            status=self.status,
            created_at=self.created_at,
            updated_at=time.time(),
            started_at=self.started_at,
            completed_at=self.completed_at,
            deadline=self.deadline,
            progress=self.progress,
            current_turn=self.current_turn,
            total_turns=self.total_turns,
            budgets=self.budgets,
            policy_ref=self.policy_ref,
            versions=self.versions,
            result_ref=result_ref,
            error=self.error,
            error_code=self.error_code,
            metadata=dict(self.metadata),
            tags=dict(self.tags),
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": self.job_id,
            "scope_id": self.scope_id,
            "principal_id": self.principal_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "parent_job_id": self.parent_job_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "deadline": self.deadline,
            "progress": self.progress,
            "current_turn": self.current_turn,
            "total_turns": self.total_turns,
            "budgets": self.budgets.to_dict() if self.budgets else None,
            "policy_ref": self.policy_ref.to_dict() if self.policy_ref else None,
            "versions": self.versions.to_dict() if self.versions else None,
            "result_ref": self.result_ref,
            "error": self.error,
            "error_code": self.error_code,
            "metadata": dict(self.metadata),
            "tags": dict(self.tags),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        """Deserialize from dictionary."""
        return cls(
            job_id=data.get("job_id", str(uuid.uuid4())),
            scope_id=data.get("scope_id"),
            principal_id=data.get("principal_id"),
            session_id=data.get("session_id"),
            run_id=data.get("run_id", str(uuid.uuid4())),
            parent_job_id=data.get("parent_job_id"),
            idempotency_key=data.get("idempotency_key"),
            status=JobStatus(data.get("status", "queued")),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            deadline=data.get("deadline"),
            progress=data.get("progress"),
            current_turn=data.get("current_turn", 0),
            total_turns=data.get("total_turns"),
            budgets=BudgetSpec.from_dict(data["budgets"]) if data.get("budgets") else None,
            policy_ref=PolicyRef.from_dict(data["policy_ref"]) if data.get("policy_ref") else None,
            versions=RunVersions.from_dict(data["versions"]) if data.get("versions") else None,
            result_ref=data.get("result_ref"),
            error=data.get("error"),
            error_code=data.get("error_code"),
            metadata=dict(data.get("metadata", {})),
            tags=dict(data.get("tags", {})),
            schema_version=data.get("schema_version", 1),
        )


__all__ = [
    "JobStatus",
    "JobRecord",
    "VALID_TRANSITIONS",
]
