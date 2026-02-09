"""
Job store implementations.

This module provides the JobStore interface and implementations
for persisting job records.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .types import JobRecord, JobStatus


@dataclass
class JobFilter:
    """Filter criteria for listing jobs."""
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    status: JobStatus | set[JobStatus] | None = None
    parent_job_id: str | None = None
    idempotency_key: str | None = None
    limit: int = 100
    offset: int = 0
    order_by: str = "created_at"
    order_desc: bool = True

    def matches(self, job: JobRecord) -> bool:
        """Check if a job matches this filter."""
        if self.scope_id and job.scope_id != self.scope_id:
            return False
        if self.principal_id and job.principal_id != self.principal_id:
            return False
        if self.session_id and job.session_id != self.session_id:
            return False
        if self.parent_job_id and job.parent_job_id != self.parent_job_id:
            return False
        if self.idempotency_key and job.idempotency_key != self.idempotency_key:
            return False
        if self.status:
            if isinstance(self.status, set):
                if job.status not in self.status:
                    return False
            elif job.status != self.status:
                return False
        return True


class JobStore(ABC):
    """Abstract interface for job persistence.
    
    Implementations must be thread-safe for concurrent access.
    """
    
    @abstractmethod
    async def create(self, job: JobRecord) -> JobRecord:
        """Create a new job record.
        
        Raises:
            ValueError: If job_id already exists
        """
        ...
    
    @abstractmethod
    async def get(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        ...
    
    @abstractmethod
    async def update(self, job: JobRecord) -> JobRecord:
        """Update an existing job record.
        
        Raises:
            ValueError: If job doesn't exist
        """
        ...
    
    @abstractmethod
    async def delete(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if deleted."""
        ...
    
    @abstractmethod
    async def list(self, filter: JobFilter | None = None) -> list[JobRecord]:
        """List jobs matching the filter."""
        ...
    
    @abstractmethod
    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        scope_id: str | None = None,
    ) -> JobRecord | None:
        """Get a job by idempotency key (for deduplication)."""
        ...
    
    @abstractmethod
    async def count(self, filter: JobFilter | None = None) -> int:
        """Count jobs matching the filter."""
        ...
    
    async def get_or_create(
        self,
        idempotency_key: str,
        job_factory: callable,
        scope_id: str | None = None,
    ) -> tuple[JobRecord, bool]:
        """Get existing job by idempotency key or create a new one.
        
        Returns:
            Tuple of (job, created) where created is True if new job was created.
        """
        existing = await self.get_by_idempotency_key(idempotency_key, scope_id)
        if existing:
            return existing, False
        
        job = job_factory()
        job = await self.create(job)
        return job, True


class InMemoryJobStore(JobStore):
    """In-memory job store implementation.
    
    Suitable for testing and single-process deployments.
    Thread-safe via asyncio.Lock.
    """
    
    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_index: dict[str, str] = {}  # key -> job_id
        self._lock = asyncio.Lock()
    
    async def create(self, job: JobRecord) -> JobRecord:
        async with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"Job {job.job_id} already exists")
            
            self._jobs[job.job_id] = job
            
            if job.idempotency_key:
                key = self._idempotency_key(job.idempotency_key, job.scope_id)
                self._idempotency_index[key] = job.job_id
            
            return job
    
    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def update(self, job: JobRecord) -> JobRecord:
        async with self._lock:
            if job.job_id not in self._jobs:
                raise ValueError(f"Job {job.job_id} not found")
            
            self._jobs[job.job_id] = job
            return job
    
    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.pop(job_id, None)
            if job and job.idempotency_key:
                key = self._idempotency_key(job.idempotency_key, job.scope_id)
                self._idempotency_index.pop(key, None)
            return job is not None
    
    async def list(self, filter: JobFilter | None = None) -> list[JobRecord]:
        async with self._lock:
            jobs = list(self._jobs.values())
            
            if filter:
                jobs = [j for j in jobs if filter.matches(j)]
                
                # Sort
                reverse = filter.order_desc
                key = lambda j: getattr(j, filter.order_by, j.created_at)
                jobs.sort(key=key, reverse=reverse)
                
                # Paginate
                jobs = jobs[filter.offset:filter.offset + filter.limit]
            
            return jobs
    
    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        scope_id: str | None = None,
    ) -> JobRecord | None:
        async with self._lock:
            key = self._idempotency_key(idempotency_key, scope_id)
            job_id = self._idempotency_index.get(key)
            if job_id:
                return self._jobs.get(job_id)
            return None
    
    async def count(self, filter: JobFilter | None = None) -> int:
        async with self._lock:
            if filter:
                return sum(1 for j in self._jobs.values() if filter.matches(j))
            return len(self._jobs)
    
    def _idempotency_key(self, key: str, scope_id: str | None) -> str:
        """Generate composite key for idempotency index."""
        return f"{scope_id or ''}:{key}"


__all__ = [
    "JobStore",
    "InMemoryJobStore",
    "JobFilter",
]
