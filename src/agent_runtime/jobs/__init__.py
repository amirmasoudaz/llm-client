"""
Job system for agent runtime.

This module provides the job lifecycle management:
- JobRecord: Persisted job state
- JobManager: Lifecycle operations (start, transition, cancel)
- JobStore: Persistence interface with implementations
"""

from .types import (
    JobStatus,
    JobRecord,
    VALID_TRANSITIONS,
)
from .manager import (
    JobManager,
    JobSpec,
)
from .store import (
    JobStore,
    InMemoryJobStore,
    JobFilter,
)

__all__ = [
    "JobStatus",
    "JobRecord",
    "VALID_TRANSITIONS",
    "JobManager",
    "JobSpec",
    "JobStore",
    "InMemoryJobStore",
    "JobFilter",
]
