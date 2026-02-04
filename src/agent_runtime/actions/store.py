"""
Action store implementations.

This module provides the ActionStore interface and implementations
for persisting action records.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .types import ActionRecord, ActionStatus


@dataclass
class ActionFilter:
    """Filter criteria for listing actions."""
    job_id: str | None = None
    status: ActionStatus | set[ActionStatus] | None = None
    type: str | None = None
    limit: int = 100
    offset: int = 0

    def matches(self, action: ActionRecord) -> bool:
        """Check if an action matches this filter."""
        if self.job_id and action.job_id != self.job_id:
            return False
        if self.type and action.type != self.type:
            return False
        if self.status:
            if isinstance(self.status, set):
                if action.status not in self.status:
                    return False
            elif action.status != self.status:
                return False
        return True


class ActionStore(ABC):
    """Abstract interface for action persistence."""
    
    @abstractmethod
    async def create(self, action: ActionRecord) -> ActionRecord:
        """Create a new action record."""
        ...
    
    @abstractmethod
    async def get(self, action_id: str) -> ActionRecord | None:
        """Get an action by ID."""
        ...
    
    @abstractmethod
    async def get_by_resume_token(self, resume_token: str) -> ActionRecord | None:
        """Get an action by resume token."""
        ...
    
    @abstractmethod
    async def update(self, action: ActionRecord) -> ActionRecord:
        """Update an existing action record."""
        ...
    
    @abstractmethod
    async def delete(self, action_id: str) -> bool:
        """Delete an action by ID."""
        ...
    
    @abstractmethod
    async def list(self, filter: ActionFilter | None = None) -> list[ActionRecord]:
        """List actions matching the filter."""
        ...
    
    @abstractmethod
    async def list_pending_for_job(self, job_id: str) -> list[ActionRecord]:
        """List pending actions for a job."""
        ...
    
    @abstractmethod
    async def list_expired(self) -> list[ActionRecord]:
        """List actions that have expired but not been marked as such."""
        ...


class InMemoryActionStore(ActionStore):
    """In-memory action store implementation.
    
    Suitable for testing and single-process deployments.
    """
    
    def __init__(self):
        self._actions: dict[str, ActionRecord] = {}
        self._resume_token_index: dict[str, str] = {}  # token -> action_id
        self._lock = asyncio.Lock()
    
    async def create(self, action: ActionRecord) -> ActionRecord:
        async with self._lock:
            if action.action_id in self._actions:
                raise ValueError(f"Action {action.action_id} already exists")
            
            self._actions[action.action_id] = action
            self._resume_token_index[action.resume_token] = action.action_id
            
            return action
    
    async def get(self, action_id: str) -> ActionRecord | None:
        async with self._lock:
            return self._actions.get(action_id)
    
    async def get_by_resume_token(self, resume_token: str) -> ActionRecord | None:
        async with self._lock:
            action_id = self._resume_token_index.get(resume_token)
            if action_id:
                return self._actions.get(action_id)
            return None
    
    async def update(self, action: ActionRecord) -> ActionRecord:
        async with self._lock:
            if action.action_id not in self._actions:
                raise ValueError(f"Action {action.action_id} not found")
            
            # Update resume token index if changed
            old_action = self._actions[action.action_id]
            if old_action.resume_token != action.resume_token:
                self._resume_token_index.pop(old_action.resume_token, None)
                self._resume_token_index[action.resume_token] = action.action_id
            
            self._actions[action.action_id] = action
            return action
    
    async def delete(self, action_id: str) -> bool:
        async with self._lock:
            action = self._actions.pop(action_id, None)
            if action:
                self._resume_token_index.pop(action.resume_token, None)
                return True
            return False
    
    async def list(self, filter: ActionFilter | None = None) -> list[ActionRecord]:
        async with self._lock:
            actions = list(self._actions.values())
            
            if filter:
                actions = [a for a in actions if filter.matches(a)]
                actions = actions[filter.offset:filter.offset + filter.limit]
            
            return actions
    
    async def list_pending_for_job(self, job_id: str) -> list[ActionRecord]:
        async with self._lock:
            return [
                a for a in self._actions.values()
                if a.job_id == job_id and a.status == ActionStatus.PENDING
            ]
    
    async def list_expired(self) -> list[ActionRecord]:
        import time
        async with self._lock:
            now = time.time()
            return [
                a for a in self._actions.values()
                if a.status == ActionStatus.PENDING
                and a.expires_at is not None
                and a.expires_at < now
            ]


__all__ = [
    "ActionStore",
    "InMemoryActionStore",
    "ActionFilter",
]
