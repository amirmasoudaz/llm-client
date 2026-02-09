"""
Ledger writer implementations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .types import LedgerEvent, LedgerEventType, UsageRecord


class LedgerWriter(ABC):
    """Abstract interface for ledger event persistence."""
    
    @abstractmethod
    async def write(self, event: LedgerEvent) -> None:
        """Write a ledger event."""
        ...
    
    @abstractmethod
    async def get_usage(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> UsageRecord:
        """Get aggregated usage for the given filters."""
        ...
    
    @abstractmethod
    async def list_events(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        job_id: str | None = None,
        event_type: LedgerEventType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LedgerEvent]:
        """List ledger events matching the filters."""
        ...


class InMemoryLedgerWriter(LedgerWriter):
    """In-memory ledger writer implementation.
    
    Stores events in memory with indexes for efficient querying.
    Suitable for testing and single-process deployments.
    """
    
    def __init__(self, max_events: int = 100000):
        self._events: list[LedgerEvent] = []
        self._by_scope: dict[str, list[LedgerEvent]] = defaultdict(list)
        self._by_principal: dict[str, list[LedgerEvent]] = defaultdict(list)
        self._by_job: dict[str, list[LedgerEvent]] = defaultdict(list)
        self._usage_cache: dict[str, UsageRecord] = {}
        self._max_events = max_events
        self._lock = asyncio.Lock()
    
    async def write(self, event: LedgerEvent) -> None:
        async with self._lock:
            # Trim if over max
            if len(self._events) >= self._max_events:
                self._events = self._events[-self._max_events // 2:]
                # Rebuild indexes
                self._rebuild_indexes()
            
            self._events.append(event)
            
            # Index by various keys
            if event.scope_id:
                self._by_scope[event.scope_id].append(event)
            if event.principal_id:
                self._by_principal[event.principal_id].append(event)
            if event.job_id:
                self._by_job[event.job_id].append(event)
            
            # Invalidate usage cache
            self._invalidate_cache(event)
    
    async def get_usage(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> UsageRecord:
        cache_key = f"{scope_id}:{principal_id}:{session_id}:{job_id}"
        
        async with self._lock:
            if cache_key in self._usage_cache:
                return self._usage_cache[cache_key]
            
            record = UsageRecord(
                scope_id=scope_id,
                principal_id=principal_id,
                session_id=session_id,
            )
            
            # Filter events
            events = self._filter_events(scope_id, principal_id, session_id, job_id)
            
            for event in events:
                record.add_event(event)
            
            self._usage_cache[cache_key] = record
            return record
    
    async def list_events(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        job_id: str | None = None,
        event_type: LedgerEventType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LedgerEvent]:
        async with self._lock:
            events = self._filter_events(scope_id, principal_id, None, job_id)
            
            if event_type:
                events = [e for e in events if e.type == event_type]
            
            # Sort by timestamp descending
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return events[offset:offset + limit]
    
    def _filter_events(
        self,
        scope_id: str | None,
        principal_id: str | None,
        session_id: str | None,
        job_id: str | None,
    ) -> list[LedgerEvent]:
        """Filter events by various criteria."""
        # Start with the most specific index
        if job_id:
            events = list(self._by_job.get(job_id, []))
        elif scope_id:
            events = list(self._by_scope.get(scope_id, []))
        elif principal_id:
            events = list(self._by_principal.get(principal_id, []))
        else:
            events = list(self._events)
        
        # Apply additional filters
        if scope_id and not job_id:
            events = [e for e in events if e.scope_id == scope_id]
        if principal_id:
            events = [e for e in events if e.principal_id == principal_id]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        
        return events
    
    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from events list."""
        self._by_scope.clear()
        self._by_principal.clear()
        self._by_job.clear()
        
        for event in self._events:
            if event.scope_id:
                self._by_scope[event.scope_id].append(event)
            if event.principal_id:
                self._by_principal[event.principal_id].append(event)
            if event.job_id:
                self._by_job[event.job_id].append(event)
    
    def _invalidate_cache(self, event: LedgerEvent) -> None:
        """Invalidate usage cache entries affected by this event."""
        # Simple approach: clear all cache entries that could be affected
        keys_to_remove = []
        for key in self._usage_cache:
            parts = key.split(":")
            if (
                (event.scope_id and parts[0] == event.scope_id) or
                (event.principal_id and parts[1] == event.principal_id) or
                (event.session_id and parts[2] == event.session_id) or
                (event.job_id and parts[3] == event.job_id)
            ):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._usage_cache[key]


__all__ = [
    "LedgerWriter",
    "InMemoryLedgerWriter",
]
