"""
Generic memory interfaces and a minimal short-term memory store.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryRecord:
    """Normalized memory entry."""

    memory_id: str
    scope: str = "global"
    content: str = ""
    relevance: float | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryQuery:
    """Memory retrieval request."""

    query: str | None = None
    scope: str | None = None
    limit: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryWrite:
    """Memory write request."""

    content: str
    scope: str = "global"
    memory_id: str | None = None
    relevance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryReader(Protocol):
    async def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Retrieve matching memory entries."""


class MemoryWriter(Protocol):
    async def write(self, entry: MemoryWrite) -> MemoryRecord:
        """Persist one memory entry."""


class MemoryStore(MemoryReader, MemoryWriter, Protocol):
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by id."""

    async def clear(self, scope: str | None = None) -> int:
        """Delete all memory entries, optionally within one scope."""


@dataclass(frozen=True)
class SummaryRecord:
    """Persisted summary associated with a scope."""

    scope: str
    summary: str
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SummaryStore(Protocol):
    async def get(self, scope: str) -> SummaryRecord | None:
        """Fetch the current summary for a scope."""

    async def put(self, scope: str, summary: str, *, metadata: dict[str, Any] | None = None) -> SummaryRecord:
        """Persist the latest summary for a scope."""

    async def clear(self, scope: str | None = None) -> int:
        """Delete one or all persisted summaries."""


class ShortTermMemoryStore(MemoryStore):
    """Minimal in-memory short-term store with scope and recency awareness."""

    def __init__(self, *, max_entries: int = 100) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: list[MemoryRecord] = []

    async def write(self, entry: MemoryWrite) -> MemoryRecord:
        now = time.time()
        record = MemoryRecord(
            memory_id=entry.memory_id or str(uuid.uuid4()),
            scope=entry.scope or "global",
            content=entry.content,
            relevance=entry.relevance,
            created_at=now,
            updated_at=now,
            metadata=dict(entry.metadata),
        )
        self._entries = [item for item in self._entries if item.memory_id != record.memory_id]
        self._entries.append(record)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        return record

    async def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        limit = max(0, int(query.limit))
        if limit == 0:
            return []

        matches = [
            entry
            for entry in self._entries
            if query.scope in {None, entry.scope}
        ]

        if query.metadata:
            matches = [
                entry
                for entry in matches
                if all(entry.metadata.get(key) == value for key, value in query.metadata.items())
            ]

        ranked = sorted(
            matches,
            key=lambda entry: (
                self._score(entry, query.query),
                entry.updated_at,
            ),
            reverse=True,
        )
        return ranked[:limit]

    async def delete(self, memory_id: str) -> bool:
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.memory_id != memory_id]
        return len(self._entries) != before

    async def clear(self, scope: str | None = None) -> int:
        if scope is None:
            cleared = len(self._entries)
            self._entries = []
            return cleared
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.scope != scope]
        return before - len(self._entries)

    async def list_all(self) -> list[MemoryRecord]:
        return list(self._entries)

    @staticmethod
    def _score(entry: MemoryRecord, query_text: str | None) -> float:
        explicit = float(entry.relevance or 0.0)
        if not query_text:
            return explicit
        entry_words = set(str(entry.content).lower().split())
        query_words = set(str(query_text).lower().split())
        if not entry_words or not query_words:
            return explicit
        overlap = len(entry_words & query_words) / max(len(query_words), 1)
        return explicit + overlap


class InMemorySummaryStore(SummaryStore):
    """Minimal in-memory persistent summary store."""

    def __init__(self) -> None:
        self._summaries: dict[str, SummaryRecord] = {}

    async def get(self, scope: str) -> SummaryRecord | None:
        return self._summaries.get(scope)

    async def put(self, scope: str, summary: str, *, metadata: dict[str, Any] | None = None) -> SummaryRecord:
        record = SummaryRecord(
            scope=scope,
            summary=summary,
            updated_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self._summaries[scope] = record
        return record

    async def clear(self, scope: str | None = None) -> int:
        if scope is None:
            cleared = len(self._summaries)
            self._summaries.clear()
            return cleared
        existed = 1 if scope in self._summaries else 0
        self._summaries.pop(scope, None)
        return existed


__all__ = [
    "InMemorySummaryStore",
    "MemoryQuery",
    "MemoryReader",
    "MemoryRecord",
    "MemoryStore",
    "MemoryWrite",
    "MemoryWriter",
    "ShortTermMemoryStore",
    "SummaryRecord",
    "SummaryStore",
]
