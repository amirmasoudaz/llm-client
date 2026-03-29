"""
Generic multi-source context assembly interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .context_planning import ContextPlan, ContextPlanner, ContextPlanningRequest
from .memory import MemoryQuery, MemoryRecord


@dataclass(frozen=True)
class ContextSourceRequest:
    current_message: str
    scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSourcePayload:
    source_name: str
    entries: list[Any] = field(default_factory=list)
    memory: list[MemoryRecord] = field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextSourceLoader(Protocol):
    async def load(self, request: ContextSourceRequest) -> ContextSourcePayload:
        """Load generic context data for one source."""


@dataclass(frozen=True)
class ContextAssemblyRequest:
    current_message: str
    base_entries: list[Any] = field(default_factory=list)
    source_request: ContextSourceRequest | None = None
    max_entries: int = 50
    memory_query: MemoryQuery | None = None
    max_memory_entries: int = 10
    summarize_when_truncated: bool = False
    summary_max_tokens: int = 256
    persist_summary: bool = False
    summary_scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextAssemblyResult:
    plan: ContextPlan
    sources: list[ContextSourcePayload] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiSourceContextAssembler:
    """Composes neutral source loaders with a context planner."""

    def __init__(
        self,
        *,
        planner: ContextPlanner,
        source_loaders: list[ContextSourceLoader] | None = None,
    ) -> None:
        self.planner = planner
        self.source_loaders = list(source_loaders or [])

    async def assemble(self, request: ContextAssemblyRequest) -> ContextAssemblyResult:
        source_request = ContextSourceRequest(
            current_message=request.current_message,
            scope=request.source_request.scope if request.source_request is not None else None,
            metadata={
                **(request.source_request.metadata if request.source_request is not None else {}),
                **request.metadata,
            },
        )
        payloads: list[ContextSourcePayload] = []
        merged_entries = list(request.base_entries)
        merged_memory: list[MemoryRecord] = []
        source_summaries: list[str] = []

        for loader in self.source_loaders:
            payload = await loader.load(source_request)
            payloads.append(payload)
            merged_entries.extend(payload.entries)
            merged_memory.extend(payload.memory)
            if payload.summary:
                source_summaries.append(payload.summary)

        plan = await self.planner.plan(
            ContextPlanningRequest(
                entries=merged_entries,
                current_message=request.current_message,
                max_entries=request.max_entries,
                memory_query=request.memory_query,
                max_memory_entries=request.max_memory_entries,
                summarize_when_truncated=request.summarize_when_truncated,
                summary_max_tokens=request.summary_max_tokens,
                persist_summary=request.persist_summary,
                summary_scope=request.summary_scope,
                metadata=dict(request.metadata),
            )
        )

        combined_memory = list(plan.memory)
        for memory_entry in merged_memory:
            if all(existing.memory_id != memory_entry.memory_id for existing in combined_memory):
                combined_memory.append(memory_entry)

        combined_summary = plan.summary
        if source_summaries:
            summary_parts = [part for part in [*source_summaries, plan.summary] if part]
            combined_summary = "\n".join(summary_parts) if summary_parts else None

        assembled_plan = ContextPlan(
            entries=plan.entries,
            memory=combined_memory,
            summary=combined_summary,
            persistent_summary=plan.persistent_summary,
            truncation=plan.truncation,
            metadata={
                **plan.metadata,
                "source_count": len(payloads),
                "source_names": [payload.source_name for payload in payloads],
            },
        )

        return ContextAssemblyResult(
            plan=assembled_plan,
            sources=payloads,
            metadata={"source_count": len(payloads)},
        )


__all__ = [
    "ContextAssemblyRequest",
    "ContextAssemblyResult",
    "ContextSourceLoader",
    "ContextSourcePayload",
    "ContextSourceRequest",
    "MultiSourceContextAssembler",
]
