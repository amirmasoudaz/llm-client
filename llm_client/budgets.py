"""
Generic usage tracking and budget enforcement primitives.

These types live in ``llm_client`` because they are reusable execution-runtime
infrastructure rather than host-runtime or product-policy concerns.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from .context import ExecutionContext


class LedgerEventType(str, Enum):
    """Types of budget and usage events."""

    MODEL_USAGE = "model_usage"
    TOOL_USAGE = "tool_usage"
    CONNECTOR_USAGE = "connector_usage"
    BUDGET_CHECK = "budget_check"
    QUOTA_CHECK = "quota_check"
    AUDIT = "audit"


@dataclass
class LedgerEvent:
    """A normalized ledger event for audit and usage tracking."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: LedgerEventType = LedgerEventType.MODEL_USAGE
    timestamp: float = field(default_factory=time.time)

    job_id: str | None = None
    run_id: str | None = None
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None

    provider: str | None = None
    model: str | None = None
    tool_name: str | None = None
    connector_name: str | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    cost: str = "0"
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def cost_decimal(self) -> Decimal:
        return Decimal(self.cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "scope_id": self.scope_id,
            "principal_id": self.principal_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "tool_name": self.tool_name,
            "connector_name": self.connector_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cost": self.cost,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEvent:
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            type=LedgerEventType(data.get("type", "model_usage")),
            timestamp=data.get("timestamp", time.time()),
            job_id=data.get("job_id"),
            run_id=data.get("run_id"),
            scope_id=data.get("scope_id"),
            principal_id=data.get("principal_id"),
            session_id=data.get("session_id"),
            provider=data.get("provider"),
            model=data.get("model"),
            tool_name=data.get("tool_name"),
            connector_name=data.get("connector_name"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            cost=data.get("cost", "0"),
            duration_ms=data.get("duration_ms"),
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", 1),
        )

    @classmethod
    def from_llm_usage(
        cls,
        usage: Any,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        scope_id: str | None = None,
        principal_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> LedgerEvent:
        return cls(
            type=LedgerEventType.MODEL_USAGE,
            job_id=job_id,
            run_id=run_id,
            scope_id=scope_id,
            principal_id=principal_id,
            provider=provider,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=getattr(usage, "input_tokens_cached", 0),
            cost=str(usage.total_cost),
        )


@dataclass
class UsageRecord:
    """Aggregated usage for a scope, principal, or session."""

    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    request_count: int = 0
    tool_call_count: int = 0
    first_event_at: float | None = None
    last_event_at: float | None = None

    def add_event(self, event: LedgerEvent) -> None:
        self.total_input_tokens += event.input_tokens
        self.total_output_tokens += event.output_tokens
        self.total_tokens += event.total_tokens
        self.total_cached_tokens += event.cached_tokens
        self.total_cost += event.cost_decimal

        if event.type == LedgerEventType.MODEL_USAGE:
            self.request_count += 1
        elif event.type == LedgerEventType.TOOL_USAGE:
            self.tool_call_count += 1

        if self.first_event_at is None:
            self.first_event_at = event.timestamp
        self.last_event_at = event.timestamp


class BudgetDecision(str, Enum):
    """Result of budget evaluation."""

    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"
    DEGRADE = "degrade"


@dataclass
class Budget:
    """Generic budget definition for one scope/principal pair."""

    scope_id: str | None = None
    principal_id: str | None = None

    max_tokens_per_request: int | None = None
    max_tokens_daily: int | None = None
    max_tokens_monthly: int | None = None
    max_tokens_total: int | None = None

    max_cost_per_request: Decimal | None = None
    max_cost_daily: Decimal | None = None
    max_cost_monthly: Decimal | None = None
    max_cost_total: Decimal | None = None

    max_requests_per_minute: int | None = None
    max_requests_daily: int | None = None

    max_tool_calls_per_request: int | None = None
    max_tool_calls_daily: int | None = None

    warning_threshold: float = 0.8
    exceed_strategy: str = "deny"

    def check(
        self,
        usage: UsageRecord,
        pending_tokens: int = 0,
        pending_cost: Decimal = Decimal("0"),
    ) -> tuple[BudgetDecision, str | None]:
        if self.max_tokens_total is not None:
            total = usage.total_tokens + pending_tokens
            if total > self.max_tokens_total:
                return self._exceed_decision("Total token limit exceeded")
            if total > self.max_tokens_total * self.warning_threshold:
                return BudgetDecision.WARN, "Approaching total token limit"

        if self.max_cost_total is not None:
            total = usage.total_cost + pending_cost
            if total > self.max_cost_total:
                return self._exceed_decision("Total cost limit exceeded")
            if total > self.max_cost_total * Decimal(str(self.warning_threshold)):
                return BudgetDecision.WARN, "Approaching total cost limit"

        return BudgetDecision.ALLOW, None

    def _exceed_decision(self, reason: str) -> tuple[BudgetDecision, str]:
        if self.exceed_strategy == "deny":
            return BudgetDecision.DENY, reason
        if self.exceed_strategy == "degrade":
            return BudgetDecision.DEGRADE, reason
        return BudgetDecision.WARN, reason


class LedgerWriter(ABC):
    """Abstract interface for ledger persistence."""

    @abstractmethod
    async def write(self, event: LedgerEvent) -> None:
        ...

    @abstractmethod
    async def get_usage(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> UsageRecord:
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
        ...


class InMemoryLedgerWriter(LedgerWriter):
    """Simple in-memory ledger writer suitable for tests and single-process use."""

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
            if len(self._events) >= self._max_events:
                self._events = self._events[-self._max_events // 2 :]
                self._rebuild_indexes()

            self._events.append(event)

            if event.scope_id:
                self._by_scope[event.scope_id].append(event)
            if event.principal_id:
                self._by_principal[event.principal_id].append(event)
            if event.job_id:
                self._by_job[event.job_id].append(event)

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
            for event in self._filter_events(scope_id, principal_id, session_id, job_id):
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
            if event_type is not None:
                events = [event for event in events if event.type == event_type]
            events.sort(key=lambda event: event.timestamp, reverse=True)
            return events[offset : offset + limit]

    def _filter_events(
        self,
        scope_id: str | None,
        principal_id: str | None,
        session_id: str | None,
        job_id: str | None,
    ) -> list[LedgerEvent]:
        if job_id:
            events = list(self._by_job.get(job_id, []))
        elif scope_id:
            events = list(self._by_scope.get(scope_id, []))
        elif principal_id:
            events = list(self._by_principal.get(principal_id, []))
        else:
            events = list(self._events)

        if scope_id and not job_id:
            events = [event for event in events if event.scope_id == scope_id]
        if principal_id:
            events = [event for event in events if event.principal_id == principal_id]
        if session_id:
            events = [event for event in events if event.session_id == session_id]
        return events

    def _rebuild_indexes(self) -> None:
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
        keys_to_remove: list[str] = []
        for key in self._usage_cache:
            parts = key.split(":")
            if (
                (event.scope_id and parts[0] == event.scope_id)
                or (event.principal_id and parts[1] == event.principal_id)
                or (event.session_id and parts[2] == event.session_id)
                or (event.job_id and parts[3] == event.job_id)
            ):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._usage_cache[key]


class BudgetExceededError(Exception):
    """Raised when a required budget check denies execution."""

    def __init__(self, reason: str, decision: BudgetDecision):
        self.reason = reason
        self.decision = decision
        super().__init__(f"Budget exceeded: {reason}")


class Ledger:
    """Generic budget and usage ledger."""

    def __init__(self, writer: LedgerWriter | None = None):
        self._writer = writer or InMemoryLedgerWriter()
        self._budgets: dict[str, Budget] = {}

    def set_budget(self, budget: Budget) -> None:
        self._budgets[self._budget_key(budget.scope_id, budget.principal_id)] = budget

    def get_budget(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
    ) -> Budget | None:
        key = self._budget_key(scope_id, principal_id)
        if key in self._budgets:
            return self._budgets[key]

        if principal_id:
            scope_key = self._budget_key(scope_id, None)
            if scope_key in self._budgets:
                return self._budgets[scope_key]

        return self._budgets.get(self._budget_key(None, None))

    async def record_usage(
        self,
        ctx: ExecutionContext,
        usage: Any,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> LedgerEvent:
        event = LedgerEvent.from_llm_usage(
            usage,
            job_id=ctx.job_id,
            run_id=ctx.run_id,
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
            provider=provider,
            model=model,
        )
        event.session_id = ctx.session_id
        await self._writer.write(event)
        return event

    async def record_tool_usage(
        self,
        ctx: ExecutionContext,
        tool_name: str,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        event = LedgerEvent(
            type=LedgerEventType.TOOL_USAGE,
            job_id=ctx.job_id,
            run_id=ctx.run_id,
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
            session_id=ctx.session_id,
            tool_name=tool_name,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        await self._writer.write(event)
        return event

    async def record_connector_usage(
        self,
        ctx: ExecutionContext,
        connector_name: str,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        event = LedgerEvent(
            type=LedgerEventType.CONNECTOR_USAGE,
            job_id=ctx.job_id,
            run_id=ctx.run_id,
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
            session_id=ctx.session_id,
            connector_name=connector_name,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        await self._writer.write(event)
        return event

    async def audit(
        self,
        ctx: ExecutionContext,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        event = LedgerEvent(
            type=LedgerEventType.AUDIT,
            job_id=ctx.job_id,
            run_id=ctx.run_id,
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
            session_id=ctx.session_id,
            metadata={"action": action, **(details or {})},
        )
        await self._writer.write(event)
        return event

    async def check_budget(
        self,
        ctx: ExecutionContext,
        pending_tokens: int = 0,
        pending_cost: Decimal = Decimal("0"),
    ) -> tuple[BudgetDecision, str | None]:
        budget = self.get_budget(ctx.scope_id, ctx.principal_id)
        if budget is None:
            return BudgetDecision.ALLOW, None

        usage = await self._writer.get_usage(
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
        )
        return budget.check(usage, pending_tokens, pending_cost)

    async def require_budget(
        self,
        ctx: ExecutionContext,
        pending_tokens: int = 0,
        pending_cost: Decimal = Decimal("0"),
    ) -> None:
        decision, reason = await self.check_budget(ctx, pending_tokens, pending_cost)
        if decision != BudgetDecision.DENY:
            return

        await self._writer.write(
            LedgerEvent(
                type=LedgerEventType.BUDGET_CHECK,
                job_id=ctx.job_id,
                run_id=ctx.run_id,
                scope_id=ctx.scope_id,
                principal_id=ctx.principal_id,
                session_id=ctx.session_id,
                metadata={
                    "decision": "deny",
                    "reason": reason,
                    "pending_tokens": pending_tokens,
                    "pending_cost": str(pending_cost),
                },
            )
        )
        raise BudgetExceededError(reason or "Budget exceeded", decision)

    async def get_usage(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> UsageRecord:
        return await self._writer.get_usage(
            scope_id=scope_id,
            principal_id=principal_id,
            session_id=session_id,
            job_id=job_id,
        )

    async def list_events(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        job_id: str | None = None,
        event_type: LedgerEventType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LedgerEvent]:
        return await self._writer.list_events(
            scope_id=scope_id,
            principal_id=principal_id,
            job_id=job_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def _budget_key(scope_id: str | None, principal_id: str | None) -> str:
        return f"{scope_id or '*'}:{principal_id or '*'}"


__all__ = [
    "Budget",
    "BudgetDecision",
    "BudgetExceededError",
    "InMemoryLedgerWriter",
    "Ledger",
    "LedgerEvent",
    "LedgerEventType",
    "LedgerWriter",
    "UsageRecord",
]
