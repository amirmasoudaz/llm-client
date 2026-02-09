"""
Ledger for usage tracking and budget enforcement.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..context import ExecutionContext
from .types import (
    LedgerEvent,
    LedgerEventType,
    UsageRecord,
    Budget,
    BudgetDecision,
)
from .writer import LedgerWriter, InMemoryLedgerWriter


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""
    def __init__(self, reason: str, decision: BudgetDecision):
        self.reason = reason
        self.decision = decision
        super().__init__(f"Budget exceeded: {reason}")


class Ledger:
    """Central ledger for usage tracking and budget enforcement.
    
    The Ledger:
    - Records all usage events
    - Tracks aggregated usage by scope/principal/session
    - Enforces budgets and quotas
    - Provides usage queries for billing/analytics
    """
    
    def __init__(
        self,
        writer: LedgerWriter | None = None,
    ):
        self._writer = writer or InMemoryLedgerWriter()
        self._budgets: dict[str, Budget] = {}  # scope:principal -> budget
    
    def set_budget(
        self,
        budget: Budget,
    ) -> None:
        """Set a budget for a scope/principal."""
        key = self._budget_key(budget.scope_id, budget.principal_id)
        self._budgets[key] = budget
    
    def get_budget(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
    ) -> Budget | None:
        """Get the budget for a scope/principal."""
        # Try exact match first
        key = self._budget_key(scope_id, principal_id)
        if key in self._budgets:
            return self._budgets[key]
        
        # Try scope-only
        if principal_id:
            scope_key = self._budget_key(scope_id, None)
            if scope_key in self._budgets:
                return self._budgets[scope_key]
        
        # Try global
        global_key = self._budget_key(None, None)
        return self._budgets.get(global_key)
    
    async def record_usage(
        self,
        ctx: ExecutionContext,
        usage: Any,  # llm_client.Usage
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> LedgerEvent:
        """Record model usage from an execution.
        
        Args:
            ctx: Execution context for correlation
            usage: Usage object from llm_client
            provider: Provider name
            model: Model name
            
        Returns:
            The created ledger event
        """
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
        """Record tool usage."""
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
        """Record connector usage."""
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
        """Record an audit event."""
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
        """Check if pending operation fits within budget.
        
        Args:
            ctx: Execution context
            pending_tokens: Estimated tokens for pending operation
            pending_cost: Estimated cost for pending operation
            
        Returns:
            Tuple of (decision, reason)
        """
        budget = self.get_budget(ctx.scope_id, ctx.principal_id)
        if not budget:
            return BudgetDecision.ALLOW, None
        
        # Get current usage
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
        """Check budget and raise if exceeded.
        
        Raises:
            BudgetExceededError: If budget is exceeded
        """
        decision, reason = await self.check_budget(ctx, pending_tokens, pending_cost)
        
        if decision == BudgetDecision.DENY:
            # Record the denial
            await self._writer.write(LedgerEvent(
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
            ))
            raise BudgetExceededError(reason or "Budget exceeded", decision)
    
    async def get_usage(
        self,
        scope_id: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> UsageRecord:
        """Get aggregated usage."""
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
        """List ledger events."""
        return await self._writer.list_events(
            scope_id=scope_id,
            principal_id=principal_id,
            job_id=job_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
    
    def _budget_key(
        self,
        scope_id: str | None,
        principal_id: str | None,
    ) -> str:
        return f"{scope_id or ''}:{principal_id or ''}"


__all__ = [
    "Ledger",
    "BudgetExceededError",
]
