"""
Ledger types for usage tracking and budget management.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class LedgerEventType(str, Enum):
    """Types of ledger events."""
    MODEL_USAGE = "model_usage"      # LLM token usage
    TOOL_USAGE = "tool_usage"        # Tool execution
    CONNECTOR_USAGE = "connector_usage"  # External connector call
    BUDGET_CHECK = "budget_check"    # Budget enforcement
    QUOTA_CHECK = "quota_check"      # Quota check
    AUDIT = "audit"                  # General audit event


@dataclass
class LedgerEvent:
    """A ledger event for audit and tracking.
    
    Every usage or budget-related action generates a ledger event.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: LedgerEventType = LedgerEventType.MODEL_USAGE
    timestamp: float = field(default_factory=time.time)
    
    # Identity
    job_id: str | None = None
    run_id: str | None = None
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    
    # Usage details
    provider: str | None = None
    model: str | None = None
    tool_name: str | None = None
    connector_name: str | None = None
    
    # Metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    
    # Cost (use Decimal for precision, stored as string)
    cost: str = "0"  # Decimal as string for JSON serialization
    
    # Duration
    duration_ms: float | None = None
    
    # Additional data
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Schema version
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
        usage: Any,  # llm_client.Usage
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        scope_id: str | None = None,
        principal_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> LedgerEvent:
        """Create a ledger event from llm_client Usage."""
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
    """Aggregated usage for a scope/principal/session."""
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    
    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    
    # Cost
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # Counts
    request_count: int = 0
    tool_call_count: int = 0
    
    # Time range
    first_event_at: float | None = None
    last_event_at: float | None = None
    
    def add_event(self, event: LedgerEvent) -> None:
        """Add a ledger event to this record."""
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
    """Result of a budget check."""
    ALLOW = "allow"        # Within budget
    WARN = "warn"          # Near budget limit
    DENY = "deny"          # Budget exceeded
    DEGRADE = "degrade"    # Allow with degraded service


@dataclass
class Budget:
    """Budget configuration for a scope/principal.
    
    Budgets can be defined at different granularities:
    - Per request
    - Daily
    - Monthly
    - Total
    """
    scope_id: str | None = None
    principal_id: str | None = None
    
    # Token limits
    max_tokens_per_request: int | None = None
    max_tokens_daily: int | None = None
    max_tokens_monthly: int | None = None
    max_tokens_total: int | None = None
    
    # Cost limits
    max_cost_per_request: Decimal | None = None
    max_cost_daily: Decimal | None = None
    max_cost_monthly: Decimal | None = None
    max_cost_total: Decimal | None = None
    
    # Request limits
    max_requests_per_minute: int | None = None
    max_requests_daily: int | None = None
    
    # Tool call limits
    max_tool_calls_per_request: int | None = None
    max_tool_calls_daily: int | None = None
    
    # Warning thresholds (0.0 - 1.0)
    warning_threshold: float = 0.8
    
    # Strategy when exceeded
    exceed_strategy: str = "deny"  # "deny", "degrade", "warn"

    def check(
        self,
        usage: UsageRecord,
        pending_tokens: int = 0,
        pending_cost: Decimal = Decimal("0"),
    ) -> tuple[BudgetDecision, str | None]:
        """Check if the pending operation fits within budget.
        
        Returns:
            Tuple of (decision, reason)
        """
        # Check token limits
        if self.max_tokens_total is not None:
            total = usage.total_tokens + pending_tokens
            if total > self.max_tokens_total:
                return self._exceed_decision("Total token limit exceeded")
            if total > self.max_tokens_total * self.warning_threshold:
                return BudgetDecision.WARN, "Approaching total token limit"
        
        # Check cost limits
        if self.max_cost_total is not None:
            total = usage.total_cost + pending_cost
            if total > self.max_cost_total:
                return self._exceed_decision("Total cost limit exceeded")
            if total > self.max_cost_total * Decimal(str(self.warning_threshold)):
                return BudgetDecision.WARN, "Approaching total cost limit"
        
        # Check request limits
        if self.max_requests_daily is not None:
            if usage.request_count >= self.max_requests_daily:
                return self._exceed_decision("Daily request limit exceeded")
        
        return BudgetDecision.ALLOW, None
    
    def _exceed_decision(self, reason: str) -> tuple[BudgetDecision, str]:
        """Get the decision for exceeded budget based on strategy."""
        if self.exceed_strategy == "deny":
            return BudgetDecision.DENY, reason
        elif self.exceed_strategy == "degrade":
            return BudgetDecision.DEGRADE, reason
        else:
            return BudgetDecision.WARN, reason


__all__ = [
    "LedgerEvent",
    "LedgerEventType",
    "UsageRecord",
    "Budget",
    "BudgetDecision",
]
