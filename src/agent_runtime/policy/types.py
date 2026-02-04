"""
Policy types for agent runtime.

This module defines the policy data structures used for
centralized policy enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PolicyDecision(str, Enum):
    """Result of a policy check."""
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"  # Allow but log a warning


class PolicyDenied(Exception):
    """Exception raised when a policy check fails."""
    def __init__(
        self,
        reason: str,
        policy_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.reason = reason
        self.policy_name = policy_name
        self.details = details or {}
        super().__init__(f"Policy denied: {reason}")


@dataclass
class PolicyRule:
    """Base class for policy rules.
    
    A policy rule evaluates a specific aspect of a request
    and returns a decision.
    """
    name: str
    description: str = ""
    enabled: bool = True
    
    def evaluate(self, context: dict[str, Any]) -> tuple[PolicyDecision, str | None]:
        """Evaluate the rule against the given context.
        
        Returns:
            Tuple of (decision, reason). Reason is set if denied.
        """
        return PolicyDecision.ALLOW, None


@dataclass
class ToolPolicy(PolicyRule):
    """Policy for tool access control.
    
    Supports allowlists and denylists, with optional per-tenant
    or per-user overrides.
    """
    allowed_tools: set[str] | None = None  # None = all allowed
    denied_tools: set[str] = field(default_factory=set)
    
    # Per-scope/principal overrides
    scope_overrides: dict[str, set[str]] = field(default_factory=dict)  # scope_id -> allowed tools
    principal_overrides: dict[str, set[str]] = field(default_factory=dict)  # principal_id -> allowed tools
    
    # Argument validators (tool_name -> validator function)
    argument_validators: dict[str, Callable[[dict], bool]] = field(default_factory=dict)

    def evaluate(self, context: dict[str, Any]) -> tuple[PolicyDecision, str | None]:
        tool_name = context.get("tool_name")
        if not tool_name:
            return PolicyDecision.ALLOW, None
        
        # Check denylist first (highest priority)
        if tool_name in self.denied_tools:
            return PolicyDecision.DENY, f"Tool '{tool_name}' is blocked by policy"
        
        # Check scope overrides
        scope_id = context.get("scope_id")
        if scope_id and scope_id in self.scope_overrides:
            if tool_name not in self.scope_overrides[scope_id]:
                return PolicyDecision.DENY, f"Tool '{tool_name}' not allowed for scope"
        
        # Check principal overrides
        principal_id = context.get("principal_id")
        if principal_id and principal_id in self.principal_overrides:
            if tool_name not in self.principal_overrides[principal_id]:
                return PolicyDecision.DENY, f"Tool '{tool_name}' not allowed for user"
        
        # Check allowlist
        if self.allowed_tools is not None:
            if tool_name not in self.allowed_tools:
                return PolicyDecision.DENY, f"Tool '{tool_name}' is not in allowlist"
        
        # Check argument validators
        if tool_name in self.argument_validators:
            arguments = context.get("arguments", {})
            if not self.argument_validators[tool_name](arguments):
                return PolicyDecision.DENY, f"Tool '{tool_name}' arguments rejected by policy"
        
        return PolicyDecision.ALLOW, None


@dataclass
class ModelPolicy(PolicyRule):
    """Policy for model access control."""
    allowed_models: set[str] | None = None  # None = all allowed
    denied_models: set[str] = field(default_factory=set)
    
    # Per-scope overrides
    scope_overrides: dict[str, set[str]] = field(default_factory=dict)
    
    # Model-specific constraints
    max_tokens_per_model: dict[str, int] = field(default_factory=dict)
    
    def evaluate(self, context: dict[str, Any]) -> tuple[PolicyDecision, str | None]:
        model_name = context.get("model_name")
        if not model_name:
            return PolicyDecision.ALLOW, None
        
        # Check denylist
        if model_name in self.denied_models:
            return PolicyDecision.DENY, f"Model '{model_name}' is blocked by policy"
        
        # Check scope overrides
        scope_id = context.get("scope_id")
        if scope_id and scope_id in self.scope_overrides:
            if model_name not in self.scope_overrides[scope_id]:
                return PolicyDecision.DENY, f"Model '{model_name}' not allowed for scope"
        
        # Check allowlist
        if self.allowed_models is not None:
            if model_name not in self.allowed_models:
                return PolicyDecision.DENY, f"Model '{model_name}' is not in allowlist"
        
        return PolicyDecision.ALLOW, None


@dataclass
class ConstraintPolicy(PolicyRule):
    """Policy for execution constraints."""
    max_turns: int | None = None
    max_tool_calls: int | None = None
    max_tool_calls_per_turn: int | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    max_reasoning_effort: str | None = None  # "low", "medium", "high"
    max_runtime_seconds: float | None = None
    
    def evaluate(self, context: dict[str, Any]) -> tuple[PolicyDecision, str | None]:
        # Check turn limit
        if self.max_turns is not None:
            current_turn = context.get("current_turn", 0)
            if current_turn >= self.max_turns:
                return PolicyDecision.DENY, f"Maximum turns ({self.max_turns}) exceeded"
        
        # Check tool call limits
        if self.max_tool_calls is not None:
            total_tool_calls = context.get("total_tool_calls", 0)
            if total_tool_calls >= self.max_tool_calls:
                return PolicyDecision.DENY, f"Maximum tool calls ({self.max_tool_calls}) exceeded"
        
        if self.max_tool_calls_per_turn is not None:
            turn_tool_calls = context.get("turn_tool_calls", 0)
            if turn_tool_calls >= self.max_tool_calls_per_turn:
                return PolicyDecision.DENY, f"Maximum tool calls per turn ({self.max_tool_calls_per_turn}) exceeded"
        
        # Check context size
        if self.max_context_tokens is not None:
            context_tokens = context.get("context_tokens", 0)
            if context_tokens > self.max_context_tokens:
                return PolicyDecision.DENY, f"Context size ({context_tokens}) exceeds limit ({self.max_context_tokens})"
        
        return PolicyDecision.ALLOW, None


@dataclass
class DataAccessPolicy(PolicyRule):
    """Policy for data source access control."""
    allowed_connectors: set[str] | None = None
    denied_connectors: set[str] = field(default_factory=set)
    
    # Per-scope connector access
    scope_connectors: dict[str, set[str]] = field(default_factory=dict)
    
    def evaluate(self, context: dict[str, Any]) -> tuple[PolicyDecision, str | None]:
        connector_name = context.get("connector_name")
        if not connector_name:
            return PolicyDecision.ALLOW, None
        
        if connector_name in self.denied_connectors:
            return PolicyDecision.DENY, f"Connector '{connector_name}' is blocked"
        
        scope_id = context.get("scope_id")
        if scope_id and scope_id in self.scope_connectors:
            if connector_name not in self.scope_connectors[scope_id]:
                return PolicyDecision.DENY, f"Connector '{connector_name}' not allowed for scope"
        
        if self.allowed_connectors is not None:
            if connector_name not in self.allowed_connectors:
                return PolicyDecision.DENY, f"Connector '{connector_name}' is not in allowlist"
        
        return PolicyDecision.ALLOW, None


@dataclass
class RedactionPolicy(PolicyRule):
    """Policy for PII/secret redaction rules."""
    redact_emails: bool = True
    redact_ssn: bool = True
    redact_api_keys: bool = True
    redact_phone_numbers: bool = False
    redact_credit_cards: bool = True
    custom_patterns: list[str] = field(default_factory=list)
    
    # Logging settings
    log_redacted_content: bool = False
    
    def evaluate(self, context: dict[str, Any]) -> tuple[PolicyDecision, str | None]:
        # Redaction policy doesn't deny, it just configures redaction behavior
        return PolicyDecision.ALLOW, None
    
    def get_patterns(self) -> list[str]:
        """Get all active redaction patterns."""
        patterns = list(self.custom_patterns)
        
        if self.redact_emails:
            patterns.append(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        if self.redact_ssn:
            patterns.append(r'\b\d{3}-\d{2}-\d{4}\b')
        if self.redact_api_keys:
            patterns.append(r'\b(?:sk-|api_key[=:]\s*)[A-Za-z0-9\-_]+')
        if self.redact_phone_numbers:
            patterns.append(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        if self.redact_credit_cards:
            patterns.append(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
        
        return patterns


__all__ = [
    "PolicyDecision",
    "PolicyDenied",
    "PolicyRule",
    "ToolPolicy",
    "ModelPolicy",
    "ConstraintPolicy",
    "DataAccessPolicy",
    "RedactionPolicy",
]
