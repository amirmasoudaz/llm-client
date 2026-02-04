"""
Policy engine for centralized policy evaluation.

This module provides the PolicyEngine that evaluates all policies
and returns a unified decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..context import ExecutionContext
from .types import (
    PolicyDecision,
    PolicyDenied,
    PolicyRule,
    ToolPolicy,
    ModelPolicy,
    ConstraintPolicy,
    RedactionPolicy,
)


@dataclass
class PolicyContext:
    """Context for policy evaluation.
    
    This aggregates all the information needed for policy checks
    into a single object that can be passed to the policy engine.
    """
    # Identity
    scope_id: str | None = None
    principal_id: str | None = None
    session_id: str | None = None
    
    # Tool context
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    
    # Model context
    model_name: str | None = None
    
    # Connector context
    connector_name: str | None = None
    
    # Execution context
    current_turn: int = 0
    total_tool_calls: int = 0
    turn_tool_calls: int = 0
    context_tokens: int = 0
    
    # Additional context
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for policy evaluation."""
        return {
            "scope_id": self.scope_id,
            "principal_id": self.principal_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "model_name": self.model_name,
            "connector_name": self.connector_name,
            "current_turn": self.current_turn,
            "total_tool_calls": self.total_tool_calls,
            "turn_tool_calls": self.turn_tool_calls,
            "context_tokens": self.context_tokens,
            **self.extra,
        }

    @classmethod
    def from_execution_context(
        cls,
        ctx: ExecutionContext,
        **kwargs: Any,
    ) -> PolicyContext:
        """Create a PolicyContext from an ExecutionContext."""
        return cls(
            scope_id=ctx.scope_id,
            principal_id=ctx.principal_id,
            session_id=ctx.session_id,
            **kwargs,
        )


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    decision: PolicyDecision
    reason: str | None = None
    policy_name: str | None = None
    warnings: list[str] = field(default_factory=list)
    
    # Approved constraints (from constraint policies)
    approved_max_turns: int | None = None
    approved_max_tool_calls: int | None = None
    approved_max_tokens: int | None = None
    
    # Approved tools/models (intersection of allowlists)
    approved_tools: set[str] | None = None
    approved_models: set[str] | None = None
    
    # Redaction config
    redaction_patterns: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.decision in {PolicyDecision.ALLOW, PolicyDecision.WARN}


class PolicyEngine:
    """Centralized policy engine for the runtime.
    
    The policy engine:
    - Aggregates multiple policy rules
    - Evaluates them in order
    - Returns a unified decision
    - Caches policy results for performance
    
    Policies can be registered globally or loaded from a policy store.
    """
    
    def __init__(self):
        self._tool_policies: list[ToolPolicy] = []
        self._model_policies: list[ModelPolicy] = []
        self._constraint_policies: list[ConstraintPolicy] = []
        self._redaction_policies: list[RedactionPolicy] = []
        self._custom_policies: list[PolicyRule] = []
    
    def add_tool_policy(self, policy: ToolPolicy) -> PolicyEngine:
        """Add a tool access policy."""
        self._tool_policies.append(policy)
        return self
    
    def add_model_policy(self, policy: ModelPolicy) -> PolicyEngine:
        """Add a model access policy."""
        self._model_policies.append(policy)
        return self
    
    def add_constraint_policy(self, policy: ConstraintPolicy) -> PolicyEngine:
        """Add an execution constraint policy."""
        self._constraint_policies.append(policy)
        return self
    
    def add_redaction_policy(self, policy: RedactionPolicy) -> PolicyEngine:
        """Add a redaction policy."""
        self._redaction_policies.append(policy)
        return self
    
    def add_policy(self, policy: PolicyRule) -> PolicyEngine:
        """Add a custom policy rule."""
        self._custom_policies.append(policy)
        return self
    
    def check_tool(
        self,
        tool_name: str,
        ctx: PolicyContext | ExecutionContext,
        arguments: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """Check if a tool is allowed.
        
        Args:
            tool_name: Name of the tool to check
            ctx: Policy context or execution context
            arguments: Optional tool arguments to validate
            
        Returns:
            PolicyResult with decision and reason
            
        Raises:
            PolicyDenied: If fail_on_deny is True and policy denies
        """
        policy_ctx = self._ensure_policy_context(ctx)
        policy_ctx.tool_name = tool_name
        if arguments:
            policy_ctx.arguments = arguments
        
        context_dict = policy_ctx.to_dict()
        warnings = []
        
        for policy in self._tool_policies:
            if not policy.enabled:
                continue
            
            decision, reason = policy.evaluate(context_dict)
            
            if decision == PolicyDecision.DENY:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=reason,
                    policy_name=policy.name,
                )
            elif decision == PolicyDecision.WARN:
                warnings.append(f"{policy.name}: {reason}")
        
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            warnings=warnings,
        )
    
    def check_model(
        self,
        model_name: str,
        ctx: PolicyContext | ExecutionContext,
    ) -> PolicyResult:
        """Check if a model is allowed."""
        policy_ctx = self._ensure_policy_context(ctx)
        policy_ctx.model_name = model_name
        
        context_dict = policy_ctx.to_dict()
        warnings = []
        
        for policy in self._model_policies:
            if not policy.enabled:
                continue
            
            decision, reason = policy.evaluate(context_dict)
            
            if decision == PolicyDecision.DENY:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=reason,
                    policy_name=policy.name,
                )
            elif decision == PolicyDecision.WARN:
                warnings.append(f"{policy.name}: {reason}")
        
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            warnings=warnings,
        )
    
    def check_constraints(
        self,
        ctx: PolicyContext | ExecutionContext,
    ) -> PolicyResult:
        """Check execution constraints."""
        policy_ctx = self._ensure_policy_context(ctx)
        context_dict = policy_ctx.to_dict()
        
        warnings = []
        approved_max_turns = None
        approved_max_tool_calls = None
        
        for policy in self._constraint_policies:
            if not policy.enabled:
                continue
            
            decision, reason = policy.evaluate(context_dict)
            
            if decision == PolicyDecision.DENY:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=reason,
                    policy_name=policy.name,
                )
            elif decision == PolicyDecision.WARN:
                warnings.append(f"{policy.name}: {reason}")
            
            # Track approved constraints (most restrictive wins)
            if policy.max_turns is not None:
                if approved_max_turns is None or policy.max_turns < approved_max_turns:
                    approved_max_turns = policy.max_turns
            if policy.max_tool_calls is not None:
                if approved_max_tool_calls is None or policy.max_tool_calls < approved_max_tool_calls:
                    approved_max_tool_calls = policy.max_tool_calls
        
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            warnings=warnings,
            approved_max_turns=approved_max_turns,
            approved_max_tool_calls=approved_max_tool_calls,
        )
    
    def evaluate(
        self,
        ctx: PolicyContext | ExecutionContext,
    ) -> PolicyResult:
        """Evaluate all policies and return a unified result.
        
        This is the main entry point for comprehensive policy checks.
        """
        policy_ctx = self._ensure_policy_context(ctx)
        context_dict = policy_ctx.to_dict()
        
        warnings = []
        
        # Evaluate all policy types
        all_policies: list[PolicyRule] = [
            *self._tool_policies,
            *self._model_policies,
            *self._constraint_policies,
            *self._custom_policies,
        ]
        
        for policy in all_policies:
            if not policy.enabled:
                continue
            
            decision, reason = policy.evaluate(context_dict)
            
            if decision == PolicyDecision.DENY:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=reason,
                    policy_name=policy.name,
                )
            elif decision == PolicyDecision.WARN:
                warnings.append(f"{policy.name}: {reason}")
        
        # Collect redaction patterns
        redaction_patterns = []
        for policy in self._redaction_policies:
            if policy.enabled:
                redaction_patterns.extend(policy.get_patterns())
        
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            warnings=warnings,
            redaction_patterns=redaction_patterns,
        )
    
    def require_allowed(
        self,
        ctx: PolicyContext | ExecutionContext,
    ) -> PolicyResult:
        """Evaluate policies and raise PolicyDenied if denied."""
        result = self.evaluate(ctx)
        if result.decision == PolicyDecision.DENY:
            raise PolicyDenied(
                reason=result.reason or "Policy denied",
                policy_name=result.policy_name,
            )
        return result
    
    def _ensure_policy_context(
        self,
        ctx: PolicyContext | ExecutionContext,
    ) -> PolicyContext:
        """Convert to PolicyContext if needed."""
        if isinstance(ctx, PolicyContext):
            return ctx
        return PolicyContext.from_execution_context(ctx)
    
    @classmethod
    def default(cls) -> PolicyEngine:
        """Create a policy engine with sensible defaults."""
        engine = cls()
        
        # Default constraint policy
        engine.add_constraint_policy(ConstraintPolicy(
            name="default_constraints",
            description="Default execution constraints",
            max_turns=50,
            max_tool_calls=100,
            max_tool_calls_per_turn=20,
        ))
        
        # Default redaction policy
        engine.add_redaction_policy(RedactionPolicy(
            name="default_redaction",
            description="Default PII redaction",
        ))
        
        return engine


__all__ = [
    "PolicyEngine",
    "PolicyContext",
    "PolicyResult",
]
