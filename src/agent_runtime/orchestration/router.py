"""
Router for selecting operators.

This module provides:
- Router: Abstract interface for routing decisions
- RoutingRule: Rule-based routing configuration
- DefaultRouter: Simple rule-based router implementation
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Operator, RoutingDecision, AgentRole


@dataclass
class RoutingRule:
    """A rule for routing requests to operators.
    
    Rules are evaluated in order; first match wins.
    
    Attributes:
        operator_name: Target operator name
        patterns: Regex patterns to match against input text
        keywords: Keywords to match (case-insensitive)
        input_fields: Required input fields
        min_confidence: Minimum confidence for this rule
        priority: Rule priority (higher = evaluated first)
        condition: Custom condition function
    """
    operator_name: str
    patterns: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    input_fields: list[str] = field(default_factory=list)
    min_confidence: float = 0.5
    priority: int = 0
    condition: Callable[[dict[str, Any]], bool] | None = None
    
    def matches(self, input_data: dict[str, Any]) -> tuple[bool, float]:
        """Check if this rule matches the input.
        
        Returns:
            Tuple of (matches, confidence)
        """
        confidence = 0.0
        match_count = 0
        total_checks = 0
        
        # Check required input fields
        if self.input_fields:
            total_checks += 1
            if all(f in input_data for f in self.input_fields):
                match_count += 1
        
        # Check keyword matches
        text = str(input_data.get("text", "")) + str(input_data.get("prompt", ""))
        text_lower = text.lower()
        
        if self.keywords:
            total_checks += 1
            matched_keywords = sum(1 for kw in self.keywords if kw.lower() in text_lower)
            if matched_keywords > 0:
                match_count += matched_keywords / len(self.keywords)
        
        # Check pattern matches
        if self.patterns:
            total_checks += 1
            matched_patterns = sum(
                1 for p in self.patterns
                if re.search(p, text, re.IGNORECASE)
            )
            if matched_patterns > 0:
                match_count += matched_patterns / len(self.patterns)
        
        # Check custom condition
        if self.condition:
            total_checks += 1
            try:
                if self.condition(input_data):
                    match_count += 1
            except Exception:
                pass
        
        # Calculate confidence
        if total_checks > 0:
            confidence = match_count / total_checks
        else:
            # No checks = always matches with min confidence
            confidence = self.min_confidence
        
        matches = confidence >= self.min_confidence
        return matches, confidence


class Router(ABC):
    """Abstract interface for routing requests to operators.
    
    A router examines the input and determines which operator(s)
    should handle the request.
    """
    
    @abstractmethod
    async def route(
        self,
        input_data: dict[str, Any],
        available_operators: dict[str, Operator],
    ) -> RoutingDecision:
        """Route a request to an operator.
        
        Args:
            input_data: Input payload to route
            available_operators: Dict of operator_name -> Operator
        
        Returns:
            RoutingDecision with selected operator(s)
        """
        ...
    
    async def route_multi(
        self,
        input_data: dict[str, Any],
        available_operators: dict[str, Operator],
        max_operators: int = 3,
    ) -> list[RoutingDecision]:
        """Route to multiple operators (for parallel execution).
        
        Default implementation calls route() once.
        Override for more sophisticated multi-routing.
        """
        decision = await self.route(input_data, available_operators)
        return [decision]


class DefaultRouter(Router):
    """Simple rule-based router.
    
    Evaluates rules in priority order and returns the first match.
    If no rules match, falls back to a default operator.
    
    Example:
        ```python
        router = DefaultRouter(
            rules=[
                RoutingRule(
                    operator_name="summarizer",
                    keywords=["summarize", "summary", "brief"],
                ),
                RoutingRule(
                    operator_name="analyzer",
                    keywords=["analyze", "analysis", "examine"],
                ),
            ],
            default_operator="general",
        )
        
        decision = await router.route(
            {"prompt": "Please summarize this text..."},
            operators,
        )
        # decision.operator_name == "summarizer"
        ```
    """
    
    def __init__(
        self,
        rules: list[RoutingRule] | None = None,
        default_operator: str = "default",
        default_confidence: float = 0.5,
    ):
        self._rules = sorted(
            rules or [],
            key=lambda r: r.priority,
            reverse=True,
        )
        self._default_operator = default_operator
        self._default_confidence = default_confidence
    
    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
    
    async def route(
        self,
        input_data: dict[str, Any],
        available_operators: dict[str, Operator],
    ) -> RoutingDecision:
        """Route using rules in priority order."""
        best_match: tuple[str, float, str | None] | None = None
        fallbacks: list[str] = []
        
        for rule in self._rules:
            # Skip if operator not available
            if rule.operator_name not in available_operators:
                continue
            
            matches, confidence = rule.matches(input_data)
            
            if matches:
                if best_match is None or confidence > best_match[1]:
                    if best_match is not None:
                        fallbacks.append(best_match[0])
                    best_match = (rule.operator_name, confidence, None)
                else:
                    fallbacks.append(rule.operator_name)
        
        if best_match is not None:
            return RoutingDecision(
                operator_name=best_match[0],
                confidence=best_match[1],
                reasoning=best_match[2],
                fallback_operators=fallbacks[:3],  # Top 3 fallbacks
            )
        
        # Fall back to default
        if self._default_operator in available_operators:
            return RoutingDecision(
                operator_name=self._default_operator,
                confidence=self._default_confidence,
                reasoning="No matching rules; using default operator",
            )
        
        # Use first available operator
        if available_operators:
            first_name = next(iter(available_operators.keys()))
            return RoutingDecision(
                operator_name=first_name,
                confidence=0.1,
                reasoning="Fallback to first available operator",
            )
        
        raise ValueError("No operators available for routing")


class LLMRouter(Router):
    """Router that uses an LLM to make routing decisions.
    
    This router prompts an LLM to analyze the input and select
    the most appropriate operator based on operator descriptions.
    
    Note: Requires llm_client to be available.
    """
    
    def __init__(
        self,
        agent: Any,  # llm_client Agent
        system_prompt: str | None = None,
    ):
        self._agent = agent
        self._system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """You are a routing assistant. Given a user request and available operators,
select the most appropriate operator to handle the request.

Respond with JSON in this format:
{
  "operator": "operator_name",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""
    
    async def route(
        self,
        input_data: dict[str, Any],
        available_operators: dict[str, Operator],
    ) -> RoutingDecision:
        """Route using LLM to analyze request."""
        import json
        
        # Build operator descriptions
        operator_desc = "\n".join([
            f"- {name}: {op.description} (role: {op.role.value})"
            for name, op in available_operators.items()
        ])
        
        prompt = f"""User request:
{input_data.get('prompt', input_data.get('text', str(input_data)))}

Available operators:
{operator_desc}

Select the best operator for this request."""
        
        result = await self._agent.run(prompt)
        
        try:
            # Parse LLM response as JSON
            response_text = result.content or ""
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                operator_name = data.get("operator", "")
                
                if operator_name in available_operators:
                    return RoutingDecision(
                        operator_name=operator_name,
                        confidence=float(data.get("confidence", 0.8)),
                        reasoning=data.get("reasoning"),
                    )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Fallback to first available
        if available_operators:
            return RoutingDecision(
                operator_name=next(iter(available_operators.keys())),
                confidence=0.5,
                reasoning="LLM routing failed; using fallback",
            )
        
        raise ValueError("No operators available")


__all__ = [
    "Router",
    "RoutingRule",
    "DefaultRouter",
    "LLMRouter",
]
