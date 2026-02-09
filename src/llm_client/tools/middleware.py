"""Middleware chain for tool execution with production-ready implementations.

This module provides a middleware pattern for tool execution, allowing
cross-cutting concerns like logging, authorization, rate limiting, and
retries to be composed declaratively.

Key design:
- ToolMiddleware is an abstract base class
- MiddlewareChain composes middlewares with canonical ordering
- Production-ready middleware set included
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Tool, ToolResult
    from ..spec import RequestContext


@dataclass
class ToolExecutionContext:
    """Context passed through middleware chain."""
    tool: Tool
    arguments: dict[str, Any]
    request_context: RequestContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


NextHandler = Callable[[ToolExecutionContext], Awaitable["ToolResult"]]


class ToolMiddleware(ABC):
    """Base class for tool middleware.
    
    Middleware wraps tool execution to add cross-cutting concerns.
    Call next(ctx) to continue the chain.
    """
    
    @abstractmethod
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler) -> "ToolResult":
        """Execute middleware logic.
        
        Args:
            ctx: Execution context with tool, arguments, and metadata.
            next: The next handler in the chain (call to continue).
        
        Returns:
            ToolResult from execution.
        """
        ...


# === Table Stakes Middleware ===


class LoggingMiddleware(ToolMiddleware):
    """Logs tool execution with timing and correlation."""
    
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("llm_client.tools")
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        start = time.monotonic()
        request_id = getattr(ctx.request_context, "request_id", "no-ctx")
        self.logger.info(f"[{request_id}] Tool {ctx.tool.name} starting with args: {list(ctx.arguments.keys())}")
        
        try:
            result = await next(ctx)
            duration_ms = (time.monotonic() - start) * 1000
            level = logging.INFO if result.success else logging.WARNING
            self.logger.log(level, f"[{request_id}] Tool {ctx.tool.name} completed in {duration_ms:.1f}ms (success={result.success})")
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            self.logger.error(f"[{request_id}] Tool {ctx.tool.name} failed after {duration_ms:.1f}ms: {e}")
            raise


class TimeoutMiddleware(ToolMiddleware):
    """Enforces execution timeout."""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default = default_timeout
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        timeout = ctx.metadata.get("timeout", self.default)
        try:
            return await asyncio.wait_for(next(ctx), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult.error_result(f"Tool {ctx.tool.name} timed out after {timeout}s")


class RetryMiddleware(ToolMiddleware):
    """Retries failed executions with exponential backoff."""
    
    def __init__(self, max_retries: int = 2, base_backoff: float = 0.5):
        self.max_retries = max_retries
        self.base_backoff = base_backoff
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            result = await next(ctx)
            if result.success:
                return result
            last_error = result.error
            if attempt < self.max_retries:
                await asyncio.sleep(self.base_backoff * (2 ** attempt))
        return ToolResult.error_result(f"Failed after {self.max_retries + 1} attempts: {last_error}")


# === Production Middleware ===


class PolicyMiddleware(ToolMiddleware):
    """Tool allowlist, argument constraints, tenant gating."""
    
    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
        argument_validators: dict[str, Callable[[dict], bool]] | None = None,
    ):
        self.allowed = allowed_tools
        self.denied = denied_tools or set()
        self.validators = argument_validators or {}
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        name = ctx.tool.name
        if name in self.denied:
            return ToolResult.error_result(f"Tool '{name}' is blocked by policy")
        if self.allowed is not None and name not in self.allowed:
            return ToolResult.error_result(f"Tool '{name}' is not in allowlist")
        if name in self.validators and not self.validators[name](ctx.arguments):
            return ToolResult.error_result(f"Tool '{name}' arguments rejected by policy")
        return await next(ctx)


class BudgetMiddleware(ToolMiddleware):
    """Enforce per-request budgets (max calls, runtime, payload size)."""
    
    def __init__(self, max_tool_calls: int = 50, max_payload_bytes: int = 100_000):
        self.max_calls = max_tool_calls
        self.max_payload = max_payload_bytes
        self._call_counts: dict[str, int] = {}
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        rid = getattr(ctx.request_context, "request_id", "global")
        self._call_counts[rid] = self._call_counts.get(rid, 0) + 1
        
        if self._call_counts[rid] > self.max_calls:
            return ToolResult.error_result(f"Budget exceeded: max {self.max_calls} tool calls per request")
        
        payload_size = len(json.dumps(ctx.arguments).encode())
        if payload_size > self.max_payload:
            return ToolResult.error_result(f"Payload too large: {payload_size} > {self.max_payload} bytes")
        
        return await next(ctx)
    
    def reset(self, request_id: str | None = None):
        """Reset budget counters."""
        if request_id:
            self._call_counts.pop(request_id, None)
        else:
            self._call_counts.clear()


class ConcurrencyLimitMiddleware(ToolMiddleware):
    """Per-tool and global concurrency caps."""
    
    def __init__(self, global_limit: int = 10, per_tool_limits: dict[str, int] | None = None):
        self._global_sem = asyncio.Semaphore(global_limit)
        self._tool_sems: dict[str, asyncio.Semaphore] = {}
        self._per_tool = per_tool_limits or {}
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        name = ctx.tool.name
        if name not in self._tool_sems and name in self._per_tool:
            self._tool_sems[name] = asyncio.Semaphore(self._per_tool[name])
        
        async with self._global_sem:
            if name in self._tool_sems:
                async with self._tool_sems[name]:
                    return await next(ctx)
            return await next(ctx)


class CircuitBreakerMiddleware(ToolMiddleware):
    """Circuit breaker pattern for tools."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        name = ctx.tool.name
        now = time.monotonic()
        
        # Check if circuit is open
        if name in self._open_until and now < self._open_until[name]:
            return ToolResult.error_result(f"Circuit open for tool '{name}'")
        
        result = await next(ctx)
        
        if result.success:
            self._failures[name] = 0
        else:
            self._failures[name] = self._failures.get(name, 0) + 1
            if self._failures[name] >= self.threshold:
                self._open_until[name] = now + self.reset_timeout
        
        return result
    
    def reset(self, tool_name: str | None = None):
        """Reset circuit breaker state."""
        if tool_name:
            self._failures.pop(tool_name, None)
            self._open_until.pop(tool_name, None)
        else:
            self._failures.clear()
            self._open_until.clear()


class ResultSizeMiddleware(ToolMiddleware):
    """Limit and truncate tool output size."""
    
    def __init__(self, max_chars: int = 50_000, truncation_suffix: str = "... [truncated]"):
        self.max_chars = max_chars
        self.suffix = truncation_suffix
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        result = await next(ctx)
        
        if result.content and isinstance(result.content, str):
            if len(result.content) > self.max_chars:
                truncated = result.content[:self.max_chars - len(self.suffix)] + self.suffix
                return ToolResult(content=truncated, success=result.success, metadata=result.metadata)
        
        return result


class RedactionMiddleware(ToolMiddleware):
    """Scrub secrets/PII from tool output before logging and model feedback."""
    
    def __init__(self, patterns: list[str] | None = None):
        default_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b(?:sk-|api_key[=:]\s*)[A-Za-z0-9\-_]+',  # API keys
        ]
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or default_patterns)]
        self.replacement = "[REDACTED]"
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        from .base import ToolResult
        
        result = await next(ctx)
        
        if result.content and isinstance(result.content, str):
            redacted = result.content
            for pattern in self.patterns:
                redacted = pattern.sub(self.replacement, redacted)
            if redacted != result.content:
                return ToolResult(content=redacted, success=result.success, metadata=result.metadata)
        
        return result


class TelemetryMiddleware(ToolMiddleware):
    """Emit spans/metrics with correlation IDs."""
    
    def __init__(self, tracer: Any = None):
        self._tracer = tracer
    
    async def __call__(self, ctx: ToolExecutionContext, next: NextHandler):
        request_id = getattr(ctx.request_context, "request_id", "no-ctx")
        start = time.monotonic()
        
        # Record in metadata for upstream hooks
        ctx.metadata["telemetry"] = {
            "start_time": start,
            "request_id": request_id,
            "tool_name": ctx.tool.name,
        }
        
        try:
            result = await next(ctx)
            ctx.metadata["telemetry"]["duration_ms"] = (time.monotonic() - start) * 1000
            ctx.metadata["telemetry"]["success"] = result.success
            return result
        except Exception as e:
            ctx.metadata["telemetry"]["duration_ms"] = (time.monotonic() - start) * 1000
            ctx.metadata["telemetry"]["error"] = str(e)
            raise


# === Chain ===


class MiddlewareChain:
    """Composes middleware into an execution chain.
    
    Middleware is applied in order: first added = outermost wrapper.
    """
    
    def __init__(self, middlewares: list[ToolMiddleware] | None = None):
        self._middlewares = list(middlewares or [])
    
    def add(self, middleware: ToolMiddleware) -> "MiddlewareChain":
        """Add middleware to the chain. Returns self for chaining."""
        self._middlewares.append(middleware)
        return self
    
    def build(self, handler: NextHandler) -> NextHandler:
        """Build the chain by wrapping the handler with all middleware."""
        result = handler
        for middleware in reversed(self._middlewares):
            captured = result
            captured_mw = middleware
            async def wrapped(ctx: ToolExecutionContext, _mw=captured_mw, _next=captured):
                return await _mw(ctx, _next)
            result = wrapped
        return result
    
    @classmethod
    def production_defaults(cls) -> "MiddlewareChain":
        """Create a chain with production-ready defaults.
        
        IMPORTANT: Middleware Ownership Boundary
        ----------------------------------------
        This middleware chain is OPT-IN for Agent and is intended for standalone
        llm-client usage where agent-runtime is NOT being used.
        
        When using agent-runtime:
        - Budget enforcement should be centralized in agent-runtime's Ledger
        - Policy enforcement should be centralized in agent-runtime's PolicyEngine
        - Telemetry should be centralized in agent-runtime's OpenTelemetryAdapter
        
        Using both this middleware chain AND agent-runtime will cause DOUBLE
        ENFORCEMENT of budgets/policies, leading to:
        - Inconsistent behavior between streaming/non-streaming calls
        - Duplicate telemetry spans
        - Subtle mismatches in multi-agent and replay scenarios
        
        Recommended patterns:
        - Standalone llm-client: Use production_defaults() via Agent(use_middleware=True)
        - With agent-runtime: Use minimal_defaults() or no middleware; let runtime enforce
        
        Canonical ordering (when used):
        1. Telemetry (outermost - captures full duration)
        2. Logging
        3. Policy (fail fast)
        4. Budget (fail fast)
        5. Concurrency limit
        6. Timeout
        7. Retry
        8. Circuit breaker
        9. [Execute tool]
        10. Result size limit
        11. Redaction
        """
        return cls([
            TelemetryMiddleware(),
            LoggingMiddleware(),
            PolicyMiddleware(),
            BudgetMiddleware(),
            ConcurrencyLimitMiddleware(),
            TimeoutMiddleware(),
            RetryMiddleware(),
            CircuitBreakerMiddleware(),
            ResultSizeMiddleware(),
            RedactionMiddleware(),
        ])

    @classmethod
    def minimal_defaults(cls) -> "MiddlewareChain":
        """Create a minimal chain suitable for use WITH agent-runtime.
        
        This chain includes only "safe but dumb" middleware that doesn't
        conflict with agent-runtime's centralized enforcement:
        - Logging: Local execution logging (agent-runtime has its own event bus)
        - Timeout: Local timeout (agent-runtime may have job-level timeouts)
        - Retry: Local retry (agent-runtime has job-level retry/resume)
        - ResultSize: Truncation (purely local concern)
        - Redaction: PII scrubbing (purely local concern)
        
        Excludes:
        - Telemetry: Use agent-runtime's OpenTelemetryAdapter instead
        - Policy: Use agent-runtime's PolicyEngine instead  
        - Budget: Use agent-runtime's Ledger instead
        - ConcurrencyLimit: Use agent-runtime's orchestration controls instead
        - CircuitBreaker: Use agent-runtime's job state management instead
        """
        return cls([
            LoggingMiddleware(),
            TimeoutMiddleware(),
            RetryMiddleware(),
            ResultSizeMiddleware(),
            RedactionMiddleware(),
        ])


__all__ = [
    "ToolExecutionContext",
    "ToolMiddleware",
    "NextHandler",
    # Table stakes
    "LoggingMiddleware",
    "TimeoutMiddleware",
    "RetryMiddleware",
    # Production
    "PolicyMiddleware",
    "BudgetMiddleware",
    "ConcurrencyLimitMiddleware",
    "CircuitBreakerMiddleware",
    "ResultSizeMiddleware",
    "RedactionMiddleware",
    "TelemetryMiddleware",
    # Chain
    "MiddlewareChain",
]
