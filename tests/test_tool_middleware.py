"""Tests for tool middleware chain."""

from __future__ import annotations

import asyncio
import pytest

from llm_client.tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    ToolMiddleware,
    ToolExecutionContext,
    MiddlewareChain,
    LoggingMiddleware,
    TimeoutMiddleware,
    RetryMiddleware,
    PolicyMiddleware,
    BudgetMiddleware,
    ConcurrencyLimitMiddleware,
    CircuitBreakerMiddleware,
    ResultSizeMiddleware,
    RedactionMiddleware,
)
from llm_client.spec import RequestContext


# === Test Tool ===


async def echo_handler(message: str) -> str:
    """Simple echo for testing."""
    return f"Echo: {message}"


async def slow_handler(delay: float = 0.5) -> str:
    """Slow handler for timeout testing."""
    await asyncio.sleep(delay)
    return "Done"


async def failing_handler() -> str:
    """Handler that always fails."""
    raise ValueError("Intentional failure")


@pytest.fixture
def echo_tool():
    return Tool(
        name="echo",
        description="Echo a message",
        parameters={"type": "object", "properties": {"message": {"type": "string"}}},
        handler=echo_handler,
    )


@pytest.fixture
def failing_tool():
    return Tool(
        name="failing",
        description="Always fails",
        parameters={"type": "object", "properties": {}},
        handler=failing_handler,
    )


@pytest.fixture
def slow_tool():
    return Tool(
        name="slow",
        description="Slow tool",
        parameters={"type": "object", "properties": {"delay": {"type": "number"}}},
        handler=slow_handler,
    )


# === Middleware Tests ===


class TestLoggingMiddleware:
    """Test LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logs_execution(self, echo_tool):
        """Should log tool execution."""
        middleware = LoggingMiddleware()
        ctx = ToolExecutionContext(tool=echo_tool, arguments={"message": "test"})
        
        async def handler(ctx):
            return ToolResult.success_result("ok")
        
        result = await middleware(ctx, handler)
        assert result.success


class TestTimeoutMiddleware:
    """Test TimeoutMiddleware."""

    @pytest.mark.asyncio
    async def test_timeout_not_triggered(self, echo_tool):
        """Fast execution should not timeout."""
        middleware = TimeoutMiddleware(default_timeout=1.0)
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        async def handler(ctx):
            await asyncio.sleep(0.01)
            return ToolResult.success_result("ok")
        
        result = await middleware(ctx, handler)
        assert result.success

    @pytest.mark.asyncio
    async def test_timeout_triggered(self, slow_tool):
        """Slow execution should timeout."""
        middleware = TimeoutMiddleware(default_timeout=0.05)
        ctx = ToolExecutionContext(tool=slow_tool, arguments={})
        
        async def handler(ctx):
            await asyncio.sleep(1.0)
            return ToolResult.success_result("ok")
        
        result = await middleware(ctx, handler)
        assert not result.success
        assert "timed out" in result.error.lower()


class TestRetryMiddleware:
    """Test RetryMiddleware."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self, echo_tool):
        """Should not retry on success."""
        call_count = 0
        
        async def handler(ctx):
            nonlocal call_count
            call_count += 1
            return ToolResult.success_result("ok")
        
        middleware = RetryMiddleware(max_retries=2)
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        result = await middleware(ctx, handler)
        assert result.success
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, failing_tool):
        """Should retry on failure."""
        call_count = 0
        
        async def handler(ctx):
            nonlocal call_count
            call_count += 1
            return ToolResult.error_result("error")
        
        middleware = RetryMiddleware(max_retries=2, base_backoff=0.01)
        ctx = ToolExecutionContext(tool=failing_tool, arguments={})
        
        result = await middleware(ctx, handler)
        assert not result.success
        assert call_count == 3  # Initial + 2 retries


class TestPolicyMiddleware:
    """Test PolicyMiddleware."""

    @pytest.mark.asyncio
    async def test_denied_tool(self, echo_tool):
        """Should block denied tools."""
        middleware = PolicyMiddleware(denied_tools={"echo"})
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        async def handler(ctx):
            return ToolResult.success_result("ok")
        
        result = await middleware(ctx, handler)
        assert not result.success
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_allowed_tool(self, echo_tool):
        """Should allow tools in allowlist."""
        middleware = PolicyMiddleware(allowed_tools={"echo"})
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        async def handler(ctx):
            return ToolResult.success_result("ok")
        
        result = await middleware(ctx, handler)
        assert result.success

    @pytest.mark.asyncio
    async def test_tool_not_in_allowlist(self, echo_tool):
        """Should block tools not in allowlist."""
        middleware = PolicyMiddleware(allowed_tools={"other_tool"})
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        async def handler(ctx):
            return ToolResult.success_result("ok")
        
        result = await middleware(ctx, handler)
        assert not result.success
        assert "allowlist" in result.error.lower()


class TestBudgetMiddleware:
    """Test BudgetMiddleware."""

    @pytest.mark.asyncio
    async def test_within_budget(self, echo_tool):
        """Should allow calls within budget."""
        middleware = BudgetMiddleware(max_tool_calls=5)
        ctx = ToolExecutionContext(
            tool=echo_tool, 
            arguments={"message": "test"}, 
            request_context=RequestContext(),
        )
        
        async def handler(ctx):
            return ToolResult.success_result("ok")
        
        for _ in range(5):
            result = await middleware(ctx, handler)
            assert result.success

    @pytest.mark.asyncio
    async def test_over_budget(self, echo_tool):
        """Should block calls over budget."""
        middleware = BudgetMiddleware(max_tool_calls=2)
        context = RequestContext()
        
        async def handler(ctx):
            return ToolResult.success_result("ok")
        
        ctx = ToolExecutionContext(tool=echo_tool, arguments={}, request_context=context)
        
        # First 2 should succeed
        await middleware(ctx, handler)
        await middleware(ctx, handler)
        
        # Third should fail
        result = await middleware(ctx, handler)
        assert not result.success
        assert "budget exceeded" in result.error.lower()


class TestCircuitBreakerMiddleware:
    """Test CircuitBreakerMiddleware."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, failing_tool):
        """Circuit should open after threshold failures."""
        middleware = CircuitBreakerMiddleware(failure_threshold=3, reset_timeout=10.0)
        ctx = ToolExecutionContext(tool=failing_tool, arguments={})
        
        async def handler(ctx):
            return ToolResult.error_result("error")
        
        # First 3 failures
        for _ in range(3):
            await middleware(ctx, handler)
        
        # Circuit should be open now
        result = await middleware(ctx, handler)
        assert not result.success
        assert "circuit open" in result.error.lower()


class TestResultSizeMiddleware:
    """Test ResultSizeMiddleware."""

    @pytest.mark.asyncio
    async def test_truncates_large_result(self, echo_tool):
        """Should truncate large results."""
        middleware = ResultSizeMiddleware(max_chars=20)
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        async def handler(ctx):
            return ToolResult.success_result("x" * 100)
        
        result = await middleware(ctx, handler)
        assert result.success
        assert len(result.content) <= 20
        assert "[truncated]" in result.content


class TestRedactionMiddleware:
    """Test RedactionMiddleware."""

    @pytest.mark.asyncio
    async def test_redacts_email(self, echo_tool):
        """Should redact email addresses."""
        middleware = RedactionMiddleware()
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        async def handler(ctx):
            return ToolResult.success_result("Contact: user@example.com for info")
        
        result = await middleware(ctx, handler)
        assert result.success
        assert "user@example.com" not in result.content
        assert "[REDACTED]" in result.content


class TestMiddlewareChain:
    """Test MiddlewareChain composition."""

    @pytest.mark.asyncio
    async def test_chain_composition(self, echo_tool):
        """Chain should compose middleware correctly."""
        call_order = []
        
        class OrderMiddleware(ToolMiddleware):
            def __init__(self, name: str):
                self.name = name
            
            async def __call__(self, ctx, next):
                call_order.append(f"{self.name}_enter")
                result = await next(ctx)
                call_order.append(f"{self.name}_exit")
                return result
        
        chain = MiddlewareChain([OrderMiddleware("A"), OrderMiddleware("B")])
        
        async def handler(ctx):
            call_order.append("handler")
            return ToolResult.success_result("ok")
        
        wrapped = chain.build(handler)
        ctx = ToolExecutionContext(tool=echo_tool, arguments={})
        
        await wrapped(ctx)
        
        assert call_order == ["A_enter", "B_enter", "handler", "B_exit", "A_exit"]

    def test_production_defaults(self):
        """Should create production chain."""
        chain = MiddlewareChain.production_defaults()
        assert len(chain._middlewares) > 0


class TestToolRegistryWithMiddleware:
    """Test ToolRegistry.execute_with_middleware."""

    @pytest.mark.asyncio
    async def test_execute_with_middleware(self, echo_tool):
        """Should execute tool with middleware chain."""
        registry = ToolRegistry()
        registry.register(echo_tool)
        
        chain = MiddlewareChain([LoggingMiddleware()])
        
        result = await registry.execute_with_middleware(
            "echo",
            {"message": "Hello"},
            middleware_chain=chain,
        )
        
        assert result.success
        assert "Echo: Hello" in result.content

    @pytest.mark.asyncio
    async def test_execute_without_middleware(self, echo_tool):
        """Should execute directly without middleware."""
        registry = ToolRegistry()
        registry.register(echo_tool)
        
        result = await registry.execute_with_middleware(
            "echo",
            {"message": "Hello"},
        )
        
        assert result.success
        assert "Echo: Hello" in result.content

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Should handle unknown tool."""
        registry = ToolRegistry()
        
        result = await registry.execute_with_middleware("unknown", {})
        
        assert not result.success
        assert "unknown tool" in result.error.lower()
