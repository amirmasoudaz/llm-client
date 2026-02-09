"""Tests for context propagation throughout the llm-client system."""

from __future__ import annotations

import pytest

from llm_client.spec import RequestContext


class TestRequestContext:
    """Test RequestContext enhancements."""

    def test_ensure_with_none_creates_default(self):
        """ensure() should create a valid context when given None."""
        ctx = RequestContext.ensure(None)
        assert ctx is not None
        assert ctx.request_id is not None
        assert len(ctx.request_id) > 0

    def test_ensure_with_context_returns_same(self):
        """ensure() should return the same context when given a valid one."""
        original = RequestContext(request_id="test-123")
        result = RequestContext.ensure(original)
        assert result is original

    def test_child_preserves_fields(self):
        """child() should preserve key fields from parent context."""
        parent = RequestContext(
            request_id="req-1",
            trace_id="trace-1",
            tenant_id="tenant-1",
            user_id="user-1",
            tags={"env": "test"},
        )
        child = parent.child()

        assert child.request_id == parent.request_id
        assert child.trace_id == parent.trace_id
        assert child.tenant_id == parent.tenant_id
        assert child.user_id == parent.user_id
        assert child.tags == parent.tags

    def test_child_generates_new_span_id_by_default(self):
        """child() should generate a new span_id by default."""
        parent = RequestContext(span_id="span-1")
        child = parent.child()

        assert child.span_id is not None
        assert child.span_id != parent.span_id

    def test_child_preserves_span_id_when_requested(self):
        """child() with new_span=False should preserve span_id."""
        parent = RequestContext(span_id="span-1")
        child = parent.child(new_span=False)

        assert child.span_id == parent.span_id

    def test_to_dict_includes_new_fields(self):
        """to_dict() should include all new context fields."""
        ctx = RequestContext(
            request_id="req-1",
            trace_id="trace-1",
            span_id="span-1",
            tenant_id="tenant-1",
            user_id="user-1",
        )
        d = ctx.to_dict()

        assert d["request_id"] == "req-1"
        assert d["trace_id"] == "trace-1"
        assert d["span_id"] == "span-1"
        assert d["tenant_id"] == "tenant-1"
        assert d["user_id"] == "user-1"

    def test_context_is_frozen(self):
        """RequestContext should be immutable (frozen)."""
        ctx = RequestContext()
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.request_id = "new-id"


class TestContextInEngine:
    """Test context propagation through ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_cache_key_includes_tenant_id(self, mock_provider):
        """Cache keys should include tenant_id for isolation."""
        from llm_client.engine import ExecutionEngine
        from llm_client.spec import RequestSpec
        from llm_client.providers.types import Message

        provider = mock_provider()
        engine = ExecutionEngine(provider)

        spec = RequestSpec(
            provider="test",
            model="test",
            messages=[Message.user("hello")],
        )

        ctx_none = None
        ctx_tenant_a = RequestContext(tenant_id="tenant-a")
        ctx_tenant_b = RequestContext(tenant_id="tenant-b")

        # Cache keys should differ by tenant
        key_none = engine._cache_key(spec, provider, ctx_none)
        key_a = engine._cache_key(spec, provider, ctx_tenant_a)
        key_b = engine._cache_key(spec, provider, ctx_tenant_b)

        assert key_none != key_a
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_engine_emits_context_to_hooks(self, mock_provider):
        """ExecutionEngine should pass context to hooks."""
        from llm_client.engine import ExecutionEngine
        from llm_client.hooks import InMemoryMetricsHook, HookManager
        from llm_client.spec import RequestSpec
        from llm_client.providers.types import Message

        hook = InMemoryMetricsHook()
        
        # Custom hook that captures context
        captured_contexts = []
        class ContextCapturingHook:
            async def emit(self, event: str, payload: dict, context) -> None:
                captured_contexts.append((event, context))
        
        capturing_hook = ContextCapturingHook()
        hooks = HookManager([hook, capturing_hook])
        
        provider = mock_provider()
        engine = ExecutionEngine(provider, hooks=hooks)

        spec = RequestSpec(
            provider="test",
            model="test",
            messages=[Message.user("hello")],
        )
        ctx = RequestContext(
            request_id="test-req",
            trace_id="test-trace",
            tenant_id="test-tenant",
        )

        await engine.complete(spec, context=ctx)

        # Verify context was passed to hooks
        assert len(captured_contexts) > 0
        for event, captured_ctx in captured_contexts:
            assert captured_ctx.request_id == "test-req"
            assert captured_ctx.trace_id == "test-trace"
            assert captured_ctx.tenant_id == "test-tenant"


class TestValidationError:
    """Test ValidationError includes request_id."""

    def test_error_includes_request_id(self):
        """ValidationError should include request_id in message."""
        from llm_client.validation import ValidationError
        
        error = ValidationError("Test error", request_id="req-123")
        assert "req-123" in str(error)
        assert error.request_id == "req-123"

    def test_error_without_request_id(self):
        """ValidationError should work without request_id."""
        from llm_client.validation import ValidationError
        
        error = ValidationError("Test error")
        assert error.request_id is None
        assert "Test error" in str(error)
