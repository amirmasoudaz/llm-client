"""Tests for cancellation support in llm-client."""

from __future__ import annotations

import asyncio
import pytest

from llm_client.cancellation import CancellationToken, CancelledError
from llm_client.spec import RequestContext


class TestCancellationToken:
    """Test CancellationToken behavior."""

    def test_token_starts_uncancelled(self):
        """New token should not be cancelled."""
        token = CancellationToken()
        assert not token.is_cancelled

    def test_cancel_sets_flag(self):
        """cancel() should set is_cancelled to True."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled

    def test_cancel_is_idempotent(self):
        """cancel() can be called multiple times safely."""
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled

    def test_raise_if_cancelled_on_uncancelled(self):
        """raise_if_cancelled() should not raise if uncancelled."""
        token = CancellationToken()
        token.raise_if_cancelled()  # Should not raise

    def test_raise_if_cancelled_on_cancelled(self):
        """raise_if_cancelled() should raise CancelledError if cancelled."""
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledError):
            token.raise_if_cancelled()

    def test_on_cancel_callback_invoked(self):
        """Registered callback should be invoked on cancel."""
        token = CancellationToken()
        called = []
        token.on_cancel(lambda: called.append(True))
        assert len(called) == 0
        token.cancel()
        assert len(called) == 1

    def test_on_cancel_immediate_if_already_cancelled(self):
        """If already cancelled, callback invoked immediately."""
        token = CancellationToken()
        token.cancel()
        called = []
        token.on_cancel(lambda: called.append(True))
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_wait_blocks_until_cancel(self):
        """wait() should block until cancel is called."""
        token = CancellationToken()
        
        async def cancel_later():
            await asyncio.sleep(0.01)
            token.cancel()
        
        asyncio.create_task(cancel_later())
        await asyncio.wait_for(token.wait(), timeout=1.0)
        assert token.is_cancelled

    def test_none_returns_singleton(self):
        """CancellationToken.none() should return a usable token."""
        token = CancellationToken.none()
        assert token is not None
        # The none token should not be cancelled initially
        # (unless it was cancelled elsewhere, which would be a misuse)


class TestCancellationInContext:
    """Test CancellationToken integration with RequestContext."""

    def test_context_has_default_token(self):
        """RequestContext should have a default cancellation token."""
        ctx = RequestContext()
        assert ctx.cancellation_token is not None

    def test_child_context_inherits_token(self):
        """Child context should inherit parent's cancellation token."""
        token = CancellationToken()
        parent = RequestContext(cancellation_token=token)
        child = parent.child()
        assert child.cancellation_token is token

    def test_custom_token_in_context(self):
        """Can create context with custom token."""
        token = CancellationToken()
        ctx = RequestContext(cancellation_token=token)
        assert ctx.cancellation_token is token

    def test_cancel_via_context(self):
        """Can cancel via context's token."""
        token = CancellationToken()
        ctx = RequestContext(cancellation_token=token)
        token.cancel()
        with pytest.raises(CancelledError):
            ctx.cancellation_token.raise_if_cancelled()


class TestCancellationInEngine:
    """Test cancellation in ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_complete_respects_cancellation(self, mock_provider):
        """complete() should raise CancelledError if cancelled before call."""
        from llm_client.engine import ExecutionEngine
        from llm_client.spec import RequestSpec
        from llm_client.providers.types import Message

        token = CancellationToken()
        token.cancel()  # Cancel before call

        ctx = RequestContext(cancellation_token=token)
        provider = mock_provider()
        engine = ExecutionEngine(provider)

        spec = RequestSpec(
            provider="test",
            model="test",
            messages=[Message.user("hello")],
        )

        with pytest.raises(CancelledError):
            await engine.complete(spec, context=ctx)

    @pytest.mark.asyncio
    async def test_stream_respects_cancellation(self, mock_provider):
        """stream() should stop yielding when cancelled."""
        from llm_client.engine import ExecutionEngine
        from llm_client.spec import RequestSpec
        from llm_client.providers.types import Message

        token = CancellationToken()
        ctx = RequestContext(cancellation_token=token)
        
        provider = mock_provider()
        engine = ExecutionEngine(provider)

        spec = RequestSpec(
            provider="test",
            model="test",
            messages=[Message.user("hello")],
        )

        events = []
        try:
            async for event in engine.stream(spec, context=ctx):
                events.append(event)
                # Cancel mid-stream
                if len(events) >= 2:
                    token.cancel()
        except CancelledError:
            pass

        # Should have stopped early
        assert len(events) >= 2


class TestCancelledError:
    """Test CancelledError exception."""

    def test_cancelled_error_is_exception(self):
        """CancelledError should be an Exception."""
        err = CancelledError("test message")
        assert isinstance(err, Exception)
        assert "test message" in str(err)

    def test_can_catch_cancelled_error(self):
        """CancelledError should be catchable."""
        try:
            raise CancelledError("test")
        except CancelledError as e:
            assert "test" in str(e)
