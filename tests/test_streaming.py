"""
Tests for the streaming infrastructure.
"""
import asyncio
from typing import AsyncIterator

import pytest

from llm_client.providers.types import (
    CompletionResult,
    StreamEvent,
    StreamEventType,
    Usage,
)


async def make_test_stream() -> AsyncIterator[StreamEvent]:
    """Create a test stream of events."""
    yield StreamEvent(type=StreamEventType.TOKEN, data="Hello")
    yield StreamEvent(type=StreamEventType.TOKEN, data=" ")
    yield StreamEvent(type=StreamEventType.TOKEN, data="world")
    yield StreamEvent(type=StreamEventType.TOKEN, data="!")
    yield StreamEvent(type=StreamEventType.USAGE, data=Usage(
        input_tokens=5,
        output_tokens=4,
        total_tokens=9,
    ))
    yield StreamEvent(type=StreamEventType.DONE, data=CompletionResult(
        content="Hello world!",
        usage=Usage(input_tokens=5, output_tokens=4, total_tokens=9),
        status=200,
    ))


class TestSSEAdapter:
    """Tests for SSE adapter."""
    
    async def test_transform_produces_sse(self):
        """Test that transform produces valid SSE format."""
        from llm_client.streaming import SSEAdapter
        
        adapter = SSEAdapter()
        
        sse_lines = []
        async for line in adapter.transform(make_test_stream()):
            sse_lines.append(line)
        
        # Should have SSE formatted lines
        assert len(sse_lines) > 0
        
        # Each line should be SSE format
        for line in sse_lines:
            assert "event:" in line or "data:" in line
            assert line.endswith("\n\n")
    
    async def test_emit_single_event(self):
        """Test emitting a single event via to_sse."""
        from llm_client.streaming import SSEAdapter
        
        event = StreamEvent(type=StreamEventType.TOKEN, data="Test")
        sse = event.to_sse()
        
        assert "event: token" in sse
        assert "data: Test" in sse


class TestCallbackAdapter:
    """Tests for callback adapter."""
    
    async def test_callbacks_invoked(self):
        """Test that callbacks are invoked for each event type."""
        from llm_client.streaming import CallbackAdapter
        
        tokens = []
        usage_received = []
        done_results = []
        
        adapter = CallbackAdapter(
            on_token=lambda t: tokens.append(t),
            on_usage=lambda u: usage_received.append(u),
            on_done=lambda r: done_results.append(r),
        )
        
        result = await adapter.consume(make_test_stream())
        
        assert tokens == ["Hello", " ", "world", "!"]
        assert len(usage_received) == 1
        assert len(done_results) == 1
        assert result is not None
        assert result.content == "Hello world!"
    
    async def test_emit_single(self):
        """Test emitting events through callbacks when consuming."""
        from llm_client.streaming import CallbackAdapter
        
        captured = []
        adapter = CallbackAdapter(
            on_token=lambda t: captured.append(t),
        )
        
        async def single_stream():
            yield StreamEvent(type=StreamEventType.TOKEN, data="X")
            yield StreamEvent(type=StreamEventType.DONE, data=CompletionResult(content="X"))
        
        await adapter.consume(single_stream())
        
        assert captured == ["X"]


class TestBufferingAdapter:
    """Tests for buffering adapter."""
    
    async def test_buffers_content(self):
        """Test that content is buffered correctly."""
        from llm_client.streaming import BufferingAdapter
        
        adapter = BufferingAdapter()
        
        async for _ in adapter.wrap(make_test_stream()):
            pass  # Consume all events
        
        result = adapter.get_result()
        
        assert result is not None
        assert result.content == "Hello world!"
    
    async def test_passes_through_events(self):
        """Test that events are passed through while buffering."""
        from llm_client.streaming import BufferingAdapter
        
        adapter = BufferingAdapter()
        
        events = []
        async for event in adapter.wrap(make_test_stream()):
            events.append(event)
        
        # Should have all original events
        token_events = [e for e in events if e.type == StreamEventType.TOKEN]
        assert len(token_events) == 4
    
    async def test_emit_single(self):
        """Test buffering via wrap method."""
        from llm_client.streaming import BufferingAdapter
        
        adapter = BufferingAdapter()
        
        async def simple_stream():
            yield StreamEvent(type=StreamEventType.TOKEN, data="A")
            yield StreamEvent(type=StreamEventType.TOKEN, data="B")
            yield StreamEvent(type=StreamEventType.DONE, data=CompletionResult(content="AB"))
        
        async for _ in adapter.wrap(simple_stream()):
            pass
        
        result = adapter.get_result()
        
        assert result.content == "AB"
    
    def test_close_clears_buffers(self):
        """Test that close clears internal buffers."""
        from llm_client.streaming import BufferingAdapter
        
        adapter = BufferingAdapter()
        adapter.emit(StreamEvent(type=StreamEventType.TOKEN, data="X"))
        
        adapter.close()
        
        result = adapter.get_result()
        assert result.content == "" or result.content is None


class TestStreamEvent:
    """Tests for StreamEvent class."""
    
    def test_to_sse_string_data(self):
        """Test SSE conversion with string data."""
        event = StreamEvent(type=StreamEventType.TOKEN, data="Hello")
        sse = event.to_sse()
        
        assert "event: token" in sse
        assert "data: Hello" in sse
    
    def test_to_sse_dict_data(self):
        """Test SSE conversion with dict data."""
        event = StreamEvent(type=StreamEventType.META, data={"key": "value"})
        sse = event.to_sse()
        
        assert "event: meta" in sse
        assert '"key"' in sse
        assert '"value"' in sse
    
    def test_to_sse_usage_data(self):
        """Test SSE conversion with Usage data."""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        event = StreamEvent(type=StreamEventType.USAGE, data=usage)
        sse = event.to_sse()
        
        assert "event: usage" in sse
        assert "input_tokens" in sse


class TestStreamUtilities:
    """Tests for stream utility functions."""
    
    async def test_collect_stream(self):
        """Test collecting stream into a result."""
        from llm_client.streaming import collect_stream
        
        result = await collect_stream(make_test_stream())
        
        assert result is not None
        assert result.content == "Hello world!"
    
    async def test_stream_to_string(self):
        """Test converting stream to string."""
        from llm_client.streaming import stream_to_string
        
        text = await stream_to_string(make_test_stream())
        
        assert text == "Hello world!"
    
    def test_format_sse_event(self):
        """Test SSE event formatting."""
        from llm_client.streaming import format_sse_event
        
        sse = format_sse_event("token", "Hello")
        
        assert sse == "event: token\ndata: Hello\n\n"
