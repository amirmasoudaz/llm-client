"""
Unified streaming infrastructure.

This module provides:
- StreamEvent types (imported from providers.types)
- Adapters for different output formats (SSE, WebSocket, Pusher)
- Utilities for consuming and transforming streams
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
)

import aiohttp
import six

# Re-export stream types from providers for convenience
from .providers.types import (
    CompletionResult,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    Usage,
)


def format_sse_event(event: str, data: str) -> str:
    """
    Format data as a Server-Sent Event (SSE).
    
    Args:
        event: The event name (e.g., "token", "done", "error")
        data: The data payload (will be stringified if not already a string)
    
    Returns:
        SSE-formatted string: "event: {event}\\ndata: {data}\\n\\n"
    """
    if not isinstance(data, str):
        data = str(data)
    return f"event: {event}\ndata: {data}\n\n"


class StreamAdapter(Protocol):
    """Protocol for stream output adapters."""
    
    async def emit(self, event: StreamEvent) -> None:
        """Emit a stream event."""
        ...
    
    async def close(self) -> None:
        """Close the adapter."""
        ...


class SSEAdapter:
    """
    Adapter that converts StreamEvents to SSE format.
    
    Use this to stream events to HTTP clients expecting SSE.
    
    Example:
        ```python
        adapter = SSEAdapter()
        async for sse_string in adapter.transform(provider.stream(messages)):
            yield sse_string  # Send to HTTP response
        ```
    """
    
    def __init__(self) -> None:
        self._closed = False
    
    async def transform(self, stream: AsyncIterator[StreamEvent]) -> AsyncIterator[str]:
        """Transform a stream of events into SSE-formatted strings."""
        async for event in stream:
            if self._closed:
                break
            yield event.to_sse()
    
    async def emit(self, event: StreamEvent) -> None:
        """Convert and return single event as SSE (for manual use)."""
        return event.to_sse()
    
    async def close(self) -> None:
        """Mark adapter as closed."""
        self._closed = True


class CallbackAdapter:
    """
    Adapter that invokes callbacks for each event type.
    
    Use this for custom event handling or UI updates.
    
    Example:
        ```python
        adapter = CallbackAdapter(
            on_token=lambda t: print(t, end=""),
            on_done=lambda r: print(f"\\nDone: {r.content}")
        )
        await adapter.consume(provider.stream(messages))
        ```
    """
    
    def __init__(
        self,
        on_token: Optional[Callable[[str], Any]] = None,
        on_reasoning: Optional[Callable[[str], Any]] = None,
        on_tool_call_start: Optional[Callable[[ToolCallDelta], Any]] = None,
        on_tool_call_delta: Optional[Callable[[ToolCallDelta], Any]] = None,
        on_tool_call_end: Optional[Callable[[ToolCall], Any]] = None,
        on_usage: Optional[Callable[[Usage], Any]] = None,
        on_done: Optional[Callable[[CompletionResult], Any]] = None,
        on_error: Optional[Callable[[Dict[str, Any]], Any]] = None,
        on_meta: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.on_token = on_token
        self.on_reasoning = on_reasoning
        self.on_tool_call_start = on_tool_call_start
        self.on_tool_call_delta = on_tool_call_delta
        self.on_tool_call_end = on_tool_call_end
        self.on_usage = on_usage
        self.on_done = on_done
        self.on_error = on_error
        self.on_meta = on_meta
    
    async def emit(self, event: StreamEvent) -> None:
        """Process a single event through callbacks."""
        handler = {
            StreamEventType.TOKEN: self.on_token,
            StreamEventType.REASONING: self.on_reasoning,
            StreamEventType.TOOL_CALL_START: self.on_tool_call_start,
            StreamEventType.TOOL_CALL_DELTA: self.on_tool_call_delta,
            StreamEventType.TOOL_CALL_END: self.on_tool_call_end,
            StreamEventType.USAGE: self.on_usage,
            StreamEventType.DONE: self.on_done,
            StreamEventType.ERROR: self.on_error,
            StreamEventType.META: self.on_meta,
        }.get(event.type)
        
        if handler:
            result = handler(event.data)
            if asyncio.iscoroutine(result):
                await result
    
    async def consume(self, stream: AsyncIterator[StreamEvent]) -> Optional[CompletionResult]:
        """
        Consume entire stream, invoking callbacks for each event.
        
        Returns the final CompletionResult if one was received.
        """
        result = None
        async for event in stream:
            await self.emit(event)
            if event.type == StreamEventType.DONE:
                result = event.data
        return result
    
    async def close(self) -> None:
        """No-op for callback adapter."""
        pass


class BufferingAdapter:
    """
    Adapter that buffers stream events and accumulates content.
    
    Useful for collecting the full response while still processing events.
    
    Example:
        ```python
        adapter = BufferingAdapter()
        async for event in adapter.wrap(provider.stream(messages)):
            # Process events as normal
            pass
        
        # After stream completes:
        print(adapter.content)       # Full accumulated content
        print(adapter.tool_calls)    # All tool calls
        print(adapter.result)        # Final CompletionResult
        ```
    """
    
    def __init__(self) -> None:
        self.events: List[StreamEvent] = []
        self.content: str = ""
        self.reasoning: str = ""
        self.tool_calls: List[ToolCall] = []
        self.usage: Optional[Usage] = None
        self.result: Optional[CompletionResult] = None
        self._tool_buffers: Dict[int, Dict[str, Any]] = {}
    
    async def wrap(self, stream: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
        """Wrap a stream, buffering events while passing them through."""
        async for event in stream:
            await self.emit(event)
            yield event
    
    async def emit(self, event: StreamEvent) -> None:
        """Buffer a single event."""
        self.events.append(event)
        
        if event.type == StreamEventType.TOKEN:
            self.content += event.data
        elif event.type == StreamEventType.REASONING:
            self.reasoning += event.data
        elif event.type == StreamEventType.TOOL_CALL_START:
            delta: ToolCallDelta = event.data
            self._tool_buffers[delta.index] = {
                "id": delta.id,
                "name": delta.name or "",
                "arguments": "",
            }
        elif event.type == StreamEventType.TOOL_CALL_DELTA:
            delta: ToolCallDelta = event.data
            if delta.index in self._tool_buffers:
                self._tool_buffers[delta.index]["arguments"] += delta.arguments_delta
        elif event.type == StreamEventType.TOOL_CALL_END:
            tc: ToolCall = event.data
            self.tool_calls.append(tc)
        elif event.type == StreamEventType.USAGE:
            self.usage = event.data
        elif event.type == StreamEventType.DONE:
            self.result = event.data
    
    async def close(self) -> None:
        """Clear buffers."""
        self.events.clear()
        self._tool_buffers.clear()
    
    def get_result(self) -> CompletionResult:
        """
        Get accumulated result.
        
        Returns the result from DONE event if available,
        otherwise constructs one from buffered data.
        """
        if self.result:
            return self.result
        
        return CompletionResult(
            content=self.content if self.content else None,
            tool_calls=self.tool_calls if self.tool_calls else None,
            usage=self.usage,
            reasoning=self.reasoning if self.reasoning else None,
        )


class PusherStreamer:
    """
    Pusher-based streaming adapter.
    
    Pushes events to a Pusher channel for real-time updates to connected clients.
    
    Requires environment variables:
    - PUSHER_AUTH_KEY
    - PUSHER_AUTH_SECRET
    - PUSHER_AUTH_VERSION
    - PUSHER_APP_ID
    - PUSHER_APP_CLUSTER
    """
    
    def __init__(self, channel: Optional[str] = None) -> None:
        self.auth_key = os.environ.get("PUSHER_AUTH_KEY")
        self.auth_secret = os.environ.get("PUSHER_AUTH_SECRET", "").encode("utf8")
        self.auth_version = os.environ.get("PUSHER_AUTH_VERSION")

        self.base = f"https://api-{os.environ.get('PUSHER_APP_CLUSTER')}.pusher.com"
        self.path = f"/apps/{os.environ.get('PUSHER_APP_ID')}/events"
        self.headers = {
            "X-Pusher-Library": f"pusher-http-python {self.auth_version}",
            "Content-Type": "application/json",
        }

        self.channel = channel
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "PusherStreamer":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.session:
            await self.session.close()

    def _generate_query_string(self, params: dict) -> str:
        body_md5 = hashlib.md5(json.dumps(params).encode("utf-8")).hexdigest()
        query_params: dict = {
            "auth_key": self.auth_key,
            "auth_timestamp": str(int(time.time())),
            "auth_version": self.auth_version,
            "body_md5": six.text_type(body_md5),
        }
        query_string = "&".join(
            map("=".join, sorted(query_params.items(), key=lambda x: x[0]))
        )
        auth_string = "\n".join(["POST", self.path, query_string]).encode("utf8")
        signature_encoded = hmac.new(
            self.auth_secret, auth_string, hashlib.sha256
        ).hexdigest()
        query_params["auth_signature"] = six.text_type(signature_encoded)
        query_string += "&auth_signature=" + query_params["auth_signature"]

        return query_string

    async def push_event(self, name: str, data: str) -> Union[dict, str]:
        """Push an event to the Pusher channel."""
        params = {"name": name, "data": data, "channels": [self.channel]}
        query_string = self._generate_query_string(params)
        url = f"{self.base}{self.path}?{query_string}"
        body = json.dumps(params)

        if not self.session:
            raise RuntimeError(
                "Streamer session not initialized. Use 'async with PusherStreamer(...)' context."
            )

        try:
            async with self.session.post(url=url, data=body, headers=self.headers) as response:
                if response.headers.get("Content-Type") == "application/json":
                    return await response.json()
                return await response.text()
        except aiohttp.ClientError as exc:
            return str(exc)
    
    async def emit(self, event: StreamEvent) -> None:
        """Emit a StreamEvent to Pusher."""
        event_name = event.type.value
        
        if isinstance(event.data, str):
            data = event.data
        elif hasattr(event.data, "to_dict"):
            data = json.dumps(event.data.to_dict())
        elif isinstance(event.data, (dict, list)):
            data = json.dumps(event.data)
        else:
            data = str(event.data)
        
        await self.push_event(event_name, data)
    
    async def consume(self, stream: AsyncIterator[StreamEvent]) -> Optional[CompletionResult]:
        """Consume stream and push all events to Pusher."""
        result = None
        await self.push_event("stream-start", "")
        
        async for event in stream:
            await self.emit(event)
            if event.type == StreamEventType.DONE:
                result = event.data
        
        await self.push_event("stream-end", "")
        return result
    
    async def close(self) -> None:
        """Close the Pusher session."""
        if self.session:
            await self.session.close()
            self.session = None


async def collect_stream(stream: AsyncIterator[StreamEvent]) -> CompletionResult:
    """
    Utility to collect a stream into a final CompletionResult.
    
    This consumes the entire stream silently and returns the result.
    """
    adapter = BufferingAdapter()
    async for event in adapter.wrap(stream):
        pass
    return adapter.get_result()


async def stream_to_string(stream: AsyncIterator[StreamEvent]) -> str:
    """
    Utility to collect just the text content from a stream.
    """
    content = ""
    async for event in stream:
        if event.type == StreamEventType.TOKEN:
            content += event.data
    return content


__all__ = [
    # Event types (re-exported)
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "CompletionResult",
    # Adapters
    "StreamAdapter",
    "SSEAdapter",
    "CallbackAdapter",
    "BufferingAdapter",
    "PusherStreamer",
    # Utilities
    "format_sse_event",
    "collect_stream",
    "stream_to_string",
]
