import json
from typing import AsyncGenerator, Any, Optional, List
from .streaming import format_sse_event

class StreamResponse:
    """
    A wrapper for streaming responses that provides easy access to text, events, and usage.
    """
    def __init__(self, stream: AsyncGenerator, stream_mode: str = "pusher"):
        self._stream = stream
        self.stream_mode = stream_mode
        self._usage: Optional[dict] = None
        self._accumulated_text: str = ""
        self._tool_calls: List[dict] = []
        self._finished = False

    async def __aiter__(self):
        """Iterate over the stream, yielding text chunks."""
        async for chunk in self.text():
            yield chunk

    async def text(self) -> AsyncGenerator[str, None]:
        """Yields text chunks from the stream."""
        async for item in self.events():
             if item["type"] == "token":
                 yield item["data"]

    async def events(self) -> AsyncGenerator[dict, None]:
        """
        Yields structured events from the stream.
        Events are dicts with keys: type (meta, token, tool, usage, error, done), data.
        """
        if self.stream_mode == "sse":
            # Pass through SSE events but parse them for internal state
            async for event_str in self._stream:
                # event_str is like "event: type\ndata: ...\n\n"
                lines = event_str.strip().split("\n")
                event_type = lines[0].split(": ", 1)[1]
                data_str = lines[1].split(": ", 1)[1]
                
                try:
                    is_json = data_str.startswith("{") or data_str.startswith("[")
                    data = json.loads(data_str) if is_json else data_str
                except json.JSONDecodeError:
                    data = data_str

                if event_type == "token":
                    self._accumulated_text += data
                elif event_type == "done" and isinstance(data, dict):
                    self._usage = data.get("usage")
                    self._finished = True
                
                yield {"type": event_type, "data": data}
        
        else:
            # Handle raw OpenAI chunks (if we refactor _call_completions to return the raw stream)
            # OR handle the existing Pusher-style stream if we are wrapping that (legacy).
            # For now, let's assume this class is used with the NEW implementation
            # where _stream is the OpenAI AsyncStream.
            
            # Wait, looking at client.py, `_call_completions` handles the iteration and yielding.
            # If we want to use THIS class, we should change `_call_completions` to return `StreamResponse`.
            # A raw OpenAI stream yields chunks.
            
            async for chunk in self._stream:
                if not chunk.choices and chunk.usage:
                     self._usage = chunk.usage.to_dict()
                     yield {"type": "usage", "data": self._usage}
                     yield {"type": "done", "data": None}
                     self._finished = True
                     break
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Text content
                if delta.content is not None:
                    self._accumulated_text += delta.content
                    yield {"type": "token", "data": delta.content}
                
                # Tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if len(self._tool_calls) <= tc.index:
                            self._tool_calls.append({"id": "", "t_type": "function", "name": "", "arguments": ""})
                        
                        current = self._tool_calls[tc.index]
                        if tc.id: current["id"] += tc.id
                        if tc.function.name: current["name"] += tc.function.name
                        if tc.function.arguments: current["arguments"] += tc.function.arguments
                        
                    yield {"type": "tool_chunk", "data": delta.tool_calls}

    @property
    def usage(self) -> Optional[dict]:
        return self._usage

    @property
    def output_text(self) -> str:
        return self._accumulated_text

    @property
    def tool_calls(self) -> List[dict]:
        return self._tool_calls
