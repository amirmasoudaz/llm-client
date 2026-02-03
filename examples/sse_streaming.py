#!/usr/bin/env python3
"""
Example: Server-Sent Events (SSE) Streaming

Demonstrates:
1. SSE streaming for web server responses
2. Building a simple SSE endpoint (FastAPI example)
3. Parsing SSE events on the client side
4. Using SSEAdapter with Provider API
5. Custom event formatting
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    OpenAIProvider,
    SSEAdapter,
    format_sse_event,
)


async def main():
    print("=" * 60)
    print("SSE STREAMING EXAMPLE")
    print("=" * 60)

    # === Example 1: Basic SSE Format ===
    print("\n" + "=" * 40)
    print("Example 1: SSE Format Basics")
    print("=" * 40)

    print("\nSSE event format: 'event: <type>\\ndata: <payload>\\n\\n'")

    # Using format_sse_event helper
    sse1 = format_sse_event("token", "Hello")
    sse2 = format_sse_event("token", " World")
    sse3 = format_sse_event("done", json.dumps({"status": "complete"}))

    print("\nFormatted events:")
    print(f"  Token 1: {repr(sse1)}")
    print(f"  Token 2: {repr(sse2)}")
    print(f"  Done: {repr(sse3)}")

    # === Example 2: StreamEvent to SSE ===
    print("\n" + "=" * 40)
    print("Example 2: StreamEvent.to_sse()")
    print("=" * 40)

    provider = OpenAIProvider(model="gpt-5-nano")

    print("\nStreaming and converting events to SSE format:")
    print("-" * 40)

    event_count = 0
    async for event in provider.stream("Say 'hello world'"):
        sse_string = event.to_sse()
        event_count += 1

        if event_count <= 5:  # Show first few
            # Display in readable format
            lines = sse_string.strip().split("\n")
            for line in lines:
                print(f"  {line}")
            print()

    print(f"  ... ({event_count} total events)")

    await provider.close()

    # === Example 3: SSEAdapter ===
    print("\n" + "=" * 40)
    print("Example 3: SSEAdapter")
    print("=" * 40)

    provider = OpenAIProvider(model="gpt-5-nano")
    adapter = SSEAdapter()

    print("\nUsing SSEAdapter.transform():")
    print("-" * 40)

    sse_events = []
    async for sse_line in adapter.transform(provider.stream("Count: 1, 2, 3")):
        sse_events.append(sse_line)
        if len(sse_events) <= 3:
            print(f"  {repr(sse_line[:50])}...")

    print(f"\n  Total SSE lines: {len(sse_events)}")

    await provider.close()

    # === Example 4: Web Server Pattern ===
    print("\n" + "=" * 40)
    print("Example 4: Web Server Pattern")
    print("=" * 40)

    print("""
SSE streaming for web frameworks (FastAPI/Starlette):

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm_client import OpenAIProvider, SSEAdapter

app = FastAPI()
provider = OpenAIProvider(model="gpt-5")
adapter = SSEAdapter()

@app.get("/stream")
async def stream_completion(prompt: str):
    async def generate():
        async for sse in adapter.transform(provider.stream(prompt)):
            yield sse
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

Client-side (JavaScript):

```javascript
const eventSource = new EventSource('/stream?prompt=Hello');

eventSource.addEventListener('token', (e) => {
    document.getElementById('output').textContent += e.data;
});

eventSource.addEventListener('done', (e) => {
    const result = JSON.parse(e.data);
    console.log('Usage:', result.usage);
    eventSource.close();
});

eventSource.addEventListener('error', (e) => {
    console.error('Error:', e.data);
    eventSource.close();
});
```
""")

    # === Example 5: Parsing SSE Events ===
    print("=" * 40)
    print("Example 5: Parsing SSE Events")
    print("=" * 40)

    provider = OpenAIProvider(model="gpt-5-nano")

    print("\nSimulating SSE client parsing:")
    print("-" * 40)

    # Collect SSE events and parse them
    collected_text = ""

    async for event in provider.stream("Say 'parsing works'"):
        sse_string = event.to_sse()

        # Parse SSE format (as a client would)
        lines = sse_string.strip().split("\n")
        event_type = None
        data = None

        for line in lines:
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()

        if event_type == "token" and data:
            collected_text += data
            print(f"  Token: {repr(data)}")
        elif event_type == "done" and data:
            json.loads(data)  # ensure payload is valid JSON
            print("  Done: received final result")
        elif event_type == "usage" and data:
            print(f"  Usage: {data[:60]}...")

    print(f"\nCollected text: {repr(collected_text)}")

    await provider.close()

    # === Example 6: Custom Event Types ===
    print("\n" + "=" * 40)
    print("Example 6: Custom SSE Events")
    print("=" * 40)

    print("\nSSE supports custom event types for rich streaming:")

    custom_events = [
        format_sse_event("thinking", "Let me think..."),
        format_sse_event("tool_call", json.dumps({"name": "search", "id": "123"})),
        format_sse_event("tool_result", json.dumps({"id": "123", "result": "found 5 items"})),
        format_sse_event("token", "Based on my search, "),
        format_sse_event("token", "I found 5 relevant items."),
        format_sse_event("done", json.dumps({"status": "success"})),
    ]

    print("\nCustom event stream:")
    for event in custom_events:
        lines = event.strip().split("\n")
        event_type = lines[0].replace("event: ", "")
        data = lines[1].replace("data: ", "")
        print(f"  [{event_type}] {data[:50]}{'...' if len(data) > 50 else ''}")

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
