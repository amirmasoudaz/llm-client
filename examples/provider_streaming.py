#!/usr/bin/env python3
"""
Example: Provider Streaming with Adapters

Demonstrates:
1. Direct streaming with OpenAIProvider
2. Using SSE adapter for web responses
3. Using callback adapter for custom handling
4. Using buffering adapter to collect full response
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv, find_dotenv

from llm_client import (
    BufferingAdapter,
    CallbackAdapter,
    OpenAIProvider,
    SSEAdapter,
    StreamEventType,
    collect_stream,
    stream_to_string,
)


load_dotenv(find_dotenv())

async def main():
    print("=" * 60)
    print("PROVIDER STREAMING EXAMPLE")
    print("=" * 60)

    provider = OpenAIProvider(model="gpt-5-nano")

    prompt = "Write a short poem about coding (4 lines max)."

    # === Example 1: Direct streaming ===
    print("\n" + "=" * 40)
    print("Example 1: Direct Streaming")
    print("=" * 40)

    print(f"\nPrompt: {prompt}")
    print("\nStreaming response:")
    print("-" * 40)

    async for event in provider.stream(prompt):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.USAGE:
            usage = event.data
            print(f"\n\nUsage: {usage.input_tokens} in, {usage.output_tokens} out")
        elif event.type == StreamEventType.DONE:
            result = event.data
            print(f"Finish reason: {result.finish_reason}")

    print("-" * 40)

    # === Example 2: SSE Adapter (for web servers) ===
    print("\n" + "=" * 40)
    print("Example 2: SSE Adapter")
    print("=" * 40)

    print("\nSSE-formatted events:")
    print("-" * 40)

    adapter = SSEAdapter()
    event_count = 0

    async for sse_string in adapter.transform(provider.stream("Say hello in 3 words")):
        # In a real web app, you'd yield this to the HTTP response
        event_count += 1
        if event_count <= 5:  # Show first few events
            print(repr(sse_string))

    print(f"... ({event_count} total SSE events)")
    print("-" * 40)

    # === Example 3: Callback Adapter ===
    print("\n" + "=" * 40)
    print("Example 3: Callback Adapter")
    print("=" * 40)

    print("\nUsing callbacks:")
    print("-" * 40)

    tokens_received = []

    def on_token(token: str):
        tokens_received.append(token)
        print("📝 ", end="")  # Visual indicator for each token

    def on_done(result):
        print(f"\n✅ Complete! Got {len(tokens_received)} tokens")

    adapter = CallbackAdapter(
        on_token=on_token,
        on_done=on_done,
    )

    await adapter.consume(provider.stream("Count from 1 to 5"))
    print("-" * 40)

    # === Example 4: Buffering Adapter ===
    print("\n" + "=" * 40)
    print("Example 4: Buffering Adapter")
    print("=" * 40)

    print("\nBuffering while streaming:")
    print("-" * 40)

    buffer = BufferingAdapter()

    # Process events while buffering
    async for event in buffer.wrap(provider.stream("What is 2+2? Answer briefly.")):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)

    print("\n")
    print(f"Buffered content: {buffer.content!r}")
    print(f"Total events: {len(buffer.events)}")

    result = buffer.get_result()
    print(f"Usage: {result.usage.total_tokens if result.usage else 'N/A'} tokens")
    print("-" * 40)

    # === Example 5: Utility Functions ===
    print("\n" + "=" * 40)
    print("Example 5: Utility Functions")
    print("=" * 40)

    # collect_stream: Get full CompletionResult
    print("\ncollect_stream():")
    result = await collect_stream(provider.stream("Say 'test'"))
    print(f"  Content: {result.content!r}")
    print(f"  Status: {result.status}")

    # stream_to_string: Get just the text
    print("\nstream_to_string():")
    text = await stream_to_string(provider.stream("Say 'hello'"))
    print(f"  Text: {text!r}")

    # === Cleanup ===
    await provider.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
