#!/usr/bin/env python3
"""
Example: Streaming Responses (SSE Mode)

Demonstrates:
1. Server-Sent Events (SSE) streaming
2. Token-by-token output
3. Usage stats after completion
"""
import asyncio

# Add src to path for development
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient


async def main():
    print("=" * 60)
    print("SSE STREAMING EXAMPLE")
    print("=" * 60)

    client = OpenAIClient(
        model="gpt-5-nano",
        cache_backend=None,  # No caching for streaming demo
    )

    prompt = "Write a haiku about programming. Be creative!"

    print(f"\n📝 Prompt: {prompt}")
    print("\n🔄 Streaming response:\n")
    print("-" * 40)

    # Get SSE stream generator
    stream = await client.get_response(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_mode="sse",
    )

    full_output = ""
    usage_info = None

    async for event in stream:
        # SSE events come as "event: <type>\ndata: <payload>\n\n"
        lines = event.strip().split("\n")
        event_type = None
        data = None

        for line in lines:
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()

        if event_type == "token" and data:
            print(data, end="", flush=True)
            full_output += data
        elif event_type == "done" and data:
            import json
            usage_info = json.loads(data)
        elif event_type == "error" and data:
            import json
            error_info = json.loads(data)
            print(f"\n❌ Error: {error_info}")

    print("\n" + "-" * 40)

    if usage_info:
        print(f"\n📊 Usage Stats:")
        usage = usage_info.get("usage", {})
        print(f"  Input tokens:  {usage.get('input_tokens', 'N/A')}")
        print(f"  Output tokens: {usage.get('output_tokens', 'N/A')}")
        print(f"  Total cost:    ${usage.get('total_cost', 0):.6f}")

    await client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
