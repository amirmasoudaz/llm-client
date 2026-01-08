#!/usr/bin/env python3
"""
Example: Robust Streaming
Demonstrates using the new StreamResponse for robust streaming with usage accumulation.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient
from llm_client.streams import StreamResponse

async def main():
    print("Initializing Client...")
    
    client = OpenAIClient(model="gpt-5-nano")

    print("\n" + "=" * 60)
    print("TEST: Raw Streaming with Usage")
    print("=" * 60)

    # Use the new 'raw' stream mode which returns StreamResponse
    stream: StreamResponse = await client.get_response(
        messages=[{"role": "user", "content": "Write a short haiku about code."}],
        stream=True,
        stream_mode="raw"
    )

    print("Stream started... receiving chunks:")
    print("-" * 30)
    
    async for chunk in stream.text():
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 30)
    print("Stream finished.")
    
    # Check usage
    print(f"\nUsage Stats: {stream.usage}")
    print(f"Full Text Collected: {stream.output_text!r}")
    
    await client.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())
