#!/usr/bin/env python3
"""
Example: Simple Generation with OpenAIProvider

Demonstrates:
1. Basic completion with the new Provider API
2. Message input formats (string, dict, Message, list)
3. Completion options (temperature, max_tokens)
4. Response handling and error checking
5. Using caching with the new Provider API
6. Retry logic with exponential backoff
"""
import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    OpenAIProvider,
    Message,
)


async def main():
    print("=" * 60)
    print("SIMPLE GENERATION EXAMPLE")
    print("=" * 60)
    
    # Create a temporary cache directory
    cache_dir = Path(tempfile.mkdtemp(prefix="llm_cache_"))
    print(f"📁 Cache directory: {cache_dir}")
    
    # === Example 1: Basic Completion ===
    print("\n" + "=" * 40)
    print("Example 1: Basic Completion")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    # Simple string input
    result = await provider.complete("What is 2 + 2? Reply with just the number.")
    
    print(f"\nPrompt: 'What is 2 + 2?'")
    print(f"Response: {result.content}")
    print(f"Status: {result.status} ({'OK' if result.ok else 'ERROR'})")
    print(f"Tokens: {result.usage.total_tokens if result.usage else 'N/A'}")
    
    await provider.close()
    
    # === Example 2: Different Input Formats ===
    print("\n" + "=" * 40)
    print("Example 2: Input Formats")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    # String input (converted to user message)
    result1 = await provider.complete("Say 'hello'")
    print(f"\nString input: {result1.content}")
    
    # Dict input
    result2 = await provider.complete({"role": "user", "content": "Say 'world'"})
    print(f"Dict input: {result2.content}")
    
    # Message object
    result3 = await provider.complete(Message.user("Say 'foo'"))
    print(f"Message input: {result3.content}")
    
    # List of messages (conversation)
    result4 = await provider.complete([
        Message.system("You respond with single words only."),
        Message.user("What color is the sky?"),
    ])
    print(f"List input: {result4.content}")
    
    await provider.close()
    
    # === Example 3: Completion Options ===
    print("\n" + "=" * 40)
    print("Example 3: Completion Options")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    # With temperature
    result = await provider.complete(
        "Generate a random 4-letter word",
        temperature=1.0,
        max_tokens=10,
    )
    print(f"\nHigh temperature: {result.content}")
    
    result = await provider.complete(
        "Generate a random 4-letter word",
        temperature=0.0,
        max_tokens=10,
    )
    print(f"Zero temperature: {result.content}")
    
    await provider.close()
    
    # === Example 4: Caching with Provider ===
    print("\n" + "=" * 40)
    print("Example 4: Caching")
    print("=" * 40)
    
    # Create provider with filesystem cache
    provider = OpenAIProvider(
        model="gpt-5-nano",
        cache_backend="fs",
        cache_dir=cache_dir,
    )
    
    prompt = "What is the capital of France? One word only."
    
    # First call - not cached
    import time
    start = time.perf_counter()
    result1 = await provider.complete(
        prompt,
        cache_response=True,
        cache_collection="geography",
    )
    time1 = time.perf_counter() - start
    print(f"\nFirst call: {result1.content} ({time1*1000:.1f}ms)")
    
    # Second call - should hit cache
    start = time.perf_counter()
    result2 = await provider.complete(
        prompt,
        cache_response=True,
        cache_collection="geography",
    )
    time2 = time.perf_counter() - start
    print(f"Cached call: {result2.content} ({time2*1000:.1f}ms)")
    print(f"Speedup: {time1/time2:.1f}x faster")
    
    await provider.close()
    
    # === Example 5: Retry Logic ===
    print("\n" + "=" * 40)
    print("Example 5: Retry Logic")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    # The provider automatically retries on 429, 500, 502, 503, 504
    result = await provider.complete(
        "Say 'retry test successful'",
        attempts=3,      # Number of retry attempts
        backoff=0.5,     # Initial backoff delay (doubles each retry)
    )
    print(f"\nWith retry logic: {result.content}")
    
    await provider.close()
    
    # === Example 6: Error Handling ===
    print("\n" + "=" * 40)
    print("Example 6: Error Handling")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    result = await provider.complete("Hello!")
    
    # Check if request was successful
    if result.ok:
        print(f"\n✅ Success: {result.content}")
    else:
        print(f"\n❌ Error ({result.status}): {result.error}")
    
    # Access detailed response info
    print(f"\nResponse details:")
    print(f"  - Model: {result.model}")
    print(f"  - Finish reason: {result.finish_reason}")
    print(f"  - Has tool calls: {result.has_tool_calls}")
    if result.usage:
        print(f"  - Input tokens: {result.usage.input_tokens}")
        print(f"  - Output tokens: {result.usage.output_tokens}")
        print(f"  - Cost: ${result.usage.total_cost:.6f}")
    
    await provider.close()
    
    # === Example 7: Context Manager ===
    print("\n" + "=" * 40)
    print("Example 7: Context Manager")
    print("=" * 40)
    
    # Provider can be used as async context manager
    async with OpenAIProvider(model="gpt-5-nano") as provider:
        result = await provider.complete("Say 'context manager works'")
        print(f"\nUsing context manager: {result.content}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
