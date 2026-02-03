#!/usr/bin/env python3
"""
Example: Basic Completions with FS Cache

Demonstrates:
1. Basic completion request
2. File-system caching with dynamic collections
3. Cache hit detection
"""

import asyncio

# Add src to path for development
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient


async def main():
    # Create a temporary cache directory
    cache_dir = Path(tempfile.mkdtemp(prefix="llm_cache_"))
    print(f"📁 Cache directory: {cache_dir}")

    # Initialize client with FS cache backend
    client = OpenAIClient(
        model="gpt-5-nano",  # Cheapest model for testing
        cache_backend="fs",
        cache_dir=cache_dir,
        cache_collection="default_collection",
    )

    # --- Test 1: Basic completion ---
    print("\n" + "=" * 60)
    print("TEST 1: Basic Completion (no caching)")
    print("=" * 60)

    response = await client.get_response(
        messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
        cache_response=False,
    )

    print(f"Status: {response.get('status')}")
    print(f"Output: {response.get('output')}")
    print(f"Usage: {response.get('usage')}")

    # --- Test 2: Cached completion ---
    print("\n" + "=" * 60)
    print("TEST 2: Completion with Caching (first call)")
    print("=" * 60)

    prompt = "What is the capital of France? Reply with just the city name."

    response1 = await client.get_response(
        messages=[{"role": "user", "content": prompt}],
        cache_response=True,
        cache_collection="geography_cache",
    )

    print(f"Status: {response1.get('status')}")
    print(f"Output: {response1.get('output')}")
    print(f"Identifier: {response1.get('identifier')}")

    # --- Test 3: Cache hit ---
    print("\n" + "=" * 60)
    print("TEST 3: Same prompt again (should hit cache)")
    print("=" * 60)

    response2 = await client.get_response(
        messages=[{"role": "user", "content": prompt}],
        cache_response=True,
        cache_collection="geography_cache",
    )

    print(f"Status: {response2.get('status')}")
    print(f"Output: {response2.get('output')}")
    print(f"Same identifier? {response1.get('identifier') == response2.get('identifier')}")

    # --- Test 4: Different collection (no cache hit) ---
    print("\n" + "=" * 60)
    print("TEST 4: Same prompt, different collection (cache miss)")
    print("=" * 60)

    response3 = await client.get_response(
        messages=[{"role": "user", "content": prompt}],
        cache_response=True,
        cache_collection="other_collection",
    )

    print(f"Status: {response3.get('status')}")
    print(f"Output: {response3.get('output')}")

    # --- Show cache structure ---
    print("\n" + "=" * 60)
    print("CACHE STRUCTURE")
    print("=" * 60)

    for item in cache_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(cache_dir)
            print(f"  📄 {rel_path}")
        elif item.is_dir():
            rel_path = item.relative_to(cache_dir)
            print(f"  📁 {rel_path}/")

    # Cleanup
    await client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
